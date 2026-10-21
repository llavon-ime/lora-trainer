using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Llavon.Lora;
using TorchSharp;
using Xunit;

namespace Llavon.Lora.Tests;

public sealed class TorchRuntimeTests : IDisposable {
    private readonly DirectoryInfo root = Directory.CreateTempSubdirectory("llavon-runtime-");

    public void Dispose() => root.Delete(recursive: true);

    [Fact]
    public void BuildVersionsMatchTheReferencedTorchSharp() {
        // The GPU runtime has to be ABI compatible with the bundled LibTorchSharp.
        Assert.Equal(TorchRuntimes.LibTorchVersion, torch.NormalizeNuGetVersion(torch.__version__));
        Assert.Equal(
            typeof(torch).Assembly.GetName().Version!.ToString(3),
            torch.NormalizeNuGetVersion(TorchRuntimes.TorchSharpVersion));
    }

    [Fact]
    public void TheLibTorchPackageComesFromTheTorchSharpNuspec() {
        var dependencies = TorchRuntimes.ParseDependencies("""
            <?xml version="1.0" encoding="utf-8"?>
            <package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
              <metadata>
                <id>TorchSharp-cuda-windows</id>
                <dependencies>
                  <group targetFramework=".NETStandard2.0">
                    <dependency id="libtorch-cuda-12.8-win-x64" version="[2.10.0]" exclude="Build,Analyzers" />
                    <dependency id="TorchSharp" version="0.106.0" exclude="Build,Analyzers" />
                  </group>
                  <group targetFramework="net6.0">
                    <dependency id="libtorch-cuda-12.8-win-x64" version="[2.10.0]" />
                  </group>
                </dependencies>
              </metadata>
            </package>
            """);
        Assert.Equal(2, dependencies.Count);

        var runtime = new TorchRuntime(TorchRuntimes.Cuda, "win-x64", "TorchSharp-cuda-windows");
        var libtorch = TorchRuntimes.SelectLibTorch(dependencies, runtime);
        Assert.Equal("libtorch-cuda-12.8-win-x64", libtorch.Id);
        Assert.Equal("cuda12.8", TorchRuntimes.VariantOf(libtorch.Id, runtime.Rid));

        // A TorchSharp that moved to another libtorch must not be mixed with this build.
        var mismatched = dependencies.Select(dependency => dependency with { Version = "9.9.9" });
        Assert.Throws<InvalidDataException>(() => TorchRuntimes.SelectLibTorch(mismatched, runtime));
    }

    [Fact]
    public void CatalogEntriesProvideSizeAndHash() {
        var hash = Convert.ToBase64String(SHA512.HashData(new byte[] { 1 }));
        using var entry = JsonDocument.Parse(
            $$"""{"packageHash":"{{hash}}","packageHashAlgorithm":"SHA512","packageSize":42}""");
        var package = TorchRuntimes.ParseCatalogEntry("part1", "https://example.test/part1.nupkg", entry.RootElement);
        Assert.Equal(42, package.Size);
        Assert.Equal(hash, package.Sha512);

        using var other = JsonDocument.Parse(
            $$"""{"packageHash":"{{hash}}","packageHashAlgorithm":"SHA256","packageSize":42}""");
        Assert.Throws<InvalidDataException>(() =>
            TorchRuntimes.ParseCatalogEntry("part1", "https://example.test/part1.nupkg", other.RootElement));
    }

    [Fact]
    public void PackageHashesAreVerified() {
        var path = Path.Combine(root.FullName, "package.nupkg");
        File.WriteAllBytes(path, [1, 2, 3]);
        TorchRuntimes.VerifySha512(path, Convert.ToBase64String(SHA512.HashData(new byte[] { 1, 2, 3 })));
        Assert.Throws<InvalidDataException>(() =>
            TorchRuntimes.VerifySha512(path, Convert.ToBase64String(SHA512.HashData(new byte[] { 3 }))));
    }

    [Fact]
    public void OnlyNativeFilesForTheRidAreExtracted() {
        var package = CreatePackage(
            ("runtimes/win-x64/native/torch_cpu.dll", "cpu"),
            ("fragments/win-x64/native/torch_cuda.dll.fragment1", "tail"),
            ("runtimes/linux-x64/native/libtorch_cpu.so", "linux"),
            ("lib/netstandard2.0/_._/empty.txt", ""),
            ("LICENSE-LIBTORCH.txt", "license"));
        var output = root.CreateSubdirectory("out").FullName;

        TorchRuntimes.ExtractNatives(package, output, "win-x64");

        Assert.Equal(
            ["torch_cpu.dll", "torch_cuda.dll.fragment1"],
            Directory.GetFiles(output).Select(Path.GetFileName).Order(StringComparer.Ordinal));
        Assert.Equal("tail", File.ReadAllText(Path.Combine(output, "torch_cuda.dll.fragment1")));
    }

    [Fact]
    public void NestedNativePathsAreRejected() {
        var package = CreatePackage(("runtimes/win-x64/native/sub/evil.dll", "x"));
        Assert.Throws<InvalidDataException>(() =>
            TorchRuntimes.ExtractNatives(package, root.CreateSubdirectory("out").FullName, "win-x64"));
    }

    [Fact]
    public void FragmentsAreJoinedAndChecked() {
        var directory = root.FullName;
        var primary = Path.Combine(directory, "torch_cuda.dll");
        File.WriteAllText(primary, "head-");
        File.WriteAllText(primary + ".fragment1", "middle-");
        File.WriteAllText(primary + ".fragment2", "tail");
        File.WriteAllText(primary + ".sha", Sha256("head-middle-tail") + "\n");
        File.WriteAllText(Path.Combine(directory, "c10.dll"), "whole");

        TorchRuntimes.Restitch(directory);

        Assert.Equal("head-middle-tail", File.ReadAllText(primary));
        Assert.Equal(
            ["c10.dll", "torch_cuda.dll"],
            Directory.GetFiles(directory).Select(Path.GetFileName).Order(StringComparer.Ordinal));
    }

    [Fact]
    public void AnInterruptedJoinIsFinishedOnTheNextRun() {
        // The joined file replaced the primary but the fragments survived.
        var primary = Path.Combine(root.FullName, "torch_cuda.dll");
        File.WriteAllText(primary, "head-tail");
        File.WriteAllText(primary + ".fragment1", "tail");
        File.WriteAllText(primary + ".sha", Sha256("head-tail"));

        TorchRuntimes.Restitch(root.FullName);

        Assert.Equal("head-tail", File.ReadAllText(primary));
        Assert.Equal(["torch_cuda.dll"], Directory.GetFiles(root.FullName).Select(Path.GetFileName));
    }

    [Fact]
    public void CorruptFragmentsAreRejected() {
        var primary = Path.Combine(root.FullName, "torch_cuda.dll");
        File.WriteAllText(primary, "head-");
        File.WriteAllText(primary + ".fragment1", "tampered");
        File.WriteAllText(primary + ".sha", Sha256("head-tail"));

        Assert.Throws<InvalidDataException>(() => TorchRuntimes.Restitch(root.FullName));
        Assert.Equal("head-", File.ReadAllText(primary));
    }

    [Fact]
    public void FragmentsWithoutAPrimaryAreRejected() {
        File.WriteAllText(Path.Combine(root.FullName, "torch_cuda.dll.fragment1"), "tail");
        Assert.Throws<InvalidDataException>(() => TorchRuntimes.Restitch(root.FullName));
    }

    private string CreatePackage(params (string Name, string Content)[] entries) {
        var path = Path.Combine(root.FullName, $"{Guid.NewGuid():N}.nupkg");
        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        foreach (var (name, content) in entries) {
            using var writer = new StreamWriter(archive.CreateEntry(name).Open());
            writer.Write(content);
        }
        return path;
    }

    private static string Sha256(string text) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
}
