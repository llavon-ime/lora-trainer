using System.IO.Compression;
using System.Net;
using System.Net.Http.Headers;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Xml.Linq;

namespace Llavon.Lora;

// A GPU libtorch that is not bundled with the trainer but restored on demand
// from the NuGet packages TorchSharp itself builds against. The bundled CPU
// libtorch stays in place; TorchNativeLibraries preloads this one instead.
// TorchSharpPackage is the TorchSharp combination package whose dependencies
// name the libtorch packages to download.
public sealed record TorchRuntime(string Backend, string Rid, string TorchSharpPackage) {
    public string LibTorchVersion => TorchRuntimes.LibTorchVersion;
    public string Name => $"libtorch-{LibTorchVersion}-{Backend}-{Rid}";
}

public sealed record TorchRuntimeProgress(
    string Phase, string? Package, int Index, int Count, long Received, long Total);

internal sealed record NuGetDependency(string Id, string Version);

internal sealed record TorchRuntimePackage(string Id, string Url, long Size, string Sha512);

public static class TorchRuntimes {
    public const string RootVariable = "LLAVON_LORA_RUNTIME_ROOT";
    public const string SourceVariable = "LLAVON_LORA_NUGET_SOURCE";
    public const string Cuda = "cuda";

    private const string DefaultSource = "https://api.nuget.org/v3/index.json";
    private const string MarkerFile = "llavon-torch-runtime.json";

    private static readonly HttpClient Http = CreateHttpClient();

    public static IReadOnlyList<TorchRuntime> All { get; } = [
        new(Cuda, "win-x64", "TorchSharp-cuda-windows"),
        new(Cuda, "linux-x64", "TorchSharp-cuda-linux")
    ];

    // Both come from Directory.Build.props through assembly metadata, so the
    // packages always match the TorchSharp this build was compiled against.
    public static string TorchSharpVersion { get; } = BuildMetadata("TorchSharpVersion");
    public static string LibTorchVersion { get; } = BuildMetadata("LibTorchVersion");

    public static string? CurrentRid =>
        RuntimeInformation.OSArchitecture != Architecture.X64 ? null :
        OperatingSystem.IsWindows() ? "win-x64" :
        OperatingSystem.IsLinux() ? "linux-x64" :
        null;

    public static IEnumerable<TorchRuntime> Available =>
        All.Where(runtime => runtime.Rid == CurrentRid);

    public static TorchRuntime? Find(string backend) =>
        Available.FirstOrDefault(runtime => runtime.Backend == backend);

    public static string DefaultRoot {
        get {
            var configured = Environment.GetEnvironmentVariable(RootVariable);
            if (!string.IsNullOrWhiteSpace(configured))
                return Path.GetFullPath(configured);
            return Path.Combine(
                Environment.GetFolderPath(
                    Environment.SpecialFolder.LocalApplicationData,
                    Environment.SpecialFolderOption.Create),
                "llavon-lora", "runtimes");
        }
    }

    public static string DirectoryOf(TorchRuntime runtime, string? root = null) =>
        Path.Combine(Path.GetFullPath(root ?? DefaultRoot), runtime.Name);

    // The installed directory, or null when the runtime is missing or incomplete.
    public static string? Locate(TorchRuntime runtime, string? root = null) {
        var directory = DirectoryOf(runtime, root);
        return IsComplete(runtime, directory) ? directory : null;
    }

    // Reads the libtorch description of an installed runtime directory.
    public static string? Describe(string directory) {
        try {
            using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, MarkerFile)));
            var root = document.RootElement;
            return $"{root.GetProperty("libtorch").GetString()}+{root.GetProperty("variant").GetString()}";
        } catch (Exception exception) when (
            exception is IOException or UnauthorizedAccessException or JsonException or
                KeyNotFoundException or InvalidOperationException) {
            return null;
        }
    }

    public static async Task<string> InstallAsync(
        TorchRuntime runtime, string? root, IProgress<TorchRuntimeProgress>? progress,
        CancellationToken cancellationToken) {
        var parent = Path.GetFullPath(root ?? DefaultRoot);
        Directory.CreateDirectory(parent);
        using var installLock = AcquireLock(parent);

        var destination = Path.Combine(parent, runtime.Name);
        if (IsComplete(runtime, destination))
            return destination;

        progress?.Report(new TorchRuntimeProgress("resolve", runtime.TorchSharpPackage, 0, 0, 0, 0));
        var (libtorch, packages) = await ResolvePackagesAsync(runtime, cancellationToken);

        // Downloads resume package by package: a package is recorded as done
        // only after it was verified and extracted, so an interrupted install
        // keeps everything it finished. The record holds the package hash, so a
        // package that changed on the feed since is downloaded again.
        var staging = StagingOf(parent, runtime);
        var done = Path.Combine(staging, ".done");
        Directory.CreateDirectory(done);
        var total = packages.Sum(package => package.Size);
        var completed = 0L;
        for (var index = 0; index < packages.Count; ++index) {
            var package = packages[index];
            var marker = Path.Combine(done, package.Id);
            if (!File.Exists(marker) || File.ReadAllText(marker) != package.Sha512) {
                var download = Path.Combine(staging, ".download.nupkg");
                var offset = completed;
                await DownloadAsync(
                    package.Url, download, package.Size,
                    received => progress?.Report(new TorchRuntimeProgress(
                        "download", package.Id, index, packages.Count, offset + received, total)),
                    cancellationToken);
                progress?.Report(new TorchRuntimeProgress(
                    "verify", package.Id, index, packages.Count, offset + package.Size, total));
                try {
                    VerifySha512(download, package.Sha512);
                    ExtractNatives(download, staging, runtime.Rid);
                } finally {
                    File.Delete(download);
                }
                File.WriteAllText(marker, package.Sha512);
            }
            completed += package.Size;
            progress?.Report(new TorchRuntimeProgress(
                "package", package.Id, index, packages.Count, completed, total));
        }

        progress?.Report(new TorchRuntimeProgress("restitch", null, 0, packages.Count, total, total));
        Restitch(staging);
        Directory.Delete(done, recursive: true);
        if (!HasCoreLibraries(staging))
            throw new InvalidDataException($"the {libtorch} packages did not contain a complete libtorch");
        WriteMarker(runtime, staging, libtorch);

        // Whatever is left at the destination failed IsComplete above.
        if (Directory.Exists(destination))
            Directory.Delete(destination, recursive: true);
        Directory.Move(staging, destination);
        return destination;
    }

    public static bool Remove(TorchRuntime runtime, string? root = null) {
        var parent = Path.GetFullPath(root ?? DefaultRoot);
        if (!Directory.Exists(parent))
            return false;
        using var installLock = AcquireLock(parent);
        var removed = false;
        foreach (var directory in new[] { Path.Combine(parent, runtime.Name), StagingOf(parent, runtime) }) {
            if (!Directory.Exists(directory))
                continue;
            try {
                Directory.Delete(directory, recursive: true);
            } catch (Exception exception) when (exception is IOException or UnauthorizedAccessException) {
                throw new IOException(
                    $"cannot remove {directory}; close any llavon-lora process that uses it: {exception.Message}",
                    exception);
            }
            removed = true;
        }
        return removed;
    }

    // TorchSharp-cuda-windows depends on libtorch-cuda-<cuda>-win-x64, which
    // only lists its -partN packages; sizes and SHA-512 hashes come from the
    // NuGet catalog, which is what `dotnet restore` checks as well.
    private static async Task<(string LibTorch, IReadOnlyList<TorchRuntimePackage> Packages)> ResolvePackagesAsync(
        TorchRuntime runtime, CancellationToken cancellationToken) {
        var source = Environment.GetEnvironmentVariable(SourceVariable);
        using var index = await GetJsonAsync(string.IsNullOrWhiteSpace(source) ? DefaultSource : source,
            cancellationToken);
        var packageBase = FindResource(index.RootElement, "PackageBaseAddress/3.0.0");
        var registrations = FindResource(index.RootElement, "RegistrationsBaseUrl/3.6.0");

        var torchSharp = ParseDependencies(await GetStringAsync(
            NuspecUrl(packageBase, runtime.TorchSharpPackage, TorchSharpVersion), cancellationToken));
        var libtorch = SelectLibTorch(torchSharp, runtime);
        var parts = ParseDependencies(await GetStringAsync(
                NuspecUrl(packageBase, libtorch.Id, libtorch.Version), cancellationToken))
            .Where(dependency => dependency.Id.StartsWith($"{libtorch.Id}-part", StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (parts.Count == 0)
            throw new InvalidDataException($"{libtorch.Id} {libtorch.Version} lists no -part packages");

        var packages = new List<TorchRuntimePackage>(parts.Count);
        foreach (var part in parts) {
            var leafUrl = $"{registrations}{part.Id.ToLowerInvariant()}/{part.Version.ToLowerInvariant()}.json";
            using var leaf = await GetJsonAsync(leafUrl, cancellationToken);
            var content = RequireHttps(leaf.RootElement.GetProperty("packageContent").GetString());
            using var catalog = await GetJsonAsync(
                RequireHttps(leaf.RootElement.GetProperty("catalogEntry").GetString()), cancellationToken);
            packages.Add(ParseCatalogEntry(part.Id, content, catalog.RootElement));
        }
        return (libtorch.Id, packages);
    }

    internal static NuGetDependency SelectLibTorch(IEnumerable<NuGetDependency> dependencies, TorchRuntime runtime) {
        var libtorch = dependencies.FirstOrDefault(dependency =>
            dependency.Id.StartsWith($"libtorch-{runtime.Backend}-", StringComparison.OrdinalIgnoreCase) &&
            dependency.Id.EndsWith($"-{runtime.Rid}", StringComparison.OrdinalIgnoreCase));
        if (libtorch is null)
            throw new InvalidDataException(
                $"{runtime.TorchSharpPackage} {TorchSharpVersion} has no libtorch-{runtime.Backend} dependency " +
                $"for {runtime.Rid}");
        if (libtorch.Version != runtime.LibTorchVersion)
            throw new InvalidDataException(
                $"{runtime.TorchSharpPackage} {TorchSharpVersion} depends on {libtorch.Id} {libtorch.Version}, " +
                $"but this build bundles libtorch {runtime.LibTorchVersion}");
        return libtorch;
    }

    // Reads every <dependency> of a nuspec; exact ranges such as [2.10.0] are
    // reduced to their version.
    internal static IReadOnlyList<NuGetDependency> ParseDependencies(string nuspec) =>
        XDocument.Parse(nuspec)
            .Descendants()
            .Where(element => element.Name.LocalName == "dependency")
            .Select(element => new NuGetDependency(
                (string?)element.Attribute("id") ?? throw new InvalidDataException("nuspec dependency has no id"),
                ((string?)element.Attribute("version") ?? "").Trim('[', ']', '(', ')', ' ').Split(',')[0].Trim()))
            .DistinctBy(dependency => dependency.Id, StringComparer.OrdinalIgnoreCase)
            .ToList();

    internal static TorchRuntimePackage ParseCatalogEntry(string id, string url, JsonElement entry) {
        var algorithm = entry.GetProperty("packageHashAlgorithm").GetString();
        if (!string.Equals(algorithm, "SHA512", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"{id} uses unsupported package hash {algorithm}");
        var hash = entry.GetProperty("packageHash").GetString() ?? "";
        if (Convert.FromBase64String(hash).Length != SHA512.HashSizeInBytes)
            throw new InvalidDataException($"{id} has an invalid package hash");
        var size = entry.GetProperty("packageSize").GetInt64();
        if (size <= 0)
            throw new InvalidDataException($"{id} has an invalid package size");
        return new TorchRuntimePackage(id, url, size, hash);
    }

    private static string FindResource(JsonElement index, string type) {
        foreach (var resource in index.GetProperty("resources").EnumerateArray()) {
            if (resource.GetProperty("@type").GetString() == type) {
                var url = RequireHttps(resource.GetProperty("@id").GetString());
                return url.EndsWith('/') ? url : url + "/";
            }
        }
        throw new InvalidDataException($"the NuGet source has no {type} resource");
    }

    private static string NuspecUrl(string packageBase, string id, string version) {
        var lowerId = id.ToLowerInvariant();
        return $"{packageBase}{lowerId}/{version.ToLowerInvariant()}/{lowerId}.nuspec";
    }

    private static string RequireHttps(string? url) =>
        Uri.TryCreate(url, UriKind.Absolute, out var uri) && uri.Scheme == Uri.UriSchemeHttps
            ? url!
            : throw new InvalidDataException($"NuGet returned a non-HTTPS URL: {url}");

    private static async Task<string> GetStringAsync(string url, CancellationToken cancellationToken) =>
        await Http.GetStringAsync(RequireHttps(url), cancellationToken);

    private static async Task<JsonDocument> GetJsonAsync(string url, CancellationToken cancellationToken) {
        await using var stream = await Http.GetStreamAsync(RequireHttps(url), cancellationToken);
        return await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
    }

    private static string BuildMetadata(string key) =>
        typeof(TorchRuntimes).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(attribute => attribute.Key == key)?.Value
        ?? throw new InvalidOperationException($"the assembly was built without {key} metadata");

    // libtorch-cuda-12.8-win-x64 -> cuda12.8
    internal static string VariantOf(string libtorchPackage, string rid) {
        var variant = libtorchPackage.StartsWith("libtorch-", StringComparison.OrdinalIgnoreCase)
            ? libtorchPackage["libtorch-".Length..]
            : libtorchPackage;
        if (variant.EndsWith($"-{rid}", StringComparison.OrdinalIgnoreCase))
            variant = variant[..^(rid.Length + 1)];
        return variant.Replace("-", "", StringComparison.Ordinal);
    }

    internal static bool IsComplete(TorchRuntime runtime, string directory) {
        if (!HasCoreLibraries(directory))
            return false;
        try {
            using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, MarkerFile)));
            var marker = document.RootElement;
            return marker.GetProperty("schema").GetInt32() == 1 &&
                   marker.GetProperty("backend").GetString() == runtime.Backend &&
                   marker.GetProperty("rid").GetString() == runtime.Rid &&
                   marker.GetProperty("libtorch").GetString() == runtime.LibTorchVersion;
        } catch (Exception exception) when (
            exception is IOException or UnauthorizedAccessException or JsonException or
                KeyNotFoundException or InvalidOperationException) {
            return false;
        }
    }

    private static bool HasCoreLibraries(string directory) =>
        File.Exists(Path.Combine(directory, TorchNativeLibraries.CoreLibraryFileName)) &&
        File.Exists(Path.Combine(directory, TorchNativeLibraries.CudaLibraryFileName));

    // NuGet places native files under runtimes/<rid>/native; the TorchSharp
    // split packages put the remaining pieces of oversized files under
    // fragments/<rid>/native with a .fragmentN suffix.
    internal static void ExtractNatives(string package, string directory, string rid) {
        string[] prefixes = [$"runtimes/{rid}/native/", $"fragments/{rid}/native/"];
        using var archive = ZipFile.OpenRead(package);
        foreach (var entry in archive.Entries) {
            var name = Uri.UnescapeDataString(entry.FullName.Replace('\\', '/'));
            var prefix = prefixes.FirstOrDefault(candidate => name.StartsWith(candidate, StringComparison.Ordinal));
            if (prefix is null)
                continue;
            var file = name[prefix.Length..];
            if (file.Length == 0)
                continue;
            if (file is "." or ".." || file.Contains('/') || Path.GetFileName(file) != file ||
                file.StartsWith('.'))
                throw new InvalidDataException($"unexpected native file in {Path.GetFileName(package)}: {name}");
            entry.ExtractToFile(Path.Combine(directory, file), overwrite: true);
        }
    }

    // Joins every <file> that has a <file>.sha with its <file>.fragmentN
    // pieces and checks the SHA-256 the packages ship for the joined file.
    internal static void Restitch(string directory) {
        // The pattern alone would also match longer extensions on Windows.
        foreach (var shaFile in Directory.GetFiles(directory, "*.sha")
                     .Where(path => path.EndsWith(".sha", StringComparison.Ordinal))) {
            var primary = shaFile[..^".sha".Length];
            if (!File.Exists(primary))
                throw new InvalidDataException($"missing primary file for {Path.GetFileName(shaFile)}");
            var expected = File.ReadAllText(shaFile).Trim();
            var fragments = new List<string>();
            for (var number = 1; File.Exists($"{primary}.fragment{number}"); ++number)
                fragments.Add($"{primary}.fragment{number}");

            // A previous run may have joined the file and stopped before it
            // removed the fragments.
            if (!Sha256Equals(HashFiles([primary], null), expected)) {
                var joined = primary + ".joined";
                string actual;
                using (var output = new FileStream(joined, FileMode.Create, FileAccess.Write, FileShare.None))
                    actual = HashFiles([primary, .. fragments], output);
                if (!Sha256Equals(actual, expected)) {
                    File.Delete(joined);
                    throw new InvalidDataException(
                        $"{Path.GetFileName(primary)} does not match its SHA-256 after joining its fragments");
                }
                File.Move(joined, primary, overwrite: true);
            }
            foreach (var fragment in fragments)
                File.Delete(fragment);
            File.Delete(shaFile);
        }

        var orphan = Directory.EnumerateFiles(directory, "*.fragment*").FirstOrDefault();
        if (orphan is not null)
            throw new InvalidDataException($"fragment without a primary file: {Path.GetFileName(orphan)}");
    }

    private static string HashFiles(IEnumerable<string> paths, Stream? output) {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[1 << 20];
        foreach (var path in paths) {
            using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                1, FileOptions.SequentialScan);
            int read;
            while ((read = input.Read(buffer)) > 0) {
                hash.AppendData(buffer.AsSpan(0, read));
                output?.Write(buffer.AsSpan(0, read));
            }
        }
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    private static bool Sha256Equals(string actual, string expected) =>
        actual.Equals(expected, StringComparison.OrdinalIgnoreCase);

    internal static void VerifySha512(string path, string expected) {
        using var stream = File.OpenRead(path);
        var actual = Convert.ToBase64String(SHA512.HashData(stream));
        if (actual != expected)
            throw new InvalidDataException(
                $"package SHA-512 mismatch for {Path.GetFileName(path)}: expected {expected}, got {actual}");
    }

    private static async Task DownloadAsync(
        string url, string destination, long size, Action<long> report, CancellationToken cancellationToken) {
        using var response = await Http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();
        if (response.Content.Headers.ContentLength is { } length && length != size)
            throw new InvalidDataException($"{url} has {length} bytes; expected {size}");

        await using var input = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var output = new FileStream(
            destination, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        var buffer = new byte[1 << 20];
        var received = 0L;
        var reported = 0L;
        int read;
        while ((read = await input.ReadAsync(buffer, cancellationToken)) > 0) {
            received += read;
            if (received > size)
                throw new InvalidDataException($"{url} is larger than {size} bytes");
            await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
            if (received - reported >= 8 << 20 || received == size) {
                reported = received;
                report(received);
            }
        }
        if (received != size)
            throw new InvalidDataException($"{url} ended after {received} of {size} bytes");
    }

    internal static void WriteMarker(TorchRuntime runtime, string directory, string libtorchPackage) =>
        File.WriteAllText(Path.Combine(directory, MarkerFile), JsonSerializer.Serialize(new {
            schema = 1,
            backend = runtime.Backend,
            variant = VariantOf(libtorchPackage, runtime.Rid),
            rid = runtime.Rid,
            libtorch = runtime.LibTorchVersion,
            package = libtorchPackage
        }));

    private static string StagingOf(string root, TorchRuntime runtime) =>
        Path.Combine(root, $".partial-{runtime.Name}");

    private static FileStream AcquireLock(string root) {
        try {
            return new FileStream(Path.Combine(root, ".lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite,
                FileShare.None, 1, FileOptions.DeleteOnClose);
        } catch (IOException exception) {
            throw new IOException($"another llavon-lora process is changing the runtimes in {root}", exception);
        }
    }

    private static HttpClient CreateHttpClient() {
        // Each package is a few hundred megabytes; cancellation, not a
        // timeout, ends a slow transfer. The registration documents are gzip
        // encoded.
        var client = new HttpClient(new SocketsHttpHandler {
            AutomaticDecompression = DecompressionMethods.All
        }) { Timeout = Timeout.InfiniteTimeSpan };
        client.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("llavon-lora", "1"));
        return client;
    }
}
