using System.IO.Compression;
using System.Security.Cryptography;

namespace Llavon.Lora;

// A libtorch build the trainer can fetch into the per-user cache. TorchSharp
// publishes no ROCm package and the CUDA package is split across many NuGet
// packages, so the ROCm backend uses PyTorch's official ROCm archive and the
// CUDA backend uses its official CUDA archive; both match LibTorch 2.10.
public sealed record LibTorchBackend(
    string Name,
    string DirectoryName,
    string Version,
    string Url,
    string Sha256,
    long Size,
    IReadOnlyList<string> KernelDirectories);

public static class LibTorchDistribution {
    public static readonly LibTorchBackend Rocm = new(
        "rocm",
        "libtorch-rocm7.0-2.10.0",
        "2.10.0+rocm7.0",
        "https://download.pytorch.org/libtorch/rocm7.0/libtorch-shared-with-deps-2.10.0%2Brocm7.0.zip",
        "d8561904e2cee6af1be8083ede61cef82fdeb2586697bcff66488bea1539e0e1",
        4850819106,
        ["rocblas/library/", "hipblaslt/library/", "hipsparselt/library/"]);

    public static readonly LibTorchBackend Cuda = new(
        "cuda",
        "libtorch-cuda12.8-2.10.0",
        "2.10.0+cu128",
        "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip",
        "429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9",
        3917843662,
        []);

    public static IReadOnlyList<LibTorchBackend> Backends { get; } = [Rocm, Cuda];

    public static LibTorchBackend ForName(string name) =>
        Backends.FirstOrDefault(backend => string.Equals(backend.Name, name, StringComparison.Ordinal))
        ?? throw new ArgumentException($"--backend must be {string.Join(", ", Backends.Select(backend => backend.Name))}");

    // The cache directory also used by scripts/fetch-libtorch-rocm.sh.
    public static string InstallDirectory(LibTorchBackend backend) =>
        Path.Combine(TorchNativeLibraries.CacheRoot, backend.DirectoryName);

    public static string LibraryDirectory(LibTorchBackend backend) =>
        Path.Combine(InstallDirectory(backend), "libtorch", "lib");

    // The trainer needs the shared libraries and the kernel databases that go
    // with them; static libraries, headers and the flash attention images used
    // by scaled dot product attention are skipped.
    internal static bool IsNeededEntry(LibTorchBackend backend, string entry) {
        if (string.Equals(entry, "libtorch/build-version", StringComparison.Ordinal))
            return true;
        const string prefix = "libtorch/lib/";
        if (!entry.StartsWith(prefix, StringComparison.Ordinal))
            return false;
        var relative = entry[prefix.Length..];
        if (relative.Length == 0 || relative.EndsWith('/'))
            return false;
        if (relative.IndexOf('/') < 0)
            return relative.Contains(".so", StringComparison.Ordinal);
        foreach (var directory in backend.KernelDirectories)
            if (relative.StartsWith(directory, StringComparison.Ordinal))
                return true;
        return false;
    }

    // Downloads and extracts the archive unless it is already installed; the
    // archive itself is removed afterwards.
    public static string Fetch(LibTorchBackend backend, string? installDirectory, bool force,
                               Action<string>? log = null) {
        if (!OperatingSystem.IsLinux())
            throw new PlatformNotSupportedException("the fetched libtorch is available for Linux only");
        var root = Path.GetFullPath(installDirectory ?? InstallDirectory(backend));
        var libraryDirectory = Path.Combine(root, "libtorch", "lib");
        if (IsInstalled(backend, libraryDirectory) && !force) {
            log?.Invoke($"libtorch {backend.Version} is already installed at {libraryDirectory}");
            return libraryDirectory;
        }

        Directory.CreateDirectory(root);
        var archive = Path.Combine(root, $"libtorch-{backend.Version}.zip.partial");
        try {
            Download(backend, archive, log);
            Extract(backend, archive, root, log);
        } finally {
            if (File.Exists(archive))
                File.Delete(archive);
        }
        if (!IsInstalled(backend, libraryDirectory))
            throw new InvalidDataException($"the libtorch archive did not install into {libraryDirectory}");
        return libraryDirectory;
    }

    // A complete installation carries the backend library and the kernel data
    // the trainer needs; an interrupted extraction must not count as installed.
    internal static bool IsInstalled(LibTorchBackend backend, string libraryDirectory) {
        var library = backend.Name == "rocm" ? "libtorch_hip" : "libtorch_cuda";
        if (!File.Exists(Path.Combine(libraryDirectory, TorchNativeLibraries.LibraryFileName(library))))
            return false;
        foreach (var directory in backend.KernelDirectories) {
            var path = Path.Combine(libraryDirectory, directory.TrimEnd('/'));
            if (!Directory.Exists(path) || !Directory.EnumerateFileSystemEntries(path).Any())
                return false;
        }
        return true;
    }

    private static void Download(LibTorchBackend backend, string path, Action<string>? log) {
        using var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        using var response = client.GetAsync(backend.Url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();
        if (response.Content.Headers.ContentLength is { } length && length != backend.Size)
            throw new InvalidDataException(
                $"the libtorch archive has {length} bytes, expected {backend.Size}");
        using var source = response.Content.ReadAsStream();
        using var destination = File.Create(path);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[4 << 20];
        long total = 0;
        long reported = 0;
        int read;
        while ((read = source.Read(buffer, 0, buffer.Length)) > 0) {
            destination.Write(buffer, 0, read);
            hash.AppendData(buffer, 0, read);
            total += read;
            if (log is not null && total - reported >= 512L * 1024 * 1024) {
                reported = total;
                log($"downloading libtorch {backend.Version}: " +
                    $"{total / (1024 * 1024)} MB of {backend.Size / (1024 * 1024)} MB");
            }
        }
        destination.Flush();
        var actual = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
        if (!string.Equals(actual, backend.Sha256, StringComparison.Ordinal))
            throw new InvalidDataException(
                $"the libtorch archive SHA-256 is {actual}, expected {backend.Sha256}");
        log?.Invoke($"downloaded libtorch {backend.Version} ({total / (1024 * 1024)} MB)");
    }

    private static void Extract(LibTorchBackend backend, string archive, string root, Action<string>? log) {
        using var zip = ZipFile.OpenRead(archive);
        var entries = zip.Entries.Where(entry => IsNeededEntry(backend, entry.FullName)).ToArray();
        if (entries.Length == 0)
            throw new InvalidDataException("the libtorch archive contains no libraries");
        long extracted = 0;
        var index = 0;
        foreach (var entry in entries) {
            var destination = Path.Combine(root, entry.FullName.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            entry.ExtractToFile(destination, overwrite: true);
            extracted += entry.Length;
            if (log is not null && ++index % 500 == 0)
                log($"extracting libtorch: {index}/{entries.Length}");
        }
        log?.Invoke($"extracted libtorch to {Path.Combine(root, "libtorch", "lib")} " +
                    $"({extracted / (1024 * 1024)} MB)");
    }
}
