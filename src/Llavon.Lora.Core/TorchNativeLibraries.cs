using System.Runtime.InteropServices;

namespace Llavon.Lora;

// TorchSharp resolves its native backend lazily, so an operator-provided libtorch
// can be loaded here first and everything TorchSharp loads afterwards resolves
// against it. That is how the released CPU build can train on an AMD GPU: point
// LLAVON_LORA_TORCH_LIB (or --torch-lib-dir) at a libtorch built for ROCm, and
// the HIP libraries take part through the CUDA entry points TorchSharp uses,
// exactly as PyTorch exposes ROCm through torch.cuda.
public static class TorchNativeLibraries {
    public const string DirectoryVariable = "LLAVON_LORA_TORCH_LIB";

    private static readonly object gate = new();
    private static string? initializedDirectory;

    public static string? InitializedDirectory {
        get { lock (gate) return initializedDirectory; }
    }

    public static void Initialize(string? requestedDirectory) {
        lock (gate) {
            if (initializedDirectory is not null) {
                // A later call only has to agree when it names a directory itself.
                if (requestedDirectory is null || IsSameDirectory(initializedDirectory, requestedDirectory))
                    return;
                throw new InvalidOperationException(
                    $"libtorch was already loaded from {initializedDirectory}; cannot switch to " +
                    $"{Path.GetFullPath(requestedDirectory)}");
            }
            var directory = ResolveDirectory(requestedDirectory);
            if (directory is null)
                return;
            Preload(directory);
            initializedDirectory = directory;
        }
    }

    private static bool IsSameDirectory(string left, string right) =>
        string.Equals(left, right, StringComparison.Ordinal) ||
        string.Equals(left, Path.GetFullPath(right), StringComparison.Ordinal);

    // The command line wins, then the environment variable, then the libraries
    // bundled next to the application (ROCm builds ship libtorch_hip.so instead
    // of libtorch_cuda.so, so they must be loaded explicitly).
    public static string? ResolveDirectory(string? requestedDirectory) =>
        ResolveDirectory(requestedDirectory, AppContext.BaseDirectory);

    internal static string? ResolveDirectory(string? requestedDirectory, string applicationDirectory) {
        var application = Path.GetFullPath(applicationDirectory);
        if (!string.IsNullOrWhiteSpace(requestedDirectory))
            return ValidateAlternativeDirectory(requestedDirectory, application);
        var fromEnvironment = Environment.GetEnvironmentVariable(DirectoryVariable);
        if (!string.IsNullOrWhiteSpace(fromEnvironment))
            return ValidateAlternativeDirectory(fromEnvironment, application);
        // ROCm builds ship libtorch_hip.so and must load their HIP libraries
        // explicitly, whether they sit next to the application or, for
        // single-file builds, in the extraction directory.
        foreach (var directory in SearchDirectories(application))
            if (HasHipLibraries(directory))
                return directory;
        // A build without bundled libraries picks up the libtorch installed by
        // `llavon-lora fetch-libtorch`; builds that ship their own must not load
        // a second one.
        if (BundlesLibTorch(application))
            return null;
        return CachedLibTorchDirectory();
    }

    // The fetched libtorch, in the order the backends are declared.
    private static string? CachedLibTorchDirectory() {
        foreach (var backend in LibTorchDistribution.Backends) {
            var directory = LibTorchDistribution.LibraryDirectory(backend);
            if (File.Exists(Path.Combine(directory, LibraryFileName("libtorch_cpu"))))
                return directory;
        }
        return null;
    }

    // Whether the application carries its own libtorch.
    public static bool BundlesLibTorch() => BundlesLibTorch(AppContext.BaseDirectory);

    // Whether the application carries the ROCm libraries itself.
    public static bool BundlesHipLibraries() =>
        SearchDirectories(AppContext.BaseDirectory).Any(HasHipLibraries);

    private static bool BundlesLibTorch(string application) =>
        SearchDirectories(application).Any(directory => File.Exists(
            Path.Combine(directory, LibraryFileName("libtorch_cpu"))));

    // Directories the runtime searches for native libraries. Single-file builds
    // extract their bundled libraries elsewhere, so the application directory
    // alone is not enough.
    private static IEnumerable<string> SearchDirectories(string application) {
        yield return application;
        if (!IsSameDirectory(application, AppContext.BaseDirectory))
            yield break;
        if (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") is not string directories)
            yield break;
        foreach (var directory in directories.Split(
                     [Path.PathSeparator, ';'], StringSplitOptions.RemoveEmptyEntries))
            if (!string.IsNullOrWhiteSpace(directory))
                yield return Path.GetFullPath(directory);
    }

    // Per-user cache root shared with scripts/fetch-libtorch-rocm.sh.
    public static string CacheRoot {
        get {
            var xdg = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
            if (!string.IsNullOrWhiteSpace(xdg))
                return Path.Combine(xdg, "llavon-lora");
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return OperatingSystem.IsMacOS()
                ? Path.Combine(home, "Library", "Caches", "llavon-lora")
                : Path.Combine(home, ".cache", "llavon-lora");
        }
    }

    public static string LibraryFileName(string stem) =>
        OperatingSystem.IsWindows() ? $"{stem}.dll" :
        OperatingSystem.IsMacOS() ? $"{stem}.dylib" :
        $"{stem}.so";

    private static string ValidateAlternativeDirectory(string directory, string application) {
        var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory));
        if (!Directory.Exists(full))
            throw new DirectoryNotFoundException($"the libtorch directory does not exist: {full}");
        var cpu = Path.Combine(full, LibraryFileName("libtorch_cpu"));
        if (!File.Exists(cpu))
            throw new FileNotFoundException(
                $"the libtorch directory has no {Path.GetFileName(cpu)}: {full}", cpu);
        // Two libtorch copies in one process abort on duplicate kernel
        // registration, so a build that ships its own copy cannot switch away
        // from it.
        if (!IsSameDirectory(application, full) && BundlesLibTorch(application))
            throw new InvalidOperationException(
                $"this build bundles a libtorch in {Path.TrimEndingDirectorySeparator(application)} and cannot load a " +
                "second one; build with -p:TorchBackend=rocm-linux to use a ROCm libtorch, or use a build that does " +
                $"not bundle a libtorch together with {DirectoryVariable} or --torch-lib-dir");
        return full;
    }

    private static bool HasHipLibraries(string directory) =>
        File.Exists(Path.Combine(directory, LibraryFileName("libtorch_hip")));

    private static void Preload(string directory) {
        // The CPU libraries go first so that the alternative libtorch_cpu wins
        // over the copy next to the application, and so that dependencies of the
        // libraries loaded below match it by soname instead of loading a second
        // libtorch.
        string[] stems = ["libc10", "libtorch_cpu", "libtorch", "libc10_hip", "libtorch_hip", "libtorch_cuda"];
        foreach (var stem in stems) {
            var path = Path.Combine(directory, LibraryFileName(stem));
            if (!File.Exists(path))
                continue;
            try {
                NativeLibrary.Load(path);
            } catch (Exception exception) when (exception is not OutOfMemoryException) {
                throw new InvalidOperationException(
                    $"failed to load {Path.GetFileName(path)} from {directory}: {exception.Message}", exception);
            }
        }
    }
}
