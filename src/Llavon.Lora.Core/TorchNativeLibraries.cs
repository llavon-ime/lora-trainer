using System.Runtime.InteropServices;

namespace Llavon.Lora;

// TorchSharp resolves its native backend lazily, so an operator-provided libtorch
// can be loaded here first and everything TorchSharp loads afterwards resolves
// against it. That is how the released CPU build trains on a GPU: point
// LLAVON_LORA_TORCH_LIB (or --torch-lib-dir) at a libtorch built for ROCm, or
// install a GPU runtime with `llavon-lora runtime install`. The HIP libraries
// take part through the CUDA entry points TorchSharp uses, exactly as PyTorch
// exposes ROCm through torch.cuda.
public static class TorchNativeLibraries {
    public const string DirectoryVariable = "LLAVON_LORA_TORCH_LIB";

    // The Windows and Linux CPU builds ship their libtorch here instead of
    // beside the application; see Llavon.Lora.Cli.csproj.
    public const string BundledDirectoryName = "libtorch";

    private static readonly object gate = new();
    private static string? initializedDirectory;

    public static string? InitializedDirectory {
        get { lock (gate) return initializedDirectory; }
    }

    // Windows libtorch drops the lib prefix that the Unix builds use.
    public static string CoreLibraryFileName =>
        LibraryFileName(OperatingSystem.IsWindows() ? "torch_cpu" : "libtorch_cpu");

    public static string CudaLibraryFileName =>
        LibraryFileName(OperatingSystem.IsWindows() ? "torch_cuda" : "libtorch_cuda");

    // The device decides whether an installed GPU runtime is used: cuda needs
    // one unless the build bundles a GPU libtorch, auto takes one when present,
    // and every other device keeps the bundled libtorch. Call this before
    // anything touches TorchSharp.
    public static void Initialize(string? requestedDirectory, string device = DeviceSelection.Cpu,
        string? runtimeRoot = null) {
        lock (gate) {
            if (initializedDirectory is not null) {
                // A later call only has to agree when it names a directory itself.
                if (requestedDirectory is null || IsSameDirectory(initializedDirectory, requestedDirectory))
                    return;
                throw new InvalidOperationException(
                    $"libtorch was already loaded from {initializedDirectory}; cannot switch to " +
                    $"{Path.GetFullPath(requestedDirectory)}");
            }
            var directory = ResolveDirectory(requestedDirectory, device, runtimeRoot);
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
    // of libtorch_cuda.so, so they must be loaded explicitly), then an installed
    // GPU runtime, then the libtorch subdirectory of the Windows CPU build.
    public static string? ResolveDirectory(string? requestedDirectory, string device = DeviceSelection.Cpu,
        string? runtimeRoot = null) =>
        ResolveDirectoryIn(requestedDirectory, AppContext.BaseDirectory, device, runtimeRoot,
            NativeSearchDirectories());

    // probedDirectories are where .NET looks, by full path, when TorchSharp
    // loads a library by name; a libtorch there is loaded whatever was
    // preloaded, so it decides what can be used.
    internal static string? ResolveDirectoryIn(string? requestedDirectory, string applicationDirectory,
        string device = DeviceSelection.Cpu, string? runtimeRoot = null,
        IEnumerable<string>? probedDirectories = null) {
        var application = Path.GetFullPath(applicationDirectory);
        var probed = (probedDirectories ?? [])
            .Select(directory => Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory)))
            .Append(Path.TrimEndingDirectorySeparator(application))
            .Distinct(StringComparer.Ordinal)
            .ToList();
        if (!string.IsNullOrWhiteSpace(requestedDirectory))
            return ValidateAlternativeDirectory(requestedDirectory, probed);
        var fromEnvironment = Environment.GetEnvironmentVariable(DirectoryVariable);
        if (!string.IsNullOrWhiteSpace(fromEnvironment))
            return ValidateAlternativeDirectory(fromEnvironment, probed);
        if (HasHipLibraries(application))
            return application;
        var runtime = device is DeviceSelection.Cuda or DeviceSelection.Auto
            ? ResolveCudaRuntime(device, probed, runtimeRoot)
            : null;
        if (runtime is not null)
            return runtime;
        var bundled = Path.Combine(application, BundledDirectoryName);
        return FindLibrary(probed, CoreLibraryFileName) is null &&
               File.Exists(Path.Combine(bundled, CoreLibraryFileName))
            ? bundled
            : null;
    }

    private static string? ResolveCudaRuntime(string device, IReadOnlyList<string> probed, string? runtimeRoot) {
        if (FindLibrary(probed, CudaLibraryFileName) is not null)
            return null;
        var runtime = TorchRuntimes.Find(TorchRuntimes.Cuda);
        if (runtime is null)
            return null;
        var installed = TorchRuntimes.Locate(runtime, runtimeRoot);
        if (installed is not null)
            return ValidateAlternativeDirectory(installed, probed);
        if (device == DeviceSelection.Cuda)
            throw new InvalidOperationException(
                $"CUDA was requested but the {runtime.Name} runtime is not installed in " +
                $"{Path.GetDirectoryName(TorchRuntimes.DirectoryOf(runtime, runtimeRoot))}; " +
                "run `llavon-lora runtime install --backend cuda` first");
        return null;
    }

    private static IEnumerable<string> NativeSearchDirectories() =>
        (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") as string)?
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) ?? [];

    private static string? FindLibrary(IEnumerable<string> directories, string fileName) =>
        directories.FirstOrDefault(directory => File.Exists(Path.Combine(directory, fileName)));

    public static string LibraryFileName(string stem) =>
        OperatingSystem.IsWindows() ? $"{stem}.dll" :
        OperatingSystem.IsMacOS() ? $"{stem}.dylib" :
        $"{stem}.so";

    private static string ValidateAlternativeDirectory(string directory, IReadOnlyList<string> probed) {
        var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory));
        if (!Directory.Exists(full))
            throw new DirectoryNotFoundException($"the libtorch directory does not exist: {full}");
        var cpu = Path.Combine(full, CoreLibraryFileName);
        if (!File.Exists(cpu))
            throw new FileNotFoundException(
                $"the libtorch directory has no {Path.GetFileName(cpu)}: {full}", cpu);
        // Two libtorch copies in one process abort on duplicate kernel
        // registration, so a build that ships its own copy where .NET probes
        // cannot switch away from it: TorchSharp loads that copy by full path
        // whatever was preloaded.
        var bundled = FindLibrary(probed.Where(candidate => !IsSameDirectory(candidate, full)), CoreLibraryFileName);
        if (bundled is not null)
            throw new InvalidOperationException(
                $"this build bundles a libtorch in {bundled} and cannot load a " +
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
        string[] stems = OperatingSystem.IsWindows()
            ? ["c10", "torch_cpu", "c10_cuda", "torch_cuda", "torch"]
            : ["libc10", "libtorch_cpu", "libc10_hip", "libtorch_hip", "libc10_cuda", "libtorch_cuda", "libtorch"];
        if (OperatingSystem.IsWindows())
            AddWindowsSearchDirectory(directory);
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

    // Loading by full path resolves static imports next to each library, and
    // TorchSharp's later loads by name bind to these modules. cuDNN and NVRTC
    // however load their siblings by name at run time, which only searches the
    // application directory, the system and PATH. Keep the selected directory
    // in PATH rather than SetDllDirectory: the latter replaces Windows' single
    // process-wide DLL directory and prevents LLamaSharp from resolving the
    // ggml libraries beside its dynamically selected AVX backend.
    private static void AddWindowsSearchDirectory(string directory) {
        const string variable = "PATH";
        var current = Environment.GetEnvironmentVariable(variable);
        var updated = PrependWindowsSearchPath(current, directory);
        if (!string.Equals(current, updated, StringComparison.Ordinal))
            Environment.SetEnvironmentVariable(variable, updated);
    }

    internal static string PrependWindowsSearchPath(string? searchPath, string directory) {
        var full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(directory));
        var entries = (searchPath ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (entries.Any(entry => IsSameDirectory(entry, full)))
            return searchPath ?? "";
        return string.IsNullOrEmpty(searchPath) ? full : $"{full}{Path.PathSeparator}{searchPath}";
    }
}
