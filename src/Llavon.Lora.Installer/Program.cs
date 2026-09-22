using System.IO.Compression;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Win32;

return await InstallerProgram.RunAsync(args);

internal static class InstallerProgram {
    private const string DefaultManifestUrl =
        "https://github.com/llavon-ime/lora-trainer/releases/download/latest/latest.json";
    private const string DefaultAsset = "win-x64-cpu";
    private const string RegistryPath = @"Software\Llavon IME\LoRA Trainer";

    private static readonly HttpClient Http = CreateHttpClient();

    internal static async Task<int> RunAsync(string[] args) {
        try {
            if (args.Length == 0 || args[0] is "--help" or "-h") {
                PrintHelp();
                return args.Length == 0 ? 2 : 0;
            }

            var options = Options.Parse(args[1..]);
            return args[0] switch {
                "status" => await StatusAsync(options),
                "install" => await InstallAsync(options),
                "uninstall" => Uninstall(options),
                _ => throw new ArgumentException($"unknown command: {args[0]}")
            };
        } catch (Exception exception) when (exception is not OutOfMemoryException) {
            Console.Error.WriteLine($"error: {exception.Message}");
            return 2;
        }
    }

    private static void PrintHelp() => Console.WriteLine("""
        llavon-lora-installer: installs the optional Llavon LoRA Trainer

        Usage:
          llavon-lora-installer status [options]
          llavon-lora-installer install [options]
          llavon-lora-installer uninstall [options]

        Options:
          --manifest-url URL   Release manifest (default: rolling latest)
          --install-dir DIR    Installation directory
          --asset ID           Release asset key (default: win-x64-cpu)
        """);

    private static async Task<int> StatusAsync(Options options) {
        var local = LoadInstalledManifest(options.InstallDirectory);
        var remote = await LoadRemoteManifestAsync(options.ManifestUrl);
        var state = GetState(local, remote, options.InstallDirectory);
        WriteStatus(state, local?.Version, remote.Version);
        return 0;
    }

    private static async Task<int> InstallAsync(Options options) {
        var remote = await LoadRemoteManifestAsync(options.ManifestUrl);
        var local = LoadInstalledManifest(options.InstallDirectory);
        var state = GetState(local, remote, options.InstallDirectory);
        WriteStatus(state, local?.Version, remote.Version);

        if (state is InstallState.Current or InstallState.Newer) {
            if (local is not null)
                WriteRegistry(local.Version, local.TrainerApi, options.InstallDirectory);
            return 0;
        }

        if (!remote.Assets.TryGetValue(options.Asset, out var asset))
            throw new InvalidDataException($"release has no asset named {options.Asset}");
        ValidateAsset(asset);

        var parent = Path.GetDirectoryName(options.InstallDirectory)
            ?? throw new InvalidOperationException("installation directory has no parent");
        Directory.CreateDirectory(parent);

        var nonce = Guid.NewGuid().ToString("N");
        var downloadPath = Path.Combine(Path.GetTempPath(), $"llavon-lora-{nonce}.zip");
        var stagingPath = Path.Combine(parent, $".lora-staging-{nonce}");
        var backupPath = Path.Combine(parent, $".lora-backup-{nonce}");

        try {
            await DownloadAsync(asset, downloadPath);
            VerifySha256(downloadPath, asset.Sha256);
            ZipFile.ExtractToDirectory(downloadPath, stagingPath);

            var stagedManifest = LoadInstalledManifest(stagingPath)
                ?? throw new InvalidDataException("artifact has no llavon-lora-manifest.json");
            if (!CalVersion.Parse(stagedManifest.Version).Equals(CalVersion.Parse(remote.Version)))
                throw new InvalidDataException("artifact version does not match the release manifest");
            if (stagedManifest.TrainerApi != remote.TrainerApi)
                throw new InvalidDataException("artifact trainerApi does not match the release manifest");
            if (!File.Exists(Path.Combine(stagingPath, "llavon-lora.exe")))
                throw new InvalidDataException("artifact has no llavon-lora.exe");

            ReplaceDirectory(options.InstallDirectory, stagingPath, backupPath);
            WriteRegistry(remote.Version, remote.TrainerApi, options.InstallDirectory);
            WriteStatus(InstallState.Current, remote.Version, remote.Version);
            return 0;
        } finally {
            TryDeleteFile(downloadPath);
            TryDeleteDirectory(stagingPath);
            TryDeleteDirectory(backupPath);
        }
    }

    private static int Uninstall(Options options) {
        if (Directory.Exists(options.InstallDirectory))
            Directory.Delete(options.InstallDirectory, recursive: true);
        using var key = Registry.LocalMachine.OpenSubKey(@"Software\Llavon IME", writable: true);
        key?.DeleteSubKeyTree("LoRA Trainer", throwOnMissingSubKey: false);
        return 0;
    }

    private static HttpClient CreateHttpClient() {
        var client = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
        client.DefaultRequestHeaders.UserAgent.Add(
            new ProductInfoHeaderValue("llavon-lora-installer", "1"));
        return client;
    }

    private static async Task<LatestManifest> LoadRemoteManifestAsync(string url) {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps)
            throw new ArgumentException("manifest URL must use HTTPS");

        await using var stream = await Http.GetStreamAsync(uri);
        var manifest = await JsonSerializer.DeserializeAsync(
            stream, InstallerJsonContext.Default.LatestManifest);
        if (manifest is null || manifest.Schema != 1)
            throw new InvalidDataException("unsupported release manifest");
        _ = CalVersion.Parse(manifest.Version);
        if (manifest.TrainerApi <= 0) throw new InvalidDataException("invalid trainerApi");
        if (manifest.Assets is null) throw new InvalidDataException("release has no assets");
        return manifest;
    }

    private static InstalledManifest? LoadInstalledManifest(string directory) {
        var path = Path.Combine(directory, "llavon-lora-manifest.json");
        if (!File.Exists(path)) return null;
        try {
            using var stream = File.OpenRead(path);
            var manifest = JsonSerializer.Deserialize(
                stream, InstallerJsonContext.Default.InstalledManifest);
            if (manifest is null || manifest.Schema != 1) return null;
            _ = CalVersion.Parse(manifest.Version);
            return manifest;
        } catch (Exception exception) when (
            exception is IOException or UnauthorizedAccessException or JsonException or FormatException) {
            return null;
        }
    }

    private static InstallState GetState(
        InstalledManifest? local, LatestManifest remote, string installDirectory) {
        if (local is null || !File.Exists(Path.Combine(installDirectory, "llavon-lora.exe")))
            return Directory.Exists(installDirectory) ? InstallState.Damaged : InstallState.Missing;
        var comparison = CalVersion.Parse(local.Version).CompareTo(CalVersion.Parse(remote.Version));
        if (comparison > 0) return InstallState.Newer;
        if (comparison < 0 || local.TrainerApi != remote.TrainerApi) return InstallState.Outdated;
        return InstallState.Current;
    }

    private static void ValidateAsset(ReleaseAsset asset) {
        if (!Uri.TryCreate(asset.Url, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttps)
            throw new InvalidDataException("asset URL must use HTTPS");
        if (asset.Size <= 0) throw new InvalidDataException("asset size must be positive");
        if (asset.Sha256.Length != 64 || asset.Sha256.Any(c => !Uri.IsHexDigit(c)))
            throw new InvalidDataException("asset SHA-256 is invalid");
        if (!asset.Name.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Windows installer requires a ZIP asset");
    }

    private static async Task DownloadAsync(ReleaseAsset asset, string destination) {
        using var response = await Http.GetAsync(asset.Url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();
        var declaredLength = response.Content.Headers.ContentLength;
        if (declaredLength is not null && declaredLength != asset.Size)
            throw new InvalidDataException("asset HTTP length does not match the release manifest");

        await using var input = await response.Content.ReadAsStreamAsync();
        await using var output = new FileStream(
            destination, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1024 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        await input.CopyToAsync(output);
        await output.FlushAsync();
        if (output.Length != asset.Size)
            throw new InvalidDataException("downloaded asset size does not match the release manifest");
    }

    private static void VerifySha256(string path, string expected) {
        using var stream = File.OpenRead(path);
        var actual = Convert.ToHexString(SHA256.HashData(stream));
        if (!actual.Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"asset SHA-256 mismatch: expected {expected}, got {actual}");
    }

    private static void ReplaceDirectory(string destination, string staging, string backup) {
        var movedExisting = false;
        try {
            if (Directory.Exists(destination)) {
                Directory.Move(destination, backup);
                movedExisting = true;
            }
            Directory.Move(staging, destination);
        } catch {
            if (!Directory.Exists(destination) && movedExisting && Directory.Exists(backup))
                Directory.Move(backup, destination);
            throw;
        }
        if (movedExisting) TryDeleteDirectory(backup);
    }

    private static void WriteRegistry(string version, int trainerApi, string installDirectory) {
        using var key = Registry.LocalMachine.CreateSubKey(RegistryPath, writable: true)
            ?? throw new InvalidOperationException("unable to create trainer registry key");
        key.SetValue("Version", version, RegistryValueKind.String);
        key.SetValue("TrainerApi", trainerApi, RegistryValueKind.DWord);
        key.SetValue("InstallPath", installDirectory, RegistryValueKind.String);
    }

    private static void WriteStatus(InstallState state, string? local, string remote) {
        var status = new StatusResult(state.ToString().ToLowerInvariant(), local, remote);
        Console.WriteLine(JsonSerializer.Serialize(status, InstallerJsonContext.Default.StatusResult));
    }

    private static void TryDeleteFile(string path) {
        try { if (File.Exists(path)) File.Delete(path); } catch (IOException) { }
    }

    private static void TryDeleteDirectory(string path) {
        try { if (Directory.Exists(path)) Directory.Delete(path, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private enum InstallState { Missing, Current, Outdated, Newer, Damaged }

    private sealed record Options(string ManifestUrl, string InstallDirectory, string Asset) {
        internal static Options Parse(string[] args) {
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var index = 0; index < args.Length; index += 2) {
                if (index + 1 >= args.Length || args[index] is not (
                    "--manifest-url" or "--install-dir" or "--asset"))
                    throw new ArgumentException($"invalid option: {args[index]}");
                values[args[index]] = args[index + 1];
            }

            var defaultDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                "Llavon IME", "tools", "lora");
            return new Options(
                values.GetValueOrDefault("--manifest-url", DefaultManifestUrl),
                Path.GetFullPath(values.GetValueOrDefault("--install-dir", defaultDirectory)),
                values.GetValueOrDefault("--asset", DefaultAsset));
        }
    }
}

internal readonly record struct CalVersion(int Year, int Month, int Day, long Revision)
    : IComparable<CalVersion> {
    internal static CalVersion Parse(string value) {
        var parts = value.Split('.');
        if (parts.Length != 4 ||
            parts[0].Length != 4 || parts[1].Length != 2 || parts[2].Length != 2 ||
            parts.Any(part => part.Length == 0 || part.Any(character => !char.IsAsciiDigit(character))) ||
            !int.TryParse(parts[0], out var year) ||
            !int.TryParse(parts[1], out var month) ||
            !int.TryParse(parts[2], out var day) ||
            !long.TryParse(parts[3], out var revision) ||
            year < 2000 || revision < 0)
            throw new FormatException($"invalid CalVer: {value}");
        try {
            _ = new DateOnly(year, month, day);
        } catch (ArgumentOutOfRangeException) {
            throw new FormatException($"invalid CalVer: {value}");
        }
        return new CalVersion(year, month, day, revision);
    }

    public int CompareTo(CalVersion other) {
        var comparison = Year.CompareTo(other.Year);
        if (comparison != 0) return comparison;
        comparison = Month.CompareTo(other.Month);
        if (comparison != 0) return comparison;
        comparison = Day.CompareTo(other.Day);
        return comparison != 0 ? comparison : Revision.CompareTo(other.Revision);
    }
}

internal sealed record LatestManifest(
    int Schema,
    string Version,
    int TrainerApi,
    Dictionary<string, ReleaseAsset> Assets);

internal sealed record ReleaseAsset(string Name, string Url, string Sha256, long Size);

internal sealed record InstalledManifest(
    int Schema,
    string Version,
    int TrainerApi,
    string Commit,
    string Rid,
    string Backend);

internal sealed record StatusResult(string State, string? LocalVersion, string RemoteVersion);

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(LatestManifest))]
[JsonSerializable(typeof(InstalledManifest))]
[JsonSerializable(typeof(StatusResult))]
internal sealed partial class InstallerJsonContext : JsonSerializerContext;
