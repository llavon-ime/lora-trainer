using System.Runtime.InteropServices;
using TorchSharp;
using static TorchSharp.torch;

namespace Llavon.Lora;

public static class DeviceSelection {
    public const string Auto = "auto";
    public const string Cpu = "cpu";
    public const string Cuda = "cuda";
    public const string Mps = "mps";

    // Names accepted by --device; everything else is a user error.
    public static IReadOnlyList<string> Supported { get; } = [Auto, Cpu, Cuda, Mps];

    public static bool IsSupported(string name) => Supported.Contains(name, StringComparer.Ordinal);

    // The selection rules take the backend availability as arguments so they can
    // be tested on machines without a GPU.
    internal static string Choose(string requested, bool cudaAvailable, bool mpsAvailable, string? mpsError = null) {
        if (!IsSupported(requested))
            throw new ArgumentException($"--device must be {string.Join(", ", Supported)}");
        if (requested == Auto)
            return cudaAvailable ? Cuda : mpsAvailable ? Mps : Cpu;
        if (requested == Cuda && !cudaAvailable)
            throw new InvalidOperationException(
                "CUDA was requested but is not available; use a build with the CUDA backend " +
                "(-p:TorchBackend=cuda-linux or cuda-windows) and a compatible driver");
        if (requested == Mps && !mpsAvailable)
            throw new InvalidOperationException(
                "MPS was requested but is not available" + (mpsError is null ? "" : $": {mpsError}"));
        return requested;
    }

    public static Device Resolve(string requested) {
        var cudaAvailable = torch.cuda.is_available();
        string? mpsError = null;
        var mpsAvailable = NeedsMpsProbe(requested, cudaAvailable) && TryInitializeMps(out mpsError);
        return Choose(requested, cudaAvailable, mpsAvailable, mpsError) switch {
            Cuda => CUDA,
            Mps => MPS,
            _ => CPU
        };
    }

    // The MPS probe initialises a backend, so it only runs when the answer can
    // change the outcome.
    private static bool NeedsMpsProbe(string requested, bool cudaAvailable) =>
        requested == Mps || (requested == Auto && !cudaAvailable);

    // TorchSharp treats MPS as available on every Apple Silicon build whatever
    // the native library was compiled with, so probe by creating a tensor. The
    // tensor is the cheapest operation that proves the Metal backend runs.
    public static bool TryInitializeMps(out string? error) {
        error = null;
        if (!OperatingSystem.IsMacOS() || RuntimeInformation.OSArchitecture != Architecture.Arm64) {
            error = "GPU training on macOS requires Apple Silicon";
            return false;
        }

        try {
            using var scope = torch.NewDisposeScope();
            using var probe = torch.zeros(1, device: MPS);
            if (probe.numel() != 1) {
                error = "the MPS backend did not produce a tensor";
                return false;
            }
            return true;
        } catch (Exception exception) when (exception is not OutOfMemoryException) {
            error = exception.Message;
            return false;
        }
    }

    // Reports what the current build and machine can run; used by `devices`.
    public static (bool Cuda, bool Mps, string? MpsError) Availability() {
        var cuda = torch.cuda.is_available();
        var mps = TryInitializeMps(out var error);
        return (cuda, mps, error);
    }
}
