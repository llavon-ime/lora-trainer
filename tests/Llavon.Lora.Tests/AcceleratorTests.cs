using System.Text.Json;
using Llavon.Lora;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Llavon.Lora.Tests;

public sealed class AcceleratorTests {
    // The macOS CI job sets this so a runner whose Metal backend cannot run the
    // training step fails instead of silently skipping it.
    private static bool MpsRequired => Environment.GetEnvironmentVariable("LLAVON_LORA_REQUIRE_MPS") == "1";

    [Theory]
    [InlineData(DeviceSelection.Cpu, false, false, DeviceSelection.Cpu)]
    [InlineData(DeviceSelection.Cpu, true, true, DeviceSelection.Cpu)]
    [InlineData(DeviceSelection.Auto, false, false, DeviceSelection.Cpu)]
    [InlineData(DeviceSelection.Auto, true, false, DeviceSelection.Cuda)]
    [InlineData(DeviceSelection.Auto, false, true, DeviceSelection.Mps)]
    [InlineData(DeviceSelection.Auto, true, true, DeviceSelection.Cuda)]
    [InlineData(DeviceSelection.Cuda, true, false, DeviceSelection.Cuda)]
    [InlineData(DeviceSelection.Mps, false, true, DeviceSelection.Mps)]
    public void SelectionPrefersCudaThenMetalThenTheProcessor(string requested, bool cuda, bool mps, string expected) =>
        Assert.Equal(expected, DeviceSelection.Choose(requested, cuda, mps));

    [Fact]
    public void RequestingAnUnavailableBackendIsAnError() {
        var cuda = Assert.Throws<InvalidOperationException>(
            () => DeviceSelection.Choose(DeviceSelection.Cuda, false, false));
        Assert.Contains("CUDA", cuda.Message);
        var mps = Assert.Throws<InvalidOperationException>(
            () => DeviceSelection.Choose(DeviceSelection.Mps, false, false, "no Metal device"));
        Assert.Contains("no Metal device", mps.Message);
    }

    [Fact]
    public void UnknownDeviceNamesAreRejected() {
        Assert.Throws<ArgumentException>(() => DeviceSelection.Choose("vulkan", false, false));
        var config = MinimalConfig() with { Device = "vulkan" };
        var error = Assert.Throws<ArgumentException>(() => config.Validate(MinimalModel()));
        Assert.Contains("--device", error.Message);
    }

    [Theory]
    [InlineData(DeviceSelection.Auto)]
    [InlineData(DeviceSelection.Cpu)]
    [InlineData(DeviceSelection.Cuda)]
    [InlineData(DeviceSelection.Mps)]
    public void SupportedDeviceNamesValidate(string device) =>
        (MinimalConfig() with { Device = device }).Validate(MinimalModel());

    [Fact]
    public void LibTorchDirectoryFollowsTheEnvironment() {
        var directory = CreateFakeLibTorch();
        var requested = CreateFakeLibTorch();
        var application = Directory.CreateTempSubdirectory("llavon-clean-");
        var previous = Environment.GetEnvironmentVariable(TorchNativeLibraries.DirectoryVariable);
        try {
            Environment.SetEnvironmentVariable(TorchNativeLibraries.DirectoryVariable, directory);
            Assert.Equal(Path.GetFullPath(directory),
                TorchNativeLibraries.ResolveDirectory(null, application.FullName));
            // An explicit directory wins over the environment.
            Assert.Equal(Path.GetFullPath(requested),
                TorchNativeLibraries.ResolveDirectory(requested, application.FullName));
        } finally {
            Environment.SetEnvironmentVariable(TorchNativeLibraries.DirectoryVariable, previous);
            Directory.Delete(directory, recursive: true);
            Directory.Delete(requested, recursive: true);
            application.Delete(recursive: true);
        }
    }

    [Fact]
    public void MissingLibTorchDirectoriesAreRejected() {
        var application = Directory.CreateTempSubdirectory("llavon-clean-");
        var missing = Path.Combine(Path.GetTempPath(), $"llavon-missing-{Guid.NewGuid():N}");
        try {
            Assert.Throws<DirectoryNotFoundException>(
                () => TorchNativeLibraries.ResolveDirectory(missing, application.FullName));

            var empty = Directory.CreateTempSubdirectory("llavon-empty-");
            try {
                Assert.Throws<FileNotFoundException>(
                    () => TorchNativeLibraries.ResolveDirectory(empty.FullName, application.FullName));
            } finally {
                empty.Delete(recursive: true);
            }
        } finally {
            application.Delete(recursive: true);
        }
    }

    [Fact]
    public void AnAlternativeLibTorchCannotJoinBundledLibraries() {
        var alternative = CreateFakeLibTorch();
        var bundled = CreateFakeLibTorch();
        var clean = Directory.CreateTempSubdirectory("llavon-clean-");
        try {
            // Without bundled libraries the alternative directory is used.
            Assert.Equal(Path.GetFullPath(alternative),
                TorchNativeLibraries.ResolveDirectory(alternative, clean.FullName));
            // A build that ships its own libtorch cannot load a second one.
            var error = Assert.Throws<InvalidOperationException>(
                () => TorchNativeLibraries.ResolveDirectory(alternative, bundled));
            Assert.Contains("rocm-linux", error.Message);
        } finally {
            Directory.Delete(alternative, recursive: true);
            Directory.Delete(bundled, recursive: true);
            clean.Delete(recursive: true);
        }
    }

    [Fact]
    public void MetalTrainingCompletesOneStep() {
        // Metal coverage lives in the dedicated macOS 15 CI job, because the
        // macOS 26 runner images have a Metal backend that disagrees with real
        // hardware (actions/runner-images#14380).
        if (!MpsRequired)
            return;

        Assert.True(DeviceSelection.TryInitializeMps(out var mpsError),
            $"this job requires Metal but MPS is unavailable: {mpsError}");
        Assert.Equal(DeviceType.MPS, DeviceSelection.Resolve(DeviceSelection.Auto).type);

        using var fixture = new TrainingFixture();
        var (loss, output) = fixture.Train(DeviceSelection.Mps, "float32");
        Assert.True(double.IsFinite(loss), $"MPS training produced a non-finite loss: {loss}");
        Assert.True(File.Exists(Path.Combine(output, "adapter_model.safetensors")));
        // Metal runs the model in single precision until bfloat16 is verified
        // on real hardware.
        Assert.Throws<ArgumentException>(() => fixture.Train(DeviceSelection.Mps, "bfloat16"));
    }

    [Fact]
    public void CachedLibTorchIsUsedWhenNothingIsBundled() {
        var cacheRoot = Directory.CreateTempSubdirectory("llavon-cache-");
        var application = Directory.CreateTempSubdirectory("llavon-clean-");
        var bundled = CreateFakeLibTorch();
        var previous = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        try {
            Environment.SetEnvironmentVariable("XDG_CACHE_HOME", cacheRoot.FullName);
            var library = Path.Combine(TorchNativeLibraries.CacheRoot, "libtorch-rocm7.0-2.10.0", "libtorch", "lib");
            Directory.CreateDirectory(library);
            File.WriteAllText(Path.Combine(library, TorchNativeLibraries.LibraryFileName("libtorch_cpu")), "");
            // A build without bundled libraries picks the cache up by itself.
            Assert.Equal(Path.GetFullPath(library),
                TorchNativeLibraries.ResolveDirectory(null, application.FullName));
            // A build that ships its own libtorch must not load the cached one.
            Assert.Null(TorchNativeLibraries.ResolveDirectory(null, bundled));
        } finally {
            Environment.SetEnvironmentVariable("XDG_CACHE_HOME", previous);
            Directory.Delete(cacheRoot.FullName, recursive: true);
            Directory.Delete(application.FullName, recursive: true);
            Directory.Delete(bundled, recursive: true);
        }
    }

    [Theory]
    [InlineData("rocm", "libtorch/build-version", true)]
    [InlineData("rocm", "libtorch/lib/libtorch_cpu.so", true)]
    [InlineData("rocm", "libtorch/lib/libtorch_hip.so", true)]
    [InlineData("rocm", "libtorch/lib/rocblas/library/TensileLibrary_lazy_gfx1201.dat", true)]
    [InlineData("rocm", "libtorch/lib/hipblaslt/library/Kernels.so-000-gfx1201.hsaco", true)]
    [InlineData("rocm", "libtorch/lib/hipsparselt/library/anything.dat", true)]
    [InlineData("rocm", "libtorch/lib/libtorch_cpu.a", false)]
    [InlineData("rocm", "libtorch/lib/aotriton.images/amd-gfx1201/anything.so", false)]
    [InlineData("rocm", "libtorch/include/torch/torch.h", false)]
    [InlineData("rocm", "libtorch/lib/", false)]
    [InlineData("cuda", "libtorch/lib/libtorch_cuda.so", true)]
    [InlineData("cuda", "libtorch/lib/libcudnn.so.9", true)]
    [InlineData("cuda", "libtorch/lib/libcublasLt.so.12", true)]
    [InlineData("cuda", "libtorch/lib/libtorch_cpu.a", false)]
    [InlineData("cuda", "libtorch/lib/rocblas/library/TensileLibrary_lazy_gfx1201.dat", false)]
    [InlineData("cuda", "libtorch/include/torch/torch.h", false)]
    public void OnlyTheNeededLibTorchEntriesAreExtracted(string backend, string entry, bool needed) =>
        Assert.Equal(needed, LibTorchDistribution.IsNeededEntry(LibTorchDistribution.ForName(backend), entry));

    [Fact]
    public void FetchLibTorchSkipsAnInstalledDirectory() {
        if (!OperatingSystem.IsLinux())
            return;
        var root = Directory.CreateTempSubdirectory("llavon-fetch-");
        try {
            var library = Path.Combine(root.FullName, "libtorch", "lib");
            Directory.CreateDirectory(library);
            File.WriteAllText(Path.Combine(library, TorchNativeLibraries.LibraryFileName("libtorch_hip")), "");
            foreach (var kernel in new[] { "rocblas/library", "hipblaslt/library", "hipsparselt/library" }) {
                var path = Path.Combine(library, kernel);
                Directory.CreateDirectory(path);
                File.WriteAllText(Path.Combine(path, "kernel.dat"), "");
            }
            var messages = new List<string>();
            var result = LibTorchDistribution.Fetch(LibTorchDistribution.Rocm, root.FullName, force: false, messages.Add);
            Assert.Equal(library, result);
            Assert.Contains(messages, message => message.Contains("already installed"));
            // An interrupted extraction is not an installation.
            File.Delete(Path.Combine(library, "rocblas", "library", "kernel.dat"));
            Assert.False(LibTorchDistribution.IsInstalled(LibTorchDistribution.Rocm, library));
        } finally {
            root.Delete(recursive: true);
        }
    }

    [Fact]
    public void CudaTrainingCompletesOneStep() {
        // Run on a machine with an AMD (ROCm) or NVIDIA GPU, in isolation so
        // that nothing loads the bundled libtorch first:
        //   LLAVON_LORA_REQUIRE_CUDA=1 LLAVON_LORA_TORCH_LIB=<libtorch dir> \
        //     dotnet test --filter FullyQualifiedName~CudaTrainingCompletesOneStep
        if (Environment.GetEnvironmentVariable("LLAVON_LORA_REQUIRE_CUDA") != "1")
            return;

        TorchNativeLibraries.Initialize(null);
        Assert.True(torch.cuda.is_available(), "this run requires a CUDA or ROCm device");
        using var fixture = new TrainingFixture();
        var (loss, _) = fixture.Train(DeviceSelection.Cuda, "float32");
        Assert.True(double.IsFinite(loss), $"CUDA training produced a non-finite loss: {loss}");
    }

    private static ModelConfig MinimalModel() =>
        new(8, 4, 8, 1, 2, 1, 2, 16, 1e-5, 10000, false, false, false, "silu");

    private static TrainConfig MinimalConfig() => new() {
        ModelConfigPath = "config.json",
        ModelPath = "model.safetensors",
        TrainDataPath = "training.jsonl",
        OutputDirectory = "adapter",
        TargetModules = new HashSet<string>(StringComparer.Ordinal) { "q_proj" },
        PadTokenId = 0,
        MaxSequenceLength = 8,
        MaxSteps = 1
    };

    private static string CreateFakeLibTorch() {
        var directory = Path.Combine(Path.GetTempPath(), $"llavon-libtorch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, TorchNativeLibraries.LibraryFileName("libtorch_cpu")), "");
        return directory;
    }

    private sealed class TrainingFixture : IDisposable {
        private readonly string root;
        private readonly string modelDirectory;
        private readonly string dataPath;

        public TrainingFixture() {
            root = Path.Combine(Path.GetTempPath(), $"llavon-accelerator-{Guid.NewGuid():N}");
            modelDirectory = Path.Combine(root, "model");
            Directory.CreateDirectory(modelDirectory);
            torch.manual_seed(11);
            var weights = new Dictionary<string, Tensor>(StringComparer.Ordinal) {
                ["model.embed_tokens.weight"] = torch.randn(8, 4) * 0.02,
                ["model.layers.0.input_layernorm.weight"] = torch.ones(4),
                ["model.layers.0.post_attention_layernorm.weight"] = torch.ones(4),
                ["model.layers.0.self_attn.q_proj.weight"] = torch.randn(4, 4) * 0.02,
                ["model.layers.0.self_attn.k_proj.weight"] = torch.randn(2, 4) * 0.02,
                ["model.layers.0.self_attn.v_proj.weight"] = torch.randn(2, 4) * 0.02,
                ["model.layers.0.self_attn.o_proj.weight"] = torch.randn(4, 4) * 0.02,
                ["model.layers.0.mlp.gate_proj.weight"] = torch.randn(8, 4) * 0.02,
                ["model.layers.0.mlp.up_proj.weight"] = torch.randn(8, 4) * 0.02,
                ["model.layers.0.mlp.down_proj.weight"] = torch.randn(4, 8) * 0.02,
                ["model.norm.weight"] = torch.ones(4),
                ["lm_head.weight"] = torch.randn(8, 4) * 0.02
            };
            try {
                SafeTensors.Save(Path.Combine(modelDirectory, "model.safetensors"), weights);
            } finally {
                foreach (var tensor in weights.Values)
                    tensor.Dispose();
            }

            File.WriteAllText(Path.Combine(modelDirectory, "config.json"), JsonSerializer.Serialize(new {
                model_type = "llama",
                vocab_size = 8,
                hidden_size = 4,
                intermediate_size = 8,
                num_hidden_layers = 1,
                num_attention_heads = 2,
                num_key_value_heads = 1,
                head_dim = 2,
                max_position_embeddings = 16,
                rms_norm_eps = 1e-5,
                rope_theta = 10000,
                attention_bias = false,
                mlp_bias = false,
                tie_word_embeddings = false,
                hidden_act = "silu"
            }));
            dataPath = Path.Combine(root, "train.jsonl");
            File.WriteAllText(dataPath,
                """
                {"tokens":[1,2,3],"labels":[-100,-100,3],"loss_weights":[0,0,1],"candidate_masks":[null,null,[3,4]]}
                """);
        }

        public (double Loss, string Directory) Train(string device, string dtype) {
            var output = Path.Combine(root, $"adapter-{device}-{dtype}");
            Trainer.Train(new TrainConfig {
                ModelConfigPath = Path.Combine(modelDirectory, "config.json"),
                ModelPath = modelDirectory,
                TrainDataPath = dataPath,
                OutputDirectory = output,
                TargetModules = new HashSet<string>(StringComparer.Ordinal) { "q_proj" },
                PadTokenId = 0,
                MaxSequenceLength = 8,
                Rank = 2,
                MaxSteps = 1,
                Device = device,
                DType = dtype
            });
            using var state = JsonDocument.Parse(
                File.ReadAllBytes(Path.Combine(output, "training_state.json")));
            return (state.RootElement.GetProperty("loss").GetDouble(), output);
        }

        public void Dispose() {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }
}
