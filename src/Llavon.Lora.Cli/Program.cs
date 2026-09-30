using System.Globalization;
using System.Reflection;
using System.Text.Json;
using Llavon.Lora;

return ProgramEntry.Run(args);

internal static class ProgramEntry {
    internal const int TrainerApiVersion = 2;

    private const string Help = """
        llavon-lora: token-level LoRA training for Hugging Face Llama checkpoints

        Usage:
          llavon-lora --version [--json]

          llavon-lora devices [--json]

          llavon-lora train --model-config FILE --model FILE_OR_DIR --train-data FILE
              --output-dir DIR --pad-token-id ID --max-seq-length N
              --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj [options]

          llavon-lora validate --train-data FILE --vocab-size N --max-seq-length N

          llavon-lora analyze-spectrum --model FILE_OR_DIR --adapter DIR [options]

          llavon-lora stabilize-adapter --model FILE_OR_DIR --adapter DIR
              --output-dir DIR [options]

          llavon-lora export-gguf --model-config FILE --model FILE_OR_DIR --vocab-file FILE
              --outfile FILE [--adapter DIR] [--outtype f16|f32]
              [--quantize TYPE --quantized-outfile FILE] [options]

        Training options:
          --rank N                         LoRA rank (default: 16)
          --alpha X                        LoRA alpha (default: 2 * rank)
          --dropout X                      LoRA dropout (default: 0)
          --batch-size N                   Samples per micro-batch (default: 1)
          --gradient-accumulation N        Micro-batches per optimizer step (default: 1)
          --epochs N                       Dataset passes (default: 1)
          --max-steps N                    Stop after N optimizer steps
          --learning-rate X                Peak AdamW learning rate (default: 2e-4)
          --weight-decay X                 AdamW weight decay (default: 0)
          --warmup-steps N                 Linear warmup steps (default: 0)
          --max-grad-norm X                Gradient clipping; 0 disables it (default: 1)
          --save-every N                   Adapter checkpoint interval; 0 disables it
          --device auto|cpu|cuda|mps       Training device (default: auto)
          --torch-lib-dir DIR              libtorch directory for an alternative
                                           backend, e.g. a ROCm build (or set
                                           LLAVON_LORA_TORCH_LIB)
          --dtype float32|bfloat16         Model compute dtype (default: float32)
          --seed N                         RNG seed (default: 42)
          --no-shuffle                     Preserve JSONL order
          --train-until-remembered          Repeat epochs until every target is predicted
          --only-train-incorrect            Compute loss only for targets that failed validation

        Spectral analysis options:
          --top-k N                        Leading singular vectors per matrix (default: 10)
          --epsilon X                      Intruder cosine threshold (default: 0.5)
          --json                           Write the complete machine-readable report

        Adapter stabilization options:
          --scale X                        Scale the top intruder in each matrix (default: 0.9)
          --output-dir DIR                 Write a directly usable stabilized PEFT adapter
          --force                          Overwrite stabilization output files

        GGUF export options:
          --adapter DIR                    Optional PEFT LoRA adapter to merge
          --outtype f16|f32                Unquantized GGUF type (default: f16)
          --quantize TYPE                  In-process llama.cpp quantization, e.g. Q4_K_M
          --quantized-outfile FILE         Quantized GGUF output
          --quantization-threads N         Quantizer threads; 0 uses its default
          --expected-outfile-sha256 HASH   Reject an unexpected unquantized artifact
          --expected-quantized-sha256 HASH Reject an unexpected quantized artifact
          --force                          Overwrite existing output files
        """;

    public static int Run(string[] args) {
        try {
            if (args.Length == 0 || args[0] is "--help" or "-h") {
                Console.WriteLine(Help);
                return args.Length == 0 ? 2 : 0;
            }

            if (args[0] is "--version" or "version") {
                if (args.Length > 2 || (args.Length == 2 && args[1] != "--json"))
                    throw new ArgumentException("usage: llavon-lora --version [--json]");

                WriteVersion(args.Length == 2);
                return 0;
            }

            var arguments = new Arguments(args[1..]);
            return args[0] switch {
                "train" => RunTrain(arguments),
                "validate" => RunValidate(arguments),
                "analyze-spectrum" => RunAnalyzeSpectrum(arguments),
                "stabilize-adapter" => RunStabilizeAdapter(arguments),
                "export-gguf" => RunExportGguf(arguments),
                "devices" => RunDevices(arguments),
                _ => throw new ArgumentException($"unknown command: {args[0]}")
            };
        } catch (Exception exception) when (exception is not OutOfMemoryException) {
            Console.Error.WriteLine($"error: {exception.Message}");
            return 2;
        }
    }

    private static void WriteVersion(bool json) {
        var assembly = Assembly.GetEntryAssembly() ?? typeof(ProgramEntry).Assembly;
        var informationalVersion = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?
            .InformationalVersion;
        var version = string.IsNullOrWhiteSpace(informationalVersion)
            ? assembly.GetName().Version?.ToString() ?? "0.0.0"
            : informationalVersion.Split('+', 2)[0];

        if (!json) {
            Console.WriteLine(version);
            return;
        }

        Console.WriteLine(JsonSerializer.Serialize(new {
            version,
            trainerApi = TrainerApiVersion
        }));
    }

    private static int RunValidate(Arguments arguments) {
        arguments.Allow("--train-data", "--vocab-size", "--max-seq-length");
        var samples = Dataset.LoadJsonLines(
            arguments.Required("--train-data"),
            arguments.Integer("--vocab-size", required: true),
            arguments.Integer("--max-seq-length", required: true));
        var positions = samples.Sum(sample => sample.Tokens.LongLength);
        var constrained = samples.Sum(sample => sample.CandidateMasks.LongCount(mask => mask is not null));
        Console.WriteLine($"valid samples={samples.Count} positions={positions} constrained_positions={constrained}");
        return 0;
    }

    private static int RunDevices(Arguments arguments) {
        arguments.Allow("--json");
        TorchNativeLibraries.Initialize(null);
        var (cuda, mps, mpsError) = DeviceSelection.Availability();
        var directory = TorchNativeLibraries.InitializedDirectory;
        var libtorch = directory is null ? null : ReadLibTorchBuildVersion(directory);

        if (arguments.Flag("--json")) {
            Console.WriteLine(JsonSerializer.Serialize(new {
                cpu = true,
                cuda,
                mps,
                libtorch,
                torchLibDir = directory,
                mpsError = mps ? null : mpsError
            }));
            return 0;
        }

        Console.WriteLine("cpu=available");
        Console.WriteLine($"cuda={(cuda ? "available" : "unavailable")}");
        Console.WriteLine($"mps={(mps ? "available" : "unavailable")}");
        if (!mps && mpsError is not null)
            Console.WriteLine($"mps-reason={mpsError}");
        Console.WriteLine($"libtorch={libtorch ?? "bundled"}");
        if (directory is not null)
            Console.WriteLine($"torch-lib-dir={directory}");
        return 0;
    }

    // build-version ships beside the libtorch libraries inside the archive.
    private static string? ReadLibTorchBuildVersion(string directory) {
        foreach (var candidate in new[] {
                     Path.Combine(directory, "build-version"),
                     Path.Combine(directory, "..", "build-version")
                 }) {
            if (File.Exists(candidate))
                return File.ReadAllText(candidate).Trim();
        }
        return null;
    }

    private static int RunTrain(Arguments arguments) {
        arguments.Allow(
            "--model-config", "--model", "--train-data", "--output-dir", "--target-modules",
            "--resume-adapter",
            "--pad-token-id", "--max-seq-length", "--rank", "--alpha", "--dropout", "--batch-size",
            "--gradient-accumulation", "--epochs", "--max-steps", "--learning-rate", "--weight-decay",
            "--warmup-steps", "--max-grad-norm", "--save-every", "--device", "--dtype", "--seed",
            "--torch-lib-dir", "--no-shuffle", "--train-until-remembered",
            "--only-train-incorrect");

        var targets = arguments.Required("--target-modules")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToHashSet(StringComparer.Ordinal);
        var rank = arguments.Integer("--rank", 16);
        var config = new TrainConfig {
            ModelConfigPath = arguments.Required("--model-config"),
            ModelPath = arguments.Required("--model"),
            TrainDataPath = arguments.Required("--train-data"),
            OutputDirectory = arguments.Required("--output-dir"),
            ResumeAdapterDirectory = arguments.Optional("--resume-adapter"),
            TargetModules = targets,
            PadTokenId = arguments.Integer("--pad-token-id", required: true),
            MaxSequenceLength = arguments.Integer("--max-seq-length", required: true),
            Rank = rank,
            Alpha = arguments.Real("--alpha", checked(2 * rank)),
            Dropout = arguments.Real("--dropout", 0),
            BatchSize = checked((int)arguments.Integer("--batch-size", 1)),
            GradientAccumulationSteps = checked((int)arguments.Integer("--gradient-accumulation", 1)),
            Epochs = checked((int)arguments.Integer("--epochs", 1)),
            MaxSteps = arguments.Integer("--max-steps", -1),
            LearningRate = arguments.Real("--learning-rate", 2e-4),
            WeightDecay = arguments.Real("--weight-decay", 0),
            WarmupSteps = arguments.Integer("--warmup-steps", 0),
            MaxGradientNorm = arguments.Real("--max-grad-norm", 1),
            SaveEvery = arguments.Integer("--save-every", 0),
            Device = arguments.Value("--device", "auto"),
            DType = arguments.Value("--dtype", "float32"),
            Seed = arguments.Integer("--seed", 42),
            Shuffle = !arguments.Flag("--no-shuffle"),
            TrainUntilRemembered = arguments.Flag("--train-until-remembered"),
            OnlyTrainIncorrect = arguments.Flag("--only-train-incorrect")
        };
        TorchNativeLibraries.Initialize(arguments.Optional("--torch-lib-dir"));
        Trainer.Train(config);
        return 0;
    }

    private static int RunAnalyzeSpectrum(Arguments arguments) {
        arguments.Allow("--model", "--adapter", "--top-k", "--epsilon", "--torch-lib-dir", "--json");
        TorchNativeLibraries.Initialize(arguments.Optional("--torch-lib-dir"));
        var result = SpectralAnalyzer.Analyze(new SpectralAnalysisConfig {
            ModelPath = arguments.Required("--model"),
            AdapterDirectory = arguments.Required("--adapter"),
            TopK = checked((int)arguments.Integer("--top-k", 10)),
            SimilarityThreshold = arguments.Real("--epsilon", 0.5)
        });

        if (arguments.Flag("--json")) {
            Console.WriteLine(JsonSerializer.Serialize(result, new JsonSerializerOptions {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            }));
            return 0;
        }

        Console.WriteLine(
            $"intruders={result.IntruderCount}/{result.ExaminedDimensions} " +
            $"matrices={result.ExaminedMatrices} rank={result.Rank} alpha={result.Alpha:G} " +
            $"scaling={result.Scaling:G} top_k={result.TopK} epsilon={result.SimilarityThreshold:G}");
        foreach (var matrix in result.Matrices.Where(matrix => matrix.IntruderCount != 0)) {
            var ranks = string.Join(',', matrix.IntruderDimensions.Select(dimension => dimension.Rank));
            Console.WriteLine($"{matrix.Weight} intruders={matrix.IntruderCount} ranks={ranks}");
        }
        return 0;
    }

    private static int RunStabilizeAdapter(Arguments arguments) {
        arguments.Allow(
            "--model", "--adapter", "--output-dir", "--top-k", "--epsilon", "--scale",
            "--torch-lib-dir", "--force", "--json");
        TorchNativeLibraries.Initialize(arguments.Optional("--torch-lib-dir"));
        var result = SpectralAnalyzer.Stabilize(new SpectralStabilizationConfig {
            ModelPath = arguments.Required("--model"),
            AdapterDirectory = arguments.Required("--adapter"),
            OutputDirectory = arguments.Required("--output-dir"),
            TopK = checked((int)arguments.Integer("--top-k", 10)),
            SimilarityThreshold = arguments.Real("--epsilon", 0.5),
            IntruderScale = arguments.Real("--scale", 0.9),
            Overwrite = arguments.Flag("--force")
        });

        if (arguments.Flag("--json")) {
            Console.WriteLine(JsonSerializer.Serialize(result, new JsonSerializerOptions {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true
            }));
            return 0;
        }

        Console.WriteLine(
            $"wrote={result.OutputDirectory} scaled_intruders={result.ScaledIntruderCount} " +
            $"scale={result.IntruderScale:G} rank={result.OriginalRank}->{result.StabilizedRank} " +
            $"alpha={result.OriginalAlpha:G}->{result.StabilizedAlpha:G}");
        return 0;
    }

    private static int RunExportGguf(Arguments arguments) {
        arguments.Allow(
            "--model-config", "--model", "--vocab-file", "--adapter", "--outfile", "--outtype",
            "--quantize", "--quantized-outfile", "--quantization-threads",
            "--expected-outfile-sha256", "--expected-quantized-sha256", "--force");
        var outputType = arguments.Value("--outtype", "f16") switch {
            "f16" => GgufOutputType.Float16,
            "f32" => GgufOutputType.Float32,
            var value => throw new ArgumentException($"invalid --outtype: {value}")
        };
        var config = new GgufExportConfig {
            ModelConfigPath = arguments.Required("--model-config"),
            ModelPath = arguments.Required("--model"),
            VocabularyPath = arguments.Required("--vocab-file"),
            AdapterDirectory = arguments.Optional("--adapter"),
            OutputPath = arguments.Required("--outfile"),
            OutputType = outputType,
            QuantizationType = arguments.Optional("--quantize"),
            QuantizedOutputPath = arguments.Optional("--quantized-outfile"),
            QuantizationThreads = checked((int)arguments.Integer("--quantization-threads", 0)),
            ExpectedOutputSha256 = arguments.Optional("--expected-outfile-sha256"),
            ExpectedQuantizedSha256 = arguments.Optional("--expected-quantized-sha256"),
            Overwrite = arguments.Flag("--force")
        };
        var result = GgufExporter.Export(config);
        Console.WriteLine($"wrote={result.OutputPath} sha256={result.OutputSha256}");
        if (result.QuantizedOutputPath is not null)
            Console.WriteLine($"wrote={result.QuantizedOutputPath} sha256={result.QuantizedOutputSha256}");
        return 0;
    }

}

internal sealed class Arguments {
    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
    private readonly HashSet<string> flags = new(StringComparer.Ordinal);

    public Arguments(IReadOnlyList<string> args) {
        for (var index = 0; index < args.Count; ++index) {
            var key = args[index];
            if (!key.StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException($"unexpected positional argument: {key}");
            if (key is "--no-shuffle" or "--force" or "--json") {
                if (!flags.Add(key))
                    throw new ArgumentException($"duplicate option: {key}");
                continue;
            }
            if (index + 1 >= args.Count)
                throw new ArgumentException($"missing value for {key}");
            if (!values.TryAdd(key, args[++index]))
                throw new ArgumentException($"duplicate option: {key}");
        }
    }

    public void Allow(params string[] names) {
        var allowed = names.ToHashSet(StringComparer.Ordinal);
        var unknown = values.Keys.Concat(flags).FirstOrDefault(key => !allowed.Contains(key));
        if (unknown is not null)
            throw new ArgumentException($"unknown option: {unknown}");
    }

    public string Required(string key) => values.TryGetValue(key, out var value)
        ? value
        : throw new ArgumentException($"missing required option: {key}");

    public string Value(string key, string fallback) => values.GetValueOrDefault(key, fallback);

    public string? Optional(string key) => values.GetValueOrDefault(key);

    public long Integer(string key, long fallback = 0, bool required = false) {
        var text = required ? Required(key) : Value(key, fallback.ToString(CultureInfo.InvariantCulture));
        return long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new ArgumentException($"invalid integer for {key}: {text}");
    }

    public double Real(string key, double fallback) {
        var text = Value(key, fallback.ToString("R", CultureInfo.InvariantCulture));
        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) &&
               double.IsFinite(value)
            ? value
            : throw new ArgumentException($"invalid number for {key}: {text}");
    }

    public bool Flag(string key) => flags.Contains(key);
}
