using System.Globalization;
using Llavon.Lora;

return ProgramEntry.Run(args);

internal static class ProgramEntry
{
    private const string Help = """
        llavon-lora: token-level LoRA training for Hugging Face Llama checkpoints

        Usage:
          llavon-lora train --model-config FILE --model FILE_OR_DIR --train-data FILE
              --output-dir DIR --pad-token-id ID --max-seq-length N
              --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj [options]

          llavon-lora validate --train-data FILE --vocab-size N --max-seq-length N

        Training options:
          --rank N                         LoRA rank (default: 16)
          --alpha X                        LoRA alpha (default: 32)
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
          --device auto|cpu|cuda           Training device (default: auto)
          --dtype float32|float16|bfloat16 Model compute dtype (default: float32)
          --seed N                         RNG seed (default: 42)
          --no-shuffle                     Preserve JSONL order
        """;

    public static int Run(string[] args)
    {
        try
        {
            if (args.Length == 0 || args[0] is "--help" or "-h")
            {
                Console.WriteLine(Help);
                return args.Length == 0 ? 2 : 0;
            }

            var arguments = new Arguments(args[1..]);
            return args[0] switch
            {
                "train" => RunTrain(arguments),
                "validate" => RunValidate(arguments),
                _ => throw new ArgumentException($"unknown command: {args[0]}")
            };
        }
        catch (Exception exception) when (exception is not OutOfMemoryException)
        {
            Console.Error.WriteLine($"error: {exception.Message}");
            return 2;
        }
    }

    private static int RunValidate(Arguments arguments)
    {
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

    private static int RunTrain(Arguments arguments)
    {
        arguments.Allow(
            "--model-config", "--model", "--train-data", "--output-dir", "--target-modules",
            "--pad-token-id", "--max-seq-length", "--rank", "--alpha", "--dropout", "--batch-size",
            "--gradient-accumulation", "--epochs", "--max-steps", "--learning-rate", "--weight-decay",
            "--warmup-steps", "--max-grad-norm", "--save-every", "--device", "--dtype", "--seed",
            "--no-shuffle");

        var targets = arguments.Required("--target-modules")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToHashSet(StringComparer.Ordinal);
        var config = new TrainConfig
        {
            ModelConfigPath = arguments.Required("--model-config"),
            ModelPath = arguments.Required("--model"),
            TrainDataPath = arguments.Required("--train-data"),
            OutputDirectory = arguments.Required("--output-dir"),
            TargetModules = targets,
            PadTokenId = arguments.Integer("--pad-token-id", required: true),
            MaxSequenceLength = arguments.Integer("--max-seq-length", required: true),
            Rank = arguments.Integer("--rank", 16),
            Alpha = arguments.Real("--alpha", 32),
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
            Shuffle = !arguments.Flag("--no-shuffle")
        };
        Trainer.Train(config);
        return 0;
    }
}

internal sealed class Arguments
{
    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
    private readonly HashSet<string> flags = new(StringComparer.Ordinal);

    public Arguments(IReadOnlyList<string> args)
    {
        for (var index = 0; index < args.Count; ++index)
        {
            var key = args[index];
            if (!key.StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException($"unexpected positional argument: {key}");
            if (key == "--no-shuffle")
            {
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

    public void Allow(params string[] names)
    {
        var allowed = names.ToHashSet(StringComparer.Ordinal);
        var unknown = values.Keys.Concat(flags).FirstOrDefault(key => !allowed.Contains(key));
        if (unknown is not null)
            throw new ArgumentException($"unknown option: {unknown}");
    }

    public string Required(string key) => values.TryGetValue(key, out var value)
        ? value
        : throw new ArgumentException($"missing required option: {key}");

    public string Value(string key, string fallback) => values.GetValueOrDefault(key, fallback);

    public long Integer(string key, long fallback = 0, bool required = false)
    {
        var text = required ? Required(key) : Value(key, fallback.ToString(CultureInfo.InvariantCulture));
        return long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new ArgumentException($"invalid integer for {key}: {text}");
    }

    public double Real(string key, double fallback)
    {
        var text = Value(key, fallback.ToString("R", CultureInfo.InvariantCulture));
        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) &&
               double.IsFinite(value)
            ? value
            : throw new ArgumentException($"invalid number for {key}: {text}");
    }

    public bool Flag(string key) => flags.Contains(key);
}
