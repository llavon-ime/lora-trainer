using System.Text.Json;

namespace Llavon.Lora;

public sealed record ModelConfig(
    long VocabSize,
    long HiddenSize,
    long IntermediateSize,
    long NumHiddenLayers,
    long NumAttentionHeads,
    long NumKeyValueHeads,
    long HeadDim,
    long MaxPositionEmbeddings,
    double RmsNormEpsilon,
    double RopeTheta,
    bool AttentionBias,
    bool MlpBias,
    bool TieWordEmbeddings,
    string HiddenActivation)
{
    public static ModelConfig Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = document.RootElement;

        static long RequiredInt64(JsonElement root, string name) =>
            root.TryGetProperty(name, out var value)
                ? value.GetInt64()
                : throw new InvalidDataException($"model config is missing '{name}'");

        static double RequiredDouble(JsonElement root, string name) =>
            root.TryGetProperty(name, out var value)
                ? value.GetDouble()
                : throw new InvalidDataException($"model config is missing '{name}'");

        static bool RequiredBoolean(JsonElement root, string name) =>
            root.TryGetProperty(name, out var value)
                ? value.GetBoolean()
                : throw new InvalidDataException($"model config is missing '{name}'");

        static string RequiredString(JsonElement root, string name) =>
            root.TryGetProperty(name, out var value)
                ? value.GetString() ?? throw new InvalidDataException($"model config '{name}' is null")
                : throw new InvalidDataException($"model config is missing '{name}'");

        var modelType = RequiredString(root, "model_type");
        if (!string.Equals(modelType, "llama", StringComparison.Ordinal))
            throw new InvalidDataException("only Hugging Face model_type='llama' is supported");

        var hiddenSize = RequiredInt64(root, "hidden_size");
        var attentionHeads = RequiredInt64(root, "num_attention_heads");
        var headDim = root.TryGetProperty("head_dim", out var headDimElement)
            ? headDimElement.GetInt64()
            : hiddenSize / attentionHeads;
        var ropeTheta = root.TryGetProperty("rope_theta", out var ropeThetaElement)
            ? ropeThetaElement.GetDouble()
            : root.TryGetProperty("rope_parameters", out var ropeParameters) &&
              ropeParameters.TryGetProperty("rope_theta", out ropeThetaElement)
                ? ropeThetaElement.GetDouble()
                : throw new InvalidDataException("model config is missing 'rope_theta'");

        var config = new ModelConfig(
            RequiredInt64(root, "vocab_size"),
            hiddenSize,
            RequiredInt64(root, "intermediate_size"),
            RequiredInt64(root, "num_hidden_layers"),
            attentionHeads,
            RequiredInt64(root, "num_key_value_heads"),
            headDim,
            RequiredInt64(root, "max_position_embeddings"),
            RequiredDouble(root, "rms_norm_eps"),
            ropeTheta,
            RequiredBoolean(root, "attention_bias"),
            RequiredBoolean(root, "mlp_bias"),
            RequiredBoolean(root, "tie_word_embeddings"),
            RequiredString(root, "hidden_act"));

        config.Validate();
        return config;
    }

    private void Validate()
    {
        if (!string.Equals(HiddenActivation, "silu", StringComparison.Ordinal))
            throw new InvalidDataException("only hidden_act='silu' is supported");
        if (VocabSize <= 0 || HiddenSize <= 0 || IntermediateSize <= 0 || NumHiddenLayers <= 0 ||
            NumAttentionHeads <= 0 || NumKeyValueHeads <= 0 || HeadDim <= 0 ||
            MaxPositionEmbeddings <= 0 || RmsNormEpsilon <= 0 || RopeTheta <= 0)
            throw new InvalidDataException("model config contains a non-positive dimension or constant");
        if (HiddenSize != NumAttentionHeads * HeadDim)
            throw new InvalidDataException("hidden_size must equal num_attention_heads * head_dim");
        if (NumAttentionHeads % NumKeyValueHeads != 0)
            throw new InvalidDataException("num_attention_heads must be divisible by num_key_value_heads");
        if (HeadDim % 2 != 0)
            throw new InvalidDataException("RoPE head_dim must be even");
    }
}

public sealed record TrainConfig
{
    public required string ModelConfigPath { get; init; }
    public required string ModelPath { get; init; }
    public required string TrainDataPath { get; init; }
    public required string OutputDirectory { get; init; }
    public required IReadOnlySet<string> TargetModules { get; init; }
    public long PadTokenId { get; init; } = -1;
    public long Rank { get; init; } = 16;
    public double Alpha { get; init; } = 32;
    public double Dropout { get; init; }
    public int BatchSize { get; init; } = 1;
    public int GradientAccumulationSteps { get; init; } = 1;
    public int Epochs { get; init; } = 1;
    public long MaxSteps { get; init; } = -1;
    public long WarmupSteps { get; init; }
    public long SaveEvery { get; init; }
    public long MaxSequenceLength { get; init; }
    public double LearningRate { get; init; } = 2e-4;
    public double WeightDecay { get; init; }
    public double MaxGradientNorm { get; init; } = 1;
    public long Seed { get; init; } = 42;
    public bool Shuffle { get; init; } = true;
    public string Device { get; init; } = "auto";
    public string DType { get; init; } = "float32";

    public void Validate(ModelConfig model)
    {
        var supported = new HashSet<string>(StringComparer.Ordinal) {
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        };

        if (string.IsNullOrWhiteSpace(ModelPath) || string.IsNullOrWhiteSpace(TrainDataPath) ||
            string.IsNullOrWhiteSpace(OutputDirectory))
            throw new ArgumentException("--model, --train-data, and --output-dir are required");
        if (TargetModules.Count == 0)
            throw new ArgumentException("--target-modules must name at least one projection");
        foreach (var name in TargetModules)
            if (!supported.Contains(name))
                throw new ArgumentException($"unsupported LoRA target module: {name}");
        if (PadTokenId < 0 || PadTokenId >= model.VocabSize)
            throw new ArgumentException("--pad-token-id must be within the model vocabulary");
        if (Rank <= 0 || Alpha <= 0 || Dropout is < 0 or >= 1)
            throw new ArgumentException("invalid LoRA rank, alpha, or dropout");
        if (BatchSize <= 0 || GradientAccumulationSteps <= 0 || Epochs <= 0)
            throw new ArgumentException("batch size, gradient accumulation, and epochs must be positive");
        if (MaxSteps == 0 || WarmupSteps < 0 || SaveEvery < 0)
            throw new ArgumentException("invalid step count");
        if (MaxSequenceLength is <= 1 || MaxSequenceLength > model.MaxPositionEmbeddings)
            throw new ArgumentException("--max-seq-length must be in [2, max_position_embeddings]");
        if (LearningRate <= 0 || WeightDecay < 0 || MaxGradientNorm < 0)
            throw new ArgumentException("invalid optimizer hyperparameter");
        if (Device is not ("auto" or "cpu" or "cuda"))
            throw new ArgumentException("--device must be auto, cpu, or cuda");
        if (DType is not ("float32" or "float16" or "bfloat16"))
            throw new ArgumentException("--dtype must be float32, float16, or bfloat16");
    }
}
