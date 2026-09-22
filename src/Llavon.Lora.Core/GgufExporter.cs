using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using LLama;
using LLama.Native;
using TorchSharp;
using static TorchSharp.torch;

namespace Llavon.Lora;

public enum GgufOutputType {
    Float16,
    Float32
}

public sealed record GgufExportConfig {
    public required string ModelConfigPath { get; init; }
    public required string ModelPath { get; init; }
    public required string VocabularyPath { get; init; }
    public string? AdapterDirectory { get; init; }
    public required string OutputPath { get; init; }
    public GgufOutputType OutputType { get; init; } = GgufOutputType.Float16;
    public string? QuantizationType { get; init; }
    public string? QuantizedOutputPath { get; init; }
    public int QuantizationThreads { get; init; }
    public bool Overwrite { get; init; }
    public string? ExpectedOutputSha256 { get; init; }
    public string? ExpectedQuantizedSha256 { get; init; }
}

public sealed record GgufExportResult(
    string OutputPath,
    string OutputSha256,
    string? QuantizedOutputPath,
    string? QuantizedOutputSha256);

public static class GgufExporter {
    private const int Alignment = 32;
    private const uint GgufMagic = 0x46554747;
    private const uint GgufVersion = 3;

    private enum GgufValueType : uint {
        Uint8 = 0,
        Int8 = 1,
        Uint16 = 2,
        Int16 = 3,
        Uint32 = 4,
        Int32 = 5,
        Float32 = 6,
        Bool = 7,
        String = 8,
        Array = 9,
        Uint64 = 10,
        Int64 = 11,
        Float64 = 12
    }

    private enum GgmlType : uint {
        Float32 = 0,
        Float16 = 1
    }

    private sealed record MetadataEntry(
        string Key,
        GgufValueType Type,
        object Value,
        GgufValueType? ElementType = null);

    private sealed record TensorSpec(
        string Name,
        string SourceName,
        bool KeepFloat32 = false,
        long? PermuteHeads = null);

    private sealed class AdapterMerger : IDisposable {
        private readonly Dictionary<string, Tensor> tensors;
        private readonly HashSet<string> consumed = new(StringComparer.Ordinal);
        private readonly double scaling;

        public AdapterMerger(string? adapterDirectory) {
            if (adapterDirectory is null) {
                tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
                scaling = 1;
                return;
            }

            ValidateAdapterMetadata(adapterDirectory);
            var config = AdapterConfig.Load(adapterDirectory);
            scaling = config.Alpha / config.Rank;
            tensors = SafeTensors.LoadModel(Path.Combine(adapterDirectory, "adapter_model.safetensors"));
            if (tensors.Count == 0)
                throw new InvalidDataException("adapter contains no tensors");
        }

        public Tensor Merge(string baseName, Tensor baseWeight) {
            var moduleName = baseName.EndsWith(".weight", StringComparison.Ordinal)
                ? baseName[..^".weight".Length]
                : throw new InvalidDataException($"cannot map non-weight tensor to LoRA: {baseName}");
            var prefix = $"base_model.model.{moduleName}";
            var aName = $"{prefix}.lora_A.weight";
            var bName = $"{prefix}.lora_B.weight";
            var hasA = tensors.TryGetValue(aName, out var a);
            var hasB = tensors.TryGetValue(bName, out var b);
            if (hasA != hasB)
                throw new InvalidDataException($"adapter is missing paired tensor: {(hasA ? bName : aName)}");
            if (!hasA)
                return baseWeight;

            var adapterA = a!;
            var adapterB = b!;
            if (baseWeight.dim() != 2 || adapterA.dim() != 2 || adapterB.dim() != 2 ||
                adapterA.shape[1] != baseWeight.shape[1] || adapterB.shape[0] != baseWeight.shape[0] ||
                adapterB.shape[1] != adapterA.shape[0])
                throw new InvalidDataException(
                    $"LoRA shape mismatch for {baseName}: base=[{string.Join(',', baseWeight.shape)}], " +
                    $"A=[{string.Join(',', adapterA.shape)}], B=[{string.Join(',', adapterB.shape)}]");

            consumed.Add(aName);
            consumed.Add(bName);
            return baseWeight.to_type(ScalarType.Float32) +
                   torch.matmul(adapterB.to_type(ScalarType.Float32), adapterA.to_type(ScalarType.Float32)) * scaling;
        }

        public void ValidateAllConsumed() {
            var unused = tensors.Keys.Where(name => !consumed.Contains(name)).Order(StringComparer.Ordinal).ToArray();
            if (unused.Length == 0)
                return;
            var preview = string.Join(", ", unused.Take(4));
            if (unused.Length > 4)
                preview += ", ...";
            throw new InvalidDataException($"adapter has {unused.Length} unused tensor(s): {preview}");
        }

        public void Dispose() {
            foreach (var tensor in tensors.Values)
                tensor.Dispose();
        }

        private static void ValidateAdapterMetadata(string adapterDirectory) {
            var path = Path.Combine(adapterDirectory, "adapter_config.json");
            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            var root = document.RootElement;
            if (root.GetProperty("peft_type").GetString() != "LORA")
                throw new InvalidDataException("only PEFT LoRA adapters are supported");
            if (root.TryGetProperty("fan_in_fan_out", out var fanInFanOut) && fanInFanOut.GetBoolean())
                throw new InvalidDataException("fan_in_fan_out LoRA adapters are not supported");
            if (root.TryGetProperty("bias", out var bias) && bias.GetString() != "none")
                throw new InvalidDataException("LoRA adapters containing trained bias are not supported");
        }
    }

    public static GgufExportResult Export(GgufExportConfig config) {
        ValidateConfig(config);
        var modelConfig = ModelConfig.Load(config.ModelConfigPath);
        var rawConfig = LoadObject(config.ModelConfigPath);
        var vocabulary = LoadObject(config.VocabularyPath);
        var tokens = ReadTokens(vocabulary);
        if (tokens.Count != modelConfig.VocabSize)
            throw new InvalidDataException(
                $"vocab size mismatch: tokens={tokens.Count}, config={modelConfig.VocabSize}");

        var weights = SafeTensors.LoadModel(config.ModelPath);
        try {
            using var adapter = new AdapterMerger(config.AdapterDirectory);
            WriteGguf(config, modelConfig, rawConfig, vocabulary, tokens, weights, adapter);
            var outputHash = VerifySha256(config.OutputPath, config.ExpectedOutputSha256);

            if (config.QuantizationType is null)
                return new GgufExportResult(config.OutputPath, outputHash, null, null);

            var quantizedPath = config.QuantizedOutputPath!;
            Quantize(config.OutputPath, quantizedPath, config.QuantizationType,
                config.QuantizationThreads, config.Overwrite);
            var quantizedHash = VerifySha256(quantizedPath, config.ExpectedQuantizedSha256);
            return new GgufExportResult(config.OutputPath, outputHash, quantizedPath, quantizedHash);
        } finally {
            foreach (var tensor in weights.Values)
                tensor.Dispose();
        }
    }

    private static void WriteGguf(
        GgufExportConfig export,
        ModelConfig model,
        JsonElement rawConfig,
        JsonElement vocabulary,
        IReadOnlyList<string> tokens,
        IReadOnlyDictionary<string, Tensor> weights,
        AdapterMerger adapter) {
        var outputPath = Path.GetFullPath(export.OutputPath);
        if (File.Exists(outputPath) && !export.Overwrite)
            throw new IOException($"{outputPath} exists; pass --force to overwrite");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        var metadata = BuildMetadata(export.OutputType, model, rawConfig, vocabulary, tokens);
        var specs = BuildTensorSpecs(model);
        foreach (var spec in specs)
            if (!weights.ContainsKey(spec.SourceName))
                throw new InvalidDataException($"base checkpoint is missing tensor: {spec.SourceName}");

        ValidatePermutation(weights, adapter, model.NumAttentionHeads);
        var temporaryPath = $"{outputPath}.{Guid.NewGuid():N}.tmp";
        try {
            using (var output = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None)) {
                WriteUInt32(output, GgufMagic);
                WriteUInt32(output, GgufVersion);
                WriteUInt64(output, checked((ulong)specs.Count));
                WriteUInt64(output, checked((ulong)metadata.Count));
                foreach (var entry in metadata)
                    WriteMetadata(output, entry);

                ulong offset = 0;
                foreach (var spec in specs) {
                    var source = weights[spec.SourceName];
                    WriteString(output, spec.Name);
                    WriteUInt32(output, checked((uint)source.shape.Length));
                    for (var index = source.shape.Length - 1; index >= 0; --index)
                        WriteUInt64(output, checked((ulong)source.shape[index]));
                    WriteUInt32(output, (uint)GetTensorType(export.OutputType, spec.KeepFloat32));
                    WriteUInt64(output, offset);
                    offset = checked(offset + (ulong)Align(GetTensorBytes(source, export.OutputType, spec.KeepFloat32)));
                }

                WritePadding(output, output.Position);
                foreach (var spec in specs) {
                    using var scope = torch.NewDisposeScope();
                    var tensor = adapter.Merge(spec.SourceName, weights[spec.SourceName]);
                    if (spec.PermuteHeads is not null)
                        tensor = PermuteQueryOrKey(tensor, spec.PermuteHeads.Value);
                    var targetType = GetTensorType(export.OutputType, spec.KeepFloat32) == GgmlType.Float16
                        ? ScalarType.Float16
                        : ScalarType.Float32;
                    using var packed = tensor.to_type(targetType).cpu().contiguous();
                    packed.WriteBytesToStream(output, 1024 * 1024);
                    WritePadding(output, checked(packed.numel() * packed.element_size()));
                }
                adapter.ValidateAllConsumed();
            }

            File.Move(temporaryPath, outputPath, export.Overwrite);
        } finally {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
        }
    }

    private static List<MetadataEntry> BuildMetadata(
        GgufOutputType outputType,
        ModelConfig model,
        JsonElement rawConfig,
        JsonElement vocabulary,
        IReadOnlyList<string> tokens) {
        var tokenTypes = new int[tokens.Count];
        Array.Fill(tokenTypes, 1);
        for (var index = 0; index < tokens.Count; ++index) {
            var token = tokens[index];
            if (token is "<PAD>" or "<BOS>" or "<EOS>" or "<UNK>")
                tokenTypes[index] = 3;
            else if (token.StartsWith('<') && token.EndsWith('>'))
                tokenTypes[index] = 4;
        }

        var unknownToken = -1;
        for (var index = 0; index < tokens.Count; ++index)
            if (tokens[index] == "<UNK>") {
                unknownToken = index;
                break;
            }
        if (unknownToken < 0)
            throw new InvalidDataException("vocabulary does not contain <UNK>");
        ValidateSpecialTokens(vocabulary, tokens);

        return [
            new("general.architecture", GgufValueType.String, "llama"),
            new("general.file_type", GgufValueType.Uint32, outputType == GgufOutputType.Float16 ? 1u : 0u),
            new("llama.context_length", GgufValueType.Uint32, checked((uint)model.MaxPositionEmbeddings)),
            new("llama.embedding_length", GgufValueType.Uint32, checked((uint)model.HiddenSize)),
            new("llama.feed_forward_length", GgufValueType.Uint32, checked((uint)model.IntermediateSize)),
            new("llama.block_count", GgufValueType.Uint32, checked((uint)model.NumHiddenLayers)),
            new("llama.attention.head_count", GgufValueType.Uint32, checked((uint)model.NumAttentionHeads)),
            new("llama.attention.head_count_kv", GgufValueType.Uint32, checked((uint)model.NumKeyValueHeads)),
            new("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, (float)model.RmsNormEpsilon),
            new("llama.rope.dimension_count", GgufValueType.Uint32, checked((uint)model.HeadDim)),
            new("llama.rope.freq_base", GgufValueType.Float32, (float)model.RopeTheta),
            new("llama.rope.scaling.type", GgufValueType.String, "none"),
            new("tokenizer.ggml.model", GgufValueType.String, "gpt2"),
            new("tokenizer.ggml.pre", GgufValueType.String, "default"),
            new("tokenizer.ggml.tokens", GgufValueType.Array, tokens, GgufValueType.String),
            new("tokenizer.ggml.scores", GgufValueType.Array, new float[tokens.Count], GgufValueType.Float32),
            new("tokenizer.ggml.token_type", GgufValueType.Array, tokenTypes, GgufValueType.Int32),
            new("tokenizer.ggml.merges", GgufValueType.Array, new[] { "<PAD> <BOS>" }, GgufValueType.String),
            new("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, ReadUInt32(rawConfig, "bos_token_id", 1)),
            new("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, ReadUInt32(rawConfig, "eos_token_id", 2)),
            new("tokenizer.ggml.unknown_token_id", GgufValueType.Uint32, checked((uint)unknownToken)),
            new("tokenizer.ggml.padding_token_id", GgufValueType.Uint32, ReadUInt32(rawConfig, "pad_token_id", 0)),
            new("tokenizer.ggml.add_bos_token", GgufValueType.Bool, true)
        ];
    }

    private static List<TensorSpec> BuildTensorSpecs(ModelConfig model) {
        var specs = new List<TensorSpec> {
            new("token_embd.weight", "model.embed_tokens.weight")
        };
        for (var layer = 0; layer < model.NumHiddenLayers; ++layer) {
            var source = $"model.layers.{layer}";
            var target = $"blk.{layer}";
            specs.Add(new($"{target}.attn_norm.weight", $"{source}.input_layernorm.weight", true));
            specs.Add(new($"{target}.attn_q.weight", $"{source}.self_attn.q_proj.weight",
                PermuteHeads: model.NumAttentionHeads));
            specs.Add(new($"{target}.attn_k.weight", $"{source}.self_attn.k_proj.weight",
                PermuteHeads: model.NumKeyValueHeads));
            specs.Add(new($"{target}.attn_v.weight", $"{source}.self_attn.v_proj.weight"));
            specs.Add(new($"{target}.attn_output.weight", $"{source}.self_attn.o_proj.weight"));
            specs.Add(new($"{target}.ffn_norm.weight", $"{source}.post_attention_layernorm.weight", true));
            specs.Add(new($"{target}.ffn_gate.weight", $"{source}.mlp.gate_proj.weight"));
            specs.Add(new($"{target}.ffn_up.weight", $"{source}.mlp.up_proj.weight"));
            specs.Add(new($"{target}.ffn_down.weight", $"{source}.mlp.down_proj.weight"));
        }
        specs.Add(new("output_norm.weight", "model.norm.weight", true));
        specs.Add(new("output.weight", "lm_head.weight"));
        return specs;
    }

    private static void ValidatePermutation(
        IReadOnlyDictionary<string, Tensor> weights,
        AdapterMerger adapter,
        long heads) {
        using var scope = torch.NewDisposeScope();
        var source = adapter.Merge("model.layers.0.self_attn.q_proj.weight",
            weights["model.layers.0.self_attn.q_proj.weight"]);
        var permuted = PermuteQueryOrKey(source, heads);
        var roundtrip = ReversePermuteQueryOrKey(permuted, heads);
        using var roundtripDifference = (source.to_type(ScalarType.Float32) - roundtrip.to_type(ScalarType.Float32)).abs().max();
        if (roundtripDifference.item<float>() != 0)
            throw new InvalidDataException("Q permutation roundtrip failed");
    }

    private static Tensor PermuteQueryOrKey(Tensor weight, long heads) {
        if (weight.dim() != 2 || weight.shape[0] % heads != 0)
            throw new InvalidDataException($"invalid Q/K shape [{string.Join(',', weight.shape)}] for {heads} heads");
        var headDimension = weight.shape[0] / heads;
        if (headDimension % 2 != 0)
            throw new InvalidDataException("Q/K head dimension must be even");
        return weight.reshape(heads, 2, headDimension / 2, weight.shape[1])
            .transpose(1, 2)
            .reshape(weight.shape)
            .contiguous();
    }

    private static Tensor ReversePermuteQueryOrKey(Tensor weight, long heads) {
        var headDimension = weight.shape[0] / heads;
        return weight.reshape(heads, headDimension / 2, 2, weight.shape[1])
            .transpose(1, 2)
            .reshape(weight.shape)
            .contiguous();
    }

    private static void Quantize(
        string source,
        string output,
        string quantizationType,
        int threads,
        bool overwrite) {
        var outputPath = Path.GetFullPath(output);
        if (File.Exists(outputPath) && !overwrite)
            throw new IOException($"{outputPath} exists; pass --force to overwrite");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        var temporaryPath = $"{outputPath}.{Guid.NewGuid():N}.tmp";
        try {
            ConfigureQuantizerLibrary();
            if (!LLamaQuantizer.Quantize(
                    Path.GetFullPath(source), temporaryPath, quantizationType,
                    threads, allowRequantize: false, quantizeOutputTensor: true))
                throw new InvalidOperationException($"llama.cpp failed to quantize the model as {quantizationType}");
            File.Move(temporaryPath, outputPath, overwrite);
        } finally {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
        }
    }

    private static void ConfigureQuantizerLibrary() {
        if (NativeLibraryConfig.LLama.LibraryHasLoaded)
            return;
        NativeLibraryConfig.LLama.WithCuda(false).WithVulkan(false);
        // LLamaModelQuantizeParams.Default() contains its own P/Invoke and does not
        // initialize NativeApi's resolver first. Force normal LLamaSharp backend
        // initialization before LLamaQuantizer reaches that call.
        NativeApi.llama_empty_call();
    }

    private static void WriteMetadata(Stream output, MetadataEntry entry) {
        WriteString(output, entry.Key);
        WriteUInt32(output, (uint)entry.Type);
        WriteValue(output, entry.Type, entry.Value, entry.ElementType);
    }

    private static void WriteValue(
        Stream output,
        GgufValueType type,
        object value,
        GgufValueType? elementType = null) {
        switch (type) {
            case GgufValueType.Uint32:
                WriteUInt32(output, (uint)value);
                break;
            case GgufValueType.Int32:
                WriteInt32(output, (int)value);
                break;
            case GgufValueType.Float32:
                WriteFloat32(output, (float)value);
                break;
            case GgufValueType.Bool:
                output.WriteByte((bool)value ? (byte)1 : (byte)0);
                break;
            case GgufValueType.String:
                WriteString(output, (string)value);
                break;
            case GgufValueType.Array:
                WriteArray(output, value, elementType ?? throw new InvalidDataException("GGUF array type is missing"));
                break;
            default:
                throw new NotSupportedException($"GGUF metadata type {type} is not supported");
        }
    }

    private static void WriteArray(Stream output, object value, GgufValueType elementType) {
        WriteUInt32(output, (uint)elementType);
        switch (value) {
            case IReadOnlyList<string> strings:
                WriteUInt64(output, checked((ulong)strings.Count));
                foreach (var item in strings)
                    WriteString(output, item);
                break;
            case float[] floats:
                WriteUInt64(output, checked((ulong)floats.Length));
                foreach (var item in floats)
                    WriteFloat32(output, item);
                break;
            case int[] integers:
                WriteUInt64(output, checked((ulong)integers.Length));
                foreach (var item in integers)
                    WriteInt32(output, item);
                break;
            default:
                throw new NotSupportedException($"GGUF array value {value.GetType()} is not supported");
        }
    }

    private static void WriteString(Stream output, string value) {
        var bytes = Encoding.UTF8.GetBytes(value);
        WriteUInt64(output, checked((ulong)bytes.Length));
        output.Write(bytes);
    }

    private static void WriteUInt32(Stream output, uint value) {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        output.Write(bytes);
    }

    private static void WriteInt32(Stream output, int value) {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        output.Write(bytes);
    }

    private static void WriteUInt64(Stream output, ulong value) {
        Span<byte> bytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        output.Write(bytes);
    }

    private static void WriteFloat32(Stream output, float value) =>
        WriteInt32(output, BitConverter.SingleToInt32Bits(value));

    private static void WritePadding(Stream output, long size) {
        var padding = Align(size) - size;
        if (padding > 0)
            output.Write(new byte[padding]);
    }

    private static long Align(long value) => checked((value + Alignment - 1) / Alignment * Alignment);

    private static long GetTensorBytes(Tensor tensor, GgufOutputType type, bool keepFloat32) =>
        checked(tensor.numel() * (type == GgufOutputType.Float16 && !keepFloat32 ? 2 : 4));

    private static GgmlType GetTensorType(GgufOutputType type, bool keepFloat32) =>
        type == GgufOutputType.Float16 && !keepFloat32 ? GgmlType.Float16 : GgmlType.Float32;

    private static JsonElement LoadObject(string path) {
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        if (document.RootElement.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException($"expected a JSON object: {path}");
        return document.RootElement.Clone();
    }

    private static IReadOnlyList<string> ReadTokens(JsonElement vocabulary) {
        if (!vocabulary.TryGetProperty("tokens", out var value) || value.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException("vocabulary is missing a tokens array");
        return value.EnumerateArray()
            .Select(item => item.GetString() ?? throw new InvalidDataException("vocabulary contains a null token"))
            .ToArray();
    }

    private static void ValidateSpecialTokens(JsonElement vocabulary, IReadOnlyList<string> tokens) {
        if (!vocabulary.TryGetProperty("special_tokens", out var specialTokens))
            return;
        foreach (var value in specialTokens.EnumerateArray()) {
            var token = value.GetString() ?? throw new InvalidDataException("special_tokens contains null");
            if (!tokens.Contains(token, StringComparer.Ordinal))
                throw new InvalidDataException($"special token is absent from tokens: {token}");
        }
    }

    private static uint ReadUInt32(JsonElement root, string name, uint fallback) =>
        root.TryGetProperty(name, out var value) ? value.GetUInt32() : fallback;

    private static string VerifySha256(string path, string? expected) {
        using var input = File.OpenRead(path);
        var actual = Convert.ToHexString(SHA256.HashData(input)).ToLowerInvariant();
        if (expected is null)
            return actual;
        var normalized = expected.Trim().ToLowerInvariant();
        if (normalized.Length != 64 || normalized.Any(character => !Uri.IsHexDigit(character)))
            throw new ArgumentException($"invalid expected SHA-256: {expected}");
        if (!string.Equals(actual, normalized, StringComparison.Ordinal))
            throw new InvalidDataException($"SHA-256 mismatch for {path}: expected {normalized}, got {actual}");
        return actual;
    }

    private static void ValidateConfig(GgufExportConfig config) {
        if (!File.Exists(config.ModelConfigPath))
            throw new FileNotFoundException("model config does not exist", config.ModelConfigPath);
        if (!File.Exists(config.VocabularyPath))
            throw new FileNotFoundException("vocabulary does not exist", config.VocabularyPath);
        if (string.IsNullOrWhiteSpace(config.ModelPath) || string.IsNullOrWhiteSpace(config.OutputPath))
            throw new ArgumentException("model and output paths are required");
        if (config.QuantizationThreads < 0)
            throw new ArgumentException("quantization threads cannot be negative");
        if (config.QuantizationType is null && config.QuantizedOutputPath is not null)
            throw new ArgumentException("quantized output requires a quantization type");
        if (config.QuantizationType is not null && string.IsNullOrWhiteSpace(config.QuantizedOutputPath))
            throw new ArgumentException("quantization requires a quantized output path");
        if (config.QuantizationType is null && config.ExpectedQuantizedSha256 is not null)
            throw new ArgumentException("expected quantized SHA-256 requires quantization");
        if (config.QuantizedOutputPath is not null &&
            string.Equals(Path.GetFullPath(config.OutputPath), Path.GetFullPath(config.QuantizedOutputPath),
                StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("unquantized and quantized output paths must be different");
        if (!config.Overwrite && File.Exists(Path.GetFullPath(config.OutputPath)))
            throw new IOException($"{Path.GetFullPath(config.OutputPath)} exists; pass --force to overwrite");
        if (!config.Overwrite && config.QuantizedOutputPath is not null &&
            File.Exists(Path.GetFullPath(config.QuantizedOutputPath)))
            throw new IOException($"{Path.GetFullPath(config.QuantizedOutputPath)} exists; pass --force to overwrite");
    }
}
