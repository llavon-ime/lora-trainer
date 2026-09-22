using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Llavon.Lora;

public sealed class LoraLinear : Module<Tensor, Tensor>
{
    private Tensor weight;
    private Tensor? bias;
    private Parameter? lora_A;
    private Parameter? lora_B;
    private Dropout? lora_dropout;
    private readonly double scaling;

    public bool IsLoraEnabled { get; }
    public Tensor LoraA => lora_A ?? throw new InvalidOperationException("LoRA is not enabled");
    public Tensor LoraB => lora_B ?? throw new InvalidOperationException("LoRA is not enabled");

    public LoraLinear(
        long inputFeatures,
        long outputFeatures,
        long rank,
        double alpha,
        double dropout,
        bool enableLora,
        bool useBias) : base(nameof(LoraLinear))
    {
        weight = torch.empty(outputFeatures, inputFeatures, dtype: ScalarType.Float32);
        bias = useBias ? torch.empty(outputFeatures, dtype: ScalarType.Float32) : null;
        IsLoraEnabled = enableLora;
        scaling = enableLora ? alpha / rank : 1;
        if (enableLora)
        {
            lora_A = Parameter(torch.empty(rank, inputFeatures, dtype: ScalarType.Float32));
            lora_B = Parameter(torch.zeros(outputFeatures, rank, dtype: ScalarType.Float32));
            init.kaiming_uniform_(lora_A, Math.Sqrt(5));
            lora_dropout = Dropout(dropout);
        }
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var output = functional.linear(input, weight, bias);
        if (!IsLoraEnabled)
            return output;

        var dropped = lora_dropout!.call(input);
        var projected = functional.linear(dropped, lora_A!);
        var update = functional.linear(projected, lora_B!);
        return output + update * scaling;
    }

    public void SetBaseWeight(Tensor value) => CopyTensor(weight, value, "linear weight");

    public void SetBaseBias(Tensor value)
    {
        if (bias is null)
            throw new InvalidOperationException("linear layer has no bias");
        CopyTensor(bias, value, "linear bias");
    }

    internal static void CopyTensor(Tensor destination, Tensor source, string name)
    {
        if (!destination.shape.SequenceEqual(source.shape))
            throw new InvalidDataException($"shape mismatch for {name}: " +
                                           $"expected [{string.Join(',', destination.shape)}], " +
                                           $"got [{string.Join(',', source.shape)}]");
        using var noGrad = torch.no_grad();
        using var converted = source.to(destination.dtype, destination.device, copy: true);
        destination.copy_(converted);
    }
}

internal sealed class RmsNorm : Module<Tensor, Tensor>
{
    private Tensor weight;
    private readonly double epsilon;

    public Tensor Weight => weight;

    public RmsNorm(long hiddenSize, double epsilon) : base(nameof(RmsNorm))
    {
        weight = torch.ones(hiddenSize, dtype: ScalarType.Float32);
        this.epsilon = epsilon;
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var inputType = input.dtype;
        var value = input.to_type(ScalarType.Float32);
        var variance = value.square().mean([-1], keepdim: true);
        var normalized = value * (variance + epsilon).rsqrt();
        return (normalized * weight.to_type(ScalarType.Float32)).to_type(inputType);
    }
}

internal sealed class LlamaAttention : Module<Tensor, Tensor, Tensor, Tensor>
{
    public LoraLinear q_proj;
    public LoraLinear k_proj;
    public LoraLinear v_proj;
    public LoraLinear o_proj;
    private readonly long heads;
    private readonly long keyValueHeads;
    private readonly long headDimension;
    private readonly double ropeTheta;

    public LlamaAttention(ModelConfig config, long rank, double alpha, double dropout, IReadOnlySet<string> targets)
        : base(nameof(LlamaAttention))
    {
        heads = config.NumAttentionHeads;
        keyValueHeads = config.NumKeyValueHeads;
        headDimension = config.HeadDim;
        ropeTheta = config.RopeTheta;
        q_proj = new LoraLinear(config.HiddenSize, heads * headDimension, rank, alpha, dropout,
            targets.Contains("q_proj"), config.AttentionBias);
        k_proj = new LoraLinear(config.HiddenSize, keyValueHeads * headDimension, rank, alpha, dropout,
            targets.Contains("k_proj"), config.AttentionBias);
        v_proj = new LoraLinear(config.HiddenSize, keyValueHeads * headDimension, rank, alpha, dropout,
            targets.Contains("v_proj"), config.AttentionBias);
        o_proj = new LoraLinear(heads * headDimension, config.HiddenSize, rank, alpha, dropout,
            targets.Contains("o_proj"), config.AttentionBias);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hidden, Tensor attentionMask, Tensor positions)
    {
        var batch = hidden.shape[0];
        var sequence = hidden.shape[1];
        var query = q_proj.call(hidden).view(batch, sequence, heads, headDimension).transpose(1, 2);
        var key = k_proj.call(hidden).view(batch, sequence, keyValueHeads, headDimension).transpose(1, 2);
        var value = v_proj.call(hidden).view(batch, sequence, keyValueHeads, headDimension).transpose(1, 2);
        (query, key) = ApplyRotary(query, key, positions);

        var groups = heads / keyValueHeads;
        if (groups != 1)
        {
            key = key.repeat_interleave(groups, 1);
            value = value.repeat_interleave(groups, 1);
        }

        var scores = torch.matmul(query, key.transpose(-2, -1)).to_type(ScalarType.Float32) /
                     Math.Sqrt(headDimension);
        var causal = torch.ones(sequence, sequence, dtype: ScalarType.Bool, device: hidden.device).triu(1);
        scores.masked_fill_(causal.view(1, 1, sequence, sequence), -1.0e30);
        var invalidKeys = attentionMask.logical_not().view(batch, 1, 1, sequence);
        scores.masked_fill_(invalidKeys, -1.0e30);
        var probabilities = torch.softmax(scores, -1).to_type(query.dtype);
        var output = torch.matmul(probabilities, value)
            .transpose(1, 2)
            .contiguous()
            .view(batch, sequence, heads * headDimension);
        return o_proj.call(output);
    }

    private (Tensor Query, Tensor Key) ApplyRotary(Tensor query, Tensor key, Tensor positions)
    {
        var half = headDimension / 2;
        var exponents = torch.arange(half, dtype: ScalarType.Float32, device: query.device);
        var inverseFrequency = (-Math.Log(ropeTheta) * exponents / half).exp();
        var frequencies = positions.to_type(ScalarType.Float32).unsqueeze(-1) * inverseFrequency;
        var cosine = frequencies.cos().unsqueeze(1).to_type(query.dtype);
        var sine = frequencies.sin().unsqueeze(1).to_type(query.dtype);

        Tensor Rotate(Tensor input)
        {
            var first = input.slice(-1, 0, half, 1);
            var second = input.slice(-1, half, half * 2, 1);
            return torch.cat([first * cosine - second * sine, second * cosine + first * sine], -1);
        }

        return (Rotate(query), Rotate(key));
    }
}

internal sealed class LlamaMlp : Module<Tensor, Tensor>
{
    public LoraLinear gate_proj;
    public LoraLinear up_proj;
    public LoraLinear down_proj;

    public LlamaMlp(ModelConfig config, long rank, double alpha, double dropout, IReadOnlySet<string> targets)
        : base(nameof(LlamaMlp))
    {
        gate_proj = new LoraLinear(config.HiddenSize, config.IntermediateSize, rank, alpha, dropout,
            targets.Contains("gate_proj"), config.MlpBias);
        up_proj = new LoraLinear(config.HiddenSize, config.IntermediateSize, rank, alpha, dropout,
            targets.Contains("up_proj"), config.MlpBias);
        down_proj = new LoraLinear(config.IntermediateSize, config.HiddenSize, rank, alpha, dropout,
            targets.Contains("down_proj"), config.MlpBias);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input) =>
        down_proj.call(gate_proj.call(input).silu() * up_proj.call(input));
}

internal sealed class LlamaDecoderLayer : Module<Tensor, Tensor, Tensor, Tensor>
{
    public LlamaAttention self_attn;
    public LlamaMlp mlp;
    public RmsNorm input_layernorm;
    public RmsNorm post_attention_layernorm;

    public LlamaDecoderLayer(
        ModelConfig config,
        long rank,
        double alpha,
        double dropout,
        IReadOnlySet<string> targets) : base(nameof(LlamaDecoderLayer))
    {
        self_attn = new LlamaAttention(config, rank, alpha, dropout, targets);
        mlp = new LlamaMlp(config, rank, alpha, dropout, targets);
        input_layernorm = new RmsNorm(config.HiddenSize, config.RmsNormEpsilon);
        post_attention_layernorm = new RmsNorm(config.HiddenSize, config.RmsNormEpsilon);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input, Tensor attentionMask, Tensor positions)
    {
        var hidden = input + self_attn.call(input_layernorm.call(input), attentionMask, positions);
        return hidden + mlp.call(post_attention_layernorm.call(hidden));
    }
}

public sealed class LlamaForCausalLm : Module<Tensor, Tensor, Tensor>
{
    private Embedding embed_tokens;
    private ModuleList<LlamaDecoderLayer> layers;
    private RmsNorm norm;
    private LoraLinear lm_head;
    private readonly ModelConfig config;
    private readonly long rank;
    private readonly double alpha;
    private readonly double dropout;
    private readonly IReadOnlySet<string> targetModules;

    public ModelConfig Config => config;

    public LlamaForCausalLm(
        ModelConfig config,
        long rank,
        double alpha,
        double dropout,
        IReadOnlySet<string> targetModules) : base(nameof(LlamaForCausalLm))
    {
        this.config = config;
        this.rank = rank;
        this.alpha = alpha;
        this.dropout = dropout;
        this.targetModules = targetModules;
        embed_tokens = Embedding(config.VocabSize, config.HiddenSize);
        embed_tokens.weight!.requires_grad = false;
        layers = ModuleList<LlamaDecoderLayer>();
        for (var index = 0L; index < config.NumHiddenLayers; ++index)
            layers.Add(new LlamaDecoderLayer(config, rank, alpha, dropout, targetModules));
        norm = new RmsNorm(config.HiddenSize, config.RmsNormEpsilon);
        lm_head = new LoraLinear(config.HiddenSize, config.VocabSize, rank, alpha, dropout, false, false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor inputIds, Tensor attentionMask)
    {
        var hidden = embed_tokens.call(inputIds);
        var positions = attentionMask.to_type(ScalarType.Int64).cumsum(-1) - 1;
        positions.clamp_min_(0);
        foreach (var layer in layers)
            hidden = layer.call(hidden, attentionMask, positions);
        return lm_head.call(norm.call(hidden));
    }

    public void LoadBaseWeights(string modelPath)
    {
        var tensors = SafeTensors.LoadModel(modelPath);
        try
        {
            LoraLinear.CopyTensor(embed_tokens.weight!, Require(tensors, "model.embed_tokens.weight"),
                "model.embed_tokens.weight");
            for (var index = 0; index < layers.Count; ++index)
            {
                var layer = layers[index];
                var prefix = $"model.layers.{index}";
                LoraLinear.CopyTensor(layer.input_layernorm.Weight,
                    Require(tensors, $"{prefix}.input_layernorm.weight"), $"{prefix}.input_layernorm.weight");
                LoraLinear.CopyTensor(layer.post_attention_layernorm.Weight,
                    Require(tensors, $"{prefix}.post_attention_layernorm.weight"),
                    $"{prefix}.post_attention_layernorm.weight");
                LoadLinear(tensors, $"{prefix}.self_attn.q_proj", layer.self_attn.q_proj);
                LoadLinear(tensors, $"{prefix}.self_attn.k_proj", layer.self_attn.k_proj);
                LoadLinear(tensors, $"{prefix}.self_attn.v_proj", layer.self_attn.v_proj);
                LoadLinear(tensors, $"{prefix}.self_attn.o_proj", layer.self_attn.o_proj);
                LoadLinear(tensors, $"{prefix}.mlp.gate_proj", layer.mlp.gate_proj);
                LoadLinear(tensors, $"{prefix}.mlp.up_proj", layer.mlp.up_proj);
                LoadLinear(tensors, $"{prefix}.mlp.down_proj", layer.mlp.down_proj);
            }
            LoraLinear.CopyTensor(norm.Weight, Require(tensors, "model.norm.weight"), "model.norm.weight");
            var headWeight = config.TieWordEmbeddings && !tensors.ContainsKey("lm_head.weight")
                ? Require(tensors, "model.embed_tokens.weight")
                : Require(tensors, "lm_head.weight");
            lm_head.SetBaseWeight(headWeight);
        }
        finally
        {
            foreach (var tensor in tensors.Values)
                tensor.Dispose();
        }
    }

    public Parameter[] TrainableParameters() => parameters().Where(parameter => parameter.requires_grad).ToArray();

    public void SavePeftAdapter(string outputDirectory, string baseModel)
    {
        var adapter = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        for (var index = 0; index < layers.Count; ++index)
        {
            var layer = layers[index];
            var prefix = $"base_model.model.model.layers.{index}";
            AddAdapter(adapter, $"{prefix}.self_attn.q_proj", layer.self_attn.q_proj);
            AddAdapter(adapter, $"{prefix}.self_attn.k_proj", layer.self_attn.k_proj);
            AddAdapter(adapter, $"{prefix}.self_attn.v_proj", layer.self_attn.v_proj);
            AddAdapter(adapter, $"{prefix}.self_attn.o_proj", layer.self_attn.o_proj);
            AddAdapter(adapter, $"{prefix}.mlp.gate_proj", layer.mlp.gate_proj);
            AddAdapter(adapter, $"{prefix}.mlp.up_proj", layer.mlp.up_proj);
            AddAdapter(adapter, $"{prefix}.mlp.down_proj", layer.mlp.down_proj);
        }

        Directory.CreateDirectory(outputDirectory);
        SafeTensors.Save(Path.Combine(outputDirectory, "adapter_model.safetensors"), adapter);
        var metadata = new
        {
            base_model_name_or_path = baseModel,
            bias = "none",
            fan_in_fan_out = false,
            inference_mode = true,
            init_lora_weights = true,
            lora_alpha = alpha,
            lora_dropout = dropout,
            peft_type = "LORA",
            r = rank,
            target_modules = targetModules.Order(StringComparer.Ordinal).ToArray(),
            task_type = "CAUSAL_LM"
        };
        File.WriteAllText(
            Path.Combine(outputDirectory, "adapter_config.json"),
            JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> tensors, string name) =>
        tensors.TryGetValue(name, out var tensor)
            ? tensor
            : throw new InvalidDataException($"base checkpoint is missing tensor: {name}");

    private static void LoadLinear(IReadOnlyDictionary<string, Tensor> tensors, string prefix, LoraLinear linear)
    {
        linear.SetBaseWeight(Require(tensors, $"{prefix}.weight"));
        if (tensors.TryGetValue($"{prefix}.bias", out var bias))
            linear.SetBaseBias(bias);
    }

    private static void AddAdapter(IDictionary<string, Tensor> adapter, string prefix, LoraLinear linear)
    {
        if (!linear.IsLoraEnabled)
            return;
        adapter.Add($"{prefix}.lora_A.weight", linear.LoraA);
        adapter.Add($"{prefix}.lora_B.weight", linear.LoraB);
    }
}
