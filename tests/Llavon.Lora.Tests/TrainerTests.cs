using Llavon.Lora;
using System.Text.Json;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Llavon.Lora.Tests;

public sealed class TrainerTests
{
    [Fact]
    public void DatasetLoadsAliasesAndCandidateMasks()
    {
        var path = TemporaryPath(".jsonl");
        try
        {
            File.WriteAllText(path,
                """
                {"input_ids":[1,4,3,7,2],"labels":[-100,-100,-100,7,2],"loss_mask":[0,0,0,1,1],"candidate_masks":[null,null,null,[8,7,7],null]}
                """);
            var samples = Dataset.LoadJsonLines(path, 10, 8);
            Assert.Single(samples);
            Assert.Equal([7L, 8L], samples[0].CandidateMasks[3]!);
            using var batch = Dataset.MakeBatch(samples, [0], 0);
            Assert.Equal([1L, 5L], batch.Tokens.shape);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void DatasetRejectsInvalidFirstToken()
    {
        var path = TemporaryPath(".jsonl");
        try
        {
            File.WriteAllText(path,
                """
                {"tokens":[99,2],"labels":[-100,2],"loss_weights":[0,1]}
                """);
            Assert.Throws<InvalidDataException>(() => Dataset.LoadJsonLines(path, 10, 8));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CandidateConstrainedLossUsesOnlyProvidedTokens()
    {
        using var batch = new TrainingBatch
        {
            Tokens = torch.tensor(new long[,] { { 0, 0 } }),
            Labels = torch.tensor(new long[,] { { -100, 1 } }),
            LossWeights = torch.tensor(new float[,] { { 0, 1 } }),
            AttentionMask = torch.tensor(new bool[,] { { true, true } }),
            CandidateMasks = new List<IReadOnlyList<long[]?>> { new long[]?[] { null, [1, 2] } }
        };
        using var logits = torch.zeros(1, 2, 4, dtype: ScalarType.Float32);
        logits[0, 0, 1] = 2;
        logits[0, 0, 2] = 1;
        using var loss = Trainer.CandidateConstrainedLoss(logits, batch);
        Assert.Equal(0.3132617, loss.item<float>(), 5);
    }

    [Fact]
    public void CandidateConstrainedLossAddsMissingTarget()
    {
        using var batch = new TrainingBatch
        {
            Tokens = torch.tensor(new long[,] { { 0, 0 } }),
            Labels = torch.tensor(new long[,] { { -100, 1 } }),
            LossWeights = torch.tensor(new float[,] { { 0, 1 } }),
            AttentionMask = torch.tensor(new bool[,] { { true, true } }),
            CandidateMasks = new List<IReadOnlyList<long[]?>> { new long[]?[] { null, [2] } }
        };
        using var logits = torch.zeros(1, 2, 4, dtype: ScalarType.Float32);
        logits[0, 0, 1] = 2;
        logits[0, 0, 2] = 1;
        using var loss = Trainer.CandidateConstrainedLoss(logits, batch);
        Assert.Equal(0.3132617, loss.item<float>(), 5);
    }

    [Fact]
    public void SafeTensorsRoundTrips()
    {
        var path = TemporaryPath(".safetensors");
        using var source = torch.arange(6, dtype: ScalarType.Float32).view(2, 3);
        try
        {
            SafeTensors.Save(path, new Dictionary<string, Tensor> { ["weight"] = source });
            var loaded = SafeTensors.LoadModel(path);
            try
            {
                Assert.True(loaded.ContainsKey("weight"));
                Assert.True(torch.allclose(source, loaded["weight"]));
            }
            finally
            {
                foreach (var tensor in loaded.Values)
                    tensor.Dispose();
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TinyLlamaTrainsAndWritesPeftAdapter()
    {
        var root = Path.Combine(Path.GetTempPath(), $"llavon-lora-{Guid.NewGuid():N}");
        var modelDirectory = Path.Combine(root, "model");
        var outputDirectory = Path.Combine(root, "adapter");
        Directory.CreateDirectory(modelDirectory);
        var weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try
        {
            torch.manual_seed(7);
            weights.Add("model.embed_tokens.weight", torch.randn(8, 4) * 0.02);
            weights.Add("model.layers.0.input_layernorm.weight", torch.ones(4));
            weights.Add("model.layers.0.post_attention_layernorm.weight", torch.ones(4));
            weights.Add("model.layers.0.self_attn.q_proj.weight", torch.randn(4, 4) * 0.02);
            weights.Add("model.layers.0.self_attn.k_proj.weight", torch.randn(2, 4) * 0.02);
            weights.Add("model.layers.0.self_attn.v_proj.weight", torch.randn(2, 4) * 0.02);
            weights.Add("model.layers.0.self_attn.o_proj.weight", torch.randn(4, 4) * 0.02);
            weights.Add("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 4) * 0.02);
            weights.Add("model.layers.0.mlp.up_proj.weight", torch.randn(8, 4) * 0.02);
            weights.Add("model.layers.0.mlp.down_proj.weight", torch.randn(4, 8) * 0.02);
            weights.Add("model.norm.weight", torch.ones(4));
            weights.Add("lm_head.weight", torch.randn(8, 4) * 0.02);
            SafeTensors.Save(Path.Combine(modelDirectory, "model.safetensors"), weights);

            var modelConfigPath = Path.Combine(modelDirectory, "config.json");
            File.WriteAllText(modelConfigPath, JsonSerializer.Serialize(new
            {
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
            var dataPath = Path.Combine(root, "train.jsonl");
            File.WriteAllText(dataPath,
                """
                {"tokens":[1,2,3],"labels":[-100,-100,3],"loss_weights":[0,0,1],"candidate_masks":[null,null,[3,4]]}
                """);

            Trainer.Train(new TrainConfig
            {
                ModelConfigPath = modelConfigPath,
                ModelPath = modelDirectory,
                TrainDataPath = dataPath,
                OutputDirectory = outputDirectory,
                TargetModules = new HashSet<string>(StringComparer.Ordinal) { "q_proj" },
                PadTokenId = 0,
                MaxSequenceLength = 8,
                Rank = 2,
                BatchSize = 1,
                MaxSteps = 1,
                Device = "cpu",
                DType = "float32"
            });

            Assert.True(File.Exists(Path.Combine(outputDirectory, "adapter_config.json")));
            Assert.True(File.Exists(Path.Combine(outputDirectory, "training_state.json")));
            var adapter = SafeTensors.LoadModel(Path.Combine(outputDirectory, "adapter_model.safetensors"));
            try
            {
                Assert.Equal(2, adapter.Count);
                Assert.Contains("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", adapter);
                Assert.Contains("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", adapter);
            }
            finally
            {
                foreach (var tensor in adapter.Values)
                    tensor.Dispose();
            }
        }
        finally
        {
            foreach (var tensor in weights.Values)
                tensor.Dispose();
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    private static string TemporaryPath(string suffix) =>
        Path.Combine(Path.GetTempPath(), $"llavon-lora-{Guid.NewGuid():N}{suffix}");
}
