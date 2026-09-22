using System.Globalization;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Llavon.Lora;

public static class Trainer {
    public static Tensor CandidateConstrainedLoss(Tensor logits, TrainingBatch batch) {
        if (logits.dim() != 3 || batch.Labels.dim() != 2 || batch.LossWeights.dim() != 2)
            throw new ArgumentException("invalid logits or batch rank");
        if (logits.shape[0] != batch.Labels.shape[0] || logits.shape[1] != batch.Labels.shape[1])
            throw new ArgumentException("logits and labels shapes do not match");

        var weightedLosses = new List<Tensor>();
        double totalWeight = 0;
        for (var row = 0L; row < batch.Labels.shape[0]; ++row) {
            for (var position = 1L; position < batch.Labels.shape[1]; ++position) {
                var label = batch.Labels[row, position].item<long>();
                var weight = batch.LossWeights[row, position].item<float>();
                var attended = batch.AttentionMask[row, position].item<bool>();
                if (!attended || label == -100 || weight <= 0)
                    continue;
                if (label < 0 || label >= logits.shape[2])
                    throw new ArgumentException("label is outside logits vocabulary");

                var positionLogits = logits[row, position - 1].to_type(ScalarType.Float32);
                Tensor tokenLoss;
                var candidate = batch.CandidateMasks[(int)row][(int)position];
                if (candidate is not null) {
                    var ids = candidate.Contains(label) ? candidate : [.. candidate, label];
                    var index = torch.tensor(ids, dtype: ScalarType.Int64, device: positionLogits.device);
                    var selected = positionLogits.index_select(0, index);
                    tokenLoss = torch.logsumexp(selected, 0) - positionLogits[label];
                } else {
                    tokenLoss = torch.logsumexp(positionLogits, 0) - positionLogits[label];
                }
                weightedLosses.Add(tokenLoss * weight);
                totalWeight += weight;
            }
        }

        return weightedLosses.Count == 0 || totalWeight <= 0
            ? logits.sum() * 0
            : torch.stack(weightedLosses).sum() / totalWeight;
    }

    public static void Train(TrainConfig config) {
        var modelConfig = ModelConfig.Load(config.ModelConfigPath);
        config.Validate(modelConfig);
        var samples = Dataset.LoadJsonLines(config.TrainDataPath, modelConfig.VocabSize, config.MaxSequenceLength);

        torch.manual_seed(config.Seed);
        var device = ResolveDevice(config.Device);
        var dtype = ResolveDType(config.DType);
        if (device.type == DeviceType.CPU && dtype != ScalarType.Float32)
            throw new ArgumentException("CPU training requires --dtype float32");

        Console.WriteLine($"loading base checkpoint from {config.ModelPath}");
        using var model = new LlamaForCausalLm(
            modelConfig, config.Rank, config.Alpha, config.Dropout, config.TargetModules);
        model.LoadBaseWeights(config.ModelPath);
        model.to(device, dtype);
        model.train();

        var trainable = model.TrainableParameters();
        if (trainable.Length == 0)
            throw new InvalidOperationException("selected target modules produced no trainable parameters");
        var trainableCount = trainable.Sum(parameter => parameter.numel());
        using var optimizer = torch.optim.AdamW(
            trainable,
            lr: config.LearningRate,
            weight_decay: config.WeightDecay);

        var order = Enumerable.Range(0, samples.Count).ToArray();
        var random = new Random(unchecked((int)config.Seed));
        var batchesPerEpoch = (samples.Count + config.BatchSize - 1) / config.BatchSize;
        var updatesPerEpoch = (batchesPerEpoch + config.GradientAccumulationSteps - 1) /
                              config.GradientAccumulationSteps;
        var totalSteps = config.MaxSteps > 0 ? config.MaxSteps : (long)updatesPerEpoch * config.Epochs;

        Console.WriteLine(
            $"samples={samples.Count} trainable_parameters={trainableCount} optimizer_steps={totalSteps} " +
            $"device={device} dtype={config.DType}");

        Directory.CreateDirectory(config.OutputDirectory);
        optimizer.zero_grad();
        long globalStep = 0;
        var accumulated = 0;
        double accumulatedLoss = 0;
        double lastMeanLoss = 0;

        for (var epoch = 0; epoch < config.Epochs && globalStep < totalSteps; ++epoch) {
            if (config.Shuffle)
                Shuffle(order, random);

            for (var begin = 0; begin < order.Length && globalStep < totalSteps; begin += config.BatchSize) {
                var end = Math.Min(order.Length, begin + config.BatchSize);
                var indices = order[begin..end];
                using var batch = Dataset.MakeBatch(samples, indices, config.PadTokenId);
                double lossValue;
                using (var scope = torch.NewDisposeScope()) {
                    using var deviceTokens = batch.Tokens.to(device);
                    using var deviceAttention = batch.AttentionMask.to(device);
                    var logits = model.call(deviceTokens, deviceAttention);
                    var loss = CandidateConstrainedLoss(logits, batch);
                    lossValue = loss.item<float>();
                    (loss / config.GradientAccumulationSteps).backward();
                }
                accumulatedLoss += lossValue;
                ++accumulated;

                var epochEnd = end == order.Length;
                if (accumulated < config.GradientAccumulationSteps && !epochEnd)
                    continue;

                var learningRate = ScheduledLearningRate(config, globalStep, totalSteps);
                foreach (var group in optimizer.ParamGroups)
                    group.LearningRate = learningRate;
                if (config.MaxGradientNorm > 0)
                    torch.nn.utils.clip_grad_norm_(trainable, config.MaxGradientNorm);
                optimizer.step();
                optimizer.zero_grad();
                ++globalStep;

                lastMeanLoss = accumulatedLoss / accumulated;
                Console.WriteLine(
                    $"step={globalStep}/{totalSteps} epoch={epoch + 1} " +
                    $"loss={lastMeanLoss.ToString("G9", CultureInfo.InvariantCulture)} " +
                    $"lr={learningRate.ToString("G9", CultureInfo.InvariantCulture)}");
                accumulated = 0;
                accumulatedLoss = 0;

                if (config.SaveEvery > 0 && globalStep % config.SaveEvery == 0) {
                    var checkpoint = Path.Combine(config.OutputDirectory, $"checkpoint-{globalStep}");
                    model.SavePeftAdapter(checkpoint, config.ModelPath);
                    WriteTrainingState(checkpoint, globalStep, lastMeanLoss, learningRate);
                }
            }
        }

        model.SavePeftAdapter(config.OutputDirectory, config.ModelPath);
        var finalLearningRate = ScheduledLearningRate(config, Math.Max(0, globalStep - 1), totalSteps);
        WriteTrainingState(config.OutputDirectory, globalStep, lastMeanLoss, finalLearningRate);
        Console.WriteLine($"adapter saved to {config.OutputDirectory}");
    }

    private static Device ResolveDevice(string requested) {
        if (requested == "cuda") {
            if (!torch.cuda.is_available())
                throw new InvalidOperationException("CUDA was requested but is not available");
            return CUDA;
        }
        return requested == "auto" && torch.cuda.is_available() ? CUDA : CPU;
    }

    private static ScalarType ResolveDType(string name) => name switch {
        "float32" => ScalarType.Float32,
        "float16" => ScalarType.Float16,
        "bfloat16" => ScalarType.BFloat16,
        _ => throw new ArgumentException($"unknown dtype: {name}")
    };

    private static double ScheduledLearningRate(TrainConfig config, long step, long totalSteps) {
        if (step < config.WarmupSteps && config.WarmupSteps > 0)
            return config.LearningRate * (step + 1d) / config.WarmupSteps;
        var decaySteps = Math.Max(1, totalSteps - config.WarmupSteps);
        var progress = Math.Clamp((double)(step - config.WarmupSteps) / decaySteps, 0, 1);
        return config.LearningRate * 0.5 * (1 + Math.Cos(Math.PI * progress));
    }

    private static void WriteTrainingState(string outputDirectory, long step, double loss, double learningRate) {
        Directory.CreateDirectory(outputDirectory);
        var state = new { step, loss, learning_rate = learningRate };
        File.WriteAllText(
            Path.Combine(outputDirectory, "training_state.json"),
            JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
    }

    private static void Shuffle<T>(T[] values, Random random) {
        for (var index = values.Length - 1; index > 0; --index) {
            var other = random.Next(index + 1);
            (values[index], values[other]) = (values[other], values[index]);
        }
    }
}
