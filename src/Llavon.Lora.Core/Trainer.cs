using System.Globalization;
using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Llavon.Lora;

public static class Trainer {
    internal readonly record struct TrainingMode(
        bool ValidatesEveryEpoch,
        bool RepeatsUntilRemembered,
        bool MasksCorrectPositions) {
        public bool ShouldContinue(long epoch, int epochs, bool remembered) =>
            RepeatsUntilRemembered ? !remembered : epoch < epochs;
    }

    internal static TrainingMode ResolveTrainingMode(
        bool trainUntilRemembered,
        bool onlyTrainIncorrect) => new(
            trainUntilRemembered || onlyTrainIncorrect,
            trainUntilRemembered,
            onlyTrainIncorrect);

    public static long CountIncorrectPredictions(Tensor logits, TrainingBatch batch) =>
        EvaluatePredictions(logits, batch).IncorrectPredictions;

    internal static bool[][] FindIncorrectPredictionPositions(Tensor logits, TrainingBatch batch) =>
        EvaluatePredictions(logits, batch).IncorrectPositions;

    private static (
        long IncorrectPredictions,
        bool[][] IncorrectPositions) EvaluatePredictions(
        Tensor logits,
        TrainingBatch batch) {
        if (logits.dim() != 3 || logits.shape[0] != batch.Labels.shape[0] ||
            logits.shape[1] != batch.Labels.shape[1])
            throw new ArgumentException("logits and labels shapes do not match");

        long incorrectPredictions = 0;
        var incorrectPositions = batch.CandidateMasks
            .Select(row => new bool[row.Count])
            .ToArray();
        for (var row = 0; row < batch.CandidateMasks.Count; ++row) {
            for (var position = 1; position < batch.CandidateMasks[row].Count; ++position) {
                using var labelTensor = batch.Labels[row, position];
                using var weightTensor = batch.LossWeights[row, position];
                using var attentionTensor = batch.AttentionMask[row, position];
                var expected = labelTensor.item<long>();
                if (expected == -100 || weightTensor.item<float>() <= 0 ||
                    !attentionTensor.item<bool>())
                    continue;

                using var positionLogits = logits[row, position - 1];
                long predicted;
                if (batch.CandidateMasks[row][position] is { } candidates) {
                    using var ids = torch.tensor(
                        candidates, dtype: ScalarType.Int64, device: logits.device);
                    using var candidateLogits = positionLogits.index_select(0, ids);
                    using var selected = candidateLogits.argmax();
                    predicted = candidates[selected.item<long>()];
                } else {
                    using var selected = positionLogits.argmax();
                    predicted = selected.item<long>();
                }
                if (predicted == expected)
                    continue;
                ++incorrectPredictions;
                incorrectPositions[row][position] = true;
            }
        }
        return (incorrectPredictions, incorrectPositions);
    }

    internal static int[] SelectSamplesForTraining(
        IReadOnlyList<int> order,
        IReadOnlyList<bool[]> incorrectPositions) =>
        order.Where(index => incorrectPositions[index].Any(incorrect => incorrect)).ToArray();

    public static Tensor CandidateConstrainedLoss(Tensor logits, TrainingBatch batch) {
        if (logits.dim() != 3 || batch.Labels.dim() != 2 || batch.LossWeights.dim() != 2)
            throw new ArgumentException("invalid logits or batch rank");
        if (logits.shape[0] != batch.Labels.shape[0] || logits.shape[1] != batch.Labels.shape[1])
            throw new ArgumentException("logits and labels shapes do not match");
        if (batch.AttentionMask.shape[0] != logits.shape[0] || batch.AttentionMask.shape[1] != logits.shape[1] ||
            batch.LossWeights.shape[0] != logits.shape[0] || batch.LossWeights.shape[1] != logits.shape[1] ||
            batch.CandidateMasks.Count != logits.shape[0] ||
            batch.CandidateMasks.Any(row => row.Count != logits.shape[1]))
            throw new ArgumentException("batch fields do not have matching shapes");

        var shiftedLength = logits.shape[1] - 1;
        if (shiftedLength <= 0)
            return logits.sum() * 0;

        var vocabularySize = logits.shape[2];
        var candidateRows = new List<long>();
        var candidateLists = new List<long[]>();
        var fullVocabularyRows = new List<long>();
        for (var row = 0; row < batch.CandidateMasks.Count; ++row) {
            for (var position = 1; position < batch.CandidateMasks[row].Count; ++position) {
                var flatRow = row * shiftedLength + position - 1;
                var candidates = batch.CandidateMasks[row][position];
                if (candidates is null) {
                    fullVocabularyRows.Add(flatRow);
                } else {
                    candidateRows.Add(flatRow);
                    candidateLists.Add(candidates);
                }
            }
        }

        var flatLogits = logits.narrow(1, 0, shiftedLength).reshape(-1, vocabularySize);
        var flatLabels = batch.Labels.narrow(1, 1, shiftedLength).reshape(-1).to(logits.device);
        var flatWeights = batch.LossWeights.narrow(1, 1, shiftedLength).reshape(-1)
            .to(logits.device).to_type(ScalarType.Float32);
        var flatAttention = batch.AttentionMask.narrow(1, 1, shiftedLength).reshape(-1).to(logits.device);
        var numerators = new List<Tensor>(2);
        var denominators = new List<Tensor>(2);

        AddCandidateLoss(
            flatLogits,
            flatLabels,
            flatWeights,
            flatAttention,
            candidateRows,
            candidateLists,
            vocabularySize,
            numerators,
            denominators);
        AddFullVocabularyLoss(
            flatLogits,
            flatLabels,
            flatWeights,
            flatAttention,
            fullVocabularyRows,
            numerators,
            denominators);

        return numerators.Count == 0
            ? logits.sum() * 0
            : torch.stack(numerators).sum() / torch.stack(denominators).sum();
    }

    private static void AddCandidateLoss(
        Tensor flatLogits,
        Tensor flatLabels,
        Tensor flatWeights,
        Tensor flatAttention,
        IReadOnlyList<long> rows,
        IReadOnlyList<long[]> candidateLists,
        long vocabularySize,
        ICollection<Tensor> numerators,
        ICollection<Tensor> denominators) {
        if (rows.Count == 0)
            return;

        var maximumCandidates = candidateLists.Max(candidates => candidates.Length);
        var candidateIds = new long[rows.Count, maximumCandidates];
        var padding = new bool[rows.Count, maximumCandidates];
        for (var row = 0; row < candidateLists.Count; ++row) {
            var candidates = candidateLists[row];
            for (var column = 0; column < candidates.Length; ++column)
                candidateIds[row, column] = candidates[column];
            for (var column = candidates.Length; column < maximumCandidates; ++column)
                padding[row, column] = true;
        }

        var rowIndices = torch.tensor(rows.ToArray(), dtype: ScalarType.Int64, device: flatLogits.device);
        var labels = flatLabels.index_select(0, rowIndices);
        var weights = flatWeights.index_select(0, rowIndices);
        var attention = flatAttention.index_select(0, rowIndices);
        var valid = labels.ne(-100).logical_and(weights.gt(0)).logical_and(attention);
        var validIndices = valid.nonzero().flatten();
        if (validIndices.numel() == 0)
            return;

        var validRows = rowIndices.index_select(0, validIndices);
        var validLabels = labels.index_select(0, validIndices);
        var validWeights = weights.index_select(0, validIndices);
        var ids = torch.tensor(candidateIds, dtype: ScalarType.Int64, device: flatLogits.device)
            .index_select(0, validIndices);
        var paddingMask = torch.tensor(padding, dtype: ScalarType.Bool, device: flatLogits.device)
            .index_select(0, validIndices);
        var linearIndices = validRows.unsqueeze(1) * vocabularySize + ids;
        var selectedLogits = flatLogits.reshape(-1)
            .index_select(0, linearIndices.reshape(-1))
            .reshape(validRows.shape[0], maximumCandidates)
            .to_type(ScalarType.Float32)
            .masked_fill(paddingMask, float.NegativeInfinity);
        var targetIndices = validRows * vocabularySize + validLabels;
        var targetLogits = flatLogits.reshape(-1).index_select(0, targetIndices).to_type(ScalarType.Float32);
        var losses = selectedLogits.logsumexp(1) - targetLogits;

        numerators.Add((losses * validWeights).sum());
        denominators.Add(validWeights.sum());
    }

    private static void AddFullVocabularyLoss(
        Tensor flatLogits,
        Tensor flatLabels,
        Tensor flatWeights,
        Tensor flatAttention,
        IReadOnlyList<long> rows,
        ICollection<Tensor> numerators,
        ICollection<Tensor> denominators) {
        if (rows.Count == 0)
            return;

        var rowIndices = torch.tensor(rows.ToArray(), dtype: ScalarType.Int64, device: flatLogits.device);
        var labels = flatLabels.index_select(0, rowIndices);
        var weights = flatWeights.index_select(0, rowIndices);
        var attention = flatAttention.index_select(0, rowIndices);
        var valid = labels.ne(-100).logical_and(weights.gt(0)).logical_and(attention);
        var validIndices = valid.nonzero().flatten();
        if (validIndices.numel() == 0)
            return;

        var validRows = rowIndices.index_select(0, validIndices);
        var validLabels = labels.index_select(0, validIndices);
        var validWeights = weights.index_select(0, validIndices);
        var selectedLogits = flatLogits.index_select(0, validRows).to_type(ScalarType.Float32);
        var targetLogits = selectedLogits.gather(1, validLabels.unsqueeze(1)).squeeze(1);
        var losses = selectedLogits.logsumexp(1) - targetLogits;

        numerators.Add((losses * validWeights).sum());
        denominators.Add(validWeights.sum());
    }

    public static void Train(TrainConfig config) {
        // An alternative libtorch has to be loaded before TorchSharp initialises
        // its own native backend.
        TorchNativeLibraries.Initialize(null);
        var modelConfig = ModelConfig.Load(config.ModelConfigPath);
        config.Validate(modelConfig);
        var samples = Dataset.LoadJsonLines(config.TrainDataPath, modelConfig.VocabSize, config.MaxSequenceLength);
        if (config.TrainUntilRemembered) {
            var conflicts = Dataset.CountConflictingTargets(samples);
            if (conflicts != 0) {
                Console.Error.WriteLine(
                    $"warning: training data has {conflicts} conflicting input target(s)");
                throw new InvalidDataException(
                    "train-until-remembered cannot continue because the same input has different answers");
            }
        }

        torch.manual_seed(config.Seed);
        var device = DeviceSelection.Resolve(config.Device);
        var dtype = ResolveDType(config.DType);
        if (device.type == DeviceType.CPU && dtype != ScalarType.Float32)
            throw new ArgumentException("CPU training requires --dtype float32");
        if (device.type == DeviceType.MPS && dtype != ScalarType.Float32)
            throw new ArgumentException("MPS training requires --dtype float32");

        Console.WriteLine($"loading base checkpoint from {config.ModelPath}");
        using var model = new LlamaForCausalLm(
            modelConfig, config.Rank, config.Alpha, config.Dropout, config.TargetModules);
        model.LoadBaseWeights(config.ModelPath);
        if (config.ResumeAdapterDirectory is not null)
            model.LoadPeftAdapter(config.ResumeAdapterDirectory);
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
        var totalSteps = config.MaxSteps > 0
            ? config.MaxSteps
            : config.TrainUntilRemembered
                ? long.MaxValue
                : (long)updatesPerEpoch * config.Epochs;
        var totalStepsLabel = config.TrainUntilRemembered && config.MaxSteps < 0
            ? "until-remembered"
            : totalSteps.ToString(CultureInfo.InvariantCulture);

        Console.WriteLine(
            $"samples={samples.Count} trainable_parameters={trainableCount} optimizer_steps={totalStepsLabel} " +
            $"device={device} dtype={config.DType}");

        Directory.CreateDirectory(config.OutputDirectory);
        optimizer.zero_grad();
        long globalStep = 0;
        var accumulated = 0;
        double accumulatedLoss = 0;
        double lastMeanLoss = 0;
        bool[][]? incorrectPositions = null;
        var remembered = false;
        var trainingMode = ResolveTrainingMode(
            config.TrainUntilRemembered, config.OnlyTrainIncorrect);
        if (trainingMode.ValidatesEveryEpoch) {
            incorrectPositions = FindIncorrectPredictionPositions(
                model, samples, config.BatchSize, config.PadTokenId, device);
            var incorrect = CountIncorrectSamples(incorrectPositions);
            remembered = incorrect == 0;
            WriteValidationResult(config, remembered, incorrect, 0);
        }

        for (long epoch = 0;
             trainingMode.ShouldContinue(epoch, config.Epochs, remembered) &&
             globalStep < totalSteps;
             ++epoch) {
            if (config.Shuffle)
                Shuffle(order, random);

            var epochOrder = incorrectPositions is null
                ? order
                : SelectSamplesForTraining(order, incorrectPositions);
            for (var begin = 0; begin < epochOrder.Length && globalStep < totalSteps; begin += config.BatchSize) {
                var end = Math.Min(epochOrder.Length, begin + config.BatchSize);
                var indices = epochOrder[begin..end];
                using var batch = Dataset.MakeBatch(
                    samples,
                    indices,
                    config.PadTokenId,
                    trainingMode.MasksCorrectPositions ? incorrectPositions : null);
                double lossValue;
                using (var scope = torch.NewDisposeScope()) {
                    using var deviceTokens = batch.Tokens.to(device);
                    using var deviceAttention = batch.AttentionMask.to(device);
                    var logits = model.call(deviceTokens, deviceAttention);
                    var loss = CandidateConstrainedLoss(logits, batch);
                    lossValue = loss.item<float>();
                    loss.backward();
                }
                accumulatedLoss += lossValue;
                ++accumulated;

                var epochEnd = end == epochOrder.Length;
                if (accumulated < config.GradientAccumulationSteps && !epochEnd)
                    continue;

                var learningRate = ScheduledLearningRate(config, globalStep);
                foreach (var group in optimizer.ParamGroups)
                    group.LearningRate = learningRate;
                AverageAccumulatedGradients(trainable, accumulated);
                if (config.MaxGradientNorm > 0)
                    torch.nn.utils.clip_grad_norm_(trainable, config.MaxGradientNorm);
                optimizer.step();
                optimizer.zero_grad();
                ++globalStep;

                lastMeanLoss = accumulatedLoss / accumulated;
                Console.WriteLine(
                    $"step={globalStep}/{totalStepsLabel} epoch={epoch + 1} " +
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

            if (trainingMode.ValidatesEveryEpoch) {
                incorrectPositions = FindIncorrectPredictionPositions(
                    model, samples, config.BatchSize, config.PadTokenId, device);
                var incorrect = CountIncorrectSamples(incorrectPositions);
                remembered = incorrect == 0;
                WriteValidationResult(config, remembered, incorrect, epoch + 1);
            }
        }

        if (config.TrainUntilRemembered && !remembered)
            throw new InvalidOperationException(
                $"max steps reached with training targets still incorrect ({globalStep} steps)");

        model.SavePeftAdapter(config.OutputDirectory, config.ModelPath);
        var finalLearningRate = ScheduledLearningRate(config, Math.Max(0, globalStep - 1));
        WriteTrainingState(config.OutputDirectory, globalStep, lastMeanLoss, finalLearningRate);
        Console.WriteLine($"adapter saved to {config.OutputDirectory}");
    }

    private static bool[][] FindIncorrectPredictionPositions(
        LlamaForCausalLm model,
        IReadOnlyList<TrainingSample> samples,
        int batchSize,
        long padTokenId,
        Device device) {
        model.eval();
        try {
            using var noGrad = torch.no_grad();
            var incorrect = samples
                .Select(sample => new bool[sample.Tokens.Length])
                .ToArray();
            for (var begin = 0; begin < samples.Count; begin += batchSize) {
                var end = Math.Min(samples.Count, begin + batchSize);
                var indices = Enumerable.Range(begin, end - begin).ToArray();
                using var batch = Dataset.MakeBatch(samples, indices, padTokenId);
                using var deviceTokens = batch.Tokens.to(device);
                using var deviceAttention = batch.AttentionMask.to(device);
                using var logits = model.call(deviceTokens, deviceAttention);
                var batchIncorrect = FindIncorrectPredictionPositions(logits, batch);
                for (var row = 0; row < batchIncorrect.Length; ++row)
                    Array.Copy(batchIncorrect[row], incorrect[begin + row], batchIncorrect[row].Length);
            }
            return incorrect;
        } finally {
            model.train();
        }
    }

    private static long CountIncorrectSamples(IEnumerable<bool[]> incorrectPositions) =>
        incorrectPositions.LongCount(row => row.Any(incorrect => incorrect));

    private static void WriteValidationResult(
        TrainConfig config,
        bool remembered,
        long incorrect,
        long epoch) {
        var mode = config.TrainUntilRemembered
            ? $"remembered={remembered.ToString().ToLowerInvariant()}"
            : "only-train-incorrect=true";
        Console.WriteLine($"{mode} incorrect={incorrect} epoch={epoch}");
    }

    internal static void AverageAccumulatedGradients(
        IEnumerable<Parameter> parameters, int accumulated) {
        if (accumulated <= 0)
            throw new ArgumentOutOfRangeException(nameof(accumulated));
        // The last group in an epoch can contain fewer micro batches.
        using var noGrad = torch.no_grad();
        foreach (var parameter in parameters) {
            using var gradient = parameter.grad;
            gradient?.mul_(1d / accumulated);
        }
    }

    private static ScalarType ResolveDType(string name) => name switch {
        "float32" => ScalarType.Float32,
        "float16" => ScalarType.Float16,
        "bfloat16" => ScalarType.BFloat16,
        _ => throw new ArgumentException($"unknown dtype: {name}")
    };

    private static double ScheduledLearningRate(TrainConfig config, long step) {
        if (step < config.WarmupSteps && config.WarmupSteps > 0)
            return config.LearningRate * (step + 1d) / config.WarmupSteps;
        return config.LearningRate;
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
