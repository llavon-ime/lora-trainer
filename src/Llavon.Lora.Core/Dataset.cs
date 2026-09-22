using System.Globalization;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Llavon.Lora;

public sealed record TrainingSample(
    long[] Tokens,
    long[] Labels,
    float[] LossWeights,
    bool[] AttentionMask,
    IReadOnlyList<long[]?> CandidateMasks);

public sealed class TrainingBatch : IDisposable {
    public required Tensor Tokens { get; init; }
    public required Tensor Labels { get; init; }
    public required Tensor LossWeights { get; init; }
    public required Tensor AttentionMask { get; init; }
    public required IReadOnlyList<IReadOnlyList<long[]?>> CandidateMasks { get; init; }

    public void Dispose() {
        Tokens.Dispose();
        Labels.Dispose();
        LossWeights.Dispose();
        AttentionMask.Dispose();
    }
}

public static class Dataset {
    public static IReadOnlyList<TrainingSample> LoadJsonLines(
        string path,
        long vocabularySize,
        long maximumSequenceLength) {
        var samples = new List<TrainingSample>();
        using var reader = new StreamReader(path);
        var lineNumber = 0;
        while (reader.ReadLine() is { } line) {
            ++lineNumber;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            try {
                using var document = JsonDocument.Parse(line);
                samples.Add(ParseSample(document.RootElement, lineNumber, vocabularySize, maximumSequenceLength));
            } catch (JsonException exception) {
                throw new InvalidDataException($"invalid JSON on line {lineNumber}: {exception.Message}", exception);
            }
        }

        return samples.Count > 0
            ? samples
            : throw new InvalidDataException($"training data contains no samples: {path}");
    }

    public static TrainingBatch MakeBatch(
        IReadOnlyList<TrainingSample> samples,
        IReadOnlyList<int> indices,
        long padTokenId) {
        if (indices.Count == 0)
            throw new ArgumentException("cannot create an empty batch", nameof(indices));

        var sequenceLength = indices.Max(index => samples[index].Tokens.Length);
        var tokens = new long[indices.Count, sequenceLength];
        var labels = new long[indices.Count, sequenceLength];
        var weights = new float[indices.Count, sequenceLength];
        var attention = new bool[indices.Count, sequenceLength];
        var candidateMasks = new List<IReadOnlyList<long[]?>>(indices.Count);

        for (var row = 0; row < indices.Count; ++row) {
            var sample = samples[indices[row]];
            var masks = new long[]?[sequenceLength];
            for (var position = 0; position < sequenceLength; ++position) {
                tokens[row, position] = padTokenId;
                labels[row, position] = -100;
                if (position >= sample.Tokens.Length)
                    continue;
                tokens[row, position] = sample.Tokens[position];
                labels[row, position] = sample.Labels[position];
                weights[row, position] = sample.LossWeights[position];
                attention[row, position] = sample.AttentionMask[position];
                masks[position] = sample.CandidateMasks[position];
            }
            candidateMasks.Add(masks);
        }

        return new TrainingBatch {
            Tokens = torch.tensor(tokens, dtype: ScalarType.Int64),
            Labels = torch.tensor(labels, dtype: ScalarType.Int64),
            LossWeights = torch.tensor(weights, dtype: ScalarType.Float32),
            AttentionMask = torch.tensor(attention, dtype: ScalarType.Bool),
            CandidateMasks = candidateMasks
        };
    }

    private static TrainingSample ParseSample(
        JsonElement row,
        int lineNumber,
        long vocabularySize,
        long maximumSequenceLength) {
        var tokens = row.TryGetProperty("tokens", out var tokensElement)
            ? ReadLongArray(tokensElement, lineNumber, "tokens")
            : row.TryGetProperty("input_ids", out tokensElement)
                ? ReadLongArray(tokensElement, lineNumber, "input_ids")
                : throw Error(lineNumber, "is missing 'tokens' (or 'input_ids')");

        if (tokens.Length < 2 || tokens.LongLength > maximumSequenceLength)
            throw Error(lineNumber, "token count must be in [2, max-seq-length]");

        var labels = row.TryGetProperty("labels", out var labelsElement)
            ? ReadLongArray(labelsElement, lineNumber, "labels")
            : [.. tokens];
        RequireLength(labels.Length, tokens.Length, lineNumber, "labels");

        JsonElement weightsElement;
        if (!row.TryGetProperty("loss_weights", out weightsElement) &&
            !row.TryGetProperty("loss_mask", out weightsElement))
            throw Error(lineNumber, "is missing 'loss_weights' (or 'loss_mask')");
        var weights = ReadFloatArray(weightsElement, lineNumber, "loss_weights");
        RequireLength(weights.Length, tokens.Length, lineNumber, "loss_weights");

        var attention = row.TryGetProperty("attention_mask", out var attentionElement)
            ? ReadAttentionMask(attentionElement, lineNumber)
            : Enumerable.Repeat(true, tokens.Length).ToArray();
        RequireLength(attention.Length, tokens.Length, lineNumber, "attention_mask");

        var candidateMasks = new long[]?[tokens.Length];
        if (row.TryGetProperty("candidate_masks", out var masksElement)) {
            if (masksElement.ValueKind != JsonValueKind.Array)
                throw Error(lineNumber, "'candidate_masks' must be an array");
            RequireLength(masksElement.GetArrayLength(), tokens.Length, lineNumber, "candidate_masks");
            var position = 0;
            foreach (var maskElement in masksElement.EnumerateArray()) {
                if (maskElement.ValueKind == JsonValueKind.Null) {
                    ++position;
                    continue;
                }
                var candidates = ReadLongArray(maskElement, lineNumber, $"candidate_masks[{position}]")
                    .Distinct()
                    .Order()
                    .ToArray();
                if (candidates.Length == 0)
                    throw Error(lineNumber, $"candidate mask at position {position} must be null or non-empty");
                if (candidates.Any(token => token < 0 || token >= vocabularySize))
                    throw Error(lineNumber, "candidate token is outside vocabulary");
                candidateMasks[position++] = candidates;
            }
        }

        var hasLoss = false;
        for (var position = 0; position < tokens.Length; ++position) {
            if (tokens[position] < 0 || tokens[position] >= vocabularySize ||
                labels[position] != -100 && (labels[position] < 0 || labels[position] >= vocabularySize))
                throw Error(lineNumber, "token or label is outside vocabulary");
            if (!float.IsFinite(weights[position]) || weights[position] < 0)
                throw Error(lineNumber, "loss weights must be finite and non-negative");
            if (position > 0 && attention[position] && labels[position] != -100 && weights[position] > 0) {
                if (candidateMasks[position] is { } candidates &&
                    Array.BinarySearch(candidates, labels[position]) < 0)
                    throw Error(
                        lineNumber,
                        $"target label {labels[position]} at position {position} is missing from its candidate mask");
                hasLoss = true;
            }
        }

        return hasLoss
            ? new TrainingSample(tokens, labels, weights, attention, candidateMasks)
            : throw Error(lineNumber, "has no trainable position");
    }

    private static long[] ReadLongArray(JsonElement element, int line, string name) {
        if (element.ValueKind != JsonValueKind.Array)
            throw Error(line, $"'{name}' must be an array");
        return element.EnumerateArray().Select(value => value.GetInt64()).ToArray();
    }

    private static float[] ReadFloatArray(JsonElement element, int line, string name) {
        if (element.ValueKind != JsonValueKind.Array)
            throw Error(line, $"'{name}' must be an array");
        return element.EnumerateArray().Select(value => value.GetSingle()).ToArray();
    }

    private static bool[] ReadAttentionMask(JsonElement element, int line) {
        var values = ReadLongArray(element, line, "attention_mask");
        if (values.Any(value => value is not (0 or 1)))
            throw Error(line, "attention_mask values must be 0 or 1");
        return values.Select(value => value == 1).ToArray();
    }

    private static void RequireLength(int actual, int expected, int line, string name) {
        if (actual != expected)
            throw Error(line, $"'{name}' length {actual.ToString(CultureInfo.InvariantCulture)} " +
                              $"does not match tokens length {expected.ToString(CultureInfo.InvariantCulture)}");
    }

    private static InvalidDataException Error(int line, string message) =>
        new($"line {line.ToString(CultureInfo.InvariantCulture)}: {message}");
}
