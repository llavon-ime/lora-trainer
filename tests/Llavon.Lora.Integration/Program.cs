using System.Globalization;
using System.Text;
using System.Text.Json;
using Llavon.Lora;
using Llavon.Lora.Integration;
using TorchSharp;
using static TorchSharp.torch;

return IntegrationProgram.Run(args);

internal static class IntegrationProgram {
    public static int Run(string[] args) {
        try {
            var options = new Options(args);
            var modelConfigPath = options.Required("--model-config");
            var modelPath = options.Required("--model");
            var vocabularyPath = options.Required("--vocab-file");
            var tablesDirectory = options.Required("--tables-dir");
            var validationPath = options.Required("--validation-data");
            var evaluationPath = options.Optional("--evaluation-data") ?? validationPath;
            var workDirectory = options.Required("--work-dir");
            var maxSteps = options.Integer("--max-steps", 84);
            var learningRate = options.Real("--learning-rate", 5e-4);
            var evaluateBase = options.Integer("--evaluate-base", 1) != 0;
            var verbose = options.Integer("--verbose", 1) != 0;
            var existingAdapter = options.Optional("--adapter");
            if (existingAdapter is not null) {
                var statePath = Path.Combine(existingAdapter, "training_state.json");
                using var state = JsonDocument.Parse(File.ReadAllBytes(statePath));
                maxSteps = state.RootElement.GetProperty("step").GetInt64();
                learningRate = state.RootElement.GetProperty("learning_rate").GetDouble();
            }

            Directory.CreateDirectory(workDirectory);
            var modelConfig = ModelConfig.Load(modelConfigPath);
            var tables = ImeRuntimeTables.Load(tablesDirectory);
            tables.ValidateVocabulary(vocabularyPath, modelConfig.VocabSize);
            var trainingCases = LoadCases(validationPath);
            var evaluationCases = string.Equals(
                Path.GetFullPath(validationPath),
                Path.GetFullPath(evaluationPath),
                StringComparison.OrdinalIgnoreCase)
                ? trainingCases
                : LoadCases(evaluationPath);
            var trainingDataPath = Path.Combine(workDirectory, "training.jsonl");
            WriteTrainingData(trainingDataPath, trainingCases, tables, modelConfig.MaxPositionEmbeddings);

            Summary? baseline = null;
            if (evaluateBase) {
                Console.WriteLine("evaluating base model");
                baseline = Evaluate(modelConfig, modelPath, null, evaluationCases, tables, verbose);
            }
            var adapterDirectory = existingAdapter ?? Path.Combine(workDirectory, "adapter");
            if (existingAdapter is null) {
                Trainer.Train(new TrainConfig {
                    ModelConfigPath = modelConfigPath,
                    ModelPath = modelPath,
                    TrainDataPath = trainingDataPath,
                    OutputDirectory = adapterDirectory,
                    TargetModules = new HashSet<string>(StringComparer.Ordinal) { "q_proj", "v_proj" },
                    PadTokenId = tables.PadTokenId,
                    MaxSequenceLength = modelConfig.MaxPositionEmbeddings,
                    Rank = 8,
                    Alpha = 16,
                    Dropout = 0,
                    BatchSize = 1,
                    Epochs = int.MaxValue,
                    MaxSteps = maxSteps,
                    LearningRate = learningRate,
                    Device = "cuda",
                    DType = "float32",
                    Seed = 42
                });
            }

            Console.WriteLine("evaluating trained adapter");
            var trainedTraining = Evaluate(
                modelConfig, modelPath, adapterDirectory, trainingCases, tables, verbose);
            var trainedEvaluation = ReferenceEquals(trainingCases, evaluationCases)
                ? trainedTraining
                : Evaluate(modelConfig, modelPath, adapterDirectory, evaluationCases, tables, verbose);
            var resultPath = Path.Combine(workDirectory, "result.json");
            File.WriteAllText(resultPath, JsonSerializer.Serialize(new {
                baseline_evaluation = baseline,
                trained_training = trainedTraining,
                trained_evaluation = trainedEvaluation,
                max_steps = maxSteps,
                learning_rate = learningRate
            }, new JsonSerializerOptions { WriteIndented = true }) + Environment.NewLine);
            if (baseline is not null)
                Console.WriteLine($"baseline exact={baseline.Exact}/{baseline.Total}, chars={baseline.CorrectCharacters}/{baseline.TotalCharacters}");
            Console.WriteLine($"trained training exact={trainedTraining.Exact}/{trainedTraining.Total}, chars={trainedTraining.CorrectCharacters}/{trainedTraining.TotalCharacters}");
            Console.WriteLine($"trained evaluation exact={trainedEvaluation.Exact}/{trainedEvaluation.Total}, chars={trainedEvaluation.CorrectCharacters}/{trainedEvaluation.TotalCharacters}");
            if (trainedEvaluation.RegressionTotal > 0 || trainedEvaluation.RetentionTotal > 0)
                Console.WriteLine(
                    $"trained evaluation regression={trainedEvaluation.RegressionExact}/{trainedEvaluation.RegressionTotal}, " +
                    $"retention={trainedEvaluation.RetentionExact}/{trainedEvaluation.RetentionTotal}");
            Console.WriteLine($"result={resultPath}");
            return 0;
        } catch (Exception exception) when (exception is not OutOfMemoryException) {
            Console.Error.WriteLine($"error: {exception}");
            return 2;
        }
    }

    private static Summary Evaluate(
        ModelConfig modelConfig,
        string modelPath,
        string? adapterDirectory,
        IReadOnlyList<ValidationCase> cases,
        ImeRuntimeTables tables,
        bool verbose) {
        var adapter = adapterDirectory is null ? null : AdapterConfig.Load(adapterDirectory);
        using var model = new LlamaForCausalLm(
            modelConfig,
            adapter?.Rank ?? 1,
            adapter?.Alpha ?? 1,
            adapter?.Dropout ?? 0,
            adapter?.TargetModules ?? new HashSet<string>(StringComparer.Ordinal));
        model.LoadBaseWeights(modelPath);
        if (adapterDirectory is not null)
            model.LoadPeftAdapter(adapterDirectory);
        model.to(CUDA, ScalarType.Float16);
        model.eval();

        var exact = 0;
        var regressionExact = 0;
        var regressionTotal = 0;
        var retentionExact = 0;
        var retentionTotal = 0;
        long correctCharacters = 0;
        long totalCharacters = 0;
        using var noGrad = torch.no_grad();
        foreach (var item in cases) {
            var bpmf = item.Padding.Select(entry => WithTone(entry.Syllable, entry.Tone)).ToArray();
            var answer = item.Answer.EnumerateRunes().Select(rune => rune.ToString()).ToArray();
            var candidates = CandidateRows(item, answer, bpmf, tables);
            var inputIds = tables.TokenizePrompt(item.Context, bpmf).ToList();
            var prediction = new StringBuilder(answer.Length);
            for (var position = 0; position < answer.Length; ++position) {
                using var scope = torch.NewDisposeScope();
                var tokens = torch.tensor(inputIds.ToArray(), dtype: ScalarType.Int64, device: CUDA).unsqueeze(0);
                var attention = torch.ones(1, inputIds.Count, dtype: ScalarType.Bool, device: CUDA);
                var logits = model.call(tokens, attention);
                var ids = torch.tensor(
                    candidates[position].Select(candidate => candidate.TokenId).ToArray(),
                    dtype: ScalarType.Int64,
                    device: CUDA);
                var selected = logits[0, -1].index_select(0, ids).argmax().item<long>();
                var predicted = candidates[position][checked((int)selected)];
                inputIds.Add(predicted.TokenId);
                prediction.Append(predicted.Character);
            }
            var predictedCharacters = prediction.ToString().EnumerateRunes().Select(rune => rune.ToString()).ToArray();
            var matches = answer.Zip(predictedCharacters).LongCount(pair => pair.First == pair.Second);
            var isExact = matches == answer.Length;
            exact += isExact ? 1 : 0;
            if (item.CaseRole == "regression") {
                ++regressionTotal;
                regressionExact += isExact ? 1 : 0;
            } else if (item.CaseRole == "retention-control") {
                ++retentionTotal;
                retentionExact += isExact ? 1 : 0;
            }
            correctCharacters += matches;
            totalCharacters += answer.Length;
            if (verbose)
                Console.WriteLine($"{(isExact ? "OK" : "FAIL")} target={item.Answer} prediction={prediction}");
        }
        return new Summary(
            exact,
            cases.Count,
            correctCharacters,
            totalCharacters,
            regressionExact,
            regressionTotal,
            retentionExact,
            retentionTotal);
    }

    private static void WriteTrainingData(
        string path,
        IReadOnlyList<ValidationCase> cases,
        ImeRuntimeTables tables,
        long maximumSequenceLength) {
        using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
        foreach (var item in cases) {
            var bpmf = item.Padding.Select(entry => WithTone(entry.Syllable, entry.Tone)).ToArray();
            var answer = item.Answer.EnumerateRunes().Select(rune => rune.ToString()).ToArray();
            var candidates = CandidateRows(item, answer, bpmf, tables);
            var prompt = tables.TokenizePrompt(item.Context, bpmf);
            var answerTokens = candidates.Select((row, position) =>
                row.Single(candidate => candidate.Character == answer[position]).TokenId).ToArray();
            var tokens = prompt.Concat(answerTokens).ToArray();
            if (tokens.LongLength > maximumSequenceLength)
                throw Error(item.Line, "prompt and answer exceed model context length");
            writer.WriteLine(JsonSerializer.Serialize(new {
                tokens,
                labels = tokens,
                loss_weights = Enumerable.Repeat(0, prompt.Length).Concat(Enumerable.Repeat(1, answer.Length)).ToArray(),
                attention_mask = Enumerable.Repeat(1, tokens.Length).ToArray(),
                candidate_masks = Enumerable.Repeat<long[]?>(null, prompt.Length)
                    .Concat(candidates.Select(row => (long[]?)row.Select(candidate => candidate.TokenId).ToArray()))
                    .ToArray()
            }));
        }
        Console.WriteLine($"wrote {cases.Count} numeric training samples to {path}");
    }

    private static IReadOnlyList<(long TokenId, string Character)>[] CandidateRows(
        ValidationCase item,
        IReadOnlyList<string> answer,
        IReadOnlyList<string> bpmf,
        ImeRuntimeTables tables) {
        if (answer.Count != bpmf.Count)
            throw Error(item.Line, "answer and padding lengths differ");
        var rows = new IReadOnlyList<(long TokenId, string Character)>[answer.Count];
        for (var position = 0; position < answer.Count; ++position) {
            rows[position] = tables.CandidateTokens(bpmf[position]);
            if (!rows[position].Any(candidate => candidate.Character == answer[position]))
                throw Error(item.Line, $"answer character {answer[position]} is missing from candidate table");
        }
        return rows;
    }

    private static IReadOnlyList<ValidationCase> LoadCases(string path) {
        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var result = new List<ValidationCase>();
        var lineNumber = 0;
        foreach (var line in File.ReadLines(path, Encoding.UTF8)) {
            ++lineNumber;
            if (string.IsNullOrWhiteSpace(line))
                continue;
            var input = JsonSerializer.Deserialize<ValidationInput>(line, options) ?? throw Error(lineNumber, "null row");
            if (input.SchemaVersion != 1 || input.Context is null || input.Answer is null || input.Padding is null)
                throw Error(lineNumber, "invalid validation row");
            if (input.CaseRole is not (null or "regression" or "retention-control"))
                throw Error(lineNumber, $"invalid caseRole: {input.CaseRole}");
            result.Add(new ValidationCase(lineNumber, input.Context, input.Answer, input.Padding, input.CaseRole));
        }
        return result;
    }

    private static string WithTone(string syllable, int tone) => tone switch {
        1 => syllable + " ",
        2 => syllable + "ˊ",
        3 => syllable + "ˇ",
        4 => syllable + "ˋ",
        5 => syllable + "˙",
        _ => throw new InvalidDataException($"tone must be in [1, 5], got {tone}")
    };

    private static InvalidDataException Error(int line, string message) => new($"validation line {line}: {message}");

    private sealed record Summary(
        int Exact,
        int Total,
        long CorrectCharacters,
        long TotalCharacters,
        int RegressionExact,
        int RegressionTotal,
        int RetentionExact,
        int RetentionTotal);
    private sealed record ValidationCase(
        int Line,
        string Context,
        string Answer,
        Padding[] Padding,
        string? CaseRole);
    private sealed record Padding(string Syllable, int Tone);
    private sealed class ValidationInput {
        public int SchemaVersion { get; init; }
        public string? Context { get; init; }
        public string? Answer { get; init; }
        public Padding[]? Padding { get; init; }
        public string? CaseRole { get; init; }
    }
}

internal sealed class Options {
    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);

    public Options(IReadOnlyList<string> args) {
        for (var index = 0; index < args.Count; index += 2) {
            if (index + 1 >= args.Count || !args[index].StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException("integration arguments must be --name value pairs");
            if (!values.TryAdd(args[index], args[index + 1]))
                throw new ArgumentException($"duplicate option: {args[index]}");
        }
    }

    public string Required(string name) => values.TryGetValue(name, out var value)
        ? value
        : throw new ArgumentException($"missing required option: {name}");

    public string? Optional(string name) => values.GetValueOrDefault(name);

    public long Integer(string name, long fallback) => values.TryGetValue(name, out var value)
        ? long.Parse(value, NumberStyles.Integer, CultureInfo.InvariantCulture)
        : fallback;

    public double Real(string name, double fallback) => values.TryGetValue(name, out var value)
        ? double.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture)
        : fallback;
}
