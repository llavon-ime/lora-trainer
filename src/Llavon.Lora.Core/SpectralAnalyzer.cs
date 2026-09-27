using System.Text.Json;
using System.Text.Json.Nodes;
using TorchSharp;
using static TorchSharp.torch;

namespace Llavon.Lora;

public sealed record SpectralAnalysisConfig {
    public required string ModelPath { get; init; }
    public required string AdapterDirectory { get; init; }
    public int TopK { get; init; } = 10;
    public double SimilarityThreshold { get; init; } = 0.5;
}

public sealed record IntruderDimension(
    int Rank,
    double SingularValue,
    double MaxAbsoluteCosineSimilarity);

public sealed record SpectralMatrixResult(
    string Weight,
    long Rows,
    long Columns,
    int ExaminedDimensions,
    IReadOnlyList<double> MaxAbsoluteCosineSimilarities,
    IReadOnlyList<IntruderDimension> IntruderDimensions) {
    public int IntruderCount => IntruderDimensions.Count;
}

public sealed record SpectralAnalysisResult(
    string Model,
    string Adapter,
    long Rank,
    double Alpha,
    double Scaling,
    int TopK,
    double SimilarityThreshold,
    int ExaminedMatrices,
    int ExaminedDimensions,
    int IntruderCount,
    IReadOnlyList<SpectralMatrixResult> Matrices);

public sealed record SpectralStabilizationConfig {
    public required string ModelPath { get; init; }
    public required string AdapterDirectory { get; init; }
    public required string OutputDirectory { get; init; }
    public int TopK { get; init; } = 10;
    public double SimilarityThreshold { get; init; } = 0.5;
    public double IntruderScale { get; init; } = 0.9;
    public bool Overwrite { get; init; }
}

public sealed record SpectralStabilizationResult(
    string OutputDirectory,
    long OriginalRank,
    long StabilizedRank,
    double OriginalAlpha,
    double StabilizedAlpha,
    double IntruderScale,
    int ScaledIntruderCount,
    SpectralAnalysisResult Analysis);

internal sealed record StabilizedMatrix(
    SpectralMatrixResult Analysis,
    Tensor AdapterA,
    Tensor AdapterB) : IDisposable {
    public void Dispose() {
        AdapterA.Dispose();
        AdapterB.Dispose();
    }
}

public static class SpectralAnalyzer {
    private const string AdapterPrefix = "base_model.model.";
    private const string AdapterASuffix = ".lora_A.weight";
    private const string AdapterBSuffix = ".lora_B.weight";

    public static SpectralAnalysisResult Analyze(SpectralAnalysisConfig config) {
        Validate(config);
        ValidateAdapterMetadata(config.AdapterDirectory);

        var adapterConfig = AdapterConfig.Load(config.AdapterDirectory);
        var adapterPath = Path.Combine(config.AdapterDirectory, "adapter_model.safetensors");
        var adapter = SafeTensors.LoadModel(adapterPath);
        try {
            var pairs = FindAdapterPairs(adapter);
            var matrices = new List<SpectralMatrixResult>(pairs.Count);
            var scaling = adapterConfig.Alpha / adapterConfig.Rank;
            foreach (var pair in pairs) {
                var wanted = new HashSet<string>(StringComparer.Ordinal) { pair.BaseWeight };
                var baseWeights = SafeTensors.LoadModel(config.ModelPath, wanted);
                try {
                    if (!baseWeights.TryGetValue(pair.BaseWeight, out var baseWeight))
                        throw new InvalidDataException($"base model is missing adapter weight: {pair.BaseWeight}");
                    var result = AnalyzeMatrix(
                        pair.BaseWeight,
                        baseWeight,
                        adapter[pair.A],
                        adapter[pair.B],
                        scaling,
                        config.TopK,
                        config.SimilarityThreshold);
                    matrices.Add(result);
                } finally {
                    DisposeAll(baseWeights.Values);
                }
            }

            return new SpectralAnalysisResult(
                config.ModelPath,
                config.AdapterDirectory,
                adapterConfig.Rank,
                adapterConfig.Alpha,
                scaling,
                config.TopK,
                config.SimilarityThreshold,
                matrices.Count,
                matrices.Sum(matrix => matrix.ExaminedDimensions),
                matrices.Sum(matrix => matrix.IntruderCount),
                matrices);
        } finally {
            DisposeAll(adapter.Values);
        }
    }

    public static SpectralStabilizationResult Stabilize(SpectralStabilizationConfig config) {
        Validate(new SpectralAnalysisConfig {
            ModelPath = config.ModelPath,
            AdapterDirectory = config.AdapterDirectory,
            TopK = config.TopK,
            SimilarityThreshold = config.SimilarityThreshold
        });
        if (string.IsNullOrWhiteSpace(config.OutputDirectory))
            throw new ArgumentException("--output-dir is required");
        if (config.IntruderScale is < 0 or >= 1 || !double.IsFinite(config.IntruderScale))
            throw new ArgumentException("--scale must be in [0, 1)");
        PrepareOutputDirectory(config.AdapterDirectory, config.OutputDirectory, config.Overwrite);
        ValidateAdapterMetadata(config.AdapterDirectory);

        var adapterConfig = AdapterConfig.Load(config.AdapterDirectory);
        var adapterPath = Path.Combine(config.AdapterDirectory, "adapter_model.safetensors");
        var adapter = SafeTensors.LoadModel(adapterPath);
        var stabilized = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try {
            var pairs = FindAdapterPairs(adapter);
            var scaling = adapterConfig.Alpha / adapterConfig.Rank;
            var matrixResults = new List<SpectralMatrixResult>(pairs.Count);
            foreach (var pair in pairs) {
                var wanted = new HashSet<string>(StringComparer.Ordinal) { pair.BaseWeight };
                var baseWeights = SafeTensors.LoadModel(config.ModelPath, wanted);
                try {
                    if (!baseWeights.TryGetValue(pair.BaseWeight, out var baseWeight))
                        throw new InvalidDataException($"base model is missing adapter weight: {pair.BaseWeight}");
                    using var stabilizedMatrix = StabilizeMatrix(
                        pair.BaseWeight,
                        baseWeight,
                        adapter[pair.A],
                        adapter[pair.B],
                        scaling,
                        config.TopK,
                        config.SimilarityThreshold,
                        config.IntruderScale);
                    matrixResults.Add(stabilizedMatrix.Analysis);
                    stabilized.Add(pair.A, stabilizedMatrix.AdapterA.clone());
                    stabilized.Add(pair.B, stabilizedMatrix.AdapterB.clone());
                } finally {
                    DisposeAll(baseWeights.Values);
                }
            }

            var scaledIntruderCount = matrixResults.Count(matrix => matrix.IntruderCount != 0);
            var newRank = scaledIntruderCount == 0
                ? adapterConfig.Rank
                : checked(adapterConfig.Rank + 1);
            var newAlpha = scaling * newRank;
            if (scaledIntruderCount == 0) {
                DisposeAll(stabilized.Values);
                stabilized.Clear();
                foreach (var (name, tensor) in adapter)
                    stabilized.Add(name, tensor.clone());
            }
            var analysis = new SpectralAnalysisResult(
                config.ModelPath,
                config.AdapterDirectory,
                adapterConfig.Rank,
                adapterConfig.Alpha,
                scaling,
                config.TopK,
                config.SimilarityThreshold,
                matrixResults.Count,
                matrixResults.Sum(matrix => matrix.ExaminedDimensions),
                matrixResults.Sum(matrix => matrix.IntruderCount),
                matrixResults);

            Directory.CreateDirectory(config.OutputDirectory);
            SafeTensors.Save(Path.Combine(config.OutputDirectory, "adapter_model.safetensors"), stabilized);
            WriteStabilizedAdapterConfig(config.AdapterDirectory, config.OutputDirectory, newRank, newAlpha);
            CopyTrainingState(config.AdapterDirectory, config.OutputDirectory);
            var stabilizationResult = new SpectralStabilizationResult(
                config.OutputDirectory,
                adapterConfig.Rank,
                newRank,
                adapterConfig.Alpha,
                newAlpha,
                config.IntruderScale,
                scaledIntruderCount,
                analysis);
            File.WriteAllText(
                Path.Combine(config.OutputDirectory, "stabilization_state.json"),
                JsonSerializer.Serialize(stabilizationResult, new JsonSerializerOptions {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                    WriteIndented = true
                }));
            return stabilizationResult;
        } finally {
            DisposeAll(stabilized.Values);
            DisposeAll(adapter.Values);
        }
    }

    internal static SpectralMatrixResult AnalyzeMatrix(
        string name,
        Tensor baseWeight,
        Tensor adapterA,
        Tensor adapterB,
        double scaling,
        int topK,
        double similarityThreshold) {
        if (baseWeight.dim() != 2 || adapterA.dim() != 2 || adapterB.dim() != 2 ||
            adapterA.shape[1] != baseWeight.shape[1] ||
            adapterB.shape[0] != baseWeight.shape[0] ||
            adapterB.shape[1] != adapterA.shape[0])
            throw new InvalidDataException(
                $"LoRA shape mismatch for {name}: base=[{string.Join(',', baseWeight.shape)}], " +
                $"A=[{string.Join(',', adapterA.shape)}], B=[{string.Join(',', adapterB.shape)}]");

        using var scope = NewDisposeScope();
        using var baseFloat = baseWeight.detach().cpu().to_type(ScalarType.Float32);
        using var aFloat = adapterA.detach().cpu().to_type(ScalarType.Float32);
        using var bFloat = adapterB.detach().cpu().to_type(ScalarType.Float32);
        using var update = matmul(bFloat, aFloat) * scaling;
        using var tuned = baseFloat + update;

        var (baseU, baseS, baseVh) = torch.linalg.svd(baseFloat, fullMatrices: true);
        using (baseU)
        using (baseS)
        using (baseVh) {
            var (tunedU, tunedS, tunedVh) = torch.linalg.svd(tuned, fullMatrices: false);
            using (tunedU)
            using (tunedS)
            using (tunedVh) {
                var examined = Math.Min(topK, checked((int)tunedS.shape[0]));
                using var topU = tunedU.narrow(1, 0, examined);
                using var similarities = matmul(baseU.transpose(0, 1), topU).abs();
                var (maximumValues, maximumIndices) = similarities.max(0);
                using (maximumValues)
                using (maximumIndices) {
                    var maxima = new double[examined];
                    var intruders = new List<IntruderDimension>();
                    for (var index = 0; index < examined; ++index) {
                        var maximum = (double)maximumValues[index].item<float>();
                        maxima[index] = maximum;
                        if (maximum < similarityThreshold) {
                            intruders.Add(new IntruderDimension(
                                index + 1,
                                tunedS[index].item<float>(),
                                maximum));
                        }
                    }

                    return new SpectralMatrixResult(
                        name,
                        baseWeight.shape[0],
                        baseWeight.shape[1],
                        examined,
                        maxima,
                        intruders);
                }
            }
        }
    }

    internal static StabilizedMatrix StabilizeMatrix(
        string name,
        Tensor baseWeight,
        Tensor adapterA,
        Tensor adapterB,
        double scaling,
        int topK,
        double similarityThreshold,
        double intruderScale) {
        var analysis = AnalyzeMatrix(
            name, baseWeight, adapterA, adapterB, scaling, topK, similarityThreshold);
        var topIntruder = analysis.IntruderDimensions.FirstOrDefault();

        using var baseFloat = baseWeight.detach().cpu().to_type(ScalarType.Float32);
        using var aFloat = adapterA.detach().cpu().to_type(ScalarType.Float32);
        using var bFloat = adapterB.detach().cpu().to_type(ScalarType.Float32);
        Tensor extraA;
        Tensor extraB;
        if (topIntruder is null) {
            extraA = zeros(1, baseWeight.shape[1], dtype: ScalarType.Float32);
            extraB = zeros(baseWeight.shape[0], 1, dtype: ScalarType.Float32);
        } else {
            using var update = matmul(bFloat, aFloat) * scaling;
            using var tuned = baseFloat + update;
            var (u, singularValues, vh) = torch.linalg.svd(tuned, fullMatrices: false);
            using (u)
            using (singularValues)
            using (vh) {
                var index = topIntruder.Rank - 1;
                extraA = vh.narrow(0, index, 1).clone();
                extraB = (u.narrow(1, index, 1) *
                          ((intruderScale - 1) * topIntruder.SingularValue / scaling)).clone();
            }
        }

        using (extraA)
        using (extraB) {
            var widenedA = cat([aFloat, extraA], 0);
            var widenedB = cat([bFloat, extraB], 1);
            return new StabilizedMatrix(analysis, widenedA, widenedB);
        }
    }

    private static List<(string BaseWeight, string A, string B)> FindAdapterPairs(
        IReadOnlyDictionary<string, Tensor> adapter) {
        var pairs = new List<(string BaseWeight, string A, string B)>();
        foreach (var aName in adapter.Keys.Where(name => name.EndsWith(AdapterASuffix, StringComparison.Ordinal))
                     .Order(StringComparer.Ordinal)) {
            if (!aName.StartsWith(AdapterPrefix, StringComparison.Ordinal))
                throw new InvalidDataException($"unsupported adapter tensor name: {aName}");
            var stem = aName[..^AdapterASuffix.Length];
            var bName = $"{stem}{AdapterBSuffix}";
            if (!adapter.ContainsKey(bName))
                throw new InvalidDataException($"adapter is missing paired tensor: {bName}");
            var baseWeight = $"{stem[AdapterPrefix.Length..]}.weight";
            pairs.Add((baseWeight, aName, bName));
        }

        if (pairs.Count == 0)
            throw new InvalidDataException("adapter contains no LoRA tensor pairs");

        var consumed = pairs.SelectMany(pair => new[] { pair.A, pair.B }).ToHashSet(StringComparer.Ordinal);
        var unused = adapter.Keys.Where(name => !consumed.Contains(name)).Order().ToArray();
        if (unused.Length != 0)
            throw new InvalidDataException(
                $"adapter has {unused.Length} unsupported tensor(s): {string.Join(", ", unused.Take(4))}");
        return pairs;
    }

    private static void Validate(SpectralAnalysisConfig config) {
        if (string.IsNullOrWhiteSpace(config.ModelPath) || string.IsNullOrWhiteSpace(config.AdapterDirectory))
            throw new ArgumentException("--model and --adapter are required");
        if (config.TopK <= 0)
            throw new ArgumentException("--top-k must be positive");
        if (config.SimilarityThreshold is <= 0 or > 1 || !double.IsFinite(config.SimilarityThreshold))
            throw new ArgumentException("--epsilon must be in (0, 1]");
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

    private static void PrepareOutputDirectory(string adapterDirectory, string outputDirectory, bool overwrite) {
        var source = Path.GetFullPath(adapterDirectory).TrimEnd(Path.DirectorySeparatorChar);
        var destination = Path.GetFullPath(outputDirectory).TrimEnd(Path.DirectorySeparatorChar);
        if (string.Equals(source, destination, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("--output-dir must differ from --adapter");
        if (Directory.Exists(destination) && Directory.EnumerateFileSystemEntries(destination).Any() && !overwrite)
            throw new IOException($"output directory is not empty: {outputDirectory}; pass --force to overwrite outputs");
    }

    private static void WriteStabilizedAdapterConfig(
        string adapterDirectory,
        string outputDirectory,
        long rank,
        double alpha) {
        var source = Path.Combine(adapterDirectory, "adapter_config.json");
        var root = JsonNode.Parse(File.ReadAllText(source))?.AsObject() ??
                   throw new InvalidDataException($"invalid adapter configuration: {source}");
        root["r"] = rank;
        root["lora_alpha"] = alpha;
        File.WriteAllText(
            Path.Combine(outputDirectory, "adapter_config.json"),
            root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void CopyTrainingState(string adapterDirectory, string outputDirectory) {
        var source = Path.Combine(adapterDirectory, "training_state.json");
        if (File.Exists(source))
            File.Copy(source, Path.Combine(outputDirectory, "training_state.json"), overwrite: true);
    }

    private static void DisposeAll(IEnumerable<Tensor> tensors) {
        foreach (var tensor in tensors)
            tensor.Dispose();
    }
}
