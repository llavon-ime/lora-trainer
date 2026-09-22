using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Llavon.Lora;

public static class SafeTensors {
    private sealed record TensorHeader(string dtype, long[] shape, long[] data_offsets);

    private static readonly JsonSerializerOptions JsonOptions = new() {
        PropertyNamingPolicy = null,
        WriteIndented = false
    };

    public static Dictionary<string, Tensor> LoadModel(string modelPath) {
        if (File.Exists(modelPath))
            return LoadFile(modelPath, null);
        if (!Directory.Exists(modelPath))
            throw new FileNotFoundException($"model path is neither a safetensors file nor a directory: {modelPath}");

        var singleFile = Path.Combine(modelPath, "model.safetensors");
        if (File.Exists(singleFile))
            return LoadFile(singleFile, null);

        var indexFile = Path.Combine(modelPath, "model.safetensors.index.json");
        if (!File.Exists(indexFile))
            throw new FileNotFoundException("model directory has no model.safetensors or model.safetensors.index.json", indexFile);

        using var index = JsonDocument.Parse(File.ReadAllBytes(indexFile));
        if (!index.RootElement.TryGetProperty("weight_map", out var weightMap) ||
            weightMap.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException($"invalid safetensors shard index: {indexFile}");

        var byFile = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var property in weightMap.EnumerateObject()) {
            var fileName = property.Value.GetString() ?? throw new InvalidDataException("null shard filename");
            if (!byFile.TryGetValue(fileName, out var names)) {
                names = new HashSet<string>(StringComparer.Ordinal);
                byFile.Add(fileName, names);
            }
            names.Add(property.Name);
        }

        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try {
            foreach (var (fileName, names) in byFile) {
                foreach (var (name, tensor) in LoadFile(Path.Combine(modelPath, fileName), names))
                    result.Add(name, tensor);
            }
            return result;
        } catch {
            foreach (var tensor in result.Values)
                tensor.Dispose();
            throw;
        }
    }

    public static void Save(string path, IReadOnlyDictionary<string, Tensor> tensors) {
        if (tensors.Count == 0)
            throw new ArgumentException("refusing to write an empty safetensors file", nameof(tensors));

        var directory = Path.GetDirectoryName(Path.GetFullPath(path));
        if (directory is not null)
            Directory.CreateDirectory(directory);

        var ordered = tensors.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToArray();
        var header = new Dictionary<string, object>(StringComparer.Ordinal);
        long offset = 0;
        foreach (var (name, tensor) in ordered) {
            var bytes = checked(tensor.numel() * tensor.element_size());
            header.Add(name, new TensorHeader(FormatDType(tensor.dtype), [.. tensor.shape], [offset, checked(offset + bytes)]));
            offset = checked(offset + bytes);
        }
        header.Add("__metadata__", new Dictionary<string, string> { ["format"] = "pt" });

        var serialized = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header, JsonOptions));
        var paddedLength = checked((serialized.Length + 7) / 8 * 8);
        var paddedHeader = new byte[paddedLength];
        serialized.CopyTo(paddedHeader, 0);
        Array.Fill(paddedHeader, (byte)' ', serialized.Length, paddedHeader.Length - serialized.Length);

        using var output = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> lengthBytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(lengthBytes, checked((ulong)paddedHeader.Length));
        output.Write(lengthBytes);
        output.Write(paddedHeader);
        foreach (var (_, tensor) in ordered) {
            using var scope = torch.NewDisposeScope();
            using var packed = tensor.detach().cpu().contiguous();
            packed.WriteBytesToStream(output, 1024 * 1024);
        }
    }

    private static Dictionary<string, Tensor> LoadFile(string path, IReadOnlySet<string>? wanted) {
        using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Span<byte> lengthBytes = stackalloc byte[8];
        input.ReadExactly(lengthBytes);
        var headerLength = BinaryPrimitives.ReadUInt64LittleEndian(lengthBytes);
        if (headerLength == 0 || headerLength > 1UL << 30 || headerLength > (ulong)(input.Length - 8))
            throw new InvalidDataException($"invalid safetensors header length: {path}");

        var headerBytes = new byte[checked((int)headerLength)];
        input.ReadExactly(headerBytes);
        using var metadata = JsonDocument.Parse(headerBytes);
        var dataStart = checked(8L + (long)headerLength);
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);

        try {
            foreach (var property in metadata.RootElement.EnumerateObject()) {
                if (property.NameEquals("__metadata__") || wanted is not null && !wanted.Contains(property.Name))
                    continue;

                var description = property.Value;
                var dtypeName = description.GetProperty("dtype").GetString() ??
                    throw new InvalidDataException($"missing dtype for tensor {property.Name}");
                var (dtype, itemSize) = ParseDType(dtypeName);
                var shape = description.GetProperty("shape").EnumerateArray().Select(item => item.GetInt64()).ToArray();
                var offsets = description.GetProperty("data_offsets").EnumerateArray().Select(item => item.GetInt64()).ToArray();
                if (offsets.Length != 2 || shape.Any(dimension => dimension < 0))
                    throw new InvalidDataException($"invalid metadata for tensor {property.Name}");

                long elementCount = 1;
                foreach (var dimension in shape)
                    elementCount = checked(elementCount * dimension);
                var byteCount = checked(elementCount * itemSize);
                if (offsets[0] < 0 || offsets[1] < offsets[0] || offsets[1] - offsets[0] != byteCount ||
                    dataStart + offsets[1] > input.Length)
                    throw new InvalidDataException($"invalid byte range for tensor {property.Name}");

                var tensor = torch.empty(shape, dtype: dtype, device: CPU);
                try {
                    input.Position = checked(dataStart + offsets[0]);
                    tensor.ReadBytesFromStream(input, 1024 * 1024);
                    tensors.Add(property.Name, tensor);
                } catch {
                    tensor.Dispose();
                    throw;
                }
            }
            return tensors;
        } catch {
            foreach (var tensor in tensors.Values)
                tensor.Dispose();
            throw;
        }
    }

    private static (ScalarType Type, long Size) ParseDType(string value) => value switch {
        "F32" => (ScalarType.Float32, 4),
        "F16" => (ScalarType.Float16, 2),
        "BF16" => (ScalarType.BFloat16, 2),
        "F64" => (ScalarType.Float64, 8),
        "I64" => (ScalarType.Int64, 8),
        "I32" => (ScalarType.Int32, 4),
        "I16" => (ScalarType.Int16, 2),
        "I8" => (ScalarType.Int8, 1),
        "U8" => (ScalarType.Byte, 1),
        "BOOL" => (ScalarType.Bool, 1),
        _ => throw new InvalidDataException($"unsupported safetensors dtype: {value}")
    };

    private static string FormatDType(ScalarType value) => value switch {
        ScalarType.Float32 => "F32",
        ScalarType.Float16 => "F16",
        ScalarType.BFloat16 => "BF16",
        ScalarType.Float64 => "F64",
        ScalarType.Int64 => "I64",
        ScalarType.Int32 => "I32",
        ScalarType.Int16 => "I16",
        ScalarType.Int8 => "I8",
        ScalarType.Byte => "U8",
        ScalarType.Bool => "BOOL",
        _ => throw new InvalidDataException($"unsupported tensor dtype for safetensors output: {value}")
    };
}
