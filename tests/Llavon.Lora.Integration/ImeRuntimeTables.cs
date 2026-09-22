using System.Text;
using System.Text.Json;

namespace Llavon.Lora.Integration;

internal sealed class ImeRuntimeTables {
    private readonly IReadOnlyDictionary<string, long> characters;
    private readonly IReadOnlyDictionary<string, long> latin;
    private readonly IReadOnlyDictionary<string, long> special;
    private readonly IReadOnlyDictionary<string, long> bopomofoTokens;
    private readonly IReadOnlyDictionary<string, string[]> bopomofoCandidates;

    private ImeRuntimeTables(
        IReadOnlyDictionary<string, long> characters,
        IReadOnlyDictionary<string, long> latin,
        IReadOnlyDictionary<string, long> special,
        IReadOnlyDictionary<string, long> bopomofoTokens,
        IReadOnlyDictionary<string, string[]> bopomofoCandidates) {
        this.characters = characters;
        this.latin = latin;
        this.special = special;
        this.bopomofoTokens = bopomofoTokens;
        this.bopomofoCandidates = bopomofoCandidates;
        PadTokenId = RequireSpecial("<PAD>");
        BosTokenId = RequireSpecial("<BOS>");
        SepTokenId = RequireSpecial("<SEP>");
        UnknownTokenId = RequireSpecial("<UNK>");
        SpaceTokenId = RequireSpecial("<SP>");
        LatinTokenId = RequireSpecial("<LATIN>");
    }

    public long PadTokenId { get; }
    public long BosTokenId { get; }
    public long SepTokenId { get; }
    public long UnknownTokenId { get; }
    public long SpaceTokenId { get; }
    public long LatinTokenId { get; }

    public static ImeRuntimeTables Load(string tablesDirectory) {
        var tokenDirectory = Path.Combine(tablesDirectory, "tokens");
        return new ImeRuntimeTables(
            LoadMap<long>(Path.Combine(tokenDirectory, "chars.json")),
            LoadMap<long>(Path.Combine(tokenDirectory, "latin.json")),
            LoadMap<long>(Path.Combine(tokenDirectory, "special_tokens.json")),
            LoadMap<long>(Path.Combine(tokenDirectory, "bpmf.json")),
            LoadMap<string[]>(Path.Combine(tablesDirectory, "bopomofo_char.json")));
    }

    public long[] TokenizePrompt(string context, IReadOnlyList<string> bopomofo) {
        var result = new List<long> { BosTokenId };
        var contextTokens = TokenizeContext(context);
        var firstKnown = contextTokens.FindIndex(token => token != UnknownTokenId);
        if (firstKnown >= 0)
            result.AddRange(contextTokens[firstKnown..]);
        foreach (var syllable in bopomofo) {
            var token = $"<{syllable}>";
            if (!bopomofoTokens.TryGetValue(token, out var tokenId))
                throw new InvalidDataException($"IME BPMF table does not contain token: {token}");
            result.Add(tokenId);
        }
        result.Add(SepTokenId);
        return result.ToArray();
    }

    public IReadOnlyList<(long TokenId, string Character)> CandidateTokens(string bopomofo) {
        if (!bopomofoCandidates.TryGetValue(bopomofo, out var candidates))
            return [];
        var result = new List<(long, string)>(candidates.Length);
        var seen = new HashSet<long>();
        foreach (var candidate in candidates) {
            var rune = candidate.EnumerateRunes().FirstOrDefault();
            if (rune.Value == 0)
                continue;
            var character = rune.ToString();
            if (characters.TryGetValue(character, out var tokenId) && seen.Add(tokenId))
                result.Add((tokenId, character));
        }
        return result;
    }

    public void ValidateVocabulary(string vocabularyPath, long vocabularySize) {
        foreach (var table in new[] { characters, latin, special, bopomofoTokens }) {
            if (table.Values.Any(id => id < 0 || id >= vocabularySize))
                throw new InvalidDataException("IME table contains a token outside the model vocabulary");
        }
        using var document = JsonDocument.Parse(File.ReadAllBytes(vocabularyPath));
        var tokens = document.RootElement.GetProperty("tokens").EnumerateArray()
            .Select(token => token.GetString() ?? throw new InvalidDataException("null vocabulary token"))
            .ToArray();
        if (tokens.LongLength != vocabularySize)
            throw new InvalidDataException("IME vocabulary size does not match the model");
        ValidateNames(tokens, characters, static token => token, StringComparison.Ordinal);
        ValidateNames(tokens, latin, static token => $"<LATIN:{token}>", StringComparison.OrdinalIgnoreCase);
        ValidateNames(tokens, special, static token => token, StringComparison.Ordinal);
        ValidateNames(tokens, bopomofoTokens, static token => token, StringComparison.Ordinal);
    }

    private List<long> TokenizeContext(string context) {
        var runes = context.EnumerateRunes().ToArray();
        var result = new List<long>(runes.Length);
        for (var index = 0; index < runes.Length; ++index) {
            var rune = runes[index];
            var text = rune.ToString();
            if (rune.Value == ' ') {
                result.Add(SpaceTokenId);
            } else if (characters.TryGetValue(text, out var characterToken)) {
                result.Add(characterToken);
            } else if (IsLatinCharacter(rune.Value)) {
                var word = new StringBuilder();
                while (index < runes.Length && IsLatinCharacter(runes[index].Value)) {
                    word.Append(ToLowerAscii(runes[index].Value));
                    ++index;
                }
                --index;
                result.Add(latin.TryGetValue(word.ToString(), out var latinToken) ? latinToken : LatinTokenId);
            } else {
                result.Add(UnknownTokenId);
            }
        }
        return result;
    }

    private long RequireSpecial(string token) => special.TryGetValue(token, out var tokenId)
        ? tokenId
        : throw new InvalidDataException($"IME special-token table does not contain {token}");

    private static bool IsLatinCharacter(int value) =>
        value is >= 'a' and <= 'z' or >= 'A' and <= 'Z' or >= '0' and <= '9' or '-' or '_' or '+';

    private static char ToLowerAscii(int value) =>
        value is >= 'A' and <= 'Z' ? (char)(value ^ 0x20) : (char)value;

    private static Dictionary<string, T> LoadMap<T>(string path) =>
        JsonSerializer.Deserialize<Dictionary<string, T>>(File.ReadAllBytes(path)) ??
        throw new InvalidDataException($"IME table is not a JSON object: {path}");

    private static void ValidateNames(
        IReadOnlyList<string> vocabulary,
        IReadOnlyDictionary<string, long> table,
        Func<string, string> format,
        StringComparison comparison) {
        foreach (var (token, id) in table)
            if (!string.Equals(vocabulary[checked((int)id)], format(token), comparison))
                throw new InvalidDataException($"IME table token mismatch at ID {id}");
    }
}
