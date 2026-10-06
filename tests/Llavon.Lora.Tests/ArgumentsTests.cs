using Xunit;

namespace Llavon.Lora.Tests;

public sealed class ArgumentsTests {
    [Theory]
    [InlineData("--only-train-incorrect")]
    [InlineData("--train-until-remembered")]
    public void TrainingFlagNeedsNoValue(string flag) {
        var arguments = new Arguments([flag]);

        arguments.Allow(flag);
        Assert.True(arguments.Flag(flag));
    }

    [Theory]
    [InlineData("--only-train-incorrect")]
    [InlineData("--train-until-remembered")]
    public void TrainingFlagPreservesFollowingOptions(string flag) {
        var arguments = new Arguments([flag, "--epochs", "5", "--no-shuffle"]);

        arguments.Allow(flag, "--epochs", "--no-shuffle");
        Assert.True(arguments.Flag(flag));
        Assert.Equal(5L, arguments.Integer("--epochs", required: true));
        Assert.True(arguments.Flag("--no-shuffle"));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void TrainingTogglesAreIndependent(bool untilRemembered, bool onlyIncorrect) {
        var args = new List<string> { "--epochs", "5" };
        if (untilRemembered) args.Add("--train-until-remembered");
        if (onlyIncorrect) args.Add("--only-train-incorrect");
        var arguments = new Arguments(args);

        arguments.Allow("--epochs", "--train-until-remembered", "--only-train-incorrect");
        Assert.Equal(untilRemembered, arguments.Flag("--train-until-remembered"));
        Assert.Equal(onlyIncorrect, arguments.Flag("--only-train-incorrect"));
        Assert.Equal(5L, arguments.Integer("--epochs", required: true));
    }

    [Theory]
    [InlineData("--only-train-incorrect")]
    [InlineData("--train-until-remembered")]
    public void DuplicateTrainingFlagsAreRejected(string flag) {
        var exception = Assert.Throws<ArgumentException>(() => new Arguments([flag, flag]));

        Assert.Equal($"duplicate option: {flag}", exception.Message);
    }
}
