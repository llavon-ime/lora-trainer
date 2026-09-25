using Xunit;

// The accelerator tests read and write process-wide state (the libtorch
// environment variable) and initialise native backends, so the suite runs one
// test at a time.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
