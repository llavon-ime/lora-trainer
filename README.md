# Llavon LoRA Trainer

`llavon-lora` is a standalone, cross-platform .NET 8 CLI and library built on
TorchSharp. It fine-tunes LoRA adapters for Hugging Face Llama checkpoints from
numeric token IDs and position-level masks supplied entirely by the caller.

The trainer contains no tokenizer, vocabulary, special-token IDs, bopomofo
table, candidate table, or language-specific data.

Its loss semantics match the historical IME training code:

- logits at position `p - 1` predict `labels[p]`;
- `loss_weights[p] == 0` excludes position `p` from loss;
- `candidate_masks[p]` restricts cross entropy at position `p` to supplied
  token IDs;
- a target absent from its candidate mask is rejected during JSONL validation;
- loss is the weighted mean across trainable positions.

## Requirements

- .NET 8 SDK for development;
- no CUDA Toolkit or NVCC is required to build or publish;
- the CUDA artifact requires a compatible NVIDIA driver when it runs.

TorchSharp and prebuilt LibTorch binaries are restored from NuGet. The CLI can
be published self-contained, so target machines do not need a .NET runtime.

## Input format

Training data is UTF-8 JSONL. Each non-empty line is one complete sequence:

```json
{
  "tokens": [1, 120, 121, 3, 901, 902, 2],
  "labels": [-100, -100, -100, -100, 901, 902, 2],
  "loss_weights": [0, 0, 0, 0, 1, 1, 1],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1],
  "candidate_masks": [null, null, null, null, [901, 950], [902, 977], null]
}
```

`input_ids` is an alias for `tokens`, and `loss_mask` is an alias for
`loss_weights`. If `labels` is omitted it equals `tokens`. If `attention_mask`
is omitted every position is attended. `candidate_masks` may be omitted for
ordinary full-vocabulary causal loss.

All IDs above are placeholders. The process creating the JSONL must supply the
actual IDs and masks.

## Build and test

```sh
dotnet restore Llavon.Lora.slnx --configfile NuGet.Config
dotnet build Llavon.Lora.slnx -c Release --no-restore
dotnet test tests/Llavon.Lora.Tests/Llavon.Lora.Tests.csproj \
  -c Release --no-build --no-restore
```

The regular test suite uses only the CPU backend and therefore needs neither a
GPU nor NVCC.

## Validate data

```sh
dotnet run --project src/Llavon.Lora.Cli -c Release -- validate \
  --train-data local-typing-tokens.jsonl \
  --vocab-size 18546 \
  --max-seq-length 384
```

Validation reads only numeric JSONL and does not load a model.

## IME integration test

The production CLI deliberately does not tokenize text or load IME tables. It
accepts only complete numeric token, label, weight, attention, and candidate-mask
arrays. The `tests/Llavon.Lora.Integration` executable contains the `ime-core`
compatibility tokenizer used to prepare the public validation fixture and to
measure the base model and trained adapter. It is test-only code and is not
included in the CLI or Core assemblies.

The table files must match the model checkpoint. The current public
`tony65535/llavon-ime-llama-250m` checkpoint uses the `ime-core` tables from
commit `00e3042`; later tables contain token IDs outside its vocabulary.

Run the test-only CUDA fixture directly when the model, validation data, and
matching tables are available locally:

```powershell
dotnet run --project tests/Llavon.Lora.Integration `
  -c Release -p:TorchBackend=cuda-windows -- `
  --model-config artifacts/integration/config.json `
  --model artifacts/integration/model.safetensors `
  --vocab-file artifacts/integration/ime_vocab.json `
  --tables-dir artifacts/integration/ime-core-00e3042/table `
  --validation-data artifacts/integration/validation.jsonl `
  --work-dir artifacts/integration/lora-run `
  --max-steps 42 --learning-rate 0.0001
```

This fixture trains on the validation rows themselves, so its post-training
score verifies the end-to-end LoRA path but is not a generalization metric.

## Train

The base model must be an unquantized Hugging Face Llama checkpoint in one
`model.safetensors` file, or a directory containing that file. Sharded models
with `model.safetensors.index.json` are supported. Architecture values are read
from the file passed through `--model-config`.

```sh
llavon-lora train \
  --model-config /models/ime/config.json \
  --model /models/ime \
  --train-data local-typing-tokens.jsonl \
  --output-dir output/ime-lora \
  --pad-token-id 0 \
  --max-seq-length 384 \
  --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --rank 16 \
  --alpha 32 \
  --dropout 0.05 \
  --batch-size 8 \
  --learning-rate 5e-4 \
  --warmup-steps 50 \
  --max-steps 500 \
  --dtype bfloat16 \
  --device cuda
```

`--target-modules` is required and is never inferred from a built-in table.
Supported projections are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`.

The learning rate is constant after the optional linear warmup. `--max-steps`
only stops training; it does not create a cosine decay cycle.

Output consists of `adapter_model.safetensors`, `adapter_config.json`, and
`training_state.json`. Adapter names and configuration follow PEFT's Llama LoRA
convention. Base weights stay frozen and are not copied into the adapter.

## Publish

CPU, self-contained:

```sh
dotnet publish src/Llavon.Lora.Cli/Llavon.Lora.Cli.csproj \
  -c Release -r linux-x64 --self-contained true \
  -p:TorchBackend=cpu -o artifacts/linux-x64-cpu
```

Windows CUDA, self-contained:

```powershell
dotnet publish src/Llavon.Lora.Cli/Llavon.Lora.Cli.csproj `
  -c Release -r win-x64 --self-contained true `
  -p:TorchBackend=cuda-windows -o artifacts/win-x64-cuda
```

Linux CUDA uses `-p:TorchBackend=cuda-linux`. These CUDA publishes restore
prebuilt native binaries and do not invoke NVCC. Publish separate CPU and CUDA
archives instead of combining both backends.

For a smaller Windows installer payload, publish framework-dependent and let
the installer provide the .NET runtime prerequisite:

```powershell
dotnet publish src/Llavon.Lora.Cli/Llavon.Lora.Cli.csproj `
  -c Release -r win-x64 --self-contained false `
  -p:TorchBackend=cpu -o artifacts/win-x64-cpu-framework-dependent
```

Native AOT and trimming are intentionally disabled because TorchSharp does not
currently guarantee compatibility with either mode.

Float16 training is rejected until FP32 optimizer master weights or proper AMP
are implemented. Float16 remains appropriate for the test-only inference pass;
use float32 or bfloat16 for training.

## Library use

Reference `src/Llavon.Lora.Core/Llavon.Lora.Core.csproj` to reuse dataset
validation, safetensors I/O, the Llama/LoRA model, or the constrained-loss API.
The C++ IME should normally invoke the CLI as a child process so CUDA failures
and GPU memory lifetime stay isolated from the input-method process.
