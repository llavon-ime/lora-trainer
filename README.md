# Llavon LoRA Trainer

`llavon-lora` 是以 TorchSharp 建置的獨立、跨平台 .NET 8 CLI 與函式庫，用完全由
呼叫端提供的數值 token ID 與位置層級遮罩，為 Hugging Face Llama checkpoint
微調 LoRA adapter。

trainer 不含斷詞器、詞彙、特殊 token ID、注音表、候選字表或任何語言特定資料。

其 loss 語意與既有的輸入法訓練程式碼一致：

- 位置 `p - 1` 的 logits 預測 `labels[p]`；
- `loss_weights[p] == 0` 會把位置 `p` 排除在 loss 之外；
- `candidate_masks[p]` 會把位置 `p` 的交叉熵限制在提供的 token ID；
- 目標不在其候選遮罩中時，會在 JSONL 驗證階段被拒絕；
- loss 是可訓練位置上的加權平均。

## 需求

- 開發需要 .NET 8 SDK；
- 自動發行版支援 CPU 與 macOS 的 Metal（MPS）GPU 訓練，不需要 CUDA 或
  NVIDIA 驅動程式；macOS 產物內含啟用 MPS 的 LibTorch，因此 `--device mps`
  不需要額外下載；
- CUDA 版本使用預先建置的 NuGet 二進位檔在本機發佈，建置時不需要 CUDA
  Toolkit 或 NVCC，執行時需要相容的 NVIDIA 驅動程式。NVIDIA GPU 也可以使用
  不含 LibTorch 的 `TorchBackend=cuda-linux-fetch` 建置，執行時由
  `llavon-lora fetch-libtorch` 下載 PyTorch 官方的 CUDA LibTorch
  （`2.10.0+cu128`，約 3.9 GB）到 `~/.cache/llavon-lora`；
- AMD GPU 使用 ROCm：LibTorch 沒有 Vulkan 訓練後端，而 PyTorch 讓 ROCm 走
  `cuda` 裝置名稱，因此 `--device cuda` 在 ROCm 版本上就是 AMD GPU。ROCm 建置
  本身很小，PyTorch 官方的 ROCm LibTorch（解開後約 9.4 GB）由
  `llavon-lora fetch-libtorch` 或 `scripts/fetch-libtorch-rocm.sh` 下載到
  `~/.cache/llavon-lora`，之後 ROCm 建置會自動使用，不需要額外參數。

TorchSharp 與預先建置的 LibTorch 二進位檔會從 NuGet 還原。CLI 可以發佈成
self-contained，因此目標機器不需要 .NET 執行階段。Apple Silicon 建置會把
LibTorch 重新指向內附的 OpenMP 執行階段，因此 macOS 使用者不需要另外用
Homebrew 安裝 `libomp`。

## 輸入格式

訓練資料是 UTF-8 JSONL。每個非空行是一筆完整序列：

```json
{
  "tokens": [1, 120, 121, 3, 901, 902, 2],
  "labels": [-100, -100, -100, -100, 901, 902, 2],
  "loss_weights": [0, 0, 0, 0, 1, 1, 1],
  "attention_mask": [1, 1, 1, 1, 1, 1, 1],
  "candidate_masks": [null, null, null, null, [901, 950], [902, 977], null]
}
```

`input_ids` 是 `tokens` 的別名，`loss_mask` 是 `loss_weights` 的別名。省略
`labels` 時會等於 `tokens`；省略 `attention_mask` 時每個位置都會被關注。
`candidate_masks` 可在一般的全詞彙 causal loss 下省略。

上述所有 ID 都是佔位符。產生 JSONL 的處理程序必須提供實際的 ID 與遮罩。

## 建置與測試

```sh
dotnet restore Llavon.Lora.slnx --configfile NuGet.Config
dotnet build Llavon.Lora.slnx -c Release --no-restore
dotnet test tests/Llavon.Lora.Tests/Llavon.Lora.Tests.csproj \
  -c Release --no-build --no-restore
```

一般測試套件只使用 CPU 後端，因此不需要 GPU 或 NVCC。CI 會在 macOS 15 上執行
Metal（MPS）冒煙測試，並建置 ROCm 後端確認可編譯。有 GPU 的機器可以手動執行
同樣的冒煙測試：

```sh
scripts/fetch-libtorch-rocm.sh
LLAVON_LORA_REQUIRE_CUDA=1 \
LLAVON_LORA_TORCH_LIB="$HOME/.cache/llavon-lora/libtorch-rocm7.0-2.10.0/libtorch/lib" \
  dotnet test tests/Llavon.Lora.Tests/Llavon.Lora.Tests.csproj -c Release \
  -p:TorchBackend=rocm-linux --filter FullyQualifiedName~CudaTrainingCompletesOneStep
```

## 驗證資料

```sh
dotnet run --project src/Llavon.Lora.Cli -c Release -- validate \
  --train-data local-typing-tokens.jsonl \
  --vocab-size 18546 \
  --max-seq-length 384
```

驗證只讀取數值 JSONL，不會載入模型。

## 輸入法整合測試

正式 CLI 刻意不對文字斷詞，也不載入輸入法表。它只接受完整的數值 token、
label、weight、attention 與候選遮罩陣列。`tests/Llavon.Lora.Integration`
執行檔內含 `ime-core` 相容的斷詞器，用來準備公開驗證 fixture，以及量測基礎模型
與訓練後的 adapter。這是僅供測試的程式碼，不會包含在 CLI 或 Core 組件中。

表檔案必須與模型 checkpoint 相符。目前公開的
`tony65535/llavon-ime-llama-250m` checkpoint 使用 commit `00e3042` 的
`ime-core` 表；更後面的表含有超出其詞彙的 token ID。

當模型、驗證資料與相符的表都在本機時，可直接執行僅供測試的 CUDA fixture：

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

這個 fixture 會直接在驗證資料列上訓練，因此其訓練後分數只驗證端到端 LoRA
流程，不是泛化指標。

## 訓練

基礎模型必須是未量化的 Hugging Face Llama checkpoint，放在單一
`model.safetensors` 檔中，或是包含該檔的目錄。支援以
`model.safetensors.index.json` 分片的模型。架構參數會從 `--model-config`
傳入的檔案讀取。

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

`--target-modules` 是必填，永遠不會從內建表推斷。支援的 projection 有
`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj` 與
`down_proj`。

`--device` 接受 `auto`、`cpu`、`cuda` 與 `mps`。`auto` 會依序選擇 CUDA（含
ROCm）、MPS 與 CPU；`mps` 只在 Apple Silicon 的 macOS 上可用，且目前限定
`--dtype float32`。`llavon-lora devices [--json]` 會列出可用的後端與目前使用
的 LibTorch，用來確認 GPU 是否被正確偵測。

學習率在選用的線性 warmup 之後保持固定。`--max-steps` 只會停止訓練，不會產生
cosine decay 週期。

輸出包含 `adapter_model.safetensors`、`adapter_config.json` 與
`training_state.json`。adapter 名稱與設定遵循 PEFT 的 Llama LoRA 慣例。基礎
權重保持凍結，不會複製到 adapter。

## 匯出 GGUF

`export-gguf` 會轉換 Hugging Face checkpoint，並可選擇合併 PEFT LoRA adapter。
匯出器與量化器都在 CLI 處理程序內執行：不會呼叫 Python、`llama-quantize` 或
任何其他執行檔。GGUF 寫入實作於 `Llavon.Lora.Core`；Q4_K_M 與其他 llama.cpp
量化型別使用固定版本的 `LLamaSharp.Backend.Cpu` NuGet 套件。

完整詞彙仍由呼叫端擁有，必須透過 `--vocab-file` 提供；函式庫內不會編入任何
詞彙或輸入法表。

```powershell
llavon-lora export-gguf `
  --model-config model/config.json `
  --model model/model.safetensors `
  --vocab-file ime_vocab.json `
  --adapter output/ime-lora `
  --outfile output/ime-lora-f16.gguf `
  --quantize Q4_K_M `
  --quantized-outfile output/ime-lora-Q4_K_M.gguf
```

在發行自動化中使用 `--expected-outfile-sha256` 與
`--expected-quantized-sha256`，讓轉換意外變動時直接失敗。已部署基礎模型的回歸
值為：

- F16: `788e435fb4f7a07a826baf499127b5c91d8c4d3df6f4fea17b4c9379878f20f6`
- Q4_K_M: `e0205904a65b735ed735b709d5e1502ebbe31fbd1393042573af9843d39e2ab8`

## 發佈

release workflow 會給每次發行一個 Asia/Taipei（UTC+8）的 CalVer，格式為
`YYYY.MM.DD.GITHUB_RUN_NUMBER`。例如 `2026.09.22.37` 是 2026-09-22 的第 37 次
release workflow 執行。相同版本會內嵌到每個平台的產物中，且不需載入 Torch
即可查詢：

```sh
llavon-lora --version --json
```

自動發行會發佈 Windows x64、Linux x64 與 macOS Arm64 的 CPU 產物。滾動的
`latest` 發行包含 `latest.json`；其下載 URL 永遠指向不可變的 CalVer 發行，並
包含 SHA-256 與位元組大小等中繼資料。

發行產物使用 .NET 單檔發佈。受控組件會打包進 `llavon-lora`；其餘原生 payload
依 RID 篩選（Windows 為 `.dll`、Linux 為 `.so`、macOS 為 `.dylib`）。

Windows 發行版另外包含一個小型的 NativeAOT 網頁安裝器。它會在下載所選壓縮檔
前，先把本機 `llavon-lora-manifest.json` 與不可變的發行 manifest 比對，拒絕
大小或 SHA-256 不符者，並只在新壓縮檔驗證通過後取代已安裝的 trainer。

CPU、self-contained：

```sh
dotnet publish src/Llavon.Lora.Cli/Llavon.Lora.Cli.csproj \
  -c Release -r linux-x64 --self-contained true \
  -p:PublishSingleFile=true -p:DebugType=None \
  -p:TorchBackend=cpu -o artifacts/linux-x64-cpu
```

CUDA 刻意不包含在自動發行中：TorchSharp 啟用 CUDA 的 LibTorch 執行階段有數
GB，不適合作為預設下載。開發者仍可手動建立本機 CUDA 發佈：

```powershell
dotnet publish src/Llavon.Lora.Cli/Llavon.Lora.Cli.csproj `
  -c Release -r win-x64 --self-contained true `
  -p:TorchBackend=cuda-windows -o artifacts/win-x64-cuda
```

Linux CUDA 使用 `-p:TorchBackend=cuda-linux`。這些 CUDA 發佈會還原預先建置的
原生二進位檔，不會呼叫 NVCC。請分開發佈 CPU 與 CUDA 壓縮檔，不要把兩種後端
合併。

ROCm 與 CUDA 的 LibTorch（分別約 9.4 GB 與 3.9 GB）不在自動發行中，但 release
仍提供**不含 LibTorch** 的 `linux-x64-rocm` 與 `linux-x64-cuda` 產物
（各約 30–40 MB）：安裝後由 `llavon-lora fetch-libtorch` 下載 PyTorch 官方的
LibTorch（ROCm 為 `2.10.0+rocm7.0`，CUDA 為 `2.10.0+cu128`，ABI 都與
TorchSharp 0.106 的 LibTorch 2.10 相符）到 `~/.cache/llavon-lora`，ROCm 版本
內含 rocBLAS 與 hipBLASLt 的 kernel 資料庫（不含 SDPA 使用的 aotriton 影像，
本訓練器不使用 SDPA）。兩者的執行階段程式庫都一併內附，只需要顯示卡驅動程式；
之後對應的建置會自動使用該 cache，不需要任何參數。

也可以自行發佈把 LibTorch 一起打包的完整版本：

```sh
scripts/fetch-libtorch-rocm.sh
dotnet publish src/Llavon.Lora.Cli/Llavon.Lora.Cli.csproj \
  -c Release -r linux-x64 --self-contained true \
  -p:TorchBackend=rocm-linux -o artifacts/linux-x64-rocm
```

發佈時會把 LibTorch 複製到輸出目錄（同一檔案系統會使用硬連結），產物約 9.5 GB。
若想讓 LibTorch 留在應用程式目錄外，可改用 `--torch-lib-dir` 或
`LLAVON_LORA_TORCH_LIB` 指向該目錄，並在沒有內附 LibTorch 的建置上執行。
已經內附 LibTorch 的建置不能切換到另一份：同一個處理程序載入兩份 LibTorch 會
因為重複註冊 kernel 而直接中止，CLI 會事先拒絕這種組合。
`TorchBackend=rocm-linux` 僅限 Linux。

若想縮小 Windows 安裝器的 payload，可改發佈 framework-dependent，並由安裝器
提供 .NET 執行階段的前置需求：

```powershell
dotnet publish src/Llavon.Lora.Cli/Llavon.Lora.Cli.csproj `
  -c Release -r win-x64 --self-contained false `
  -p:TorchBackend=cpu -o artifacts/win-x64-cpu-framework-dependent
```

Native AOT 與 trimming 刻意停用，因為 TorchSharp 目前不保證與這兩種模式相容。

在實作 FP32 optimizer master weights 或正確的 AMP 之前，Float16 訓練會被
拒絕。Float16 仍適用於僅供測試的推論階段；訓練請使用 float32 或 bfloat16。

## 以函式庫使用

參考 `src/Llavon.Lora.Core/Llavon.Lora.Core.csproj`，即可重用資料集驗證、
safetensors I/O、Llama/LoRA 模型或受限 loss API。呼叫 GGUF 量化的應用程式必須
參考一個 LLamaSharp 後端；CLI 使用 `LLamaSharp.Backend.Cpu`。C++ 輸入法通常
應以子處理程序方式呼叫 CLI，讓 CUDA 失敗與 GPU 記憶體生命週期與輸入法處理
程序隔離。
