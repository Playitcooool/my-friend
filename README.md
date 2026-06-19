# My Friend

Utilities for parsing WeChat chat screenshots with a local OpenAI-compatible
vision model endpoint.

## Setup

Create a project virtual environment with `uv`, then install dependencies:

```bash
uv venv
uv pip install openai "mlx-lm[train]" pyobjc-framework-Vision pyobjc-framework-Quartz pillow
```

## Usage

Put extracted screenshot frames in `frames/`, then run the default OCR extractor:

```bash
python process_frames.py
```

Useful environment variables:

```bash
LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_MODEL=lmstudio-community:Qwen3.5-4B-MLX-8bit
FRAMES_DIR=frames
OUTPUT_PATH=raw_frame_items.jsonl
MAX_CONCURRENCY=2
MAX_RETRIES=2
INCLUDE_RAW=1
```

Choose the extraction backend explicitly when needed:

```bash
python process_frames.py --method ocr
python process_frames.py --method vlm
python process_frames.py --method ocr-with-vlm-fallback
```

OCR uses Apple Vision and groups OCR text lines into detected WeChat bubbles by
default. Use `--disable-bubble-grouping` to compare against the previous
line-level OCR items. VLM extraction is available through the same
`process_frames.py` entry point.

Local videos, frame images, and JSONL extraction outputs are intentionally
ignored and should not be committed.

## Clean Conversation Data

Clean the frame-level extraction into bubble-level JSONL and SFT pair JSONL:

```bash
python clean_wechat_conversation.py
```

Output:

```bash
data/train.jsonl
```

`data/train.jsonl` contains one training example per line:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The cleaner removes overlapping frame duplicates, recall notices, quoted-message
prefixes, incomplete edge bubbles, and dangling final user messages.

## LoRA Fine-Tuning with MLX-LM

MLX-LM expects local fine-tuning data in a directory containing `train.jsonl`.
Create the data directory and use the cleaned JSONL as the training split:

```bash
mkdir -p data
python clean_wechat_conversation.py
```

Optional validation split:

```bash
cp data/train.jsonl data/valid.jsonl
```

Fine-tune using the project config:

```bash
mlx_lm.lora --config lora_config.yaml
```

Equivalent explicit command:

```bash
mlx_lm.lora \
  --model "Qwen:Qwen3-14B-MLX-8bit" \
  --train \
  --fine-tune-type lora \
  --data ./data \
  --adapter-path ./adapters/contact \
  --batch-size 1 \
  --iters 1000 \
  --learning-rate 1e-5 \
  --num-layers 4 \
  --max-seq-length 1024 \
  --grad-checkpoint \
  --mask-prompt
```

The adapter weights are written to:

```bash
adapters/contact/
```

## Inference with MLX-LM

Generate with the base model plus the trained LoRA adapter:

```bash
mlx_lm.generate \
  --model "Qwen:Qwen3-14B-MLX-8bit" \
  --adapter-path ./adapters/contact \
  --prompt "晚上想吃什么？" \
  --max-tokens 256 \
  --temp 0.7
```

For chat-style prompts, format the prompt as a user turn:

```bash
mlx_lm.generate \
  --model "Qwen:Qwen3-14B-MLX-8bit" \
  --adapter-path ./adapters/contact \
  --prompt '<|im_start|>user
晚上想吃什么？
<|im_end|>
<|im_start|>assistant
' \
  --max-tokens 256 \
  --temp 0.7
```
