# My Friend

Build a private WeChat-style chat dataset from screenshots, fine-tune an MLX-LM
LoRA adapter, filter weak examples with an LLM judge, and chat with the adapter
from the terminal.

The repository is designed around local artifacts. Personal media, extracted
JSONL files, trained adapters, and model configs are ignored by git by default.

## Workflow

```text
screen recording -> extract frames -> OCR/VLM extraction -> clean context SFT data -> judge/split -> LoRA -> terminal chat
```

## Quick Start

Create the environment:

```bash
uv venv
uv pip install openai "mlx-lm[train]" pyobjc-framework-Vision pyobjc-framework-Quartz pillow
```

Use the default private artifact names:

```bash
uv run python extract_frames.py --video screen_recording.mp4
uv run python process_frames.py
uv run python clean_wechat_conversation.py
uv run python judge_roleplay_value.py --model YOUR_JUDGE_MODEL
mlx_lm.lora --config lora_config.yaml
uv run python chat_with_adapter.py
```

Use an explicit local contact prefix when you want named artifacts:

```bash
uv run python extract_frames.py --video screen_recording.mp4 --name CONTACT_ALIAS
uv run python process_frames.py --name CONTACT_ALIAS
uv run python clean_wechat_conversation.py --name CONTACT_ALIAS
uv run python judge_roleplay_value.py --name CONTACT_ALIAS --model YOUR_JUDGE_MODEL
uv run python chat_with_adapter.py --name CONTACT_ALIAS
```

With `--name CONTACT_ALIAS`, the default artifacts become:

| Step | Default artifact |
| --- | --- |
| Extract frames | `CONTACT_ALIAS_frames/`, `CONTACT_ALIAS_raw_frame_items.jsonl` |
| Clean data | `CONTACT_ALIAS_raw_frame_items.jsonl` -> `data/CONTACT_ALIAS_train.jsonl` |
| Quality report | `data/CONTACT_ALIAS_quality_report.json` |
| Judge/filter | `data/CONTACT_ALIAS_train.filtered.jsonl`, `data/CONTACT_ALIAS_valid.filtered.jsonl`, and `data/CONTACT_ALIAS.roleplay_judgments.jsonl` |
| Chat | `adapters/CONTACT_ALIAS/` |

Explicit paths always override `--name`.

## From Screen Recording To Frames

The easiest data collection path is to screen record the WeChat chat history,
scroll slowly through the conversation, then use `ffmpeg` to cut the recording
into image frames for OCR.

1. Open the WeChat conversation.
2. Start a screen recording.
3. Scroll upward or downward slowly and steadily through the chat history.
4. Stop recording and save the video locally.
5. Extract frames with `extract_frames.py`.

Install `ffmpeg` if needed:

```bash
brew install ffmpeg
```

Extract one frame every second:

```bash
uv run python extract_frames.py --video screen_recording.mp4
```

For a named local artifact set:

```bash
uv run python extract_frames.py --video screen_recording.mp4 --name CONTACT_ALIAS
```

If the recording scrolls quickly, use more frames:

```bash
uv run python extract_frames.py --video screen_recording.mp4 --name CONTACT_ALIAS --fps 2
```

To cut only part of a recording:

```bash
uv run python extract_frames.py --video screen_recording.mp4 --start 00:01:00 --duration 00:02:30 --overwrite
```

Tips:

- Scroll slowly enough that each chat bubble appears fully in at least one frame.
- Avoid notification banners, floating windows, and cursor movement over text.
- Keep the chat column in a stable position and avoid zoom changes mid-recording.
- Use `fps=1` for slow scrolling and `fps=2` or `fps=3` for faster scrolling.
- The frame directory is ignored by git, but it can still contain private data.

## 中文指南

这个项目的推荐流程是：录制微信聊天记录的视频，切成图片帧，做 OCR/视觉模型解析，
清洗成微调数据，再用 LoRA 训练一个本地聊天风格适配器。

完整流程：

```text
微信录屏 -> ffmpeg 切帧 -> OCR/VLM 提取聊天气泡 -> 清洗 SFT 数据 -> LLM 过滤低质量样本 -> LoRA 微调 -> 终端多轮聊天
```

### 1. 录制微信聊天记录

1. 打开要处理的微信聊天窗口。
2. 开始屏幕录制。
3. 慢慢滚动聊天记录，尽量保持匀速。
4. 确保每个聊天气泡至少在某一帧里完整出现。
5. 结束录制，把视频放到项目目录或你方便引用的位置。

建议：

- 不要滚动太快，否则 OCR 可能只能看到被截断的气泡。
- 尽量避免通知横幅、悬浮窗、鼠标遮挡文字。
- 录制过程中不要频繁缩放或改变窗口位置。
- 录屏和切出来的图片帧都可能包含隐私信息，不要提交到 git。

### 2. 用 ffmpeg 切图片帧

如果没有安装 `ffmpeg`：

```bash
brew install ffmpeg
```

默认匿名产物：

```bash
uv run python extract_frames.py --video screen_recording.mp4
```

如果你想在本地用联系人别名区分不同数据集：

```bash
uv run python extract_frames.py --video screen_recording.mp4 --name CONTACT_ALIAS
```

如果滚动速度比较快，可以提高抽帧频率：

```bash
uv run python extract_frames.py --video screen_recording.mp4 --name CONTACT_ALIAS --fps 2
```

### 3. 提取、清洗、过滤

匿名默认流程：

```bash
uv run python process_frames.py
uv run python clean_wechat_conversation.py
uv run python judge_roleplay_value.py --model YOUR_JUDGE_MODEL
```

带本地别名前缀的流程：

```bash
uv run python process_frames.py --name CONTACT_ALIAS
uv run python clean_wechat_conversation.py --name CONTACT_ALIAS
uv run python judge_roleplay_value.py --name CONTACT_ALIAS --model YOUR_JUDGE_MODEL
```

生成文件对应关系：

| 步骤 | 产物 |
| --- | --- |
| 切帧 | `CONTACT_ALIAS_frames/` |
| OCR/VLM 提取 | `CONTACT_ALIAS_raw_frame_items.jsonl` |
| 清洗训练数据 | `data/CONTACT_ALIAS_train.jsonl` |
| 清洗质量报告 | `data/CONTACT_ALIAS_quality_report.json` |
| 过滤后训练数据 | `data/CONTACT_ALIAS_train.filtered.jsonl` |
| 验证集 | `data/CONTACT_ALIAS_valid.filtered.jsonl` |
| 审核记录 | `data/CONTACT_ALIAS.roleplay_judgments.jsonl` |

### 4. 微调和聊天

编辑 `lora_config.yaml` 里的模型路径和 adapter 路径，然后运行：

```bash
mlx_lm.lora --config lora_config.yaml
```

终端多轮聊天：

```bash
uv run python chat_with_adapter.py --name CONTACT_ALIAS
```

聊天内命令：

```text
/reset    清空上下文
/history  查看当前上下文
/help     查看命令
/exit     退出
```

## Commands

### 1. Extract Frames

```bash
uv run python extract_frames.py --video screen_recording.mp4
```

Useful options:

```bash
uv run python extract_frames.py --video screen_recording.mp4 --name CONTACT_ALIAS
uv run python extract_frames.py --video screen_recording.mp4 --fps 2 --frames-dir frames
uv run python extract_frames.py --video screen_recording.mp4 --start 00:00:30 --duration 60 --overwrite
```

`--name CONTACT_ALIAS` maps to `CONTACT_ALIAS_frames/`. Explicit path flags
always override `--name`.

### 2. Extract Frame Items

Put extracted frames in `contact_frames/` or pass `--frames`.

```bash
uv run python process_frames.py --method ocr
```

Useful options:

```bash
uv run python process_frames.py --frames frames --output raw_items.jsonl
uv run python process_frames.py --method vlm --vlm-base-url http://127.0.0.1:1234/v1
uv run python process_frames.py --method ocr-with-vlm-fallback
```

OCR uses Apple Vision and groups OCR text lines into chat bubbles by default.
Use `--disable-bubble-grouping` only when you need line-level debugging output.
After processing, the script prints a quality summary with total frames,
successful frames, failed frames, empty frames, average items per frame, and
warnings when the output count looks mismatched.

### 3. Clean SFT Examples

```bash
uv run python clean_wechat_conversation.py
```

Default output:

```text
data/train.jsonl
```

Each row has the standard SFT shape and ends with an assistant message. The
default mode builds multi-turn context windows:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Useful options:

```bash
uv run python clean_wechat_conversation.py --input raw_items.jsonl --output data/train.jsonl
uv run python clean_wechat_conversation.py --items-output debug_clean_items.jsonl
uv run python clean_wechat_conversation.py --example-mode pair
uv run python clean_wechat_conversation.py --context-turns 3
uv run python clean_wechat_conversation.py --redact-term "Local Name" --redact-terms-file private_terms.txt
uv run python clean_wechat_conversation.py --no-redact
```

The cleaner removes overlapping frame duplicates, recall notices, quoted-message
prefixes, media placeholders by default, incomplete edge bubbles, leading
assistant turns, dangling final user turns, exact duplicate messages, and exact
duplicate examples. Privacy redaction is enabled by default for phone numbers,
emails, URLs, long ID-like numbers, and custom terms. The quality report is
written to `data/train_quality_report.json` or `data/CONTACT_ALIAS_quality_report.json`.

The quality report records counts for dropped rows, duplicate removal,
redaction, suspicious OCR samples, and final example totals. If many frames are
empty or suspicious OCR counts are high, slow down scrolling, increase `--fps`,
or inspect `--items-output`.

### 4. Judge, Filter, And Split

```bash
uv run python judge_roleplay_value.py --model YOUR_JUDGE_MODEL
```

Default outputs:

```text
data/train.roleplay_filtered.jsonl
data/valid.roleplay_filtered.jsonl
data/train.roleplay_judgments.jsonl
```

The filtered JSONL files are training-ready train/validation splits. The
validation split is deterministic by default with `--valid-ratio 0.1` and
`--split-seed 42`. The judgments JSONL is kept so you can audit why rows were
kept or dropped.

For OpenAI-compatible local servers:

```bash
uv run python judge_roleplay_value.py \
  --base-url http://127.0.0.1:1234/v1 \
  --model YOUR_LOCAL_MODEL \
  --max-concurrency 1 \
  --resume
```

Named artifacts default to:

```text
data/CONTACT_ALIAS_train.filtered.jsonl
data/CONTACT_ALIAS_valid.filtered.jsonl
data/CONTACT_ALIAS.roleplay_judgments.jsonl
```

### 5. Fine-Tune

Edit `lora_config.yaml` for your local base model and adapter path, then run:

```bash
mlx_lm.lora --config lora_config.yaml
```

Minimal explicit example:

```bash
mlx_lm.lora \
  --model "Qwen:Qwen3-14B-MLX-8bit" \
  --train \
  --fine-tune-type lora \
  --data ./data \
  --adapter-path ./adapters/contact \
  --batch-size 1 \
  --iters 300 \
  --learning-rate 1e-5 \
  --num-layers 4 \
  --max-seq-length 1024 \
  --grad-checkpoint \
  --mask-prompt
```

### 6. Chat In The Terminal

```bash
uv run python chat_with_adapter.py
```

Useful options:

```bash
uv run python chat_with_adapter.py --name CONTACT_ALIAS
uv run python chat_with_adapter.py --model "Qwen:Qwen3-14B-MLX-8bit" --adapter-path adapters/contact
uv run python chat_with_adapter.py --max-tokens 128 --temp 0.7
```

Inside chat:

```text
/reset    clear history
/history  print current context
/help     show commands
/exit     quit
```

Thinking is disabled by default for chat templates that support
`enable_thinking`. Pass `--enable-thinking` to opt in.

## Environment Variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| `FRAMES_DIR` | `process_frames.py` | Default frames directory |
| `OUTPUT_PATH` | `process_frames.py` | Default raw extraction JSONL |
| `MAX_CONCURRENCY` | `process_frames.py` | Concurrent VLM workers |
| `MAX_RETRIES` | `process_frames.py` | Retry count |
| `INCLUDE_RAW` | `process_frames.py` | Include raw model/OCR details |
| `LOCAL_LLM_BASE_URL` | `process_frames.py` | OpenAI-compatible VLM endpoint |
| `LOCAL_LLM_MODEL` | `process_frames.py` | VLM model name |
| `LOCAL_LLM_API_KEY` | `process_frames.py` | VLM API key |

## Privacy Notes

- Real names are not required in committed code or docs.
- Use `--name` only for local artifacts you intentionally want to identify.
- `*.jsonl`, media files, frame directories, adapters, and local configs are
  ignored by git.
- Before publishing, scan with:

```bash
rg -n --no-ignore "REAL_NAME|ABSOLUTE_PATH" . --glob '!.git/**' --glob '!.venv/**'
```
