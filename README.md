# My Friend

Utilities for parsing WeChat chat screenshots with a local OpenAI-compatible
vision model endpoint.

## Setup

Create a project virtual environment with `uv`, then install dependencies:

```bash
uv venv
uv pip install openai
```

## Usage

Put extracted screenshot frames in `frames/`, then run:

```bash
python process_frames_async.py
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

Local videos, frame images, and JSONL extraction outputs are intentionally
ignored and should not be committed.
