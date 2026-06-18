import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LOCAL_LLM_MODEL", "lmstudio-community:Qwen3.5-4B-MLX-8bit")

FRAMES_DIR = Path(os.getenv("FRAMES_DIR", "huxinyi_frames"))

# 注意：换新文件，别和旧的 messages schema 混在一起
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "huxinyi_raw_frame_items.jsonl"))

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# debug 阶段保留 raw；稳定后可 export INCLUDE_RAW=0
INCLUDE_RAW = os.getenv("INCLUDE_RAW", "1") != "0"

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
    timeout=180,
)


PROMPT = """
你是一个微信聊天截图解析器。

任务：
从截图中提取所有完整或部分可见的微信聊天气泡，输出气泡级 JSON。
不要整理成最终训练 messages，不要合并多个气泡。

角色映射：
- 右侧气泡 = user
- 左侧气泡 = assistant
- 居中时间、系统提示、日期分割线 = system
- 顶部标题、联系人名、状态栏、输入框、底部按钮 = 忽略，不要输出

重要规则：
1. 每一个聊天气泡输出一个 item。
2. 不要合并多个气泡，即使它们来自同一个人。
3. 不要改写、不要润色、不要补全。
4. 按截图中从上到下的顺序输出。
5. 只输出真实可见内容。
6. 如果气泡在截图顶部或底部被截断，complete=false。
7. 如果不确定某条气泡是否完整，complete=false。
8. 对 complete=false 的气泡，只能写可见部分，不能根据上下文补全。
9. 图片、表情、语音、文件、转账等非纯文本消息也必须作为单独 item 输出。
10. 非文本消息的 content 使用：[图片]、[表情]、[语音]、[文件]、[转账]。
11. 如果是完整可见气泡，edge="none"。
12. 如果被顶部截断，edge="top"。
13. 如果被底部截断，edge="bottom"。
14. 如果上下都被截断，edge="both"。
15. vertical_position 表示该气泡大致在截图中的位置：top / middle / bottom。
16. 只输出严格 JSON。
17. 第一个字符必须是 {，最后一个字符必须是 }。
18. 不要 Markdown，不要解释，不要代码块。

输出格式必须是：
{
  "items": [
    {
      "order": 1,
      "side": "left|right|center",
      "role": "assistant|user|system",
      "type": "text|image|sticker|voice|file|transfer|system",
      "content": "...",
      "complete": true,
      "edge": "none|top|bottom|both",
      "vertical_position": "top|middle|bottom"
    }
  ]
}
"""


def frame_index_from_name(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        return -1
    return int(match.group(1))


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image type: {path}")

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def extract_json(text: str) -> dict:
    if text is None:
        raise ValueError("Model returned None content")

    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"No valid JSON found: {text[:300]}")


def get_raw_content(resp: Any) -> str:
    message = resp.choices[0].message
    raw = message.content

    if raw is None:
        message_dict = message.model_dump()
        raw = (
            message_dict.get("content")
            or message_dict.get("reasoning_content")
            or message_dict.get("reasoning")
            or message_dict.get("text")
        )

    if isinstance(raw, list):
        parts = []
        for part in raw:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        raw = "\n".join(parts)

    if raw is None:
        raise ValueError(f"Model returned no content: {resp.model_dump()}")

    return raw


def already_done() -> set[str]:
    done = set()
    if not OUTPUT_PATH.exists():
        return done

    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("ok") is True:
                    done.add(item["frame"])
            except Exception:
                continue

    return done


def get_frames() -> list[Path]:
    frames = (
        list(FRAMES_DIR.glob("frame*.png"))
        + list(FRAMES_DIR.glob("frame*.jpg"))
        + list(FRAMES_DIR.glob("frame*.jpeg"))
    )
    return sorted(frames, key=lambda p: frame_index_from_name(p))


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return default


def normalize_item(raw_item: dict, fallback_order: int) -> dict | None:
    side = raw_item.get("side")
    role = raw_item.get("role")
    item_type = raw_item.get("type")
    content = raw_item.get("content")
    edge = raw_item.get("edge", "none")
    vertical_position = raw_item.get("vertical_position", "middle")

    if side not in {"left", "right", "center"}:
        return None

    # 用 side 强制修正 role，避免模型 role 写错
    if side == "right":
        role = "user"
    elif side == "left":
        role = "assistant"
    elif side == "center":
        role = "system"

    if role not in {"user", "assistant", "system"}:
        return None

    if item_type not in {
        "text",
        "image",
        "sticker",
        "voice",
        "file",
        "transfer",
        "system",
    }:
        # 兜底：如果模型乱写 type，但有文字，就当 text
        item_type = "text"

    if not isinstance(content, str):
        return None

    content = content.strip()
    if not content:
        return None

    if edge not in {"none", "top", "bottom", "both"}:
        edge = "none"

    if vertical_position not in {"top", "middle", "bottom"}:
        vertical_position = "middle"

    complete = normalize_bool(raw_item.get("complete"), default=False)

    # 如果 edge 不是 none，就强制 complete=false
    if edge != "none":
        complete = False

    order = raw_item.get("order", fallback_order)
    try:
        order = int(order)
    except Exception:
        order = fallback_order

    return {
        "order": order,
        "side": side,
        "role": role,
        "type": item_type,
        "content": content,
        "complete": complete,
        "edge": edge,
        "vertical_position": vertical_position,
    }


def normalize_items(parsed: dict) -> list[dict]:
    raw_items = parsed.get("items", [])

    if not isinstance(raw_items, list):
        return []

    cleaned = []
    for i, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            continue

        item = normalize_item(raw_item, fallback_order=i)
        if item is not None:
            cleaned.append(item)

    cleaned.sort(key=lambda x: x["order"])

    # 重新编号，确保 order 连续
    for i, item in enumerate(cleaned, start=1):
        item["order"] = i

    return cleaned


async def parse_one_frame(frame_path: Path) -> dict:
    started_at = time.perf_counter()
    frame_index = frame_index_from_name(frame_path)

    try:
        image_url = image_to_data_url(frame_path)
    except Exception as e:
        return {
            "frame": frame_path.name,
            "frame_index": frame_index,
            "ok": False,
            "error": repr(e),
            "_elapsed_seconds": time.perf_counter() - started_at,
            "_attempts": 0,
        }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=2048,
                messages=messages,
            )

            raw = get_raw_content(resp)
            parsed = extract_json(raw)
            items = normalize_items(parsed)

            result = {
                "frame": frame_path.name,
                "frame_index": frame_index,
                "ok": True,
                "items": items,
            }

            if INCLUDE_RAW:
                result["raw"] = raw

            result["_elapsed_seconds"] = time.perf_counter() - started_at
            result["_attempts"] = attempt
            return result

        except Exception as e:
            if attempt <= MAX_RETRIES:
                await asyncio.sleep(0.8 * attempt)
                continue

            return {
                "frame": frame_path.name,
                "frame_index": frame_index,
                "ok": False,
                "error": repr(e),
                "_elapsed_seconds": time.perf_counter() - started_at,
                "_attempts": attempt,
            }


async def main():
    frames = get_frames()
    done = already_done()

    todo = [f for f in frames if f.name not in done]

    print(f"BASE_URL = {BASE_URL}")
    print(f"MODEL = {MODEL}")
    print(f"FRAMES_DIR = {FRAMES_DIR}")
    print(f"OUTPUT_PATH = {OUTPUT_PATH}")
    print(f"MAX_CONCURRENCY = {MAX_CONCURRENCY}")
    print(f"INCLUDE_RAW = {INCLUDE_RAW}")
    print(f"Found {len(frames)} frames.")
    print(f"Already done {len(done)} frames.")
    print(f"Todo {len(todo)} frames.")

    if not todo:
        print("Nothing to do.")
        return

    write_lock = asyncio.Lock()
    frame_queue: asyncio.Queue[Path | None] = asyncio.Queue()

    completed = 0
    total = len(todo)
    run_started_at = time.perf_counter()

    for frame in todo:
        frame_queue.put_nowait(frame)
    for _ in range(MAX_CONCURRENCY):
        frame_queue.put_nowait(None)

    async def handle_result(frame: Path, item: dict, out: Any):
        nonlocal completed

        elapsed_seconds = item.pop("_elapsed_seconds", None)
        attempts = item.pop("_attempts", None)

        async with write_lock:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()

            completed += 1
            status = "OK" if item.get("ok") else "ERR"
            n_items = len(item.get("items", []))
            total_elapsed = time.perf_counter() - run_started_at
            avg_seconds = total_elapsed / completed
            frames_per_minute = completed / total_elapsed * 60
            remaining = total - completed
            eta_seconds = remaining * avg_seconds

            timing = ""
            if elapsed_seconds is not None:
                timing = f" time={elapsed_seconds:.1f}s"
            if attempts is not None and attempts > 1:
                timing += f" attempts={attempts}"

            print(
                f"[{completed}/{total}] {status} {frame.name} items={n_items}"
                f"{timing} avg={avg_seconds:.1f}s/frame"
                f" rate={frames_per_minute:.1f}/min eta={eta_seconds / 60:.1f}m"
            )

    async def worker(out: Any):
        while True:
            frame = await frame_queue.get()
            try:
                if frame is None:
                    return

                item = await parse_one_frame(frame)
                await handle_result(frame, item, out)
            finally:
                frame_queue.task_done()

    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        tasks = [asyncio.create_task(worker(out)) for _ in range(MAX_CONCURRENCY)]
        await frame_queue.join()
        await asyncio.gather(*tasks)

    total_elapsed = time.perf_counter() - run_started_at
    print(
        f"Done in {total_elapsed / 60:.1f}m "
        f"({total_elapsed / total:.1f}s/frame, {total / total_elapsed * 60:.1f}/min)."
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
