import base64
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LOCAL_LLM_MODEL", "lmstudio-community:Qwen3.5-4B-MLX-8bit")

FRAMES_DIR = Path("frames")
OUTPUT_PATH = Path("raw_frame_messages.jsonl")

client = OpenAI(
    base_url=BASE_URL,
    api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
)

PROMPT = """
你是一个微信聊天截图转 SFT 数据的解析器。

任务：
从这张微信聊天截图中提取所有可见的聊天消息，并直接整理成接近训练数据的 messages 格式。

角色映射：
- 右侧气泡 = user
- 左侧气泡 = assistant
- 居中时间、系统提示、日期分割线 = 忽略，不要输出
- 聊天窗口顶部标题、联系人名、状态栏、输入框、底部按钮 = 忽略，不要输出

整理规则：
1. 只提取截图中完整可见的聊天消息。
2. 如果消息气泡被截图顶部或底部截断，直接忽略，不要输出。
3. 不要根据半截文字补全消息。
4. 按截图中从上到下的顺序输出。
5. 右侧气泡 = user，左侧气泡 = assistant。
6. 居中时间、系统提示、日期分割线、顶部标题、输入框全部忽略。
7. 连续同一个 role 的多条完整气泡必须合并成一个 message，用 \n 保留分条。
8. 图片、表情、语音、文件、转账等完整可见的非文本消息，用占位符。
9. 只输出严格 JSON。

输出格式必须是：
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""


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
    text = text.strip()

    # 去掉模型可能输出的代码块
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 容错：截取第一个 JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"No valid JSON found: {text[:300]}")


def already_done() -> set[str]:
    done = set()
    if not OUTPUT_PATH.exists():
        return done

    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["frame"])
            except Exception:
                continue

    return done


def parse_one_frame(frame_path: Path) -> dict:
    image_url = image_to_data_url(frame_path)

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )

    raw = resp.choices[0].message.content
    parsed = extract_json(raw)

    return {
        "frame": frame_path.name,
        "ok": True,
        "messages": parsed.get("messages", []),
        "raw": raw,
    }


def main():
    frames = sorted(FRAMES_DIR.glob("frame*.png")) + sorted(
        FRAMES_DIR.glob("frame*.jpg")
    )
    done = already_done()

    print(f"BASE_URL = {BASE_URL}")
    print(f"MODEL = {MODEL}")
    print(f"Found {len(frames)} frames.")
    print(f"Already done {len(done)} frames.")

    with OUTPUT_PATH.open("a", encoding="utf-8") as out:
        for i, frame in enumerate(frames, start=1):
            if frame.name in done:
                continue

            print(f"[{i}/{len(frames)}] {frame.name}")

            try:
                item = parse_one_frame(frame)
            except Exception as e:
                item = {
                    "frame": frame.name,
                    "ok": False,
                    "error": repr(e),
                }

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()

            time.sleep(0.2)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
