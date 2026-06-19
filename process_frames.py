import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_FRAMES_DIR = Path(os.getenv("FRAMES_DIR", "contact_frames"))
DEFAULT_OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "contact_raw_frame_items.jsonl"))
DEFAULT_MAX_WORKERS = int(os.getenv("MAX_CONCURRENCY", "2"))
DEFAULT_MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
DEFAULT_INCLUDE_RAW = os.getenv("INCLUDE_RAW", "0") != "0"
DEFAULT_METHOD = os.getenv("EXTRACTION_METHOD", "ocr")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".heic", ".tiff", ".tif"}

VLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
VLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "lmstudio-community:Qwen3.5-4B-MLX-4bit")
VLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "not-needed")


def artifact_prefix(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")
    if not prefix:
        raise ValueError("--name must contain at least one filename-safe character.")
    return prefix


def option_was_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])

VLM_PROMPT = """
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
9. 文本气泡里的 Unicode emoji 必须保留原始 emoji 字符，例如 😂、🥱、😭、😷、💀、🦷。不要把这些 emoji 写成 [表情]。
10. 只有独立的微信贴纸/表情包/大表情消息，才输出 type="sticker" 且 content="[表情]"。
11. 图片、独立表情包、语音、文件、转账等非纯文本消息也必须作为单独 item 输出。
12. 非文本消息的 content 使用：[图片]、[表情]、[语音]、[文件]、[转账]。
13. 如果是完整可见气泡，edge="none"。
14. 如果被顶部截断，edge="top"。
15. 如果被底部截断，edge="bottom"。
16. 如果上下都被截断，edge="both"。
17. vertical_position 表示该气泡大致在截图中的位置：top / middle / bottom。
18. 只输出严格 JSON。
19. 第一个字符必须是 {，最后一个字符必须是 }。
20. 不要 Markdown，不要解释，不要代码块。

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


def get_frames(frames_dir: Path) -> list[Path]:
    frames = [
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(frames, key=frame_index_from_name)


def already_done(output_path: Path) -> set[str]:
    done = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("ok") is True:
                done.add(item["frame"])
    return done


def has_failed_frames(output_path: Path) -> bool:
    if not output_path.exists():
        return False

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("ok") is False:
                return True
    return False


def extraction_quality_summary(frames_dir: Path, output_path: Path) -> dict[str, Any]:
    frames = get_frames(frames_dir) if frames_dir.exists() else []
    successful_frames = 0
    failed_frames = 0
    empty_frames = 0
    total_items = 0
    output_rows = 0

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                output_rows += 1
                if row.get("ok") is True:
                    successful_frames += 1
                    items = row.get("items")
                    item_count = len(items) if isinstance(items, list) else 0
                    total_items += item_count
                    if item_count == 0:
                        empty_frames += 1
                elif row.get("ok") is False:
                    failed_frames += 1

    average_items = total_items / successful_frames if successful_frames else 0.0
    warnings = []
    if successful_frames and empty_frames / successful_frames >= 0.6:
        warnings.append("most_successful_frames_are_empty")
    if frames and output_rows and abs(output_rows - len(frames)) > max(2, len(frames) * 0.1):
        warnings.append("output_row_count_mismatches_frame_count")

    return {
        "total_frames": len(frames),
        "successful_frames": successful_frames,
        "failed_frames": failed_frames,
        "empty_frames": empty_frames,
        "average_items_per_frame": round(average_items, 2),
        "output_rows": output_rows,
        "warnings": warnings,
    }


def print_extraction_quality_summary(frames_dir: Path, output_path: Path) -> None:
    summary = extraction_quality_summary(frames_dir, output_path)
    print("Extraction quality summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def vertical_position(y_center_from_top: float) -> str:
    if y_center_from_top < 0.34:
        return "top"
    if y_center_from_top > 0.66:
        return "bottom"
    return "middle"


def side_for_line(x_center: float, center_band: tuple[float, float]) -> str:
    if center_band[0] <= x_center <= center_band[1]:
        return "center"
    if x_center < 0.5:
        return "left"
    return "right"


def should_ignore_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in {"微信", "WeChat"}:
        return True
    return False


def load_apple_vision() -> tuple[Any, Any, Any]:
    try:
        import Quartz
        import Vision
        from Foundation import NSURL
    except ImportError as exc:
        raise RuntimeError(
            "Apple Vision OCR requires PyObjC. Install it in the project venv with: "
            "uv pip install pyobjc-framework-Vision pyobjc-framework-Quartz pillow"
        ) from exc
    return Quartz, Vision, NSURL


def load_pillow_image(path: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Bubble grouping requires Pillow. Install it in the project venv with: "
            "uv pip install pillow"
        ) from exc
    return Image.open(path).convert("RGB")


def get_image_size(path: Path, quartz: Any, nsurl: Any) -> tuple[float, float]:
    url = nsurl.fileURLWithPath_(str(path.resolve()))
    image_source = quartz.CGImageSourceCreateWithURL(url, None)
    if image_source is None:
        raise ValueError(f"Could not open image: {path}")

    image = quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")

    width = float(quartz.CGImageGetWidth(image))
    height = float(quartz.CGImageGetHeight(image))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for {path}: {width}x{height}")
    return width, height


def recognize_text(path: Path, recognition_level: str) -> list[dict[str, Any]]:
    quartz, vision, nsurl = load_apple_vision()
    image_width, image_height = get_image_size(path, quartz, nsurl)

    request = vision.VNRecognizeTextRequest.alloc().init()
    if recognition_level == "fast":
        request.setRecognitionLevel_(vision.VNRequestTextRecognitionLevelFast)
    else:
        request.setRecognitionLevel_(vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    request.setRecognitionLanguages_(["zh-Hans", "zh-Hant", "en-US"])

    url = nsurl.fileURLWithPath_(str(path.resolve()))
    handler = vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    ok = handler.performRequests_error_([request], None)
    if isinstance(ok, tuple):
        ok = ok[0]
    if not ok:
        raise RuntimeError(f"Vision OCR request failed for {path}")

    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue

        text = str(candidates[0].string()).strip()
        if not text:
            continue

        bbox = observation.boundingBox()
        x = float(bbox.origin.x)
        y = float(bbox.origin.y)
        width = float(bbox.size.width)
        height = float(bbox.size.height)
        lines.append(
            {
                "text": text,
                "confidence": float(candidates[0].confidence()),
                "bbox": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "pixel_x": x * image_width,
                    "pixel_y": y * image_height,
                    "pixel_width": width * image_width,
                    "pixel_height": height * image_height,
                },
            }
        )

    return sorted(
        lines,
        key=lambda line: (
            -(line["bbox"]["y"] + line["bbox"]["height"]),
            line["bbox"]["x"],
        ),
    )


def is_cjk_char(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff"


def join_wrapped_text(chunks: list[str]) -> str:
    content = ""
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if not chunk:
            continue
        if not content:
            content = chunk
            continue
        if is_cjk_char(content[-1]) or is_cjk_char(chunk[0]):
            content += chunk
        else:
            content += " " + chunk
    return content


def is_right_bubble_pixel(r: int, g: int, b: int) -> bool:
    return (
        40 <= r <= 120
        and 135 <= g <= 220
        and 75 <= b <= 160
        and g - r >= 55
        and g - b >= 35
    )


def is_left_bubble_pixel(r: int, g: int, b: int) -> bool:
    return (
        45 <= r <= 105
        and 45 <= g <= 105
        and 45 <= b <= 105
        and abs(r - g) <= 18
        and abs(g - b) <= 18
    )


def connected_component_boxes(
    mask: bytearray, width: int, height: int, min_pixels: int
) -> list[dict[str, float]]:
    seen = bytearray(width * height)
    boxes = []

    for y in range(height):
        row_offset = y * width
        for x in range(width):
            offset = row_offset + x
            if seen[offset] or not mask[offset]:
                continue

            queue = deque([(x, y)])
            seen[offset] = 1
            min_x = max_x = x
            min_y = max_y = y
            count = 0

            while queue:
                px, py = queue.popleft()
                count += 1
                min_x = min(min_x, px)
                max_x = max(max_x, px)
                min_y = min(min_y, py)
                max_y = max(max_y, py)

                for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    neighbor_offset = ny * width + nx
                    if seen[neighbor_offset] or not mask[neighbor_offset]:
                        continue
                    seen[neighbor_offset] = 1
                    queue.append((nx, ny))

            box_width = max_x - min_x + 1
            box_height = max_y - min_y + 1
            if count < min_pixels or box_width < 18 or box_height < 14:
                continue

            boxes.append(
                {
                    "x": float(min_x),
                    "y": float(min_y),
                    "width": float(box_width),
                    "height": float(box_height),
                    "pixels": float(count),
                }
            )
    return boxes


def merge_nearby_boxes(boxes: list[dict[str, float]]) -> list[dict[str, float]]:
    merged = []
    for box in sorted(boxes, key=lambda item: (item["y"], item["x"])):
        matched = None
        for candidate in merged:
            x_gap = max(candidate["x"], box["x"]) - min(
                candidate["x"] + candidate["width"], box["x"] + box["width"]
            )
            y_gap = max(candidate["y"], box["y"]) - min(
                candidate["y"] + candidate["height"], box["y"] + box["height"]
            )
            if x_gap <= 8 and y_gap <= 8:
                matched = candidate
                break

        if matched is None:
            merged.append(dict(box))
            continue

        min_x = min(matched["x"], box["x"])
        min_y = min(matched["y"], box["y"])
        max_x = max(matched["x"] + matched["width"], box["x"] + box["width"])
        max_y = max(matched["y"] + matched["height"], box["y"] + box["height"])
        matched.update(
            {
                "x": min_x,
                "y": min_y,
                "width": max_x - min_x,
                "height": max_y - min_y,
                "pixels": matched.get("pixels", 0) + box.get("pixels", 0),
            }
        )
    return merged


def detect_bubble_regions(
    path: Path, content_top: float, content_bottom: float
) -> list[dict[str, Any]]:
    image = load_pillow_image(path)
    width, height = image.size
    pixels = image.load()
    top_px = int(height * content_top)
    bottom_px = int(height * content_bottom)
    min_pixels = max(40, int(width * height * 0.00008))
    masks = {"right": bytearray(width * height), "left": bytearray(width * height)}

    for y in range(top_px, bottom_px):
        row_offset = y * width
        for x in range(width):
            r, g, b = pixels[x, y]
            x_norm = x / width
            if x_norm >= 0.42 and is_right_bubble_pixel(r, g, b):
                masks["right"][row_offset + x] = 1
            elif 0.12 <= x_norm <= 0.72 and is_left_bubble_pixel(r, g, b):
                masks["left"][row_offset + x] = 1

    regions = []
    for side, mask in masks.items():
        boxes = merge_nearby_boxes(connected_component_boxes(mask, width, height, min_pixels))
        for box in boxes:
            x_center = (box["x"] + box["width"] / 2) / width
            if side == "right" and x_center < 0.5:
                continue
            if side == "left" and x_center > 0.68:
                continue
            y_center = (box["y"] + box["height"] / 2) / height
            if y_center < content_top or y_center > content_bottom:
                continue
            regions.append(
                {
                    "side": side,
                    "role": "user" if side == "right" else "assistant",
                    "type": "text",
                    "bbox": {
                        "x": box["x"] / width,
                        "y": box["y"] / height,
                        "width": box["width"] / width,
                        "height": box["height"] / height,
                    },
                }
            )

    return sorted(regions, key=lambda region: (region["bbox"]["y"], region["bbox"]["x"]))


def line_top_origin_box(line: dict[str, Any]) -> dict[str, float]:
    bbox = line["bbox"]
    return {
        "x": bbox["x"],
        "y": 1 - (bbox["y"] + bbox["height"]),
        "width": bbox["width"],
        "height": bbox["height"],
    }


def line_center_top_origin(line: dict[str, Any]) -> tuple[float, float]:
    box = line_top_origin_box(line)
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def point_in_region(
    point: tuple[float, float],
    region: dict[str, Any],
    padding_x: float = 0.02,
    padding_y: float = 0.01,
) -> bool:
    x, y = point
    bbox = region["bbox"]
    return (
        bbox["x"] - padding_x <= x <= bbox["x"] + bbox["width"] + padding_x
        and bbox["y"] - padding_y <= y <= bbox["y"] + bbox["height"] + padding_y
    )


def edge_for_box(y_top: float, y_bottom: float, edge_margin: float) -> str:
    top_cut = y_top <= edge_margin
    bottom_cut = y_bottom >= 1 - edge_margin
    if top_cut and bottom_cut:
        return "both"
    if top_cut:
        return "top"
    if bottom_cut:
        return "bottom"
    return "none"


def make_item(
    order: int,
    side: str,
    role: str,
    item_type: str,
    content: str,
    y_top: float,
    y_bottom: float,
    edge_margin: float,
) -> dict[str, Any]:
    y_center_from_top = (y_top + y_bottom) / 2
    edge = edge_for_box(y_top, y_bottom, edge_margin)
    return {
        "order": order,
        "side": side,
        "role": role,
        "type": item_type,
        "content": content,
        "complete": edge == "none",
        "edge": edge,
        "vertical_position": vertical_position(y_center_from_top),
    }


def lines_to_items(
    lines: list[dict[str, Any]],
    center_band: tuple[float, float],
    edge_margin: float,
    min_confidence: float,
    content_top: float,
    content_bottom: float,
) -> list[dict[str, Any]]:
    items = []
    for raw_line in lines:
        text = raw_line["text"].strip()
        confidence = raw_line["confidence"]
        if confidence < min_confidence or should_ignore_text(text):
            continue

        bbox = raw_line["bbox"]
        x_center = bbox["x"] + bbox["width"] / 2
        y_top = 1 - (bbox["y"] + bbox["height"])
        y_bottom = 1 - bbox["y"]
        y_center_from_top = (y_top + y_bottom) / 2
        if y_center_from_top < content_top or y_center_from_top > content_bottom:
            continue

        side = side_for_line(x_center, center_band)
        if side == "right":
            role, item_type = "user", "text"
        elif side == "left":
            role, item_type = "assistant", "text"
        else:
            role, item_type = "system", "system"

        items.append(
            make_item(
                len(items) + 1,
                side,
                role,
                item_type,
                text,
                y_top,
                y_bottom,
                edge_margin,
            )
        )
    return items


def lines_to_bubble_items(
    lines: list[dict[str, Any]],
    bubble_regions: list[dict[str, Any]],
    center_band: tuple[float, float],
    edge_margin: float,
    min_confidence: float,
    content_top: float,
    content_bottom: float,
) -> list[dict[str, Any]]:
    eligible_lines = []
    for index, raw_line in enumerate(lines):
        text = raw_line["text"].strip()
        confidence = raw_line["confidence"]
        if confidence < min_confidence or should_ignore_text(text):
            continue

        x_center, y_center = line_center_top_origin(raw_line)
        if y_center < content_top or y_center > content_bottom:
            continue
        eligible_lines.append((index, raw_line, x_center, y_center))

    assigned_line_indexes: set[int] = set()
    grouped = []
    for region in bubble_regions:
        region_lines = []
        for index, raw_line, x_center, y_center in eligible_lines:
            if index in assigned_line_indexes:
                continue
            if point_in_region((x_center, y_center), region):
                region_lines.append((index, raw_line))
        if not region_lines:
            continue

        for index, _raw_line in region_lines:
            assigned_line_indexes.add(index)
        region_lines.sort(
            key=lambda entry: (
                line_top_origin_box(entry[1])["y"],
                line_top_origin_box(entry[1])["x"],
            )
        )

        content = join_wrapped_text([line["text"] for _index, line in region_lines])
        if not content:
            continue

        bbox = region["bbox"]
        grouped.append(
            {
                "sort_y": bbox["y"],
                "sort_x": bbox["x"],
                "item": make_item(
                    0,
                    region["side"],
                    region["role"],
                    region["type"],
                    content,
                    bbox["y"],
                    bbox["y"] + bbox["height"],
                    edge_margin,
                ),
            }
        )

    for index, raw_line, _x_center, _y_center in eligible_lines:
        if index in assigned_line_indexes:
            continue
        line_item = lines_to_items(
            [raw_line],
            center_band,
            edge_margin,
            min_confidence,
            content_top,
            content_bottom,
        )
        if not line_item:
            continue
        box = line_top_origin_box(raw_line)
        grouped.append({"sort_y": box["y"], "sort_x": box["x"], "item": line_item[0]})

    grouped.sort(key=lambda item: (item["sort_y"], item["sort_x"]))
    items = [entry["item"] for entry in grouped]
    for order, item in enumerate(items, start=1):
        item["order"] = order
    return items


def parse_ocr_frame(frame_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    frame_index = frame_index_from_name(frame_path)
    center_band = (args.center_left, args.center_right)

    try:
        lines = recognize_text(frame_path, args.recognition_level)
        bubble_regions = []
        if args.disable_bubble_grouping:
            items = lines_to_items(
                lines,
                center_band,
                args.edge_margin,
                args.min_confidence,
                args.content_top,
                args.content_bottom,
            )
        else:
            bubble_regions = detect_bubble_regions(
                frame_path, args.content_top, args.content_bottom
            )
            items = lines_to_bubble_items(
                lines,
                bubble_regions,
                center_band,
                args.edge_margin,
                args.min_confidence,
                args.content_top,
                args.content_bottom,
            )

        result: dict[str, Any] = {
            "frame": frame_path.name,
            "frame_index": frame_index,
            "ok": True,
            "items": items,
        }
        if args.include_raw:
            result["raw"] = lines
            result["_bubble_regions"] = bubble_regions
        result["_elapsed_seconds"] = time.perf_counter() - started_at
        return result
    except Exception as exc:
        return {
            "frame": frame_path.name,
            "frame_index": frame_index,
            "ok": False,
            "error": repr(exc),
            "_elapsed_seconds": time.perf_counter() - started_at,
        }


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "1"}:
            return True
        if value in {"false", "no", "0"}:
            return False
    return default


def normalize_vlm_item(raw_item: dict[str, Any], fallback_order: int) -> dict[str, Any] | None:
    side = raw_item.get("side")
    role = raw_item.get("role")
    item_type = raw_item.get("type")
    content = raw_item.get("content")
    edge = raw_item.get("edge", "none")
    position = raw_item.get("vertical_position", "middle")

    if side not in {"left", "right", "center"}:
        return None
    if side == "right":
        role = "user"
    elif side == "left":
        role = "assistant"
    else:
        role = "system"

    if item_type not in {"text", "image", "sticker", "voice", "file", "transfer", "system"}:
        item_type = "text"
    if not isinstance(content, str):
        return None

    content = content.strip()
    if not content:
        return None
    if edge not in {"none", "top", "bottom", "both"}:
        edge = "none"
    if position not in {"top", "middle", "bottom"}:
        position = "middle"

    complete = normalize_bool(raw_item.get("complete"), default=False)
    if edge != "none":
        complete = False

    try:
        order = int(raw_item.get("order", fallback_order))
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
        "vertical_position": position,
    }


def normalize_vlm_items(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = parsed.get("items", [])
    if not isinstance(raw_items, list):
        return []

    cleaned = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            continue
        item = normalize_vlm_item(raw_item, fallback_order=index)
        if item is not None:
            cleaned.append(item)

    cleaned.sort(key=lambda item: item["order"])
    for order, item in enumerate(cleaned, start=1):
        item["order"] = order
    return cleaned


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image type for VLM extraction: {path}")

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def extract_json(text: str) -> dict[str, Any]:
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


async def parse_vlm_frame(frame_path: Path, args: argparse.Namespace, client: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    frame_index = frame_index_from_name(frame_path)
    try:
        image_url = image_to_data_url(frame_path)
    except Exception as exc:
        return {
            "frame": frame_path.name,
            "frame_index": frame_index,
            "ok": False,
            "error": repr(exc),
            "_elapsed_seconds": time.perf_counter() - started_at,
            "_attempts": 0,
        }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    for attempt in range(1, args.max_retries + 2):
        try:
            resp = await client.chat.completions.create(
                model=args.vlm_model,
                temperature=0,
                max_tokens=1024,
                messages=messages,
            )
            raw = get_raw_content(resp)
            parsed = extract_json(raw)
            result: dict[str, Any] = {
                "frame": frame_path.name,
                "frame_index": frame_index,
                "ok": True,
                "items": normalize_vlm_items(parsed),
            }
            if args.include_raw:
                result["raw"] = raw
            result["_elapsed_seconds"] = time.perf_counter() - started_at
            result["_attempts"] = attempt
            return result
        except Exception as exc:
            if attempt <= args.max_retries:
                await asyncio.sleep(0.8 * attempt)
                continue
            return {
                "frame": frame_path.name,
                "frame_index": frame_index,
                "ok": False,
                "error": repr(exc),
                "_elapsed_seconds": time.perf_counter() - started_at,
                "_attempts": attempt,
            }


def print_progress(
    completed: int,
    total: int,
    frame: Path,
    item: dict[str, Any],
    elapsed_seconds: float | None,
    attempts: int | None,
    run_started_at: float,
) -> None:
    status = "OK" if item.get("ok") else "ERR"
    n_items = len(item.get("items", []))
    total_elapsed = time.perf_counter() - run_started_at
    avg_seconds = total_elapsed / completed
    frames_per_minute = completed / total_elapsed * 60
    remaining = total - completed
    eta_seconds = remaining * avg_seconds
    timing = f" time={elapsed_seconds:.1f}s" if elapsed_seconds is not None else ""
    if attempts is not None and attempts > 1:
        timing += f" attempts={attempts}"
    print(
        f"[{completed}/{total}] {status} {frame.name} items={n_items}"
        f"{timing} avg={avg_seconds:.1f}s/frame"
        f" rate={frames_per_minute:.1f}/min eta={eta_seconds / 60:.1f}m"
    )


def run_ocr(args: argparse.Namespace) -> None:
    frames = get_frames(args.frames_dir)
    done = already_done(args.output_path)
    todo = [frame for frame in frames if frame.name not in done]

    print("EXTRACTION_METHOD = ocr")
    print("OCR_ENGINE = Apple Vision")
    print(f"FRAMES_DIR = {args.frames_dir}")
    print(f"OUTPUT_PATH = {args.output_path}")
    print(f"MAX_WORKERS = {args.max_workers}")
    print(f"RECOGNITION_LEVEL = {args.recognition_level}")
    print(f"BUBBLE_GROUPING = {not args.disable_bubble_grouping}")
    print(f"Found {len(frames)} frames.")
    print(f"Already done {len(done)} frames.")
    print(f"Todo {len(todo)} frames.")
    if not todo:
        print("Nothing to do.")
        print_extraction_quality_summary(args.frames_dir, args.output_path)
        return

    completed = 0
    total = len(todo)
    run_started_at = time.perf_counter()
    with args.output_path.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(parse_ocr_frame, frame, args): frame for frame in todo}
            for future in as_completed(futures):
                frame = futures[future]
                item = future.result()
                elapsed_seconds = item.pop("_elapsed_seconds", None)
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
                print_progress(
                    completed,
                    total,
                    frame,
                    item,
                    elapsed_seconds,
                    None,
                    run_started_at,
                )

    total_elapsed = time.perf_counter() - run_started_at
    print(
        f"Done in {total_elapsed / 60:.1f}m "
        f"({total_elapsed / total:.1f}s/frame, {total / total_elapsed * 60:.1f}/min)."
    )
    print(f"Saved to {args.output_path}")
    print_extraction_quality_summary(args.frames_dir, args.output_path)


async def run_vlm(args: argparse.Namespace) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "VLM extraction requires the OpenAI Python package. Install it with: "
            "uv pip install openai"
        ) from exc

    client = AsyncOpenAI(
        base_url=args.vlm_base_url,
        api_key=args.vlm_api_key,
        timeout=180,
    )
    frames = get_frames(args.frames_dir)
    done = already_done(args.output_path)
    todo = [frame for frame in frames if frame.name not in done]

    print("EXTRACTION_METHOD = vlm")
    print(f"BASE_URL = {args.vlm_base_url}")
    print(f"MODEL = {args.vlm_model}")
    print(f"FRAMES_DIR = {args.frames_dir}")
    print(f"OUTPUT_PATH = {args.output_path}")
    print(f"MAX_CONCURRENCY = {args.max_workers}")
    print(f"INCLUDE_RAW = {args.include_raw}")
    print(f"Found {len(frames)} frames.")
    print(f"Already done {len(done)} frames.")
    print(f"Todo {len(todo)} frames.")
    if not todo:
        print("Nothing to do.")
        print_extraction_quality_summary(args.frames_dir, args.output_path)
        return

    write_lock = asyncio.Lock()
    frame_queue: asyncio.Queue[Path | None] = asyncio.Queue()
    completed = 0
    total = len(todo)
    run_started_at = time.perf_counter()

    for frame in todo:
        frame_queue.put_nowait(frame)
    for _ in range(args.max_workers):
        frame_queue.put_nowait(None)

    async def handle_result(frame: Path, item: dict[str, Any], out: Any) -> None:
        nonlocal completed
        elapsed_seconds = item.pop("_elapsed_seconds", None)
        attempts = item.pop("_attempts", None)
        async with write_lock:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            completed += 1
            print_progress(
                completed,
                total,
                frame,
                item,
                elapsed_seconds,
                attempts,
                run_started_at,
            )

    async def worker(out: Any) -> None:
        while True:
            frame = await frame_queue.get()
            try:
                if frame is None:
                    return
                item = await parse_vlm_frame(frame, args, client)
                await handle_result(frame, item, out)
            finally:
                frame_queue.task_done()

    with args.output_path.open("a", encoding="utf-8") as out:
        tasks = [asyncio.create_task(worker(out)) for _ in range(args.max_workers)]
        await frame_queue.join()
        await asyncio.gather(*tasks)

    total_elapsed = time.perf_counter() - run_started_at
    print(
        f"Done in {total_elapsed / 60:.1f}m "
        f"({total_elapsed / total:.1f}s/frame, {total / total_elapsed * 60:.1f}/min)."
    )
    print(f"Saved to {args.output_path}")
    print_extraction_quality_summary(args.frames_dir, args.output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract WeChat frame items. OCR is the default primary extractor; "
            "VLM can be selected explicitly or used as fallback."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=["ocr", "vlm", "ocr-with-vlm-fallback"],
        default=DEFAULT_METHOD,
        help="Extraction backend. Default: ocr.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional local artifact prefix, e.g. creates NAME_frames and NAME_raw_frame_items.jsonl.",
    )
    parser.add_argument("--frames-dir", "--frames", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output-path", "--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--include-raw", action="store_true", default=DEFAULT_INCLUDE_RAW)
    parser.add_argument("--recognition-level", choices=["accurate", "fast"], default="accurate")
    parser.add_argument("--center-left", type=float, default=0.36)
    parser.add_argument("--center-right", type=float, default=0.64)
    parser.add_argument("--edge-margin", type=float, default=0.025)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--content-top", type=float, default=0.11)
    parser.add_argument("--content-bottom", type=float, default=0.95)
    parser.add_argument(
        "--disable-bubble-grouping",
        action="store_true",
        help="OCR only: keep line-level OCR items instead of bubble grouping.",
    )
    parser.add_argument("--vlm-base-url", default=VLM_BASE_URL)
    parser.add_argument("--vlm-model", default=VLM_MODEL)
    parser.add_argument("--vlm-api-key", default=VLM_API_KEY)
    args = parser.parse_args()
    if args.name:
        try:
            prefix = artifact_prefix(args.name)
        except ValueError as exc:
            parser.error(str(exc))
        if not option_was_provided("--frames-dir") and not option_was_provided("--frames"):
            args.frames_dir = Path(f"{prefix}_frames")
        if not option_was_provided("--output-path") and not option_was_provided("--output"):
            args.output_path = Path(f"{prefix}_raw_frame_items.jsonl")
    return args


def main() -> int:
    args = parse_args()
    if args.method == "ocr":
        run_ocr(args)
        return 0
    if args.method == "vlm":
        asyncio.run(run_vlm(args))
        return 0

    run_ocr(args)
    if not has_failed_frames(args.output_path):
        return 0

    print(
        "OCR did not fully succeed; running VLM fallback for frames not already "
        f"completed in {args.output_path}."
    )
    asyncio.run(run_vlm(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
