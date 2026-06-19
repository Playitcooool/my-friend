import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("contact_raw_frame_items.jsonl")
DEFAULT_MESSAGES_OUTPUT = Path("data/train.jsonl")

TEXT_TYPES = {"text"}
MEDIA_TYPES = {"image", "sticker", "voice", "file", "transfer"}
VALID_TYPES = TEXT_TYPES | MEDIA_TYPES | {"system"}
RECALL_PATTERNS = [
    re.compile(r"^'.+'\s+recalled a message\.?$", re.IGNORECASE),
    re.compile(r"^you recalled a message\.?$", re.IGNORECASE),
    re.compile(r"^.+撤回了一条消息$"),
]
QUOTE_PREFIX_PATTERNS = [
    re.compile(r"^money is all you need\s*[:：]\s*.+$", re.IGNORECASE),
    re.compile(r"^CONTACT_NAME\s*[:：]\s*.+$"),
]


def artifact_prefix(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")
    if not prefix:
        raise ValueError("--name must contain at least one filename-safe character.")
    return prefix


def option_was_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def frame_index_from_name(name: str) -> int:
    match = re.search(r"(\d+)", Path(name).stem)
    if not match:
        return -1
    return int(match.group(1))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_recall_notice(content: str) -> bool:
    return any(pattern.match(content) for pattern in RECALL_PATTERNS)


def is_quote_notice(content: str) -> bool:
    return any(pattern.match(content) for pattern in QUOTE_PREFIX_PATTERNS)


def normalize_item(
    raw_item: dict[str, Any], frame: dict[str, Any]
) -> dict[str, Any] | None:
    content = raw_item.get("content")
    if not isinstance(content, str):
        return None

    content = normalize_space(content)
    if not content:
        return None
    if is_quote_notice(content):
        return None

    side = raw_item.get("side")
    role = raw_item.get("role")
    item_type = raw_item.get("type")
    edge = raw_item.get("edge", "none")
    vertical_position = raw_item.get("vertical_position", "middle")

    if side == "right":
        role = "user"
    elif side == "left":
        role = "assistant"
    elif side == "center":
        role = "system"

    if role not in {"user", "assistant", "system"}:
        return None

    if item_type not in VALID_TYPES:
        item_type = "text"

    if edge not in {"none", "top", "bottom", "both"}:
        edge = "none"

    if vertical_position not in {"top", "middle", "bottom"}:
        vertical_position = "middle"

    complete = raw_item.get("complete")
    if not isinstance(complete, bool):
        complete = edge == "none"
    if edge != "none":
        complete = False

    try:
        order = int(raw_item.get("order", 0))
    except (TypeError, ValueError):
        order = 0

    frame_index = frame.get("frame_index")
    if not isinstance(frame_index, int):
        frame_index = frame_index_from_name(str(frame.get("frame", "")))

    return {
        "frame": frame.get("frame"),
        "frame_index": frame_index,
        "order": order,
        "role": role,
        "type": item_type,
        "content": content,
        "complete": complete,
        "edge": edge,
        "vertical_position": vertical_position,
    }


def load_frames(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    frames = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json_lines"] += 1
                continue

            if frame.get("ok") is not True:
                stats["failed_frames"] += 1
                continue

            raw_items = frame.get("items")
            if not isinstance(raw_items, list):
                stats["frames_without_items"] += 1
                continue

            normalized_items = []
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    stats["bad_items"] += 1
                    continue

                item = normalize_item(raw_item, frame)
                if item is None:
                    stats["bad_items"] += 1
                    continue
                normalized_items.append(item)

            normalized_items.sort(key=lambda item: item["order"])
            frame["_line_number"] = line_number
            frame["_items"] = normalized_items
            frames.append(frame)

    frames.sort(
        key=lambda frame: (
            frame.get("frame_index")
            if isinstance(frame.get("frame_index"), int)
            else frame_index_from_name(str(frame.get("frame", ""))),
            frame.get("_line_number", 0),
        )
    )
    stats["input_frames"] = len(frames)
    stats["input_items"] = sum(len(frame["_items"]) for frame in frames)
    return frames, stats


def item_signature(item: dict[str, Any]) -> tuple[str, str, str]:
    return (item["role"], item["type"], item["content"])


def filter_items(
    items: list[dict[str, Any]],
    keep_incomplete: bool,
    keep_media: bool,
    keep_system: bool,
) -> list[dict[str, Any]]:
    filtered = []
    for item in items:
        if is_recall_notice(item["content"]):
            continue
        if item["role"] == "system" and not keep_system:
            continue
        if not item["complete"] and not keep_incomplete:
            continue
        if item["type"] in MEDIA_TYPES and not keep_media:
            continue
        filtered.append(item)
    return filtered


def find_overlap(
    conversation: list[dict[str, Any]],
    current: list[dict[str, Any]],
    lookback: int,
    min_overlap: int,
) -> tuple[int, int]:
    if not conversation or not current:
        return (0, 0)

    tail = conversation[-lookback:]
    tail_signatures = [item_signature(item) for item in tail]
    current_signatures = [item_signature(item) for item in current]

    best_start = 0
    best_length = 0

    for tail_start in range(len(tail_signatures)):
        for current_start in range(len(current_signatures)):
            length = 0
            while (
                tail_start + length < len(tail_signatures)
                and current_start + length < len(current_signatures)
                and tail_signatures[tail_start + length]
                == current_signatures[current_start + length]
            ):
                length += 1

            if length > best_length:
                best_start = current_start
                best_length = length

    if best_length >= min_overlap:
        return (best_start, best_length)

    return (0, 0)


def build_conversation(
    frames: list[dict[str, Any]],
    keep_incomplete: bool,
    keep_media: bool,
    keep_system: bool,
    collapse_adjacent_frame_duplicates: bool,
    lookback: int,
    min_overlap: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    conversation: list[dict[str, Any]] = []

    for frame in frames:
        items = filter_items(frame["_items"], keep_incomplete, keep_media, keep_system)
        stats["candidate_items"] += len(items)
        if not items:
            continue

        overlap_start, overlap_length = find_overlap(
            conversation=conversation,
            current=items,
            lookback=lookback,
            min_overlap=min_overlap,
        )
        stats["deduped_items"] += overlap_length

        if overlap_length:
            append_from = overlap_start + overlap_length
        else:
            append_from = 0

        conversation.extend(items[append_from:])

    if collapse_adjacent_frame_duplicates:
        collapsed = []
        for item in conversation:
            if (
                collapsed
                and item_signature(collapsed[-1]) == item_signature(item)
                and collapsed[-1].get("frame") != item.get("frame")
            ):
                stats["adjacent_frame_duplicates"] += 1
                continue
            collapsed.append(item)
        conversation = collapsed

    for index, item in enumerate(conversation, start=1):
        item["index"] = index

    stats["clean_items"] = len(conversation)
    return conversation, stats


def merge_messages(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in items:
        if item["role"] == "system":
            continue

        content = item["content"]
        if messages and messages[-1]["role"] == item["role"]:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": item["role"], "content": content})

    return messages


def trim_dangling_final_user(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    if messages and messages[-1]["role"] == "user":
        return messages[:-1], 1
    return messages, 0


def build_sft_pairs(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, list[dict[str, str]]]], Counter[str]]:
    stats: Counter[str] = Counter()
    pairs = []
    pending_user: dict[str, str] | None = None

    for message in messages:
        role = message["role"]
        if role == "user":
            if pending_user is not None:
                stats["dropped_unanswered_user_messages"] += 1
            pending_user = message
            continue

        if role == "assistant":
            if pending_user is None:
                stats["dropped_leading_assistant_messages"] += 1
                continue

            pairs.append({"messages": [pending_user, message]})
            pending_user = None
            continue

    if pending_user is not None:
        stats["dropped_unanswered_user_messages"] += 1

    stats["pairs"] = len(pairs)
    return pairs, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean overlapping WeChat frame extraction JSONL into SFT chat pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional local artifact prefix, e.g. reads NAME_raw_frame_items.jsonl and writes data/NAME_train.jsonl.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--items-output",
        type=Path,
        default=None,
        help="Optional debug output for cleaned bubble-level items.",
    )
    parser.add_argument(
        "--messages-output",
        "--output",
        type=Path,
        default=DEFAULT_MESSAGES_OUTPUT,
        help="Output JSONL containing {'messages': [...]} SFT pairs.",
    )
    parser.add_argument("--lookback", type=int, default=80)
    parser.add_argument("--min-overlap", type=int, default=2)
    parser.add_argument("--keep-incomplete", action="store_true")
    parser.add_argument("--drop-media", action="store_true")
    parser.add_argument("--keep-system", action="store_true")
    parser.add_argument(
        "--keep-adjacent-frame-duplicates",
        action="store_true",
        help="Keep identical adjacent bubbles when they come from different frames.",
    )
    parser.add_argument(
        "--keep-dangling-final-user",
        action="store_true",
        help="Keep a final user message even when it has no assistant answer.",
    )
    args = parser.parse_args()
    if args.name:
        prefix = artifact_prefix(args.name)
        if not option_was_provided("--input"):
            args.input = Path(f"{prefix}_raw_frame_items.jsonl")
        if not option_was_provided("--messages-output") and not option_was_provided("--output"):
            args.messages_output = Path("data") / f"{prefix}_train.jsonl"
        if args.items_output is not None and not option_was_provided("--items-output"):
            args.items_output = Path(f"{prefix}_clean_items.jsonl")
    return args


def main() -> None:
    args = parse_args()
    frames, load_stats = load_frames(args.input)
    items, clean_stats = build_conversation(
        frames=frames,
        keep_incomplete=args.keep_incomplete,
        keep_media=not args.drop_media,
        keep_system=args.keep_system,
        collapse_adjacent_frame_duplicates=not args.keep_adjacent_frame_duplicates,
        lookback=args.lookback,
        min_overlap=args.min_overlap,
    )
    messages = merge_messages(items)
    trimmed_final_user_messages = 0
    if not args.keep_dangling_final_user:
        messages, trimmed_final_user_messages = trim_dangling_final_user(messages)
    pairs, pair_stats = build_sft_pairs(messages)

    if args.items_output is not None:
        write_jsonl(args.items_output, items)
    write_jsonl(args.messages_output, pairs)

    stats = load_stats + clean_stats + pair_stats
    stats["messages"] = len(messages)
    stats["trimmed_final_user_messages"] = trimmed_final_user_messages

    print(f"Read {stats['input_frames']} frames and {stats['input_items']} raw items.")
    print(
        f"Kept {stats['clean_items']} cleaned items, "
        f"deduped {stats['deduped_items']} overlaps, "
        f"collapsed {stats['adjacent_frame_duplicates']} adjacent frame duplicates, "
        f"trimmed {stats['trimmed_final_user_messages']} dangling final user messages, "
        f"merged into {stats['messages']} messages, "
        f"wrote {stats['pairs']} SFT pairs."
    )
    if args.items_output is not None:
        print(f"Wrote {args.items_output}")
    print(f"Wrote {args.messages_output}")


if __name__ == "__main__":
    main()
