import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_INPUT = Path("data/train.jsonl")
DEFAULT_OUTPUT = Path("data/train.roleplay_filtered.jsonl")
DEFAULT_JUDGMENTS_OUTPUT = Path("data/train.roleplay_judgments.jsonl")

RATINGS = ("high", "medium", "low", "incomplete", "invalid")
KEEP_RATINGS = {"high", "medium"}
MIN_RATING_ORDER = {
    "high": 4,
    "medium": 3,
    "low": 2,
    "incomplete": 1,
    "invalid": 0,
}

SYSTEM_PROMPT = """You are judging Chinese WeChat chat fine-tuning examples for personal chat style value.

The goal is to select examples that help a model learn the assistant speaker's natural WeChat style:
- wording habits
- rhythm and line breaks
- emotional stance
- teasing, humor, hesitation, complaints, reactions
- relationship dynamic
- casual short replies when they are meaningful in context

Do NOT judge by whether the assistant sounds formal, helpful, or like an AI assistant.
Natural WeChat messages can be short, fragmented, emotional, repetitive, or informal.

Return strict JSON only with these keys:
- rating: one of high, medium, low, incomplete, invalid
- keep: boolean
- style_score: integer from 1 to 5
- context_score: integer from 1 to 5
- quality_score: integer from 1 to 5
- privacy_risk: one of low, medium, high
- short_reply_type: one of none, useful_style, generic_ack, duplicate_noise
- style_signal: short description of the useful style signal
- issues: list of short issue labels
- reason: concise explanation

Rating criteria:
- high: strong assistant style signal. The assistant reply shows distinctive wording, rhythm, emotion, humor, teasing, hesitation, catchphrases, or relationship dynamic.
- medium: coherent and usable. Some style signal, even if not very strong.
- low: coherent but weak style signal, too generic, mostly factual, or not very useful for learning this person's style.
- incomplete: context is missing, OCR corruption breaks meaning, role assignment seems wrong, message appears cut off, or assistant answer is dangling.
- invalid: malformed message schema.

Short reply policy:
- Do NOT automatically reject short replies.
- Short replies like "?", "哦哦哦", "笑死", "不是吧", "我好累", "好讨厌" can be useful if they express style or fit the context.
- Generic acknowledgements like "嗯", "对", "好", "ok" are usually low unless the context makes them stylistically meaningful.
- Repeated generic short replies should be marked as duplicate_noise or generic_ack.

Privacy policy:
- privacy_risk=high if the sample contains phone numbers, addresses, IDs, exact private locations, health/financial details, or highly sensitive personal information.
- privacy_risk=medium if it contains real names, schools, workplaces, travel plans, relationship details, or identifiable third-party references.
- privacy_risk=low if there is no obvious sensitive or identifying information.

Keep policy:
- keep=true only if rating is high or medium, quality_score >= 4, context_score >= 3, and privacy_risk is not high.
- keep=false for incomplete, invalid, obvious OCR errors, role confusion, cut-off replies, high privacy risk, or duplicate noise.
"""


def artifact_prefix(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")
    if not prefix:
        raise ValueError("--name must contain at least one filename-safe character.")
    return prefix


def option_was_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge roleplay value for JSONL chat examples and write filtered SFT data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional local artifact prefix, e.g. reads data/NAME_train.jsonl and writes data/NAME.roleplay_filtered.jsonl.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--judgments-output", type=Path, default=DEFAULT_JUDGMENTS_OUTPUT
    )
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="mlx-community:Qwen3.5-9B-MLX-4bit")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--min-rating",
        choices=tuple(MIN_RATING_ORDER),
        default="medium",
        help="Lowest rating to keep. Default keeps high and medium.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N non-empty input lines.",
    )
    args = parser.parse_args()
    if args.name:
        prefix = artifact_prefix(args.name)
        if not option_was_provided("--input"):
            args.input = Path("data") / f"{prefix}_train.jsonl"
        if not option_was_provided("--output"):
            args.output = Path("data") / f"{prefix}.roleplay_filtered.jsonl"
        if not option_was_provided("--judgments-output"):
            args.judgments_output = Path("data") / f"{prefix}.roleplay_judgments.jsonl"
    return args


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None

    parsed = urlparse(base_url)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return base_url.rstrip("/") + "/v1"
    return base_url


def normalize_judgment(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    rating = raw.get("rating")
    if rating not in RATINGS:
        rating = "invalid"

    issues = raw.get("issues")
    if not isinstance(issues, list):
        issues = []
    issues = [str(issue)[:80] for issue in issues]

    return {
        "rating": rating,
        "keep": bool(raw.get("keep")) and rating in KEEP_RATINGS,
        "style_signal": str(raw.get("style_signal", ""))[:300],
        "issues": issues,
        "reason": str(raw.get("reason", ""))[:600],
    }


def invalid_judgment(reason: str, issues: list[str] | None = None) -> dict[str, Any]:
    return {
        "rating": "invalid",
        "keep": False,
        "style_signal": "",
        "issues": issues or ["invalid_schema"],
        "reason": reason,
    }


def validate_messages(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "row is not a JSON object"

    messages = obj.get("messages")
    if not isinstance(messages, list):
        return False, "messages is not a list"
    if len(messages) != 2:
        return False, "messages must contain exactly two items"

    expected_roles = ("user", "assistant")
    for index, (message, expected_role) in enumerate(zip(messages, expected_roles)):
        if not isinstance(message, dict):
            return False, f"message {index} is not an object"
        if message.get("role") != expected_role:
            return False, f"message {index} role is not {expected_role}"
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return False, f"message {index} content is empty or not a string"

    return True, ""


def message_signature(obj: dict[str, Any]) -> str:
    return json.dumps(
        obj["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def make_record(
    line_number: int,
    raw_line: str,
    obj: dict[str, Any] | None,
    judgment: dict[str, Any],
    duplicate_of_line: int | None = None,
) -> dict[str, Any]:
    return {
        "line_number": line_number,
        "row": raw_line,
        "messages": obj.get("messages") if obj else None,
        "duplicate_of_line": duplicate_of_line,
        "judgment": normalize_judgment(judgment),
    }


def load_input(
    path: Path, limit: int | None
) -> list[tuple[int, str, dict[str, Any] | None, str | None]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                rows.append((line_number, raw_line, None, f"malformed JSON: {exc.msg}"))
                continue

            ok, error = validate_messages(obj)
            rows.append((line_number, raw_line, obj, None if ok else error))
    return rows


def load_existing_judgments(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    existing = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            line_number = record.get("line_number")
            if isinstance(line_number, int) and isinstance(
                record.get("judgment"), dict
            ):
                existing[line_number] = record
    return existing


def build_user_prompt(messages: list[dict[str, str]]) -> str:
    return json.dumps(
        {
            "messages": messages,
            "required_output": {
                "rating": "high|medium|low|incomplete|invalid",
                "keep": "boolean",
                "style_signal": "short string",
                "issues": ["short issue labels"],
                "reason": "short string",
            },
        },
        ensure_ascii=False,
        indent=2,
    )


async def judge_one(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(messages)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return normalize_judgment(json.loads(content))
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                await asyncio.sleep(min(2**attempt, 8))

    return {
        "rating": "incomplete",
        "keep": False,
        "style_signal": "",
        "issues": ["judge_error"],
        "reason": f"judge call failed: {last_error[:500]}",
    }


async def judge_pending(
    pending: list[tuple[int, dict[str, Any]]],
    api_key: str | None,
    base_url: str | None,
    model: str,
    max_concurrency: int,
    max_retries: int,
    request_timeout: float,
    on_result: Any | None = None,
) -> dict[int, dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit(
            "The openai package is required. Install it in this project venv with: "
            "uv pip install openai"
        ) from exc

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    elif base_url:
        client_kwargs["api_key"] = "not-needed"
    if base_url:
        client_kwargs["base_url"] = base_url
    client_kwargs["timeout"] = request_timeout
    client = AsyncOpenAI(**client_kwargs)

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    results: dict[int, dict[str, Any]] = {}
    total = len(pending)
    completed = 0

    log(
        f"Starting judge calls: {total} rows, model={model}, "
        f"base_url={base_url or 'OpenAI SDK default'}, concurrency={max_concurrency}"
    )

    async def run(line_number: int, obj: dict[str, Any]) -> None:
        nonlocal completed
        async with semaphore:
            log(f"Requesting judgment for line {line_number}")
            results[line_number] = await judge_one(
                client=client,
                model=model,
                messages=obj["messages"],
                max_retries=max_retries,
            )
            if on_result is not None:
                on_result(line_number, results[line_number])
            completed += 1
            log(f"Completed {completed}/{total} judgments")

    await asyncio.gather(*(run(line_number, obj) for line_number, obj in pending))
    return results


def should_keep(judgment: dict[str, Any], min_rating: str) -> bool:
    rating = judgment.get("rating")
    if rating not in MIN_RATING_ORDER:
        return False
    return (
        bool(judgment.get("keep"))
        and MIN_RATING_ORDER[rating] >= MIN_RATING_ORDER[min_rating]
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()


async def async_main() -> int:
    args = parse_args()
    args.base_url = normalize_base_url(args.base_url)
    if not args.model:
        raise SystemExit(
            "A judge model is required. Pass --model or set ROLEPLAY_JUDGE_MODEL."
        )
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1.")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be at least 0.")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be greater than 0.")

    input_rows = load_input(args.input, args.limit)
    existing = load_existing_judgments(args.judgments_output) if args.resume else {}
    log(f"Loaded {len(input_rows)} input rows from {args.input}")

    records_by_line: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    seen_signatures: dict[str, int] = {}

    for line_number, raw_line, obj, validation_error in input_rows:
        if args.resume and line_number in existing:
            record = existing[line_number]
            records_by_line[line_number] = record
            if obj and record.get("duplicate_of_line") is None:
                seen_signatures.setdefault(message_signature(obj), line_number)
            continue

        if validation_error is not None or obj is None:
            records_by_line[line_number] = make_record(
                line_number,
                raw_line,
                obj,
                invalid_judgment(validation_error or "invalid row"),
            )
            continue

        signature = message_signature(obj)
        duplicate_of_line = seen_signatures.get(signature)
        if duplicate_of_line is not None:
            records_by_line[line_number] = make_record(
                line_number,
                raw_line,
                obj,
                {
                    "rating": "low",
                    "keep": False,
                    "style_signal": "",
                    "issues": ["duplicate"],
                    "reason": f"exact duplicate of line {duplicate_of_line}",
                },
                duplicate_of_line=duplicate_of_line,
            )
            continue

        seen_signatures[signature] = line_number
        pending.append((line_number, obj))

    log(f"Prepared {len(pending)} rows for LLM judging")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    log(f"Initialized filtered output at {args.output}")

    args.judgments_output.parent.mkdir(parents=True, exist_ok=True)
    args.judgments_output.write_text("", encoding="utf-8")
    for line_number in sorted(records_by_line):
        record = records_by_line[line_number]
        append_jsonl(args.judgments_output, record)
        if record.get("duplicate_of_line") is None and should_keep(
            record["judgment"], args.min_rating
        ):
            try:
                append_jsonl(args.output, json.loads(record["row"]))
            except (json.JSONDecodeError, TypeError):
                pass
    log(f"Initialized incremental judgments at {args.judgments_output}")

    if pending:
        raw_by_line = {
            line_number: raw_line for line_number, raw_line, _, _ in input_rows
        }
        obj_by_line = {line_number: obj for line_number, _, obj, _ in input_rows if obj}

        def record_result(line_number: int, judgment: dict[str, Any]) -> None:
            record = make_record(
                line_number,
                raw_by_line[line_number],
                obj_by_line[line_number],
                judgment,
            )
            records_by_line[line_number] = record
            append_jsonl(args.judgments_output, record)
            if should_keep(record["judgment"], args.min_rating):
                append_jsonl(args.output, obj_by_line[line_number])

        judgments = await judge_pending(
            pending=pending,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            max_concurrency=args.max_concurrency,
            max_retries=args.max_retries,
            request_timeout=args.request_timeout,
            on_result=record_result,
        )
        for line_number, judgment in judgments.items():
            if line_number not in records_by_line:
                records_by_line[line_number] = make_record(
                    line_number,
                    raw_by_line[line_number],
                    obj_by_line[line_number],
                    judgment,
                )

    judgment_rows = [records_by_line[line_number] for line_number, *_ in input_rows]
    kept_rows = []
    for record in judgment_rows:
        obj = None
        try:
            obj = json.loads(record["row"])
        except (json.JSONDecodeError, TypeError):
            pass
        if (
            obj
            and record.get("duplicate_of_line") is None
            and should_keep(record["judgment"], args.min_rating)
        ):
            kept_rows.append(obj)

    write_jsonl(args.judgments_output, judgment_rows)
    write_jsonl(args.output, kept_rows)
    log(f"Wrote sorted judgments to {args.judgments_output}")
    log(f"Wrote filtered rows to {args.output}")

    rating_counts: dict[str, int] = {}
    for record in judgment_rows:
        rating = record["judgment"].get("rating", "invalid")
        rating_counts[rating] = rating_counts.get(rating, 0) + 1

    print(
        json.dumps(
            {
                "input_rows": len(input_rows),
                "judged_rows": len(pending),
                "kept_rows": len(kept_rows),
                "judgments_output": str(args.judgments_output),
                "filtered_output": str(args.output),
                "ratings": rating_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
