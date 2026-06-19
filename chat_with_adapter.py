import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_ADAPTER_PATH = Path("adapters/contact")
DEFAULT_MODEL = "Qwen:Qwen3-14B-MLX-8bit"
DEFAULT_SYSTEM_PROMPT = (
    "你正在微信聊天。用自然、简短、口语化的中文回复，保持训练数据里的语气和关系感。"
)


def artifact_prefix(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")
    if not prefix:
        raise ValueError("--name must contain at least one filename-safe character.")
    return prefix


def option_was_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an MLX-LM base model with LoRA adapters and chat in the terminal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional local artifact prefix, e.g. uses adapters/NAME.",
    )
    parser.add_argument("--model", default=None, help="Base model path or HF repo.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help="Directory containing MLX-LM LoRA adapter_config.json and safetensors.",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=12,
        help="Number of recent user/assistant turns to keep in the prompt.",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Do not include a system message in the chat prompt.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print each assistant reply only after generation finishes.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to supported chat templates.",
    )
    args = parser.parse_args()
    if args.name and not option_was_provided("--adapter-path"):
        args.adapter_path = Path("adapters") / artifact_prefix(args.name)
    return args


def load_adapter_model_path(adapter_path: Path) -> str | None:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    model = config.get("model")
    return model if isinstance(model, str) and model else None


def build_messages(
    history: list[dict[str, str]],
    system_prompt: str,
    include_system: bool,
    history_turns: int,
) -> list[dict[str, str]]:
    if history_turns > 0:
        trimmed_history = history[-history_turns * 2 :]
    else:
        trimmed_history = history

    messages = []
    if include_system and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(trimmed_history)
    return messages


def apply_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def format_prompt(
    tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return apply_chat_template(tokenizer, messages, enable_thinking)

    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and hasattr(inner, "apply_chat_template"):
        return apply_chat_template(inner, messages, enable_thinking)

    parts = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def clean_response(text: str) -> str:
    stop_markers = ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>")
    for marker in stop_markers:
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    return {
        "max_tokens": args.max_tokens,
        "sampler": make_sampler(temp=args.temp, top_p=args.top_p),
        "logits_processors": make_logits_processors(
            repetition_penalty=args.repetition_penalty,
            repetition_context_size=64,
        ),
    }


def print_help() -> None:
    print(
        "Commands: /exit or /quit to leave, /reset to clear history, "
        "/history to print the current context, /help for this message."
    )


def main() -> None:
    args = parse_args()
    model_path = (
        args.model or load_adapter_model_path(args.adapter_path) or DEFAULT_MODEL
    )

    if not args.adapter_path.exists():
        raise SystemExit(f"Adapter path does not exist: {args.adapter_path}")

    print(f"Loading model: {model_path}", file=sys.stderr, flush=True)
    print(f"Loading adapters: {args.adapter_path}", file=sys.stderr, flush=True)

    try:
        from mlx_lm import generate, load, stream_generate
    except ImportError as exc:
        raise SystemExit(
            "mlx-lm is required. Install it in this project venv with: "
            'uv pip install "mlx-lm[train]"'
        ) from exc

    model, tokenizer = load(str(model_path), adapter_path=str(args.adapter_path))
    print("Ready. Type /help for commands.", file=sys.stderr, flush=True)

    history: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/reset":
            history.clear()
            print("History cleared.")
            continue
        if user_text == "/history":
            print(json.dumps(history, ensure_ascii=False, indent=2))
            continue

        history.append({"role": "user", "content": user_text})
        prompt = format_prompt(
            tokenizer,
            build_messages(
                history=history,
                system_prompt=args.system,
                include_system=not args.no_system,
                history_turns=args.history_turns,
            ),
            enable_thinking=args.enable_thinking,
        )

        print("Assistant: ", end="", flush=True)
        kwargs = generation_kwargs(args)
        if args.no_stream:
            response = generate(
                model,
                tokenizer,
                prompt,
                **kwargs,
            )
            response = clean_response(response)
            print(response)
        else:
            chunks = []
            for chunk in stream_generate(
                model,
                tokenizer,
                prompt,
                **kwargs,
            ):
                text = chunk.text
                chunks.append(text)
                if any(
                    marker in text
                    for marker in ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>")
                ):
                    break
                print(text, end="", flush=True)
            response = clean_response("".join(chunks))
            print()

        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
