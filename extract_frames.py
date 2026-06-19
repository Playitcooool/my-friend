import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_FPS = 1.0
DEFAULT_FRAMES_DIR = Path("contact_frames")
FRAME_PATTERN = "frame%06d.png"


def artifact_prefix(name: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._-")
    if not prefix:
        raise ValueError("--name must contain at least one filename-safe character.")
    return prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PNG frames from a WeChat screen recording with ffmpeg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--frames-dir", "--frames", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument(
        "--name",
        default=None,
        help="Optional local artifact prefix, e.g. writes CONTACT_ALIAS_frames/.",
    )
    parser.add_argument("--start", default=None, help="Optional ffmpeg -ss start time.")
    parser.add_argument("--duration", default=None, help="Optional ffmpeg -t duration.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def option_was_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def main() -> int:
    args = parse_args()
    if args.name:
        try:
            prefix = artifact_prefix(args.name)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not option_was_provided("--frames-dir") and not option_was_provided("--frames"):
            args.frames_dir = Path(f"{prefix}_frames")
    if args.fps <= 0:
        raise SystemExit("--fps must be greater than 0.")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = args.frames_dir / FRAME_PATTERN
    cmd = ["ffmpeg", "-hide_banner"]
    cmd.append("-y" if args.overwrite else "-n")
    if args.start:
        cmd.extend(["-ss", args.start])
    cmd.extend(["-i", str(args.video)])
    if args.duration:
        cmd.extend(["-t", args.duration])
    cmd.extend(["-vf", f"fps={args.fps:g}", str(output_pattern)])

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    frames = sorted(args.frames_dir.glob("frame*.png"))
    print(f"Wrote {len(frames)} frames to {args.frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
