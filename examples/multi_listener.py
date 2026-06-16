"""Real-time wake word detection for the local hey_jarvis model."""

import argparse
import asyncio
from pathlib import Path

from livekit.wakeword import WakeWordListener, WakeWordModel

DEFAULT_MODEL_DIR = Path("/home/jasper/models/wake_word")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for one or more wake-word models.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_MODEL_DIR],
        help="ONNX model path(s) or directory paths. Defaults to /~/models/wake_word.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--debounce", type=float, default=2.0)
    return parser.parse_args()


def collect_model_paths(paths: list[Path]) -> list[Path]:
    model_paths: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            model_paths.extend(sorted(resolved.glob("*.onnx")))
        else:
            model_paths.append(resolved)
    return model_paths


async def main() -> None:
    args = parse_args()
    model_paths = collect_model_paths(args.paths)
    if not model_paths:
        raise FileNotFoundError(f"No .onnx models found in: {DEFAULT_MODEL_DIR}")

    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Model not found: {missing[0]}")

    model = WakeWordModel(models=model_paths)
    model_names = ", ".join(path.stem for path in model_paths)

    async with WakeWordListener(
        model,
        threshold=args.threshold,
        debounce=args.debounce,
    ) as listener:
        print(f"Listening for {model_names}... Press Ctrl+C to stop.")
        while True:
            detection = await listener.wait_for_detection()
            print(f"Detected {detection.name}! ({detection.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
