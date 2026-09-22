"""One-frame Gemini Robotics ER probe; proposal only, never robot actuation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .api import query_image
from .er_client import build_prompt, normalized_yx_to_pixel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="MuJoCo RGB image")
    parser.add_argument("--task", required=True, help="natural-language simulation task")
    parser.add_argument("--model", default="gemini-robotics-er-2-preview")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt without an API call")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.image.is_file():
        raise SystemExit(f"image not found: {args.image}")
    prompt = build_prompt(args.task)
    if args.dry_run:
        print(prompt)
        return 0

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set; use --dry-run to inspect the request")
    try:
        proposal = query_image(
            args.image, args.task, model=args.model, api_key=api_key
        )
    except Exception as exc:
        # API exceptions may include request metadata. Keep credentials out of
        # terminal logs, issue reports, and copied stack traces.
        status = getattr(exc, "code", "unknown")
        raise SystemExit(f"Gemini API failed: {type(exc).__name__} (status {status})") from None
    result = {"model": args.model, "proposal": proposal.to_dict()}

    if proposal.point_yx_norm is not None:
        try:
            from PIL import Image

            with Image.open(args.image) as image:
                result["point_pixel_xy"] = normalized_yx_to_pixel(
                    proposal.point_yx_norm, image.width, image.height
                )
        except ImportError:
            result["point_pixel_xy"] = "install pillow to convert normalized coordinates"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
