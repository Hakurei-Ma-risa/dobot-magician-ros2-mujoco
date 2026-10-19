"""Gemini look at the conveyor: is a cube on the belt, and which colour.

One D435i colour frame comes from the running RealSense node (chip_tray.grab_rgb),
is saved as a JPEG and sent inline to Gemini (no file upload: that alone cost
about 4 s). A box overlay is saved when the model gives one, and each call is
appended to log.jsonl next to the images.

Measured on 2026-10-17 with the same prompt: gemini-flash-lite-latest 1.2 s,
gemini-robotics-er-2-preview with thinking "low" 4.2 s; single requests vary
a lot (up to 30+ s), hence the timeout and the manual fallback in the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import time

import cv2

from .chip_tray import grab_rgb


DEFAULT_MODEL = "gemini-flash-lite-latest"
COMPARE_MODELS = ("gemini-flash-lite-latest", "gemini-robotics-er-2-preview")
DEFAULT_DIR = Path("reports/conveyor_sort/vision")
COLOURS = ("red", "green", "blue")
PROMPT = """This photo shows a conveyor belt next to a small robot arm.
Coloured cubes (red, green or blue, about 25 mm) are carried on the belt.
Look only at the belt surface; ignore cubes in boxes, trays or on the table.
Answer with JSON only, no other text:
{"block_on_belt": true or false,
 "colour": "red" or "green" or "blue" or "unknown",
 "box_2d": [ymin, xmin, ymax, xmax] of that cube, normalised to 0-1000, or null}"""


@dataclass
class BeltView:
    present: bool
    colour: str | None  # None when absent or not clearly red/green/blue
    box_2d: list[float] | None
    seconds: float
    image: str
    raw: str


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("没有设置 GEMINI_API_KEY。先在这个终端运行：export GEMINI_API_KEY='你的 key'")
    return key


def parse_answer(text: str) -> tuple[bool, str | None, list[float] | None]:
    """(present, colour, box) from the model text; tolerant of ``` fences and a one-element list."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    payload = json.loads(cleaned)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    present = bool(payload.get("block_on_belt"))
    colour = str(payload.get("colour") or "").strip().lower()
    box = payload.get("box_2d")
    box = [float(v) for v in box] if isinstance(box, (list, tuple)) and len(box) == 4 else None
    return present, (colour if present and colour in COLOURS else None), box


def ask_gemini(image_path: Path, *, model: str, key: str, timeout_s: float) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key, http_options={"timeout": int(timeout_s * 1000)})
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # Gemini 3 / ER 2 think by default; "low" keeps them at a few seconds
        thinking_config=(types.ThinkingConfig(thinking_level="low")
                         if "gemini-3" in model or "robotics-er-2" in model else None),
    )
    image = types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type="image/jpeg")
    return client.models.generate_content(model=model, contents=[image, PROMPT], config=config).text or ""


def snapshot(out_dir: Path = DEFAULT_DIR, label: str = ""):
    """(BGR frame, saved JPEG path) from the D435i."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image = grab_rgb(timeout_s=10.0)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}{'-' + label if label else ''}.jpg"
    cv2.imwrite(str(path), image)
    return image, path


def ask(image, path: Path, *, model: str, key: str, timeout_s: float = 30.0) -> BeltView:
    """Ask ``model`` about a saved frame; overlay and log line next to it."""
    started = time.monotonic()
    raw = ask_gemini(path, model=model, key=key, timeout_s=timeout_s)
    seconds = time.monotonic() - started
    try:
        present, colour, box = parse_answer(raw)
    except (ValueError, AttributeError, TypeError):
        present, colour, box = False, None, None
    view = BeltView(present, colour, box, round(seconds, 2), str(path), raw)
    if box:
        overlay = image.copy()
        height, width = overlay.shape[:2]
        y0, x0, y1, x1 = box
        cv2.rectangle(overlay, (int(x0 * width / 1000), int(y0 * height / 1000)),
                      (int(x1 * width / 1000), int(y1 * height / 1000)), (0, 255, 255), 2)
        cv2.putText(overlay, colour or "?", (int(x0 * width / 1000), max(15, int(y0 * height / 1000) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(path.with_name(f"{path.stem}-{model}.jpg")), overlay)
    with (path.parent / "log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"model": model, **asdict(view)}, ensure_ascii=False) + "\n")
    return view


def look(*, model: str = DEFAULT_MODEL, key: str, out_dir: Path = DEFAULT_DIR, timeout_s: float = 30.0,
         label: str = "") -> BeltView:
    """Grab a frame and ask ``model`` about it."""
    image, path = snapshot(out_dir, label)
    return ask(image, path, model=model, key=key, timeout_s=timeout_s)
