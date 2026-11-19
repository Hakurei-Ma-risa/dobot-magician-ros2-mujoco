"""Module 3 "choose target": pick ONE candidate for an instruction.

Two interchangeable selectors share one signature:

    select(instruction, image_rgb, boxes, positions) -> Choice

- RuleSelector: free, offline, understands a few words (colours, left/right,
  nearest/farthest, biggest/smallest, tallest/shortest).
- DeepSeekSelector: draws numbered boxes on the image (set-of-marks) and asks
  deepseek-flash which number matches. The model only returns a number; the
  3-D position still comes from depth, so the AI never invents coordinates.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    index: int
    bbox_xywh: tuple[int, int, int, int]
    position_m: tuple[float, float, float]  # base frame
    size_m: tuple[float, float, float]


@dataclass(frozen=True)
class Choice:
    index: int  # -1 = nothing matches
    reason: str
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------- drawing


def draw_marks(image_rgb: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Green boxes with big white numbers (black outline) for people and AI."""

    marked = image_rgb.copy()
    for box in boxes:
        x, y, w, h = box.bbox_xywh
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = str(box.index)
        origin = (x + 4, y + 26)
        cv2.putText(marked, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(marked, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return marked


# ---------------------------------------------------------------- colours

_HUE_NAMES = (  # OpenCV hue 0..179
    (0, 8, "red"),
    (8, 20, "orange"),
    (20, 35, "yellow"),
    (35, 85, "green"),
    (85, 100, "cyan"),
    (100, 130, "blue"),
    (130, 165, "purple"),
    (165, 180, "red"),
)

_COLOUR_WORDS = {
    "red": {"red"},
    "orange": {"orange", "yellow"},
    "yellow": {"yellow", "orange"},
    "gold": {"yellow", "orange"},
    "green": {"green"},
    "cyan": {"cyan", "blue"},
    "blue": {"blue", "cyan"},
    "purple": {"purple"},
    "violet": {"purple"},
    "magenta": {"purple"},
    "pink": {"purple", "red"},
    "white": {"white"},
    "black": {"black"},
    "grey": {"grey"},
    "gray": {"grey"},
}


def colour_name(image_rgb: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> str:
    """Dominant colour inside a box, ignoring pixels that look like the table.

    The table colour is sampled from a thin frame just outside the box.
    """

    height, width = image_rgb.shape[:2]
    x, y, w, h = bbox_xywh
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x0 >= x1 or y0 >= y1:
        return "unknown"
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    pad = 6
    ox0, oy0 = max(0, x0 - pad), max(0, y0 - pad)
    ox1, oy1 = min(width, x1 + pad), min(height, y1 + pad)
    frame = np.ones((oy1 - oy0, ox1 - ox0), bool)
    frame[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    inside = hsv[y0:y1, x0:x1].reshape(-1, 3)
    outside = hsv[oy0:oy1, ox0:ox1][frame]
    if outside.size:
        background = np.median(outside, axis=0)
        hue_gap = np.abs(inside[:, 0] - background[0])
        hue_gap = np.minimum(hue_gap, 180.0 - hue_gap)
        looks_like_table = (hue_gap < 6) & (np.abs(inside[:, 1] - background[1]) < 50)
        inside = inside[~looks_like_table]
    if inside.shape[0] < 20:
        return "unknown"
    saturated = inside[(inside[:, 1] > 80) & (inside[:, 2] > 50)]
    if saturated.shape[0] < 0.2 * inside.shape[0]:
        value = float(np.median(inside[:, 2]))
        return "white" if value > 180 else "black" if value < 60 else "grey"
    hist = np.bincount(saturated[:, 0].astype(int), minlength=180)
    votes: dict[str, int] = {}
    for low, high, name in _HUE_NAMES:
        votes[name] = votes.get(name, 0) + int(hist[low:high].sum())
    return max(votes, key=votes.get)


def _object_pixels_hsv(image_rgb: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
    """Saturated pixels inside the box that do not look like the table."""

    height, width = image_rgb.shape[:2]
    x, y, w, h = bbox_xywh
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((0, 3), np.float32)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    pad = 6
    ox0, oy0 = max(0, x0 - pad), max(0, y0 - pad)
    ox1, oy1 = min(width, x1 + pad), min(height, y1 + pad)
    frame = np.ones((oy1 - oy0, ox1 - ox0), bool)
    frame[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    inside = hsv[y0:y1, x0:x1].reshape(-1, 3)
    outside = hsv[oy0:oy1, ox0:ox1][frame]
    if outside.size:
        background = np.median(outside, axis=0)
        gap = np.abs(inside[:, 0] - background[0])
        gap = np.minimum(gap, 180.0 - gap)
        inside = inside[~((gap < 6) & (np.abs(inside[:, 1] - background[1]) < 50))]
    return inside[(inside[:, 1] > 80) & (inside[:, 2] > 50)]


def object_hue(image_rgb: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> float | None:
    """Typical hue (OpenCV 0..179) of the coloured object in the box, or None."""

    pixels = _object_pixels_hsv(image_rgb, bbox_xywh)
    if pixels.shape[0] < 30:
        return None
    angles = pixels[:, 0] * (np.pi / 90.0)  # hue is circular
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return float((mean * 90.0 / np.pi) % 180.0)


def hue_pixel_count(image_rgb: np.ndarray, bbox_xywh: tuple[int, int, int, int], hue: float, tolerance: float = 12.0) -> int:
    """How many object pixels in the box have roughly this hue."""

    pixels = _object_pixels_hsv(image_rgb, bbox_xywh)
    gap = np.abs(pixels[:, 0] - hue)
    gap = np.minimum(gap, 180.0 - gap)
    return int(np.count_nonzero(gap <= tolerance))


# ---------------------------------------------------------------- rule


class RuleSelector:
    """Keyword rules. Good enough for 'the red one on the left'."""

    name = "rule"

    def select(self, instruction: str, image_rgb: np.ndarray, boxes: list[Box]) -> Choice:
        start = time.monotonic()
        if not boxes:
            return Choice(-1, "no objects found")
        words = re.findall(r"[a-z]+", instruction.lower())
        pool = list(boxes)
        notes = []
        wanted = set()
        for word in words:
            wanted |= _COLOUR_WORDS.get(word, set())
        if wanted:
            colours = {box.index: colour_name(image_rgb, box.bbox_xywh) for box in pool}
            pool = [box for box in pool if colours[box.index] in wanted]
            notes.append(
                "colours seen: "
                + ", ".join(f"{i}={c}" for i, c in sorted(colours.items()))
            )
            if not pool:
                return Choice(-1, "no object has colour " + "/".join(sorted(wanted)) + "; " + notes[0])

        def distance(box: Box) -> float:
            return float(np.hypot(box.position_m[0], box.position_m[1]))

        def footprint(box: Box) -> float:
            return box.size_m[0] * box.size_m[1]

        rules = (
            ({"left", "leftmost"}, lambda b: b.bbox_xywh[0] + b.bbox_xywh[2] / 2, min, "leftmost in the image"),
            ({"right", "rightmost"}, lambda b: b.bbox_xywh[0] + b.bbox_xywh[2] / 2, max, "rightmost in the image"),
            ({"far", "farthest", "furthest"}, distance, max, "farthest from the robot"),
            ({"big", "bigger", "biggest", "large", "larger", "largest"}, footprint, max, "largest footprint"),
            ({"small", "smaller", "smallest", "tiny"}, footprint, min, "smallest footprint"),
            ({"tall", "taller", "tallest", "high", "highest"}, lambda b: b.size_m[2], max, "tallest"),
            ({"short", "shorter", "shortest", "flat", "low", "lowest"}, lambda b: b.size_m[2], min, "shortest"),
        )
        for keys, score, pick, text in rules:
            if keys & set(words):
                chosen = pick(pool, key=score)
                break
        else:
            chosen = min(pool, key=distance)
            text = "nearest to the robot"
        reason = text if not wanted else f"{'/'.join(sorted(wanted))} and {text}"
        if notes:
            reason += "; " + notes[0]
        return Choice(chosen.index, reason, time.monotonic() - start)


# ---------------------------------------------------------------- DeepSeek

SYSTEM_PROMPT = (
    "You help a robot arm choose ONE object to pick. The image shows the "
    "workspace; every candidate object has a green box with a white number. "
    "Read the user's instruction and choose the number of the best matching "
    "box. Ignore the robot itself. Reply with JSON only, like "
    '{"index": 2, "reason": "red can on the left"}. '
    'If no box matches, reply {"index": -1, "reason": "..."}.'
)


class DeepSeekSelector:
    """Set-of-marks selection with deepseek-flash (Anthropic-compatible API)."""

    name = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com/anthropic",
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.model = model
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.timeout_s = timeout_s

    def select(self, instruction: str, image_rgb: np.ndarray, boxes: list[Box]) -> Choice:
        start = time.monotonic()
        if not boxes:
            return Choice(-1, "no objects found")
        marked = draw_marks(image_rgb, boxes)
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(marked, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("could not encode image")
        numbers = ", ".join(str(box.index) for box in boxes)
        body = {
            "model": self.model,
            "max_tokens": 400,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(jpeg.tobytes()).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Box numbers: {numbers}.\nInstruction: {instruction}",
                        },
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise RuntimeError(f"DeepSeek HTTP {error.code}: {detail}") from error
        text = "".join(
            part.get("text", "") for part in reply.get("content", []) if part.get("type") == "text"
        )
        usage = reply.get("usage", {})
        index, reason = parse_choice(text, {box.index for box in boxes})
        return Choice(
            index,
            reason,
            time.monotonic() - start,
            int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )


def parse_choice(text: str, allowed: set[int]) -> tuple[int, str]:
    """Read {"index": n, "reason": "..."} from a model reply; be forgiving."""

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            index = int(data.get("index", -1))
            reason = str(data.get("reason", ""))
            if index in allowed or index == -1:
                return index, reason
            return -1, f"model chose {index}, which is not a box number"
        except (ValueError, TypeError):
            pass
    number = re.search(r"-?\d+", text)
    if number and int(number.group(0)) in allowed:
        return int(number.group(0)), text.strip()[:120]
    return -1, "could not read the model reply: " + text.strip()[:120]


def make_selector(name: str):
    if name == "rule":
        return RuleSelector()
    if name == "deepseek":
        return DeepSeekSelector()
    raise ValueError(f"unknown selector {name!r} (use rule or deepseek)")
