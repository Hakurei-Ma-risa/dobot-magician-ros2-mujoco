"""Small, dependency-light parser for Gemini Robotics ER proposals.

The model returns image coordinates in normalized ``[y, x]`` order on a
0--1000 scale.  This module deliberately stops at a validated proposal; it
does not command a robot or write MuJoCo state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ERProposal:
    label: str
    point_yx_norm: tuple[float, float] | None = None
    bbox_yxyx_norm: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prompt(task: str) -> str:
    """Build a strict, low-ambiguity prompt for a single simulator frame."""

    return f"""You are the perception and task-planning layer for a MuJoCo robot simulation.
Task: {task}

Inspect the image and return ONLY valid JSON (no Markdown and no prose) using this schema:
{{
  \"label\": \"specific visible object name\",
  \"point_yx_norm\": [y, x],
  \"bbox_yxyx_norm\": [y_min, x_min, y_max, x_max],
  \"confidence\": 0.0,
  \"reason\": \"short explanation\"
}}

Coordinates must be integers or floats in [0, 1000], in [y, x] order, normalized to
the image. If the target is not visible at all, return {{\"label\": \"none\"}}.
If it is partly hidden by the gripper, use the visible part to locate it.
This is an object-location request; do not judge grasp safety from the image.
Replace every schema placeholder with observations from this image. Name the
actual target in the label field; do not repeat "specific visible object name".
Do not output joint angles, raw qpos, or a real-robot command."""


def normalized_yx_to_pixel(point_yx_norm: Sequence[float], width: int, height: int) -> tuple[int, int]:
    """Convert ER's normalized ``[y, x]`` point to integer ``(x, y)`` pixels."""

    if len(point_yx_norm) != 2:
        raise ValueError("point_yx_norm must contain [y, x]")
    y_norm, x_norm = (float(point_yx_norm[0]), float(point_yx_norm[1]))
    if not 0 <= y_norm <= 1000 or not 0 <= x_norm <= 1000:
        raise ValueError("normalized coordinates must be in [0, 1000]")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x = min(width - 1, max(0, round(x_norm / 1000 * width)))
    y = min(height - 1, max(0, round(y_norm / 1000 * height)))
    return int(x), int(y)


def normalized_bbox_to_xywh(
    bbox_yxyx_norm: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    """Convert normalized ``[y0, x0, y1, x1]`` to a bounded pixel ROI."""

    if len(bbox_yxyx_norm) != 4:
        raise ValueError("bbox_yxyx_norm must contain [y0, x0, y1, x1]")
    y0, x0, y1, x1 = (float(value) for value in bbox_yxyx_norm)
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        raise ValueError("bbox must have ordered coordinates in [0, 1000]")
    px0, py0 = normalized_yx_to_pixel((y0, x0), width, height)
    px1, py1 = normalized_yx_to_pixel((y1, x1), width, height)
    return px0, py0, max(1, px1 - px0), max(1, py1 - py0)


def _json_payload(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    payload = json.loads(cleaned)
    # The official examples often use a one-element JSON array for point or
    # box queries. Accept that shape as well as the stricter object schema used
    # by this adapter.
    if isinstance(payload, list):
        if not payload or not isinstance(payload[0], Mapping):
            raise ValueError("ER response array must contain an object")
        payload = payload[0]
    if not isinstance(payload, Mapping):
        raise ValueError("ER response must be a JSON object or array")
    return payload


def _coord_tuple(value: Any, length: int, field: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain {length} values")
    values = tuple(float(item) for item in value)
    if any(not 0 <= item <= 1000 for item in values):
        raise ValueError(f"{field} values must be in [0, 1000]")
    return values


def parse_proposal(text: str) -> ERProposal:
    """Parse and validate one structured ER response."""

    payload = _json_payload(text)
    label = str(payload.get("label", "none")).strip() or "none"
    point_value = payload.get("point_yx_norm", payload.get("point"))
    point = _coord_tuple(point_value, 2, "point_yx_norm")
    bbox_value = payload.get("bbox_yxyx_norm")
    if bbox_value is None and all(key in payload for key in ("y", "x", "y2", "x2")):
        bbox_value = [payload["y"], payload["x"], payload["y2"], payload["x2"]]
    bbox = _coord_tuple(bbox_value, 4, "bbox_yxyx_norm")
    confidence = payload.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
    reason = str(payload.get("reason", ""))
    return ERProposal(label, point, bbox, confidence, reason)
