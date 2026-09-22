"""Gemini Robotics ER adapters for simulation experiments."""

from .er_client import (
    ERProposal,
    build_prompt,
    normalized_yx_to_pixel,
    parse_proposal,
)

__all__ = [
    "ERProposal",
    "build_prompt",
    "normalized_yx_to_pixel",
    "parse_proposal",
]
