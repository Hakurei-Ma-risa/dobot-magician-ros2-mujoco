"""Gemini Robotics ER image request shared by the offline and ROS probes."""

from __future__ import annotations

from pathlib import Path

from .er_client import ERProposal, build_prompt, parse_proposal


def query_image(
    image_path: Path,
    task: str,
    *,
    model: str,
    api_key: str,
) -> ERProposal:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("install the optional client with: python -m pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(image_path))
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "image", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
            {"type": "text", "text": build_prompt(task)},
        ],
    )
    raw = getattr(interaction, "output_text", None)
    if not raw:
        raise ValueError("Gemini returned no output_text")
    return parse_proposal(raw)


def describe_image(
    image_path: Path,
    question: str,
    *,
    model: str,
    api_key: str,
) -> str:
    """Answer a free-form scene question from the camera image."""

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("install the optional client with: python -m pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(image_path))
    interaction = client.interactions.create(
        model=model,
        input=[
            {"type": "image", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
            {
                "type": "text",
                "text": (
                    "Answer the user's question using only visible information "
                    "in this RGB image. Do not claim to know hidden simulator "
                    "state or exact metric coordinates. User: " + question
                ),
            },
        ],
    )
    return str(getattr(interaction, "output_text", "") or "")
