import pytest

from mani_gemini.er_client import (
    build_prompt,
    normalized_bbox_to_xywh,
    normalized_yx_to_pixel,
    parse_proposal,
)


def test_parse_and_project_point():
    proposal = parse_proposal(
        '{"label":"card","point_yx_norm":[250,500],"bbox_yxyx_norm":[100,400,400,600],"confidence":0.9}'
    )
    assert proposal.label == "card"
    assert normalized_yx_to_pixel(proposal.point_yx_norm, 640, 480) == (320, 120)
    assert normalized_bbox_to_xywh(proposal.bbox_yxyx_norm, 640, 480) == (256, 48, 128, 144)


def test_markdown_fence_is_accepted():
    assert parse_proposal("```json\n{\"label\":\"none\"}\n```").label == "none"


def test_official_array_point_shape_is_accepted():
    proposal = parse_proposal('[{"point":[100,200],"label":"card"}]')
    assert proposal.label == "card"
    assert proposal.point_yx_norm == (100.0, 200.0)


def test_bad_coordinate_is_rejected():
    with pytest.raises(ValueError):
        parse_proposal('{"label":"card","point_yx_norm":[-1,500]}')


def test_reversed_bbox_is_rejected():
    with pytest.raises(ValueError):
        normalized_bbox_to_xywh((500, 500, 400, 600), 640, 480)


def test_prompt_mentions_yx_order_and_no_actuation():
    prompt = build_prompt("pick the card")
    assert "[y, x]" in prompt
    assert "raw qpos" in prompt
