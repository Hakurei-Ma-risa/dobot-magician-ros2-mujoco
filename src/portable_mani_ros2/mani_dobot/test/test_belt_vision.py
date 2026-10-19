import pytest

from mani_dobot.belt_vision import parse_answer


def test_parse_answer_reads_fenced_json_and_lists():
    text = '```json\n{"block_on_belt": true, "colour": "Red", "box_2d": [100, 200, 300, 400]}\n```'
    assert parse_answer(text) == (True, "red", [100.0, 200.0, 300.0, 400.0])
    assert parse_answer('[{"block_on_belt": true, "colour": "blue", "box_2d": null}]') == (True, "blue", None)


def test_parse_answer_gives_no_colour_when_absent_or_unclear():
    assert parse_answer('{"block_on_belt": false, "colour": "red"}')[:2] == (False, None)
    assert parse_answer('{"block_on_belt": true, "colour": "unknown"}')[:2] == (True, None)
    with pytest.raises(ValueError):
        parse_answer("I see a red cube")
