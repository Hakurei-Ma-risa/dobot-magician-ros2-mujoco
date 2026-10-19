from mani_dobot.sensor_check import COLOUR, DI, IR, changes, read_all


class FakeBot:
    def __init__(self, colour=b"\x00\x00\x00", ir=b"\x00", high=()):
        self.colour, self.ir, self.high = colour, ir, set(high)

    def send(self, command_id, params=b"", **_):
        if command_id == COLOUR:
            return self.colour
        if command_id == IR:
            return self.ir
        if command_id == DI:
            return bytes((params[0], params[0] in self.high))
        raise AssertionError(command_id)


def test_read_all_names_colours_and_levels():
    values = read_all([("1号", FakeBot(colour=b"\x01\x00\x00", ir=b"\x01", high={5}))])
    assert values["1号 GP2 颜色"] == "红"
    assert values["1号 GP4 红外/开关"] == "01"
    assert values["1号 EIO5 电平"] == "01" and values["1号 EIO6 电平"] == "00"
    assert len(values) == 22


def test_changes_lists_only_channels_that_moved():
    baseline = read_all([("2号", FakeBot())])
    pressed = read_all([("2号", FakeBot(high={11}, colour=b"\x00\x00\x01"))])
    assert changes(baseline, pressed) == ["2号 GP2 颜色: 无 → 蓝", "2号 EIO11 电平: 00 → 01"]
    assert changes(baseline, baseline) == []
