import struct

from mani_dobot.dobot_home import HOME, describe
from mani_dobot.fast_pick_place import Dobot


def test_describe_names_known_and_unknown_alarms():
    assert describe([]) == "无"
    assert describe([0x43, 0x49, 0x77]) == "J2 负向限位（大臂太往后）、平行四边形负向限位（小臂相对大臂太低）、0x77"


def test_home_sets_end_pose_before_queueing_the_home_command():
    bot = Dobot.__new__(Dobot)
    calls = []
    bot.send = lambda cid, params=b"", **kw: calls.append(("send", cid, params)) or b""
    bot._queue = lambda cid, params: calls.append(("queue", cid, params))
    bot.last_index = 7
    bot._wait_index = lambda target, stall_s: calls.append(("wait", target))
    bot._wait_pose = lambda pose, **kw: calls.append(("pose", pose))
    bot.home(HOME)
    assert calls == [("send", 30, struct.pack("<4f", *HOME)), ("queue", 31, struct.pack("<I", 0)),
                     ("wait", 7), ("pose", HOME)]
