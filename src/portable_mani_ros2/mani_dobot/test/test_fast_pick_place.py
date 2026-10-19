import struct

from mani_dobot.fast_pick_place import MAX_PENDING, Dobot, alarm_codes, build_cycle, packet


PICK = (204.8, 69.4, -45.7, 18.7)
PLACE = (207.0, -63.4, -45.3, -17.0)
MOTION = dict(safe_z=-10.0, xy_speed=150.0, z_speed=80.0, grip_ms=400, release_ms=300)


def test_packet_matches_protocol_checksum():
    assert packet(10) == bytes.fromhex("aaaa020a00f6")  # GetPose
    queued = packet(62, b"\x01\x01", write=True, queued=True)
    assert queued[3:5] == bytes((62, 0b11)) and sum(queued[3:]) % 256 == 0


def test_alarm_bits_decode_to_codes():
    state = bytearray(16)
    state[8] = 0b1000  # code 0x43
    state[9] = 0b0001_0010  # codes 0x49, 0x4c
    assert alarm_codes(bytes(state)) == [0x43, 0x49, 0x4C]


def test_cycle_holds_r_and_moves_only_vertically_near_contacts():
    steps = build_cycle((150.0, 0.0, -40.0), PICK, PLACE, **MOTION)
    moves = [value for kind, value in steps if kind == "movl"]
    assert all(pose[3] == PICK[3] for pose in moves)
    assert moves[0] == (150.0, 0.0, -10.0, PICK[3])  # escape upwards before any XY
    for previous, pose in zip(moves, moves[1:]):
        if pose[:2] != previous[:2]:
            assert previous[2] == pose[2] == -10.0  # XY only on the transfer plane
    kinds = [kind for kind, _ in steps]
    on, off = kinds.index("suction"), len(kinds) - 1 - kinds[::-1].index("suction")
    assert steps[on] == ("suction", True) and moves.index((*PICK[:3], PICK[3])) < moves.index((*PLACE[:3], PICK[3]))
    assert steps[off] == ("suction", False) and steps[off + 1] == ("wait", 300)
    assert moves[-1] == (*PLACE[:2], -10.0, PICK[3])


def test_cycle_from_transfer_height_skips_initial_lift():
    steps = build_cycle((207.0, -63.4, -10.0), PICK, PLACE, **MOTION)
    assert steps[:2] == [("speed", 150.0), ("movl", (*PICK[:2], -10.0, PICK[3]))]


class FakeDobot(Dobot):
    """Controller queue that executes one command per index poll."""

    def __init__(self):
        self.accel, self.speed, self.last_index, self.executed = 100.0, None, 0, 0
        self.queued, self.position, self.max_pending = [], (0.0, 0.0, 0.0), 0
        self.done = 0

    def send(self, command_id, params=b"", *, write=False, queued=False):
        if queued:
            self.queued.append((command_id, params))
            self.max_pending = max(self.max_pending, len(self.queued) - self.done)
            return struct.pack("<Q", len(self.queued))
        if command_id == 246:
            if self.done < len(self.queued):
                command, payload = self.queued[self.done]
                if command == 84:
                    self.position = struct.unpack("<B4f", payload)[1:4]
                self.done += 1
            return struct.pack("<Q", self.done)
        if command_id == 20:
            return bytes(16)
        if command_id == 10:
            return struct.pack("<8f", *self.position, 0, 0, 0, 0, 0)
        raise AssertionError(command_id)


def test_run_streams_a_bounded_queue_and_waits_for_the_final_pose():
    bot = FakeDobot()
    steps = build_cycle((150.0, 0.0, -40.0), PICK, PLACE, **MOTION)
    bot.run(steps)
    assert bot.done == len(bot.queued)
    assert bot.max_pending <= MAX_PENDING
    speeds = [struct.unpack("<4f", p)[0] for c, p in bot.queued if c == 81]
    assert all(a != b for a, b in zip(speeds, speeds[1:]))  # no redundant speed commands
    assert bot.queued[-1] == (110, struct.pack("<I", 0))
    assert abs(bot.position[0] - PLACE[0]) < 1e-3 and bot.position[2] == -10.0
