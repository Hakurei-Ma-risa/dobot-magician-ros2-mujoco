from mani_perception.camera_healthcheck import StreamStats


class Stamp:
    sec = 10
    nanosec = 0


class Header:
    stamp = Stamp()


class Image:
    header = Header()
    width = 640
    height = 480
    encoding = "rgb8"


def test_stream_stats_summary() -> None:
    stats = StreamStats()
    message = Image()
    for index in range(3):
        message.header.stamp.sec = 10
        message.header.stamp.nanosec = index * 100_000_000
        stats.add(message, 20.0 + index * 0.1)

    summary = stats.summary()
    assert summary["frames"] == 3
    assert summary["receive_hz"] == 10.0
    assert summary["source_hz"] == 10.0
    assert summary["max_gap_sec"] == 0.1
    assert summary["encoding"] == "rgb8"

