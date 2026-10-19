"""Move chips between cells of a fixed chip tray with the Dobot suction cup.

Robot cell coordinates come from three hand-taught corner cells (affine grid).
The D435i colour image tells which cells hold a chip: an empty pocket shows the
white table through its through-hole, a filled pocket is dark. Image cell
positions come from a homography fitted to four clicked corner cells and then
refined on the visible holes.

  chip_tray view                       # live view for aiming the camera
  chip_tray teach --rows 12 --cols 6   # touch 3 corner chips, then pick a park pose
  chip_tray click                      # click the 4 corner cells in the camera image
  chip_tray status                     # occupancy grid + overlay image
  chip_tray move 1,1:5,3 2,6:7,2       # row,col source:destination, taught positions only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .fast_pick_place import DEFAULT_PORT, TAUGHT_TOOL_BIAS_MM, Dobot


DEFAULT_TRAY = Path("reports/chip_tray/tray.json")


# ---------------------------------------------------------------- geometry

# Datasheet joint ranges; J2 >= -5 deg matches the 0x43 alarm seen on this arm.
JOINT_LIMITS_DEG = ((-125.0, 125.0), (-5.0, 90.0), (-15.0, 95.0), (-150.0, 150.0))


def joint_angles(pose) -> tuple[float, float, float, float]:
    """Dobot firmware IK (135/147 mm links, radial tool bias); ValueError if out of range."""
    x, y, z, yaw = pose
    radial = math.hypot(x, y) - TAUGHT_TOOL_BIAS_MM[0]
    distance = math.hypot(radial, z)
    c_shoulder = (distance**2 + 135.0**2 - 147.0**2) / (270.0 * distance)
    c_elbow = (distance**2 - 135.0**2 - 147.0**2) / (2.0 * 135.0 * 147.0)
    if radial <= 0.0 or abs(c_shoulder) > 1.0 or abs(c_elbow) > 1.0:
        raise ValueError(f"({x:.1f}, {y:.1f}, {z:.1f}) is out of reach")
    first = math.degrees(math.atan2(y, x))
    rear = 90.0 - math.degrees(math.atan2(z, radial) + math.acos(c_shoulder))
    front = math.degrees(math.acos(c_elbow)) - 90.0 + rear
    joints = (first, rear, front, (yaw - first + 180.0) % 360.0 - 180.0)
    for index, (angle, (low, high)) in enumerate(zip(joints, JOINT_LIMITS_DEG)):
        if not low <= angle <= high:
            raise ValueError(f"joint {index+1}={angle:.1f} deg outside [{low}, {high}] at ({x:.1f}, {y:.1f}, {z:.1f})")
    # Parallelogram coupling: with the upper arm far forward the forearm may not rise as much.
    # The firmware planner rejects such targets with alarm 0x12 (seen at J2 59, J3 -2.3).
    coupled_low = JOINT_LIMITS_DEG[2][0] + max(rear - 40.0, 0.0)
    if front < coupled_low:
        raise ValueError(f"joint 3={front:.1f} deg below the coupled limit {coupled_low:.1f} "
                         f"(joint 2={rear:.1f}) at ({x:.1f}, {y:.1f}, {z:.1f})")
    return joints


class Grid:
    """Robot xyz of every cell from taught corner cells.

    XY is a least-squares affine map of (row, col). With the four corners
    taught, Z is bilinear between them, so a slightly warped tray still fits.
    """

    def __init__(self, corners) -> None:
        a = np.array([[c["row"], c["col"], 1.0] for c in corners])
        b = np.array([c["pose"][:3] for c in corners])
        if np.linalg.matrix_rank(a) < 3:
            raise ValueError("Taught cells must not lie on one row/column line")
        self.affine = np.linalg.lstsq(a, b, rcond=None)[0]
        z = {(c["row"], c["col"]): c["pose"][2] for c in corners}
        rows, cols = sorted({r for r, _ in z}), sorted({c for _, c in z})
        self.corner_z = z if len(z) == 4 and len(rows) == 2 and len(cols) == 2 else None
        self.rows, self.cols = rows, cols

    def xyz(self, row, col) -> np.ndarray:
        point = np.array([row, col, 1.0]) @ self.affine
        if self.corner_z is not None:
            (r0, r1), (c0, c1), z = self.rows, self.cols, self.corner_z
            u, v = (row - r0) / (r1 - r0), (col - c0) / (c1 - c0)
            point[2] = ((1-u)*(1-v)*z[r0, c0] + (1-u)*v*z[r0, c1] + u*(1-v)*z[r1, c0] + u*v*z[r1, c1])
        return point


def fit_grid(corners) -> Grid:
    return Grid(corners)


def cell_xyz(grid: Grid, row: int, col: int) -> np.ndarray:
    return grid.xyz(row, col)


def plan_moves(start, park, cells, *, safe_z, r, xy_speed, z_speed, grip_ms, release_ms):
    """Queued steps: start -> each (src, dst) chip move -> park.

    XY travel inside the tray happens at ``safe_z``. Legs to and from the park
    pose cross at most 40 mm higher and change height only above the park XY or
    the cell, since lifting high next to the base folds the upper arm back past
    its limit. Contacts are approached vertically.
    """
    high = max(safe_z, min(park[2], safe_z + 40.0))
    start_r = start[3] if len(start) > 3 else r
    park_r = park[3] if len(park) > 3 else r
    steps, last = [], tuple(start[:3])

    def movl(x, y, z, speed, yaw=r):
        nonlocal last
        if math.dist(last, (x, y, z)) > 0.05:
            steps.extend([("speed", speed), ("movl", (float(x), float(y), float(z), float(yaw)))])
            last = (x, y, z)

    movl(start[0], start[1], high, z_speed, start_r)
    for index, (src, dst) in enumerate(cells):
        travel = high if index == 0 else safe_z
        movl(src[0], src[1], travel, xy_speed)
        movl(src[0], src[1], safe_z, z_speed)
        movl(*src, z_speed)
        steps += [("suction", True), ("wait", grip_ms)]
        movl(src[0], src[1], safe_z, z_speed)
        movl(dst[0], dst[1], safe_z, xy_speed)
        movl(*dst, z_speed)
        steps += [("suction", False), ("wait", release_ms)]
        movl(dst[0], dst[1], safe_z, z_speed)
    movl(last[0], last[1], high, z_speed)
    movl(park[0], park[1], high, xy_speed, park_r)
    movl(*park[:3], z_speed, park_r)
    return steps


def check_path(start, steps) -> None:
    """Raise ValueError if any queued MOVL leaves the firmware joint limits."""
    previous = tuple(start[:4])
    for kind, value in steps:
        if kind != "movl":
            continue
        count = max(1, math.ceil(math.dist(previous[:3], value[:3]) / 2.0))
        try:
            for i in range(count + 1):
                joint_angles(tuple(a + (b - a) * i / count for a, b in zip(previous, value)))
        except ValueError as exc:
            raise ValueError(f"MOVL {np.round(previous[:3], 1).tolist()} -> {np.round(value[:3], 1).tolist()} "
                             f"leaves the joint limits ({exc}); move the arm inside its limits first") from exc
        previous = value


# ------------------------------------------------------------------ vision

def grid_points(rows: int, cols: int) -> np.ndarray:
    """(col, row) of every cell, row-major from (1, 1)."""
    return np.array([(c, r) for r in range(1, rows+1) for c in range(1, cols+1)], dtype=float)


def project(h, points) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), h).reshape(-1, 2)


def _local_pitch(h, rows, cols) -> np.ndarray:
    pts = grid_points(rows, cols)
    right = project(h, pts + [1, 0]) - project(h, pts)
    down = project(h, pts + [0, 1]) - project(h, pts)
    return np.minimum(np.linalg.norm(right, axis=1), np.linalg.norm(down, axis=1))


def _bright_mask(gray, h, rows, cols):
    outline = project(h, [(0.5, 0.5), (cols+0.5, 0.5), (cols+0.5, rows+0.5), (0.5, rows+0.5)])
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(outline).astype(np.int32), 255)
    # Far holes are dim in an oblique view, so threshold relative to the black tray.
    dark, light = np.percentile(gray[mask > 0], (40, 99))
    return ((gray > dark + 0.2 * (light - dark)) & (mask > 0)).astype(np.uint8)


def refine_homography(image, h, rows, cols, iterations: int = 3):
    """Snap the grid onto the visible through-holes; returns (H, matched count)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grid = grid_points(rows, cols)
    matched = 0
    for _ in range(iterations):
        bright = _bright_mask(gray, h, rows, cols)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(bright)
        pitch = float(np.median(_local_pitch(h, rows, cols)))
        blobs = centroids[1:][(stats[1:, cv2.CC_STAT_AREA] >= 3)
                              & (stats[1:, cv2.CC_STAT_AREA] <= 0.8 * pitch**2)]
        if len(blobs) < 8:
            break
        predicted = project(h, grid)
        distance = np.linalg.norm(predicted[:, None] - blobs[None], axis=2)
        nearest = distance.argmin(axis=1)
        keep = distance[np.arange(len(grid)), nearest] < 0.35 * pitch
        # One blob per cell: drop blobs claimed by several predictions.
        claimed = np.bincount(nearest[keep], minlength=len(blobs))
        keep &= claimed[nearest] == 1
        if keep.sum() < 8:
            break
        h, _ = cv2.findHomography(grid[keep], blobs[nearest[keep]], cv2.RANSAC, 0.2 * pitch)
        matched = int(keep.sum())
    return h, matched


def occupancy(image, h, rows, cols) -> np.ndarray:
    """Boolean (rows, cols): True where the through-hole is covered by a chip."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright = _bright_mask(gray, h, rows, cols)
    centres, pitch = project(h, grid_points(rows, cols)), _local_pitch(h, rows, cols)
    full = np.zeros(rows * cols, bool)
    for index, ((u, v), p) in enumerate(zip(centres, pitch)):
        radius = max(1, int(round(0.1 * p)))
        x, y = int(round(u)), int(round(v))
        patch = bright[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
        full[index] = patch.size == 0 or patch.mean() < 0.25
    return full.reshape(rows, cols)


def draw_overlay(image, h, rows, cols, full) -> np.ndarray:
    out = image.copy()
    for (u, v), (c, r) in zip(project(h, grid_points(rows, cols)), grid_points(rows, cols)):
        colour = (0, 0, 255) if full[int(r)-1, int(c)-1] else (0, 200, 0)
        cv2.circle(out, (int(round(u)), int(round(v))), 4, colour, -1)
        if r == 1 or c == 1:
            label = str(int(c)) if r == 1 else str(int(r))
            cv2.putText(out, label, (int(u)+5, int(v)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return out


def grid_text(full) -> str:
    rows, cols = full.shape
    lines = ["     " + " ".join(f"{c:>2}" for c in range(1, cols+1))]
    lines += [f"{r:>3}  " + " ".join(" #" if full[r-1, c-1] else " ." for c in range(1, cols+1))
              for r in range(1, rows+1)]
    return "\n".join(lines)


def tray_holes(image, dark_level: int = 50) -> list[np.ndarray]:
    """Contours of bright through-holes enclosed by black tray plastic."""
    dark = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < dark_level).astype(np.uint8)
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    holes = [(parent, c) for c, (_, _, _, parent) in zip(contours, hierarchy[0])
             if parent >= 0 and 5 <= cv2.contourArea(c) <= 600]
    if not holes:
        return []
    # The tray is the dark region enclosing the most holes (not the phone or robot).
    tray = np.bincount([parent for parent, _ in holes]).argmax()
    return [c for parent, c in holes if parent == tray]


def grab_rgb(timeout_s: float = 5.0, skip: int = 2) -> np.ndarray:
    """One fresh colour frame from the running RealSense node."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("chip_tray_snapshot")
    frames = []

    def on_image(message):
        data = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        frame = data[:, :message.width * 3].reshape(message.height, message.width, 3)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if message.encoding == "rgb8" else frame.copy())

    node.create_subscription(Image, "/camera/camera/color/image_raw", on_image, qos_profile_sensor_data)
    deadline = time.monotonic() + timeout_s
    try:
        while len(frames) <= skip:
            if time.monotonic() > deadline:
                raise RuntimeError("No image on /camera/camera/color/image_raw; is the D435i launch running?")
            rclpy.spin_once(node, timeout_sec=0.1)
        return frames[-1]
    finally:
        node.destroy_node()


# ---------------------------------------------------------------- commands

def load_tray(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_tray(path: Path, tray: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tray, indent=2) + "\n", encoding="utf-8")


def _open_robot(port) -> Dobot:
    bot = Dobot(port, accel=100.0)
    bias = bot.tool_bias()
    if math.dist(bias, TAUGHT_TOOL_BIAS_MM) > 0.5:
        bot.close()
        raise SystemExit(f"Tool bias {bias} != {TAUGHT_TOOL_BIAS_MM}; reconfigure the suction tool first")
    return bot


def _ask_pose(bot, prompt) -> list[float]:
    input(prompt)
    return [round(v, 3) for v in bot.pose()[:4]]


def corner_report(corners) -> tuple[float, float, float, float]:
    """(XY corner mismatch mm, row pitch mm, column pitch mm, axis angle deg).

    Four true corners of a rectangular grid satisfy P11 + Pmn = P1n + Pm1 in XY,
    so a large mismatch means one corner was touched in a neighbouring cell.
    Which one cannot be told from four points alone. Z is left out: a warped
    tray or uneven touch pressure is absorbed by the bilinear height map.
    """
    pose = {(c["row"], c["col"]): np.array(c["pose"][:3]) for c in corners}
    (r0, c0), (r1, c1) = min(pose), max(pose)
    mismatch = float(np.linalg.norm((pose[r0, c0] + pose[r1, c1] - pose[r0, c1] - pose[r1, c0])[:2]))
    grid = fit_grid(corners)
    row_axis = cell_xyz(grid, 2, 1) - cell_xyz(grid, 1, 1)
    col_axis = cell_xyz(grid, 1, 2) - cell_xyz(grid, 1, 1)
    angle = float(np.degrees(np.arccos(row_axis @ col_axis / np.linalg.norm(row_axis) / np.linalg.norm(col_axis))))
    return mismatch, float(np.linalg.norm(row_axis)), float(np.linalg.norm(col_axis)), angle


def unreachable_cells(tray, clearance: float = 10.0) -> list[list[int]]:
    """Cells whose contact or lifted pose violates the firmware joint limits."""
    grid, r = fit_grid(tray["corners"]), tray["corners"][0]["pose"][3]
    bad = []
    for row in range(1, tray["rows"] + 1):
        for col in range(1, tray["cols"] + 1):
            x, y, z = cell_xyz(grid, row, col)
            try:
                joint_angles((x, y, z, r))
                joint_angles((x, y, z + clearance, r))
            except ValueError:
                bad.append([row, col])
    return bad


def _ask_park(bot) -> list[float]:
    while True:
        park = _ask_pose(bot, "\nMove the arm to a park pose that does not hide the tray from the camera, "
                              "then press Enter: ")
        try:
            joint_angles(park)
            return park
        except ValueError as exc:
            print(f"  Outside the joint limits ({exc}). Keep the upper arm from leaning back: "
                  "move the cup further out from the base or lower, then try again.")


def cmd_park(args) -> None:
    tray = load_tray(args.tray)
    bot = _open_robot(args.port)
    try:
        tray["park"] = _ask_park(bot)
    finally:
        bot.close()
    tray["unreachable"] = unreachable_cells(tray)
    save_tray(args.tray, tray)
    print(f"Saved park pose {tray['park']} to {args.tray}")


def cmd_teach(args) -> None:
    rows, cols = args.rows, args.cols
    bot = _open_robot(args.port)
    corners = []
    try:
        for row, col in ((1, 1), (1, cols), (rows, cols), (rows, 1)):
            pose = _ask_pose(bot, f"\nPut a chip in row {row}, col {col}. Hold Unlock, rest the suction cup on "
                                  "the centre of the chip top, release Unlock, then press Enter: ")
            corners.append({"row": row, "col": col, "pose": pose})
            print(f"  row {row}, col {col}: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")
        while True:
            mismatch, pitch_row, pitch_col, angle = corner_report(corners)
            print(f"\nRow pitch {pitch_row:.2f} mm, column pitch {pitch_col:.2f} mm, axis angle {angle:.1f} deg, "
                  f"corner mismatch {mismatch:.1f} mm")
            if mismatch <= 4.0:
                break
            labels = ", ".join(f"{i+1}=({c['row']},{c['col']})" for i, c in enumerate(corners))
            answer = input(f"The corners do not form a rectangle: one was probably touched in a neighbouring "
                           f"cell. Re-teach which corner? {labels}, or k to keep: ").strip().lower()
            if answer == "k":
                break
            if answer not in {str(i+1) for i in range(len(corners))}:
                continue
            corner = corners[int(answer) - 1]
            corner["pose"] = _ask_pose(bot, f"Rest the cup on the chip in row {corner['row']}, col {corner['col']}, "
                                            "then press Enter: ")
        park = _ask_park(bot)
    finally:
        bot.close()
    tray = load_tray(args.tray) if args.tray.exists() else {}
    tray.update(rows=rows, cols=cols, corners=corners, park=park)
    tray["unreachable"] = unreachable_cells(tray)
    save_tray(args.tray, tray)
    print(f"Saved {args.tray}; park {park[:3]}")
    if tray["unreachable"]:
        print(f"Cells outside the joint limits (never used): {tray['unreachable']}")


def cmd_view(args) -> None:
    """Live colour view for aiming the camera: more and larger holes is better."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("chip_tray_view")
    latest = {}

    def on_image(message):
        data = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
        frame = data[:, :message.width * 3].reshape(message.height, message.width, 3)
        latest["frame"] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if message.encoding == "rgb8" else frame.copy()

    node.create_subscription(Image, "/camera/camera/color/image_raw", on_image, qos_profile_sensor_data)
    window = "D435i live - aim at the chip tray (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    shown = False
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            if "frame" in latest:
                frame = latest.pop("frame")
                view = frame.copy()
                holes = tray_holes(frame)
                cv2.drawContours(view, holes, -1, (0, 255, 0), 1)
                size = np.median([math.sqrt(cv2.contourArea(c)) for c in holes]) if holes else 0.0
                h, w = view.shape[:2]
                cv2.line(view, (w//2, 0), (w//2, h), (255, 255, 0), 1)
                cv2.line(view, (0, h//2), (w, h//2), (255, 255, 0), 1)
                cv2.putText(view, f"holes {len(holes)}   median hole size {size:.1f} px", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window, view)
                shown = True
            key = cv2.waitKey(1) & 0xFF
            # GTK reports the window property only after the first imshow.
            if key in (ord("q"), 27) or (shown and cv2.getWindowProperty(window, cv2.WND_PROP_AUTOSIZE) < 0):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


def cmd_click(args) -> None:
    tray = load_tray(args.tray)
    rows, cols = tray["rows"], tray["cols"]
    order = [(1, 1), (1, cols), (rows, cols), (rows, 1)]
    image = grab_rgb()
    if args.corners_px:
        clicks = [tuple(map(float, item.split(","))) for item in args.corners_px]
    else:
        clicks, window = [], "click corner cells (hole centres)"

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append((float(x), float(y)))

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_mouse)
        while len(clicks) < 4:
            view = image.copy()
            for point in clicks:
                cv2.circle(view, tuple(map(int, point)), 4, (0, 0, 255), -1)
            row, col = order[len(clicks)]
            cv2.putText(view, f"click row {row}, col {col}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(window, view)
            if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                raise SystemExit("Cancelled")
        cv2.destroyWindow(window)
    if len(clicks) != 4:
        raise SystemExit("Need exactly 4 corner points")
    h0 = cv2.getPerspectiveTransform(np.float32([(c, r) for r, c in order]), np.float32(clicks))
    h, matched = refine_homography(image, h0, rows, cols)
    tray["image_homography"] = h.tolist()
    save_tray(args.tray, tray)
    full = occupancy(image, h, rows, cols)
    overlay_path = args.tray.parent / "click_overlay.png"
    cv2.imwrite(str(overlay_path), draw_overlay(image, h, rows, cols, full))
    print(f"Homography refined on {matched} visible holes; overlay {overlay_path}")
    print(grid_text(full))


def bright_wall_fraction(image, h, rows, cols) -> float:
    """Share of in-image wall crossings that are not black plastic.

    The hole lattice is periodic, so a grid shifted by whole cells still lands
    on holes; its outer walls then fall on the table instead of the tray.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    walls = np.round(project(h, [(c + 0.5, r + 0.5) for r in range(rows + 1) for c in range(cols + 1)])).astype(int)
    inside = (walls[:, 0] >= 0) & (walls[:, 0] < gray.shape[1]) & (walls[:, 1] >= 0) & (walls[:, 1] < gray.shape[0])
    return float(bright[walls[inside, 1], walls[inside, 0]].mean())


def locate_grid(tray, image) -> np.ndarray:
    """Stored grid re-fitted to this frame; refuses if the camera moved too far."""
    rows, cols = tray["rows"], tray["cols"]
    h, matched = refine_homography(image, np.array(tray["image_homography"]), rows, cols)
    if matched < 20 or bright_wall_fraction(image, h, rows, cols) > 0.05:
        raise SystemExit("The tray grid no longer fits the camera image (camera or tray moved). "
                         "Run: ros2 run mani_dobot chip_tray click")
    return h


def cmd_status(args) -> np.ndarray:
    tray = load_tray(args.tray)
    rows, cols = tray["rows"], tray["cols"]
    image = grab_rgb()
    h = locate_grid(tray, image)
    tray["image_homography"] = h.tolist()  # follow small drifts
    save_tray(args.tray, tray)
    full = occupancy(image, h, rows, cols)
    path = args.tray.parent / "status.png"
    cv2.imwrite(str(path), draw_overlay(image, h, rows, cols, full))
    print(grid_text(full))
    print(f"(# = chip, . = empty)  overlay {path}")
    return full


def parse_move(text: str):
    try:
        src, dst = text.split(":")
        return tuple(map(int, src.split(","))), tuple(map(int, dst.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Use row,col:row,col, got {text!r}") from exc


def cmd_move(args) -> None:
    tray = load_tray(args.tray)
    rows, cols = tray["rows"], tray["cols"]
    for src, dst in args.moves:
        for row, col in (src, dst):
            if not (1 <= row <= rows and 1 <= col <= cols):
                raise SystemExit(f"Cell {row},{col} is outside the {rows}x{cols} tray")
    if args.camera:
        full = cmd_status(args).copy()
        for src, dst in args.moves:
            if not full[src[0]-1, src[1]-1]:
                raise SystemExit(f"No chip at {src[0]},{src[1]}")
            if full[dst[0]-1, dst[1]-1]:
                raise SystemExit(f"Destination {dst[0]},{dst[1]} is occupied")
            full[src[0]-1, src[1]-1], full[dst[0]-1, dst[1]-1] = False, True
        expected = full

    unreachable = {tuple(cell) for cell in tray.get("unreachable", [])}
    for cell in (cell for move in args.moves for cell in move):
        if tuple(cell) in unreachable:
            raise SystemExit(f"Cell {cell[0]},{cell[1]} is outside the arm's joint limits")
    grid = fit_grid(tray["corners"])
    corner_z = [cell_xyz(grid, r, c)[2] for r in (1, rows) for c in (1, cols)]
    safe_z = max(corner_z) + args.clearance
    r = tray["corners"][0]["pose"][3]
    deeper = np.array([0.0, 0.0, args.pick_depth])
    cells = [(cell_xyz(grid, *src) - deeper, cell_xyz(grid, *dst)) for src, dst in args.moves]
    bot = _open_robot(args.port)
    bot.accel = args.accel
    try:
        start = bot.pose()[:4]
        steps = plan_moves(start[:3], tray["park"], cells, safe_z=safe_z, r=r,
                           xy_speed=args.speed, z_speed=args.z_speed,
                           grip_ms=args.grip_ms, release_ms=args.release_ms)
        check_path(start, steps)
    except BaseException as exc:
        bot.close()
        raise SystemExit(str(exc)) if isinstance(exc, ValueError) else exc
    try:
        codes = bot.alarms()
        if codes:
            print(f"Clearing latched alarms {[hex(c) for c in codes]}")
            bot.clear_alarms()
        bot.start()
        started = time.monotonic()
        bot.run(steps)
        print(f"{len(cells)} chip move(s) done in {time.monotonic() - started:.1f} s")
    except BaseException:
        bot.stop_and_clear()
        print("Stopped and cleared the motion queue; suction left as is.")
        raise
    finally:
        bot.close()
    if args.camera:
        time.sleep(0.3)
        after = cmd_status(args)
        wrong = [f"{r+1},{c+1}" for r, c in zip(*np.nonzero(after != expected))]
        print("Camera check: OK" if not wrong else f"Camera check: unexpected cells {wrong}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tray", type=Path, default=DEFAULT_TRAY)
    parser.add_argument("--port", default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="command", required=True)
    teach = sub.add_parser("teach", help="touch three corner chips and record a park pose")
    teach.add_argument("--rows", type=int, required=True)
    teach.add_argument("--cols", type=int, required=True)
    click = sub.add_parser("click", help="click the four corner cells in the camera image")
    click.add_argument("--corners-px", nargs=4, metavar="U,V",
                       help="skip the window: pixels of (1,1) (1,cols) (rows,cols) (rows,1)")
    sub.add_parser("park", help="record only the park pose")
    sub.add_parser("view", help="live camera view for aiming at the tray")
    sub.add_parser("status", help="print which cells hold a chip")
    move = sub.add_parser("move", help="move chips: row,col:row,col ...")
    move.add_argument("moves", nargs="+", type=parse_move)
    move.add_argument("--speed", type=float, default=100.0, help="horizontal mm/s")
    move.add_argument("--z-speed", type=float, default=50.0, help="vertical mm/s")
    move.add_argument("--accel", type=float, default=150.0, help="mm/s^2")
    move.add_argument("--clearance", type=float, default=10.0, help="lift above the chip tops, mm")
    move.add_argument("--pick-depth", type=float, default=3.0,
                      help="go this many mm below the taught chip top when picking up")
    move.add_argument("--grip-ms", type=int, default=300)
    move.add_argument("--release-ms", type=int, default=300)
    move.add_argument("--camera", action="store_true", help="check occupancy with the D435i before and after")
    args = parser.parse_args(argv)
    commands = {"teach": cmd_teach, "park": cmd_park, "view": cmd_view, "click": cmd_click, "status": cmd_status, "move": cmd_move}
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
