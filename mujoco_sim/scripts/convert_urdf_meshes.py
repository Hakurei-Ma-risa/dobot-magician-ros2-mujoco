#!/usr/bin/env python3
"""Convert the upstream Collada visuals to MuJoCo-compatible OBJ parts.

The source DAE files contain many CAD submeshes with a handful of flat
materials.  MuJoCo does not load Collada directly, so this script applies the
Collada scene transforms, groups geometry by source color and exports OBJ
parts.  Collision and inertia remain the simplified MJCF primitives.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS_WORKSPACE = PROJECT_ROOT.parent
SOURCE_DIR = (
    ROS_WORKSPACE
    / "src"
    / "magician_ros2"
    / "dobot_description"
    / "meshes"
    / "dae"
)
OUTPUT_DIR = PROJECT_ROOT / "model" / "meshes"

SOURCE_MESHES = (
    "magicianBase",
    "magicianLink1",
    "magicianLink2",
    "magicianLink3",
    "magicianLink4",
    "gripper_core",
    "jaw_left",
    "jaw_right",
)


def material_color(mesh: trimesh.Trimesh) -> tuple[int, int, int, int]:
    material = getattr(mesh.visual, "material", None)
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        return (180, 180, 180, 255)
    values = np.asarray(color, dtype=np.uint8).reshape(-1)
    if len(values) == 3:
        values = np.append(values, 255)
    return tuple(int(value) for value in values[:4])


def convert(source_name: str) -> list[dict[str, object]]:
    scene = trimesh.load(SOURCE_DIR / f"{source_name}.dae", force="scene")
    grouped: dict[tuple[int, int, int, int], list[trimesh.Trimesh]] = defaultdict(list)

    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        color = material_color(mesh)
        mesh.apply_transform(transform)
        grouped[color].append(mesh)

    manifest_entries: list[dict[str, object]] = []
    for color, meshes in sorted(grouped.items()):
        combined = trimesh.util.concatenate(meshes)
        combined.remove_unreferenced_vertices()
        suffix = "_".join(str(channel) for channel in color[:3])
        filename = f"{source_name}_{suffix}.obj"
        combined.export(OUTPUT_DIR / filename, file_type="obj", include_color=False)
        manifest_entries.append(
            {
                "file": filename,
                "rgba": [round(channel / 255.0, 6) for channel in color],
                "vertices": int(len(combined.vertices)),
                "faces": int(len(combined.faces)),
            }
        )
    return manifest_entries


def main() -> None:
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source mesh directory not found: {SOURCE_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_obj in OUTPUT_DIR.glob("*.obj"):
        old_obj.unlink()

    manifest = {name: convert(name) for name in SOURCE_MESHES}
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
