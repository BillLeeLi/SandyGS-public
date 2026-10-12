#!/usr/bin/env python3
"""Convert an endpoint-sampled height map into a P1 exchange bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sandygs.exchange import validate_bundle
from sandygs.configuration import load_scene_config
from sandygs.prepare import prepare_rainygs_terrain_bundle


def tuple_values(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of length {length}")
    return tuple(float(item) for item in value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height-map", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene, _ = load_scene_config(args.scene_config.resolve())
    renderer = scene["renderer"]

    prepare_rainygs_terrain_bundle(
        source_height_map=args.height_map.resolve(),
        output=args.output.resolve(),
        scene_id=str(scene["scene_id"]),
        domain=tuple_values(renderer["height_map_range"], 4, "height_map_range"),
        rotation_degrees=tuple_values(
            renderer["align_rotation_angles"], 3, "align_rotation_angles"
        ),
        translation=tuple_values(
            renderer["align_translation_vector"], 3, "align_translation_vector"
        ),
        scale=tuple_values(
            renderer.get("align_scale", [1, 1, 1]), 3, "align_scale"
        ),
    )
    metadata = validate_bundle(args.output.resolve())
    print(
        f"validated {args.output.resolve()} "
        f"({metadata['grid']['resolution'][0]}x{metadata['grid']['resolution'][1]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
