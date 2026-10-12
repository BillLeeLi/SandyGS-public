from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from sandygs.render_schedule import expand_render_schedule
from scripts.run_isolated_render_schedule import (
    isolated_schedule,
    merge_isolated_frame,
)


class IsolatedRenderScheduleTests(unittest.TestCase):
    def test_one_frame_schedule_preserves_time_camera_and_flags(self) -> None:
        source = {
            "schema_version": 1,
            "camera_path": "generated_orbit",
            "camera_path_frames": 180,
            "camera_path_azimuth_degrees": [80.0, 440.0],
            "sequences": [
                {
                    "name": "sync",
                    "type": "synchronized_multiview",
                    "camera_indices": [60, 120],
                    "times": [0.0, 0.12],
                    "debug": [1],
                    "intermediates": "debug",
                }
            ],
        }
        frames, _ = expand_render_schedule(source)
        schedule = isolated_schedule(frames[-1], source)
        isolated, camera_count = expand_render_schedule(schedule)
        self.assertEqual(camera_count, 180)
        self.assertEqual(schedule["camera_path_azimuth_degrees"], [80.0, 440.0])
        self.assertEqual(len(isolated), 1)
        self.assertEqual(isolated[0].time_seconds, 0.12)
        self.assertEqual(isolated[0].camera_index, 120)
        self.assertTrue(isolated[0].save_debug)
        self.assertTrue(isolated[0].save_intermediates)

    def test_merge_renames_all_frame_products_to_original_output(self) -> None:
        frames, _ = expand_render_schedule(
            {
                "schema_version": 1,
                "camera_path_frames": 180,
                "sequences": [
                    {
                        "name": "sync",
                        "type": "synchronized_multiview",
                        "camera_indices": [120],
                        "times": [0.0, 0.12],
                        "debug": "all",
                    }
                ],
            }
        )
        frame = frames[-1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            isolated = root / "job/render/release_check/isolated_frame"
            (isolated / "renders").mkdir(parents=True)
            (isolated / "sand_debug/sand_owner").mkdir(parents=True)
            (isolated / "regression_raw/render").mkdir(parents=True)
            (isolated / "regression_raw/manifests").mkdir(parents=True)
            (isolated / "renders/00000.png").write_bytes(b"rgb")
            (isolated / "sand_debug/sand_owner/00000.png").write_bytes(b"owner")
            (isolated / "regression_raw/render/00000.npy").write_bytes(b"array")
            (isolated / "regression_raw/manifests/00000.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "frame": "00000",
                        "arrays": [
                            {
                                "key": "render",
                                "relative_path": "render/00000.npy",
                                "sha256": "0" * 64,
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            copied = merge_isolated_frame(root / "job/render", root / "final", frame)
            destination = root / "final/release_check/sync/time_0001"
            self.assertEqual(copied, 4)
            self.assertEqual(
                (destination / "renders/camera_120.png").read_bytes(), b"rgb"
            )
            self.assertEqual(
                (
                    destination
                    / "sand_debug/sand_owner/camera_120.png"
                ).read_bytes(),
                b"owner",
            )
            manifest = yaml.safe_load(
                (
                    destination
                    / "regression_raw/manifests/camera_120.yaml"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["frame"], "camera_120")
            self.assertEqual(
                manifest["arrays"][0]["relative_path"],
                "render/camera_120.npy",
            )


if __name__ == "__main__":
    unittest.main()
