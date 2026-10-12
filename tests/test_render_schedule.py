from __future__ import annotations

import unittest

from sandygs.render_schedule import (
    expand_render_schedule,
    resolve_profile_render_window,
)


class RenderScheduleTests(unittest.TestCase):
    def test_expands_fixed_synchronized_and_moving_sequences(self) -> None:
        frames, camera_count = expand_render_schedule(
            {
                "schema_version": 1,
                "camera_path": "generated_orbit",
                "camera_path_frames": 10,
                "sequences": [
                    {
                        "name": "fixed",
                        "type": "fixed_camera",
                        "camera_index": 2,
                        "times": [0.0, 0.1],
                        "debug": [1],
                        "intermediates": "debug",
                    },
                    {
                        "name": "sync",
                        "type": "synchronized_multiview",
                        "camera_indices": [0, 5],
                        "times": [0.2],
                        "debug": "all",
                        "intermediates": "none",
                    },
                    {
                        "name": "moving",
                        "type": "moving_camera",
                        "camera_indices": {"start": 0, "end": 9, "count": 3},
                        "times": {"start": 0.0, "end": 0.2, "count": 3},
                        "debug": [-1],
                    },
                ],
            }
        )
        self.assertEqual(camera_count, 10)
        self.assertEqual(len(frames), 7)
        self.assertEqual(
            [(frame.camera_index, frame.time_seconds) for frame in frames[:2]],
            [(2, 0.0), (2, 0.1)],
        )
        self.assertFalse(frames[0].save_debug)
        self.assertTrue(frames[1].save_debug)
        self.assertTrue(frames[1].save_intermediates)
        self.assertEqual(frames[2].output_group, "sync/time_0000")
        self.assertFalse(frames[2].save_intermediates)
        self.assertEqual(
            [frame.camera_index for frame in frames[-3:]], [0, 4, 9]
        )
        self.assertTrue(frames[-1].save_debug)

    def test_rejects_camera_outside_generated_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds generated path"):
            expand_render_schedule(
                {
                    "schema_version": 1,
                    "camera_path_frames": 3,
                    "sequences": [
                        {
                            "name": "bad",
                            "type": "fixed_camera",
                            "camera_index": 3,
                            "times": [0.0],
                        }
                    ],
                }
            )

    def test_observed_camera_schedule_uses_captured_view_indices(self) -> None:
        spec = {
            "schema_version": 1,
            "camera_path": "observed_cameras",
            "camera_path_frames": 3,
            "sequences": [{
                "name": "captured",
                "type": "synchronized_multiview",
                "camera_indices": [0, 1, 2],
                "times": [0.0, 0.12],
            }],
        }
        frames, count = expand_render_schedule(spec)
        self.assertEqual(count, 3)
        self.assertEqual(len(frames), 6)
        self.assertEqual([(frame.camera_index, frame.time_seconds) for frame in frames[:3]],
                         [(0, 0.0), (1, 0.0), (2, 0.0)])
        with self.assertRaisesRegex(ValueError, "observed camera list"):
            expand_render_schedule({**spec, "camera_path_frames": 2})
        with self.assertRaisesRegex(ValueError, "does not use"):
            expand_render_schedule({**spec, "camera_path_azimuth_degrees": [0.0, 360.0]})

    def test_validates_optional_azimuth_override(self) -> None:
        frames, camera_count = expand_render_schedule(
            {
                "schema_version": 1,
                "camera_path_frames": 361,
                "camera_path_azimuth_degrees": [80.0, 440.0],
                "sequences": [
                    {
                        "name": "full_orbit",
                        "type": "synchronized_multiview",
                        "camera_indices": [0, 180, 360],
                        "times": [0.0],
                    }
                ],
            }
        )
        self.assertEqual(camera_count, 361)
        self.assertEqual([frame.camera_index for frame in frames], [0, 180, 360])

        with self.assertRaisesRegex(ValueError, "must be increasing"):
            expand_render_schedule(
                {
                    "schema_version": 1,
                    "camera_path_frames": 3,
                    "camera_path_azimuth_degrees": [260.0, 80.0],
                    "sequences": [
                        {
                            "name": "bad_azimuth",
                            "type": "fixed_camera",
                            "camera_index": 0,
                            "times": [0.0],
                        }
                    ],
                }
            )

    def test_resolves_explicit_profile_window_independently_of_view_count(self) -> None:
        render = {
            "frame_time_seconds": 0.0,
            "fps": 30.0,
            "render_schedule": {
                "schema_version": 1,
                "camera_path_frames": 180,
                "sequences": [
                    {
                        "name": "regression",
                        "type": "synchronized_multiview",
                        "camera_indices": [0, 60, 120],
                        "times": [0.0, 0.125, 0.25],
                    }
                ],
            },
        }
        plan = resolve_profile_render_window(
            render, {"duration_seconds": 0.25}, max_views=1
        )
        self.assertEqual(len(plan["frames"]), 9)
        self.assertEqual(plan["first_time_seconds"], 0.0)
        self.assertEqual(plan["last_time_seconds"], 0.25)
        self.assertEqual(plan["camera_path_frames"], 180)

        with self.assertRaisesRegex(ValueError, "ends after"):
            resolve_profile_render_window(
                render, {"duration_seconds": 0.249}, max_views=1
            )

        legacy = resolve_profile_render_window(
            {"frame_time_seconds": 0.001, "fps": 30.0},
            {"steps": 2},
            max_views=3,
        )
        self.assertIsNone(legacy["frames"])
        self.assertEqual(legacy["first_time_seconds"], 0.001)
        self.assertAlmostEqual(
            legacy["last_time_seconds"], 0.001 + 2.0 / 30.0
        )


if __name__ == "__main__":
    unittest.main()
