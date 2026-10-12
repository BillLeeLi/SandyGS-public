from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from sandygs.existing_sand.workspace import ExistingSandError
from sandygs.render_schedule import expand_render_schedule
from scripts.experiments.existing_sand_e4_demo import (
    _camera_frame_table,
    _demo_settings,
    _encoder_common,
    _find_program,
    build_demo_schedule,
    encode_demo,
)
from scripts.experiments.existing_sand_e4_render import (
    _surface_render_overrides,
)


class ExistingSandE4TextureOverrideTests(unittest.TestCase):
    def texture_overrides(self) -> dict:
        return {
            "sand_texture_enabled": True,
            "sand_texture_diagnostics": True,
            "sand_texture_material": "texturecan_ground_0024_2k",
            "sand_texture_material_mode": "texture_primary",
            "sand_texture_root": "/tmp/material",
            "sand_texture_projection": "simulation_triplanar",
            "sand_texture_triplanar_sharpness": 4.0,
            "sand_texture_phase_mode": "world_locked",
            "sand_texture_period_sim": 0.8,
            "sand_texture_rotation_degrees": 0.0,
            "sand_texture_offset": [0.0, 0.0],
            "sand_texture_primary_albedo_gain": [1.2, 1.2, 1.2],
            "sand_texture_primary_roughness_scale": 1.0,
            "sand_texture_normal_strength": 1.25,
            "sand_texture_normal_highpass_enabled": True,
            "sand_texture_normal_highpass_sigma_texels": 8.0,
            "sand_texture_normal_highpass_strength": 1.0,
            "sand_texture_roughness_min": 0.08,
            "sand_texture_roughness_max": 1.0,
            "sand_texture_lod_bias": -1.0,
            "sand_texture_wet_adjustment_enabled": True,
            "sand_texture_wet_albedo_scale": 0.7,
            "sand_texture_wet_response_rate": 4.0,
            "sand_texture_ao_enabled": False,
        }

    def test_accepts_reviewed_texture_primary_triplanar_fields(self):
        overrides = self.texture_overrides()
        self.assertEqual(
            _surface_render_overrides(
                {"surface_render_overrides": overrides}
            ),
            overrides,
        )

    def test_texture_enable_requires_material_root(self):
        overrides = self.texture_overrides()
        del overrides["sand_texture_root"]
        with self.assertRaisesRegex(ExistingSandError, "sand_texture_root"):
            _surface_render_overrides(
                {"surface_render_overrides": overrides}
            )

    def test_wet_adjustment_requires_texture_primary(self):
        overrides = self.texture_overrides()
        overrides["sand_texture_material_mode"] = "perturbation"
        with self.assertRaisesRegex(ExistingSandError, "texture_primary"):
            _surface_render_overrides(
                {"surface_render_overrides": overrides}
            )


class ExistingSandE4DemoTests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "camera_ids": ["a.jpg", "b.jpg", "c.jpg"],
            "demo": {
                "start_time_seconds": 0.0,
                "end_time_seconds": 3.0,
                "fps": 24.0,
                "diagnostic_times_seconds": [0.0, 0.5, 1.0, 2.0, 3.0],
                "video": {
                    "codec": "libx264",
                    "preset": "slow",
                    "crf": 18,
                    "pixel_format": "yuv420p",
                    "multiview": True,
                },
            },
        }

    def test_schedule_has_dense_rgb_and_sparse_synchronized_diagnostics(self):
        config = self.config()
        settings = _demo_settings(config)
        self.assertEqual(len(settings["times_seconds"]), 73)
        self.assertEqual(settings["diagnostic_indices"], [0, 12, 24, 48, 72])
        schedule = build_demo_schedule(config, settings)
        frames, camera_count = expand_render_schedule(schedule)
        self.assertEqual(camera_count, 3)
        self.assertEqual(len(frames), 219)
        self.assertEqual(sum(frame.save_debug for frame in frames), 15)
        self.assertEqual(sum(frame.save_intermediates for frame in frames), 15)
        for time_index in range(73):
            group = frames[time_index * 3:(time_index + 1) * 3]
            self.assertEqual(
                {frame.time_seconds for frame in group},
                {settings["times_seconds"][time_index]},
            )

    def test_rejects_diagnostic_time_between_video_frames(self):
        config = self.config()
        config["demo"]["diagnostic_times_seconds"] = [0.1]
        with self.assertRaisesRegex(ExistingSandError, "rendered frame"):
            _demo_settings(config)

    def test_camera_frame_table_requires_every_rendered_image(self):
        config = self.config()
        config["demo"]["end_time_seconds"] = 1.0 / 24.0
        config["demo"]["diagnostic_times_seconds"] = [0.0]
        settings = _demo_settings(config)
        schedule = build_demo_schedule(config, settings)
        frames, _ = expand_render_schedule(schedule)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for frame in frames:
                path = (
                    output / "release_check" / frame.output_group / "renders"
                    / f"{frame.output_name}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
            table = _camera_frame_table(output, schedule)
            self.assertEqual(sorted(table), [0, 1, 2])
            self.assertTrue(all(len(values) == 2 for values in table.values()))
            table[0][0][1].unlink()
            with self.assertRaisesRegex(ExistingSandError, "missing"):
                _camera_frame_table(output, schedule)

    def test_encoder_contract_is_fixed_to_reviewed_h264_settings(self):
        settings = _demo_settings(self.config())
        options = _encoder_common(settings)
        self.assertIn("libx264", options)
        self.assertIn("yuv420p", options)
        self.assertIn("bt709", options)

    @unittest.skipIf(_find_program("ffmpeg") is None, "FFmpeg is unavailable")
    def test_encode_writes_three_camera_videos_and_multiview_manifest(self):
        config = self.config()
        config["render_id"] = "demo_test_001"
        config["demo"]["end_time_seconds"] = 1.0 / 24.0
        config["demo"]["diagnostic_times_seconds"] = [0.0]
        settings = _demo_settings(config)
        schedule = build_demo_schedule(config, settings)
        frames, _ = expand_render_schedule(schedule)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            output = root / "visualizations/demo_test_001/water"
            output.mkdir(parents=True)
            (output / "render.yaml").write_text(
                yaml.safe_dump({"render_schedule": schedule}, sort_keys=False),
                encoding="utf-8",
            )
            camera_map = [
                {"camera_index": index, "camera_id": camera_id}
                for index, camera_id in enumerate(config["camera_ids"])
            ]
            (output / "result.yaml").write_text(
                yaml.safe_dump(
                    {
                        "status": "completed_e4_demo_render",
                        "camera_map": camera_map,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            colors = ((220, 60, 60), (60, 220, 60), (60, 60, 220))
            for frame in frames:
                path = (
                    output / "release_check" / frame.output_group / "renders"
                    / f"{frame.output_name}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB", (64, 48), colors[frame.camera_index]
                ).save(path)
            videos = encode_demo(root, config_path)
            manifest = yaml.safe_load(
                (videos / "manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["videos"]), 4)
            self.assertEqual(manifest["frame_count_per_camera"], 2)
            self.assertTrue(all(
                (output / item["path"]).is_file()
                for item in manifest["videos"]
            ))


if __name__ == "__main__":
    unittest.main()
