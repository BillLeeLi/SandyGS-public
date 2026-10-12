from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from sandygs.storage_guard import (
    StorageGuardError,
    validate_artifact_output,
    validate_artifact_storage,
)


class ArtifactStorageGuardTests(unittest.TestCase):
    def _fixture(self, root: Path, *, volume_id: str = "test-volume") -> Path:
        mounted = root / "mounted"
        artifacts = mounted / "artifacts"
        mounted.mkdir()
        artifacts.mkdir()
        (mounted / ".volume-id").write_text(volume_id + "\n")
        link = root / "artifact-link"
        link.symlink_to(artifacts)
        config = {
            "schema_version": 1,
            "roots": {"artifacts": str(link)},
            "storage_guard": {
                "artifacts": {
                    "enabled": True,
                    # The test tree lives on the real root filesystem, whose
                    # mount identity is stable in every Linux test runner.
                    "mountpoint": "/",
                    "expected_real_path": str(artifacts),
                    "require_symlink": True,
                    "volume_id_file": str(
                        (mounted / ".volume-id").relative_to("/")
                    ),
                    "volume_id": volume_id,
                    "write_probe": True,
                }
            },
        }
        path = root / "paths.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    def test_valid_link_identity_and_write_probe_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            result = validate_artifact_storage(config)
            self.assertTrue(result["enabled"])
            self.assertEqual(result["artifact_root"], str(root / "mounted/artifacts"))

    def test_wrong_volume_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            (root / "mounted/.volume-id").write_text("wrong\n")
            with self.assertRaisesRegex(StorageGuardError, "identity mismatch"):
                validate_artifact_storage(config)

    def test_output_outside_artifacts_does_not_require_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            (root / "mounted/.volume-id").write_text("wrong\n")
            result = validate_artifact_output(config, root / "unrelated")
            self.assertFalse(result["enabled"])

    def test_guard_is_optional_for_server_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "paths.yaml"
            path.write_text(yaml.safe_dump({"roots": {"artifacts": "/tmp/a"}}))
            self.assertFalse(validate_artifact_storage(path)["enabled"])


if __name__ == "__main__":
    unittest.main()
