#!/usr/bin/env python3
"""
Tests for the subprocess isolation in analyze_image_gender.

The key behaviour being tested: when DeepFace crashes (e.g. segfault, any
non-zero exit code), analyze_image_gender() must return (None, None) instead
of letting the crash propagate to the caller.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the repo root is on the path so we can import analyze_image_gender
sys.path.insert(0, str(Path(__file__).parent.parent))

import analyze_image_gender as aig


class TestAnalyzeViaSubprocess(unittest.TestCase):
    """Unit tests for _analyze_via_subprocess()."""

    def _make_proc(self, returncode, stdout="", stderr=""):
        """Create a mock CompletedProcess object."""
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = stderr
        return proc

    def test_segfault_returns_none(self):
        """A subprocess killed by SIGSEGV (exit -11) must return (None, None)."""
        with patch("subprocess.run", return_value=self._make_proc(-11)):
            faces, dims = aig._analyze_via_subprocess("https://example.com/img.jpg")
        self.assertIsNone(faces)
        self.assertIsNone(dims)

    def test_nonzero_exit_returns_none(self):
        """Any non-zero exit code must return (None, None)."""
        with patch("subprocess.run", return_value=self._make_proc(1, stderr="error")):
            faces, dims = aig._analyze_via_subprocess("https://example.com/img.jpg")
        self.assertIsNone(faces)
        self.assertIsNone(dims)

    def test_timeout_returns_none(self):
        """A subprocess that times out must return (None, None)."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(
            cmd=[], timeout=aig._SUBPROCESS_TIMEOUT
        )):
            faces, dims = aig._analyze_via_subprocess("https://example.com/img.jpg")
        self.assertIsNone(faces)
        self.assertIsNone(dims)

    def test_invalid_json_returns_none(self):
        """Garbled stdout (e.g. from a crash dump) must return (None, None)."""
        with patch("subprocess.run", return_value=self._make_proc(0, stdout="not json")):
            faces, dims = aig._analyze_via_subprocess("https://example.com/img.jpg")
        self.assertIsNone(faces)
        self.assertIsNone(dims)

    def test_valid_json_parsed_correctly(self):
        """Valid JSON output is parsed into faces + dimensions."""
        payload = {
            "faces": [{"gender": "Woman", "confidence": 0.9, "prominence": 0.5}],
            "dimensions": {"width": 800, "height": 600},
        }
        with patch("subprocess.run", return_value=self._make_proc(0, stdout=json.dumps(payload))):
            faces, dims = aig._analyze_via_subprocess("https://example.com/img.jpg")
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["gender"], "Woman")
        self.assertEqual(dims, (800, 600))

    def test_missing_dimensions_returns_none_dims(self):
        """When dimensions are missing from the JSON, dims should be None."""
        payload = {"faces": [], "dimensions": {"width": None, "height": None}}
        with patch("subprocess.run", return_value=self._make_proc(0, stdout=json.dumps(payload))):
            faces, dims = aig._analyze_via_subprocess("https://example.com/img.jpg")
        self.assertEqual(faces, [])
        self.assertIsNone(dims)


class TestAnalyzeImageGenderDispatch(unittest.TestCase):
    """Tests that analyze_image_gender() routes correctly."""

    def test_url_dispatches_to_subprocess_by_default(self):
        """A URL source should be sent to _analyze_via_subprocess when not in worker mode."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure worker key is absent
            os.environ.pop(aig._WORKER_ENV_KEY, None)
            with patch.object(aig, "_analyze_via_subprocess", return_value=([], (100, 100))) as mock_sub:
                aig.analyze_image_gender("https://example.com/img.jpg")
                mock_sub.assert_called_once_with("https://example.com/img.jpg")

    def test_path_dispatches_to_subprocess_by_default(self):
        """A file-path source should also be routed through the subprocess."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(aig._WORKER_ENV_KEY, None)
            with patch.object(aig, "_analyze_via_subprocess", return_value=([], (100, 100))) as mock_sub:
                aig.analyze_image_gender("/tmp/test_image.jpg")
                mock_sub.assert_called_once_with("/tmp/test_image.jpg")

    def test_worker_env_bypasses_subprocess(self):
        """When _DEEPFACE_WORKER is set, DeepFace is called directly (no subprocess)."""
        with patch.dict(os.environ, {aig._WORKER_ENV_KEY: "1"}):
            with patch.object(aig, "_analyze_via_subprocess") as mock_sub, \
                 patch.object(aig, "_load_image", side_effect=Exception("load error")):
                # _load_image raises, so analyze_image_gender returns (None, None)
                faces, dims = aig.analyze_image_gender("https://example.com/img.jpg")
                mock_sub.assert_not_called()
                self.assertIsNone(faces)
                self.assertIsNone(dims)


if __name__ == "__main__":
    unittest.main()
