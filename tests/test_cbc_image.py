"""CBC image URL regressions, with an opt-in real download/DeepFace reproduction."""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import requests

import analyze_image_gender as aig
from test_processing_history import load_pipeline_functions


CBC_IMAGE_URL = (
    "https://i.cbc.ca/ais/0beadf88-d8a7-4a33-9422-b1df70808d17,1793223288722/"
    "full/max/0/default.jpg?im=Crop%2Crect%3D%280%2C269%2C5160%2C2902%29%3BResize%3D620"
)


class CBCImageRegressionTests(unittest.TestCase):
    def test_encoded_url_is_downloaded_unchanged(self):
        image_bytes = BytesIO()
        Image.new("RGB", (620, 349)).save(image_bytes, format="JPEG")
        response = Mock(content=image_bytes.getvalue())
        with patch.object(aig.requests, "get", return_value=response) as get:
            image = aig._load_image(CBC_IMAGE_URL)
        get.assert_called_once_with(CBC_IMAGE_URL, timeout=10)
        response.raise_for_status.assert_called_once()
        self.assertEqual(image.shape, (349, 620, 3))

    def test_dns_failure_has_diagnostics_and_never_calls_deepface(self):
        diagnostics = {}
        with patch.dict(os.environ, {aig._WORKER_ENV_KEY: "1"}), \
                patch.object(aig.requests, "get", side_effect=requests.ConnectionError(
                    "Failed to resolve 'i.cbc.ca'")), \
                patch.object(aig, "_run_deepface_gender") as analyze, \
                patch.object(aig, "get_timing_tracker") as tracker:
            result = aig.analyze_image_gender(CBC_IMAGE_URL, diagnostics=diagnostics)
        self.assertEqual(result, (None, None))
        self.assertEqual(diagnostics["stage"], "image_load")
        self.assertIn("Failed to resolve 'i.cbc.ca'", diagnostics["error"])
        analyze.assert_not_called()
        tracker.return_value.record_error.assert_called_once_with("image_load_error")

    def test_worker_failure_is_visible_to_parent_even_on_zero_exit(self):
        worker_diagnostics = {"stage": "image_load", "error": "ConnectionError: Failed to resolve 'i.cbc.ca'"}
        payload = {"faces": None, "dimensions": {"width": None, "height": None},
                   "diagnostics": worker_diagnostics}
        proc = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="Worker warning")
        diagnostics = {}
        with patch.object(aig.subprocess, "run", return_value=proc) as run, \
                patch.object(aig, "get_timing_tracker") as tracker, \
                self.assertLogs(aig.logger, level="WARNING") as logs:
            result = aig._analyze_via_subprocess(CBC_IMAGE_URL, diagnostics=diagnostics)
        self.assertEqual(result, (None, None))
        self.assertEqual(diagnostics, worker_diagnostics)
        self.assertEqual(run.call_args.args[0][2], CBC_IMAGE_URL)
        self.assertIn("Failed to resolve", logs.output[0])
        tracker.return_value.record_error.assert_called_once_with("image_load_error")

    def test_analysis_failure_is_distinct_from_download_failure(self):
        diagnostics = {}
        with patch.dict(os.environ, {aig._WORKER_ENV_KEY: "1"}), \
                patch.object(aig, "_load_image", return_value=np.zeros((349, 620, 3), dtype=np.uint8)), \
                patch.object(aig, "_run_deepface_gender", side_effect=RuntimeError("Model unavailable")), \
                patch.object(aig, "get_timing_tracker"):
            result = aig.analyze_image_gender(CBC_IMAGE_URL, diagnostics=diagnostics)
        self.assertEqual(result, (None, (620, 349)))
        self.assertEqual(diagnostics["stage"], "face_analysis")
        self.assertIn("Model unavailable", diagnostics["error"])

    def test_worker_cli_includes_original_failure_in_json(self):
        output = StringIO()
        with patch.dict(os.environ, {aig._WORKER_ENV_KEY: "1"}), \
                patch.object(aig.requests, "get", side_effect=requests.ConnectionError("DNS unavailable")), \
                patch.object(aig, "get_timing_tracker"), redirect_stdout(output):
            aig.main([CBC_IMAGE_URL])
        payload = json.loads(output.getvalue())
        self.assertIsNone(payload["faces"])
        self.assertEqual(payload["diagnostics"]["stage"], "image_load")
        self.assertIn("DNS unavailable", payload["diagnostics"]["error"])


@unittest.skipUnless(os.environ.get("RUN_CBC_IMAGE_TEST") == "1",
                     "Set RUN_CBC_IMAGE_TEST=1 for the live CBC download and DeepFace test")
class CBCLiveImageTests(unittest.TestCase):
    def test_real_image_analysis(self):
        source = os.environ.get("CBC_IMAGE_SOURCE", CBC_IMAGE_URL)
        diagnostics = {}
        faces, dimensions = aig.analyze_image_gender(source, diagnostics=diagnostics)
        report = {"url": source, "faces": faces, "dimensions": dimensions, "diagnostics": diagnostics}
        if faces is not None:
            pipeline = load_pipeline_functions("news_filter.py")
            status, prominence = pipeline.female_faces(faces)
            article = {"most_relevant_image": source, "most_relevant_status": status}
            report.update(image_status=status, prominence=prominence,
                          female_confidence_threshold=aig.FEMALE_CONFIDENCE_THRESHOLD,
                          default_max_width_px=pipeline.get_image_size_and_url(article)[0])
        print(json.dumps(report, indent=2))
        self.assertIsNotNone(faces, f"Image processing failed: {diagnostics}")
        self.assertIsNotNone(dimensions)
        self.assertGreater(len(faces), 0, "No faces returned; inspect the detector output")
        for face in faces:
            self.assertIn("raw_scores", face)
            self.assertGreaterEqual(face["confidence"], 0)
            self.assertLessEqual(face["confidence"], 1)


if __name__ == "__main__":
    unittest.main()
