import argparse
import json
import logging
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import requests
from PIL import Image

# before importing deepface turn off debugging problem
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # optional
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["KERAS_BACKEND"] = "tensorflow"
#
# #slower, for debugging
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)

# Environment variable used to prevent recursive subprocess spawning.
# When set, analyze_image_gender() calls DeepFace directly instead of
# delegating to a subprocess.
_WORKER_ENV_KEY = "_DEEPFACE_WORKER"

# Timeout (seconds) for the DeepFace worker subprocess.
_SUBPROCESS_TIMEOUT = 120

FEMALE_CONFIDENCE_THRESHOLD = 0.6
MIN_FACE_WIDTH_RATIO       = 0.05

ImageSource = Union[str, Path, np.ndarray]


# -----------------------------
# Image loading with safety
# -----------------------------
def _load_image(source: ImageSource) -> np.ndarray:
    """Load an image and return RGB numpy array. Raises on failure."""

    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    if isinstance(source, (str, Path)):
        src = str(source)

        # Remote URL
        if src.startswith("http://") or src.startswith("https://"):
            logger.info("[ImageGender] Downloading image from URL: %s", src)
            resp = requests.get(src, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return np.array(img)

        # Local path
        logger.info("[ImageGender] Loading image from file: %s", src)
        img = Image.open(src).convert("RGB")
        return np.array(img)

    raise TypeError(f"Unsupported image source type: {type(source)!r}")


# -----------------------------
# DeepFace gender analysis
# -----------------------------
def _run_deepface_gender(np_img: np.ndarray) -> List[Dict[str, Any]]:
    """Run DeepFace.analyze exactly like the working tests."""
    from deepface import DeepFace  # imported late to avoid top-level TF init

    logger.info("[ImageGender] Calling DeepFace.analyze(actions=['gender'], enforce_detection=False)")

    result = DeepFace.analyze(
        np_img,
        actions=["gender"],
        enforce_detection=False,
    )

    # Normalizing for single vs multi-face return
    if isinstance(result, dict):
        return [result]
    return list(result)


# -----------------------------
# Subprocess isolation wrapper
# -----------------------------
def _analyze_via_subprocess(source_str: str):
    """
    Run DeepFace analysis in a fully isolated child process.

    DeepFace/TensorFlow can cause a segmentation fault (SIGSEGV) that kills
    the entire Python process and cannot be caught with try/except.  By
    delegating the work to a subprocess we ensure that a crash only terminates
    the child; the parent receives a non-zero exit code and can return a safe
    fallback value instead of dying.
    """
    # Derive the effective log level name for the child process.
    effective_level = logging.getLevelName(logger.getEffectiveLevel())
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        source_str,
        "--log-level", effective_level,
    ]
    # Set the worker flag so the child calls DeepFace directly.
    env = {**os.environ, _WORKER_ENV_KEY: "1"}

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("[ImageGender] Subprocess timed out for %s", source_str)
        return None, None
    except Exception as exc:
        logger.warning("[ImageGender] Subprocess launch error: %s", exc)
        return None, None

    if proc.returncode != 0:
        # A negative exit code means the process was killed by a signal
        # (e.g. -11 == SIGSEGV).  A positive non-zero code indicates a
        # handled error inside the child.
        logger.warning(
            "[ImageGender] Subprocess exited with code %d for %s",
            proc.returncode,
            source_str,
        )
        if proc.stderr:
            logger.debug("[ImageGender] Subprocess stderr: %s", proc.stderr[:500])
        return None, None

    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[ImageGender] Could not parse subprocess output: %s", exc)
        return None, None

    faces = data.get("faces")
    dims = data.get("dimensions") or {}
    w = dims.get("width")
    h = dims.get("height")
    dimensions = (w, h) if w is not None and h is not None else None
    return faces, dimensions


# -----------------------------
# Main analysis wrapper
# -----------------------------
def analyze_image_gender(source: ImageSource):
    """Return (faces, (width, height)) or (None, None) on failure.

    When *source* is a URL or file path and we are not already running inside
    a worker subprocess, the analysis is delegated to an isolated child
    process.  This prevents a segmentation fault inside DeepFace/TensorFlow
    from killing the caller's process.
    """

    # Delegate to subprocess for URL/path sources to isolate segfaults.
    # Skip when we are already the worker (avoid infinite recursion) or
    # when the caller passed a numpy array (can't be serialised cheaply).
    if not os.environ.get(_WORKER_ENV_KEY) and isinstance(source, (str, Path)):
        return _analyze_via_subprocess(str(source))

    try:
        np_img = _load_image(source)
    except Exception as e:
        logger.warning("[ImageGender] Image load failed: %s", e)
        return None, None

    height, width = np_img.shape[:2]

    try:
        faces_raw = _run_deepface_gender(np_img)
    except Exception as e:
        logger.warning("[ImageGender] DeepFace failed: %s", e)
        return None, (width, height)

    # Log how many faces were detected
    logger.info("[ImageGender] Detected %d face(s)", len(faces_raw))

    processed_faces: List[Dict[str, Any]] = []

    for idx, f in enumerate(faces_raw):
        gender_scores = f.get("gender") or {}
        dominant_gender = f.get("dominant_gender")
        dominant_score = None

        if dominant_gender and isinstance(gender_scores, dict):
            dominant_score = gender_scores.get(dominant_gender)

        region = f.get("region") or {}

        # Build a normalized face record matching what female_faces() expects
        face_rec = {
            "dominant_gender": dominant_gender,
            "dominant_gender_score": dominant_score,
            "gender_scores": gender_scores,
            "face_confidence": f.get("face_confidence"),
            "region": {
                "x": region.get("x"),
                "y": region.get("y"),
                "w": region.get("w"),
                "h": region.get("h"),
            },
        }

        # Map to expected classifier fields
        face_rec["gender"] = dominant_gender              # e.g. "Woman"
        face_rec["raw_scores"] = gender_scores            # Raw DeepFace gender scores

        # Convert DeepFace's 0–100 score into 0–1 confidence
        if dominant_score is not None:
            face_rec["confidence"] = float(dominant_score) / 100.0
        else:
            face_rec["confidence"] = 0.0

        # Compute prominence (face area ÷ full image area)
        w = face_rec["region"]["w"]
        h = face_rec["region"]["h"]
        if width and height and w and h:
            face_rec["prominence"] = (w * h) / (width * height)
        else:
            face_rec["prominence"] = 0.0

        logger.info(
            "[ImageGender] Face %d: gender=%s score=%.2f confidence=%.2f region=%s",
            idx,
            face_rec["dominant_gender"],
            face_rec["dominant_gender_score"] or 0.0,
            face_rec["face_confidence"] or 0.0,
            face_rec["region"],
        )

        processed_faces.append(face_rec)

    return processed_faces, (width, height)


# -----------------------------
# CLI Wrapper
# -----------------------------
def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze image gender using DeepFace.")
    parser.add_argument("image", help="Image path or URL")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    faces, dims = analyze_image_gender(args.image)
    width, height = dims if dims else (None, None)

    out = {
        "faces": faces,
        "dimensions": {"width": width, "height": height},
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
