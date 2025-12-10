import argparse
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import requests
from PIL import Image
from deepface import DeepFace

logger = logging.getLogger(__name__)

FEMALE_CONFIDENCE_THRESHOLD = 0.6
MIN_FACE_WIDTH_RATIO       = 0.05

from typing import Union
ImageSource = Union[str, Path, np.ndarray]

def _load_image(source: ImageSource) -> np.ndarray:
    """Load an image from path / URL / numpy array and return an RGB numpy array.

    This mirrors the way tests call DeepFace: they pass a numpy RGB array
    created from a PIL image.
    """

    # Already a numpy array
    if isinstance(source, np.ndarray):
        arr = source
        # Ensure 3 channels
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
            arr = arr[:, :, :3]
        return arr

    # Path or URL
    if isinstance(source, (str, Path)):
        src = str(source)
        if src.startswith("http://") or src.startswith("https://"):
            logger.info("[ImageGender] Downloading image from URL: %s", src)
            resp = requests.get(src, timeout=20)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            path = Path(src)
            logger.info("[ImageGender] Loading image from file: %s", path)
            img = Image.open(path).convert("RGB")

        return np.array(img)

    raise TypeError(f"Unsupported image source type: {type(source)!r}")


def _run_deepface_gender(np_img: np.ndarray) -> List[Dict[str, Any]]:
    """Run DeepFace.analyze for gender on the full image.

    This is intentionally the same call pattern as in tests:
    - Pass a numpy RGB array
    - actions=["gender"]
    - enforce_detection=False
    We rely on DeepFace's own face detection and region handling.
    """

    logger.info("[ImageGender] Calling DeepFace.analyze(actions=['gender'], enforce_detection=False)")

    result = DeepFace.analyze(
        np_img,
        actions=["gender"],
        enforce_detection=False,
    )

    # DeepFace sometimes returns a dict for a single face, or a list for multiple.
    if isinstance(result, dict):
        return [result]
    return list(result)


def analyze_image_gender(source: ImageSource):
    """High-level API used by the rest of the project.

    Returns a tuple: (faces, (width, height))
    """

    np_img = _load_image(source)
    height, width = np_img.shape[:2]

    try:
        faces = _run_deepface_gender(np_img)
    except Exception as e:  # noqa: BLE001 - we want to log and bubble up
        logger.warning("[ImageGender] DeepFace failed: %s", e)
        raise

    processed_faces: List[Dict[str, Any]] = []

    for f in faces:
        gender_scores = f.get("gender") or {}
        dominant_gender = f.get("dominant_gender")
        dominant_score = None
        if dominant_gender and isinstance(gender_scores, dict):
            dominant_score = gender_scores.get(dominant_gender)

        region = f.get("region") or {}
        processed_faces.append(
            {
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
        )

    return processed_faces, (width, height)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze image gender using DeepFace.")
    parser.add_argument("image", help="Image path or URL")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("[ImageGender] Starting analysis for %s", args.image)

    faces, (width, height) = analyze_image_gender(args.image)
    result = {
        "faces": faces,
        "dimensions": {
            "width": width,
            "height": height,
        },
    }
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
