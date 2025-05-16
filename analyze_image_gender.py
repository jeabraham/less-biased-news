# analyze_image_gender.py

import logging, time, requests
import numpy as np
from io import BytesIO
from PIL import Image
from deepface import DeepFace
from facenet_pytorch import MTCNN
import torch

logger = logging.getLogger(__name__)

FEMALE_CONFIDENCE_THRESHOLD = 0.6
MIN_FACE_WIDTH_RATIO       = 0.05

# Initialize MTCNN once
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn  = MTCNN(keep_all=True, device=device, thresholds=[0.6,0.7,0.7])

import requests
from requests.exceptions import RequestException
from urllib.parse import urlparse

def download_image(url: str, timeout: int = 5) -> bytes | None:
    """
    Try downloading `url` with a browser-like User-Agent and Referer.
    Returns the raw bytes on success, or None on any error.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15"
        ),
        # Some CDNs require a Referer header matching the page that embedded the image:
        "Referer": f"{urlparse(url).scheme}://{urlparse(url).netloc}/",
        "Accept": "image/avif,image/webp,image/apng,*/*;q=0.8"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except RequestException as e:
        logger.warning(f"[ImageGender] Download/Open failed for {url}: {e}")
        return None


def analyze_image_gender(image_url: str):
    """
    Download an image, detect ALL faces with MTCNN,
    then classify each crop with DeepFace.
    Catches and logs detection errors (e.g. empty lists → torch.cat() failures).
    """
    start = time.time()
    logger.info(f"[ImageGender] Starting analysis for {image_url}")

    # 1) Download + load
    try:
        data = download_image(image_url, timeout=5)
        if not data:
            return []  # abort face analysis
        img = Image.open(BytesIO(data)).convert("RGB")
        img_np  = np.array(img)
        H, W, _ = img_np.shape
        logger.debug(f"[ImageGender] Downloaded image in {time.time()-start:.2f}s")
    except Exception as e:
        logger.warning(f"[ImageGender] Download/Open failed: {e}")
        return []

    # 2) Face detection (fully wrapped)
    t0 = time.time()
    try:
        boxes, _ = mtcnn.detect(img)
    except Exception as e:
        # This will catch the torch.cat() error or any MTCNN internals
        logger.warning(f"[ImageGender] Face detection error (took {time.time()-t0:.2f}s): {e}")
        return []
    finally:
        logger.debug(f"[ImageGender] detect() call finished in {time.time()-t0:.2f}s")

    # 3) Handle zero faces
    if boxes is None or len(boxes) == 0:
        logger.info(f"[ImageGender] No faces detected (in {time.time()-t0:.2f}s)")
        return []

    logger.debug(f"[ImageGender] MTCNN found {len(boxes)} face(s)")

    faces = []
    # 4) Classify each face crop
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]
        w, h = x2 - x1, y2 - y1
        # drop tiny detections
        if w < W * MIN_FACE_WIDTH_RATIO:
            logger.debug(f"[ImageGender] Dropped tiny face {i}: w={w}px")
            continue

        face_crop = img_np[y1:y2, x1:x2]
        try:
            od = DeepFace.analyze(
                face_crop,
                actions=["gender"],
                enforce_detection=False,
                detector_backend="mtcnn"
            )
        except Exception as e:
            logger.warning(f"[ImageGender] DeepFace failed on face {i}: {e}")
            continue

        rec = od if isinstance(od, dict) else od[0]
        raw = rec.get("gender", {})
        man_score   = raw.get("Man",   0.0)
        woman_score = raw.get("Woman", 0.0)
        total = man_score + woman_score

        if total > 0:
            if woman_score > man_score:
                gender     = "Woman"
                confidence = woman_score / total
            else:
                gender     = "Man"
                confidence = man_score / total
        else:
            gender     = rec.get("dominant_gender", "Unknown")
            confidence = rec.get("gender_confidence", 0.0)

        prominence = (w * h) / (W * H) if (W * H) else 0.0

        face_rec = {
            "region":      (x1, y1, w, h),
            "raw_scores":  {"Man": man_score, "Woman": woman_score},
            "gender":      gender,
            "confidence":  confidence,
            "prominence":  prominence
        }
        logger.debug(f"[ImageGender] Face {i}: {face_rec}")
        faces.append(face_rec)

    logger.info(f"[ImageGender] Completed in {time.time()-start:.2f}s: {len(faces)} valid face(s)")
    return faces
