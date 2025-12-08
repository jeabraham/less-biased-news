# analyze_image_gender.py

import logging, time, requests
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from deepface import DeepFace
from facenet_pytorch import MTCNN
import torch
import cv2
from timing_tracker import get_timing_tracker

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

def load_image(data: bytes):
    """
    Try loading `data` first with PIL, then with OpenCV if PIL fails.
    Returns a RGB PIL Image on success, or None on failure.
    """
    # --- PIL attempt ---
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as e:
        logger.debug(f"[ImageGender] PIL failed to identify image: {e}")
    except Exception as e:
        logger.warning(f"[ImageGender] Unexpected PIL error: {e}")

    # --- OpenCV fallback ---
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if cv_img is None:
            raise ValueError("cv2.imdecode returned None")
        # convert BGR → RGB and to PIL
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception as e:
        logger.warning(f"[ImageGender] OpenCV fallback failed: {e}")
        return None

def analyze_image_gender(image_url: str):
    """
    Download an image, detect faces using MTCNN, and classify each detected face's gender using DeepFace.

    This method performs the following steps:
    1. Downloads the image from the given URL.
    2. Detects faces in the image using MTCNN.
    3. For each detected face, classifies the gender (Male/Woman) using the DeepFace library.
    4. Calculates a prominence score for each face based on its relative size in the image.

    Parameters:
    ----------
    image_url : str
        The URL of the image to analyze.

    Returns:
    --------
    tuple:
        - A list of detected faces (or None if no faces are detected).
        - A tuple containing the image dimensions (width, height) or None if unavailable.
    """
    tracker = get_timing_tracker()
    with tracker.time_task("image_analysis"):
        logger.info(f"[ImageGender] Starting analysis for {image_url}")

        try:
            # 1) Download and load the image
            data = download_image(image_url, timeout=5)
            if not data:
                return None, None  # Return with no faces and dimensions if download fails
            img_pil = load_image(data)
            if img_pil is None:
                logger.info(f"[ImageGender] Unable to parse image, skipping: {image_url}")
                return None, None

            img_np = np.array(img_pil)
            H, W, _ = img_np.shape  # Extract image dimensions
            logger.debug(f"[ImageGender] Downloaded image (W={W}, H={H})")
        except Exception as e:
            logger.warning(f"[ImageGender] Download/Open failed: {e}")
            return None, None

        try:
            # 2) Face detection with MTCNN
            boxes, _ = mtcnn.detect(img_pil)
        except Exception as e:
            logger.warning(f"[ImageGender] Face detection error: {e}")
            return None, (W, H)

        # Handle zero faces
        if boxes is None or len(boxes) == 0:
            logger.info(f"[ImageGender] No faces detected")
            return None, (W, H)

        logger.debug(f"[ImageGender] MTCNN found {len(boxes)} face(s)")

        faces = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            w, h = x2 - x1, y2 - y1

            # Ignore faces that are too small
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
            man_score = raw.get("Man", 0.0)
            woman_score = raw.get("Woman", 0.0)
            total = man_score + woman_score

            if total > 0:
                if woman_score > man_score:
                    gender = "Woman"
                    confidence = woman_score / total
                else:
                    gender = "Man"
                    confidence = man_score / total
            else:
                gender = rec.get("dominant_gender", "Unknown")
                confidence = rec.get("gender_confidence", 0.0)

            prominence = (w * h) / (W * H) if (W * H) else 0.0

            face_rec = {
                "region": (x1, y1, w, h),
                "raw_scores": {"Man": man_score, "Woman": woman_score},
                "gender": gender,
                "confidence": confidence,
                "prominence": prominence
            }
            logger.debug(f"[ImageGender] Face {i}: {face_rec}")
            faces.append(face_rec)

        logger.info(f"[ImageGender] Analysis complete: {len(faces)} valid face(s)")
        return faces, (W, H)
