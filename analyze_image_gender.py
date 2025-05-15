import requests
import numpy as np
from io import BytesIO
from PIL import Image
from deepface import DeepFace
import logging

logger = logging.getLogger(__name__)

def analyze_image_gender(image_url: str):
    """
    Download an image and run DeepFace to detect all faces and predict gender.
    Returns a list of dicts:
      [ { "region": (x,y,w,h), "gender": "Man"|"Woman", "confidence": 0.95, "prominence": 0.24 }, ... ]
    where 'prominence' is face_area / image_area.
    """
    try:
        resp = requests.get(image_url, timeout=5)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        logger.warning(f"Failed to download or open image: {e}")
        return []

    # run DeepFace analysis (gender only)
    try:
        analysis = DeepFace.analyze(
            img_np,
            actions=["gender"],
            enforce_detection=False,
            detector_backend="opencv"
        )
    except Exception as e:
        logger.warning(f"DeepFace analysis failed: {e}")
        return []

    faces = []
    H, W, _ = img_np.shape
    # DeepFace returns either a single dict or a list
    records = analysis if isinstance(analysis, list) else [analysis]
    for rec in records:
        x, y, w, h = rec["region"]["x"], rec["region"]["y"], rec["region"]["w"], rec["region"]["h"]
        area = w * h
        prominence = (area / (W * H)) if W * H > 0 else 0
        faces.append({
            "region": (x, y, w, h),
            "gender": rec["gender"],          # "Man" or "Woman"
            "confidence": rec.get("gender_confidence", None),
            "prominence": prominence
        })
    return faces
