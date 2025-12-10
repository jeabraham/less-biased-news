import sys
import numpy as np
from PIL import Image, ImageDraw
from deepface import DeepFace
import os

def save_debug_face_crop(np_img, region, label):
    """Save the cropped face and an annotated version of the full image."""

    x, y, w, h = region["x"], region["y"], region["w"], region["h"]

    # Ensure bounds are valid
    H, W = np_img.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)

    # Extract face
    crop = np_img[y:y2, x:x2]
    crop_img = Image.fromarray(crop)
    crop_img.save(f"debug_face_{label}.png")

    # Draw on full image
    full = Image.fromarray(np_img.copy())
    draw = ImageDraw.Draw(full)
    draw.rectangle([x, y, x2, y2], outline="red", width=4)
    full.save(f"debug_box_{label}.png")

    print(f"Saved debug_face_{label}.png and debug_box_{label}.png")


def run_test(path):
    print(f"🔍 Testing DeepFace.analyze on: {path}")

    img = Image.open(path).convert("RGB")
    np_img = np.array(img)

    # --- Full Image ---
    print("\n=== FULL IMAGE ANALYSIS ===")
    out_full = DeepFace.analyze(np_img, actions=["gender"], enforce_detection=False)
    if isinstance(out_full, list):
        out_full = out_full[0]

    print(out_full)
    save_debug_face_crop(np_img, out_full["region"], "full")

    # --- Cropped Image ---
    print("\n=== CROPPED FACE ANALYSIS ===")
    region = out_full["region"]
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    face = np_img[y:y+h, x:x+w]

    out_crop = DeepFace.analyze(face, actions=["gender"], enforce_detection=False)
    if isinstance(out_crop, list):
        out_crop = out_crop[0]

    print(out_crop)
    save_debug_face_crop(face, {"x":0, "y":0, "w":face.shape[1], "h":face.shape[0]}, "crop")


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_data/08-01OALC111325-scaled.jpg"
    run_test(img_path)
