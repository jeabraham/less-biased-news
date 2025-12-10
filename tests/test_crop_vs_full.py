from deepface import DeepFace
from PIL import Image
import numpy as np
from pathlib import Path

IMG = Path("test_data/08-10OALC111325-2048x1365.jpg")

def test_crop_vs_full():
    import pprint

    # 1. Load image
    img = Image.open(IMG).convert("RGB")
    np_full = np.array(img)

    print(f"🔍 Testing DeepFace.analyze on: {IMG}")

    print("\n=== FULL IMAGE ANALYSIS ===")
    full = DeepFace.analyze(np_full, actions=["gender"], enforce_detection=False)
    pprint.pprint(full)

    # 2. Simulate your crop using DeepFace's detected region
    region = full[0]["region"]
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    crop = np_full[y:y+h, x:x+w]

    print("\n=== CROPPED FACE ANALYSIS ===")
    try:
        cropped = DeepFace.analyze(crop, actions=["gender"], enforce_detection=False)
        pprint.pprint(cropped)
    except Exception as e:
        print("DeepFace failed on crop:")
        print(type(e).__name__, str(e))


if __name__ == "__main__":
    test_crop_vs_full()
