from deepface import DeepFace
from pathlib import Path
import pprint

DEFAULT_IMAGE = Path("test_data/08-10OALC111325-2048x1365.jpg")

def test_deepface(image_path: str | Path = DEFAULT_IMAGE):
    """
    Simple DeepFace test that loads an image and runs DeepFace.analyze.
    Defaults to: test_data/08-01OALC111325-scaled.jpg
    """

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"🔍 Testing DeepFace.analyze on: {image_path}")

    try:
        result = DeepFace.analyze(
            img_path=str(image_path),
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False  # prevents failure if no face found
        )

        print("✅ DeepFace.analyze succeeded.")
        pprint.pprint(result)

    except Exception as e:
        print("❌ DeepFace.analyze failed:")
        print(type(e).__name__, str(e))


if __name__ == "__main__":
    test_deepface()
