"""
test_predictions.py
-------------------
Tests the running web app with real images from the dataset
and prints a prediction report.

Usage:
    python scripts/test_predictions.py
"""

import sys, os, io, random, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import requests
from PIL import Image

APP_URL  = "http://localhost:8080/predict"
TEST_DIR = Path("src/data/Testing Data/Testing Data")

CLASS_NAMES = [
    "Beetle", "Butterfly", "Cat", "Cow", "Dog",
    "Elephant", "Gorilla", "Hippo", "Lizard", "Monkey",
    "Mouse", "Panda", "Spider", "Tiger", "Zebra",
]

def predict_image(img_path: Path) -> dict:
    with open(img_path, "rb") as f:
        resp = requests.post(APP_URL, files={"file": (img_path.name, f, "image/jpeg")})
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return resp.json()

def main():
    print("\n" + "=" * 60)
    print("  Animal Classifier — Live Prediction Test")
    print("=" * 60)

    correct = 0
    total   = 0
    rows    = []

    for cls in CLASS_NAMES:
        cls_dir = TEST_DIR / cls
        if not cls_dir.exists():
            continue
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + \
                 list(cls_dir.glob("*.png"))
        if not images:
            continue

        # Pick 1 random image per class
        img_path = random.choice(images)
        try:
            result = predict_image(img_path)
        except Exception as e:
            print(f"  ERROR for {cls}: {e}")
            continue

        top_pred  = result["top_prediction"]
        top_conf  = result["confidence"] * 100
        top3      = [p["class"] for p in result["predictions"][:3]]
        is_correct = top_pred.lower() == cls.lower()
        in_top3    = cls.lower() in [c.lower() for c in top3]

        mark = "✓" if is_correct else ("~" if in_top3 else "✗")
        rows.append((cls, top_pred, top_conf, top3, mark))

        if is_correct:
            correct += 1
        total += 1

    # Print table
    print(f"\n  {'True Class':<12} {'Predicted':<12} {'Conf%':>6}  {'Top-3':<35} {'OK?'}")
    print("  " + "-" * 75)
    for cls, pred, conf, top3, mark in rows:
        top3_str = ", ".join(top3)
        print(f"  {cls:<12} {pred:<12} {conf:>5.1f}%  {top3_str:<35} {mark}")

    print("\n" + "=" * 60)
    print(f"  Accuracy on this sample : {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  ✓ = correct   ~ = in top-3   ✗ = wrong")
    print("=" * 60)

    # Also show full evaluation report if it exists
    report = Path("outputs/reports/evaluation_report_test.txt")
    if report.exists():
        print("\n  Full Test Set Evaluation Report:")
        print("  " + "-" * 55)
        for line in report.read_text().splitlines():
            print("  " + line)

if __name__ == "__main__":
    main()
