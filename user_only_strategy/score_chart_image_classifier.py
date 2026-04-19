from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a chart image with a trained classifier")
    p.add_argument("--model", required=True, help="pickle created by train_chart_image_classifier.py")
    p.add_argument("--image", required=True, help="image path to score")
    return p.parse_args()


def load_flattened_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


def main() -> None:
    args = parse_args()
    with Path(args.model).open("rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    image_size = int(payload["image_size"])
    x = load_flattened_image(Path(args.image), image_size)
    pred = int(model.predict(x)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0].tolist()
    else:
        proba = []
    print(json.dumps({"image": args.image, "pred": pred, "proba": proba}, ensure_ascii=False))


if __name__ == "__main__":
    main()
