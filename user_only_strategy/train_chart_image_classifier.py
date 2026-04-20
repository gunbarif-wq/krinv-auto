from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a lightweight chart image classifier")
    p.add_argument("--dataset-dir", required=True, help="dataset directory created by build_chart_image_dataset.py")
    p.add_argument("--model-out", required=True, help="output pickle path")
    p.add_argument("--report-out", default="", help="optional report json path")
    p.add_argument("--image-size", type=int, default=96, help="resize image to NxN before training")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def load_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def load_dataset(dataset_dir: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")
    xs: List[np.ndarray] = []
    ys: List[int] = []
    names: List[str] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_path = Path(row["image_path"])
            if raw_path.is_absolute():
                img_path = raw_path
            else:
                img_path = dataset_dir / raw_path
                if not img_path.exists():
                    img_path = dataset_dir.parent / raw_path
            if not img_path.exists():
                continue
            try:
                xs.append(load_image(img_path, image_size))
                ys.append(int(row["label"]))
                names.append(str(img_path))
            except Exception:
                continue
    if not xs:
        raise RuntimeError("no valid images found in dataset")
    return np.vstack(xs), np.array(ys, dtype=int), names


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    model_out = Path(args.model_out)
    report_out = Path(args.report_out) if args.report_out else model_out.with_suffix(".report.json")

    x, y, names = load_dataset(dataset_dir, args.image_size)
    label_counts = {str(k): int(v) for k, v in sorted(Counter(y).items())}

    if len(np.unique(y)) >= 2 and len(y) >= 8:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        pca_dim = max(8, min(64, x_train.shape[0] - 1, x_train.shape[1]))
        candidates = {
            "logistic_raw": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state),
            "pca_logistic": Pipeline(
                [
                    ("pca", PCA(n_components=pca_dim, random_state=args.random_state)),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)),
                ]
            ),
            "pca_rf": Pipeline(
                [
                    ("pca", PCA(n_components=pca_dim, random_state=args.random_state)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=args.random_state,
                            class_weight="balanced_subsample",
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            "pca_et": Pipeline(
                [
                    ("pca", PCA(n_components=pca_dim, random_state=args.random_state)),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=400,
                            random_state=args.random_state,
                            class_weight="balanced",
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        }
        best_name = ""
        best_model = None
        best_metrics = None
        comparison = {}
        for name, candidate in candidates.items():
            candidate.fit(x_train, y_train)
            y_pred = candidate.predict(x_test)
            cand_metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            }
            comparison[name] = cand_metrics
            if (
                best_metrics is None
                or cand_metrics["balanced_accuracy"] > best_metrics["balanced_accuracy"]
                or (
                    cand_metrics["balanced_accuracy"] == best_metrics["balanced_accuracy"]
                    and cand_metrics["accuracy"] > best_metrics["accuracy"]
                )
            ):
                best_name = name
                best_model = candidate
                best_metrics = cand_metrics
        model = best_model
        metrics = {
            "mode": str(best_name),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "accuracy": float(best_metrics["accuracy"]),
            "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "model_comparison": comparison,
            "pca_dim": int(pca_dim),
        }
    else:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x, y)
        y_pred = model.predict(x)
        metrics = {
            "mode": "dummy_most_frequent",
            "train_samples": int(len(y)),
            "test_samples": int(len(y)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "image_size": int(args.image_size),
        "label_counts": label_counts,
        "dataset_dir": str(dataset_dir),
    }
    with model_out.open("wb") as f:
        pickle.dump(payload, f)

    report = {
        "dataset_dir": str(dataset_dir),
        "samples": int(len(y)),
        "label_counts": label_counts,
        "metrics": metrics,
        "model_out": str(model_out),
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
