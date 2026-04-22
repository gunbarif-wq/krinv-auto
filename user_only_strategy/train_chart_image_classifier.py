from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import sklearn

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a lightweight chart image classifier")
    p.add_argument("--dataset-dir", required=True, help="dataset directory created by build_chart_image_dataset.py")
    p.add_argument("--model-out", required=True, help="output pickle path")
    p.add_argument("--report-out", default="", help="optional report json path")
    p.add_argument("--image-size", type=int, default=96, help="resize image to NxN before training")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--target-side", default="long", choices=["long", "short"])
    p.add_argument(
        "--preferred-model",
        default="",
        choices=["", "logistic_raw", "pca_logistic", "pca_random_forest", "pca_gradient_boosting", "pca_catboost"],
        help="force a specific candidate model instead of selecting by balanced accuracy",
    )
    p.add_argument(
        "--preferred-threshold",
        type=float,
        default=-1.0,
        help="record a preferred decision threshold in the saved payload/report (e.g. 0.40)",
    )
    return p.parse_args()


def load_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def load_dataset(dataset_dir: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")
    xs: List[np.ndarray] = []
    ys: List[int] = []
    rets: List[float] = []
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
                rets.append(float(row.get("fwd_close_ret", 0.0) or 0.0))
                names.append(str(img_path))
            except Exception:
                continue
    if not xs:
        raise RuntimeError("no valid images found in dataset")
    return np.vstack(xs), np.array(ys, dtype=int), np.array(rets, dtype=np.float32), names


def build_portable_payload(model) -> dict | None:
    if isinstance(model, LogisticRegression):
        return {
            "kind": "logistic_raw",
            "coef": np.asarray(model.coef_, dtype=np.float32).tolist(),
            "intercept": np.asarray(model.intercept_, dtype=np.float32).tolist(),
            "classes": np.asarray(model.classes_, dtype=np.int32).tolist(),
        }
    if isinstance(model, Pipeline):
        steps = dict(model.named_steps)
        pca = steps.get("pca")
        clf = steps.get("clf")
        if isinstance(pca, PCA) and isinstance(clf, LogisticRegression):
            return {
                "kind": "pca_logistic",
                "mean": np.asarray(pca.mean_, dtype=np.float32).tolist(),
                "components": np.asarray(pca.components_, dtype=np.float32).tolist(),
                "coef": np.asarray(clf.coef_, dtype=np.float32).tolist(),
                "intercept": np.asarray(clf.intercept_, dtype=np.float32).tolist(),
                "classes": np.asarray(clf.classes_, dtype=np.int32).tolist(),
            }
    return None


def build_constant_portable_payload(label: int) -> dict:
    return {
        "kind": "constant",
        "label": int(label),
        "classes": [int(label)],
    }


def strategy_returns_from_forward_returns(fwd_returns: np.ndarray, target_side: str) -> np.ndarray:
    if str(target_side).lower() == "short":
        return -fwd_returns
    return fwd_returns


def predict_positive_scores(model, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float32)
    preds = np.asarray(model.predict(x_test), dtype=np.float32)
    return preds


def summarize_threshold_metrics(
    scores: np.ndarray,
    y_true: np.ndarray,
    strategy_returns: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    signal_mask = y_pred == 1
    signal_count = int(np.sum(signal_mask))
    signal_returns = strategy_returns[signal_mask]
    mean_ret = float(np.mean(signal_returns)) if signal_count else 0.0
    median_ret = float(np.median(signal_returns)) if signal_count else 0.0
    hit_rate = float(np.mean(signal_returns > 0)) if signal_count else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_pred)),
        "signal_count": signal_count,
        "strategy_mean_return": mean_ret,
        "strategy_median_return": median_ret,
        "strategy_hit_rate": hit_rate,
    }


def evaluate_candidate(
    model,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fwd_returns_test: np.ndarray,
    target_side: str,
) -> Dict[str, object]:
    model.fit(x_train, y_train)
    scores = predict_positive_scores(model, x_test)
    strategy_returns = strategy_returns_from_forward_returns(fwd_returns_test, target_side)
    default_metrics = summarize_threshold_metrics(scores, y_test, strategy_returns, 0.5)
    sweep = []
    best_by_return = None
    for threshold in np.arange(0.35, 0.81, 0.05):
        threshold_metrics = summarize_threshold_metrics(scores, y_test, strategy_returns, float(threshold))
        sweep.append(threshold_metrics)
        enough_signals = threshold_metrics["signal_count"] >= max(10, int(len(y_test) * 0.03))
        if not enough_signals:
            continue
        if (
            best_by_return is None
            or threshold_metrics["strategy_mean_return"] > best_by_return["strategy_mean_return"]
            or (
                threshold_metrics["strategy_mean_return"] == best_by_return["strategy_mean_return"]
                and threshold_metrics["balanced_accuracy"] > best_by_return["balanced_accuracy"]
            )
        ):
            best_by_return = threshold_metrics
    return {
        "model": model,
        "default_metrics": default_metrics,
        "best_return_threshold_metrics": best_by_return,
        "threshold_sweep": sweep,
    }


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    model_out = Path(args.model_out)
    report_out = Path(args.report_out) if args.report_out else model_out.with_suffix(".report.json")

    x, y, fwd_returns, names = load_dataset(dataset_dir, args.image_size)
    label_counts = {str(k): int(v) for k, v in sorted(Counter(y).items())}

    comparison = {}
    portable_kind = ""
    portable = None
    if len(np.unique(y)) >= 2 and len(y) >= 8:
        x_train, x_test, y_train, y_test, ret_train, ret_test = train_test_split(
            x,
            y,
            fwd_returns,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        del ret_train
        pca_dim = max(8, min(64, x_train.shape[0] - 1, x_train.shape[1]))
        candidates = {
            "logistic_raw": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state),
            "pca_logistic": Pipeline(
                [
                    ("pca", PCA(n_components=pca_dim, random_state=args.random_state)),
                    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)),
                ]
            ),
            "pca_random_forest": Pipeline(
                [
                    ("pca", PCA(n_components=pca_dim, random_state=args.random_state)),
                    ("clf", RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=4, class_weight="balanced_subsample", random_state=args.random_state, n_jobs=1)),
                ]
            ),
            "pca_gradient_boosting": Pipeline(
                [
                    ("pca", PCA(n_components=pca_dim, random_state=args.random_state)),
                    ("clf", GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=args.random_state)),
                ]
            ),
        }
        if CatBoostClassifier is not None:
            candidates["pca_catboost"] = Pipeline(
                [
                    ("pca", PCA(n_components=pca_dim, random_state=args.random_state)),
                    (
                        "clf",
                        CatBoostClassifier(
                            iterations=250,
                            depth=8,
                            learning_rate=0.05,
                            loss_function="Logloss",
                            eval_metric="BalancedAccuracy",
                            verbose=False,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            )
        best_name = ""
        best_model = None
        best_metrics = None
        for name, candidate in candidates.items():
            result = evaluate_candidate(candidate, x_train, x_test, y_train, y_test, ret_test, args.target_side)
            cand_metrics = dict(result["default_metrics"])
            if result["best_return_threshold_metrics"] is not None:
                cand_metrics["best_return_threshold_metrics"] = result["best_return_threshold_metrics"]
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
                best_model = result["model"]
                best_metrics = cand_metrics
        if args.preferred_model:
            forced_metrics = comparison.get(str(args.preferred_model))
            forced_model = candidates.get(str(args.preferred_model))
            if forced_metrics is None or forced_model is None:
                raise RuntimeError(f"preferred model not available: {args.preferred_model}")
            forced_model.fit(x_train, y_train)
            best_name = str(args.preferred_model)
            best_model = forced_model
            best_metrics = forced_metrics
        model = best_model
        portable = build_portable_payload(model)
        portable_kind = str(portable.get("kind", "")) if portable else ""
        preferred_threshold = float(args.preferred_threshold) if float(args.preferred_threshold) > 0 else None
        metrics = {
            "mode": str(best_name),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "accuracy": float(best_metrics["accuracy"]),
            "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "positive_rate": float(best_metrics["positive_rate"]),
            "strategy_mean_return": float(best_metrics["strategy_mean_return"]),
            "strategy_median_return": float(best_metrics["strategy_median_return"]),
            "strategy_hit_rate": float(best_metrics["strategy_hit_rate"]),
            "best_return_threshold_metrics": best_metrics.get("best_return_threshold_metrics"),
            "model_comparison": comparison,
            "pca_dim": int(pca_dim),
            "portable_kind": portable_kind,
            "target_side": str(args.target_side),
            "preferred_threshold": preferred_threshold,
        }
    else:
        model = None
        constant_label = int(y[0]) if len(y) else 0
        y_pred = np.full_like(y, constant_label)
        portable = build_constant_portable_payload(constant_label)
        portable_kind = str(portable.get("kind", "")) if portable else ""
        metrics = {
            "mode": "constant_fallback",
            "train_samples": int(len(y)),
            "test_samples": int(len(y)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
            "portable_kind": portable_kind,
            "target_side": str(args.target_side),
        }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model if portable is None else None,
        "portable_model": portable,
        "image_size": int(args.image_size),
        "label_counts": label_counts,
        "dataset_dir": str(dataset_dir),
        "target_side": str(args.target_side),
        "preferred_threshold": float(args.preferred_threshold) if float(args.preferred_threshold) > 0 else None,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
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
