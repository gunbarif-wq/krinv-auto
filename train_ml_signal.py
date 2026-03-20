from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from build_ml_dataset import TRIO_FEATURE_COLUMNS, build_rows, load_split as load_raw_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ML signal model and choose threshold on validation set")
    p.add_argument("--dataset-dir", default="data/ml/047810")
    p.add_argument("--raw-csv", default="", help="if set, build train/val/test from raw 1m csv directly")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--horizon-bars", type=int, default=30)
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr"])
    p.add_argument("--up-threshold", type=float, default=0.012)
    p.add_argument("--down-threshold", type=float, default=0.007)
    p.add_argument("--atr-up-mult", type=float, default=2.0)
    p.add_argument("--atr-down-mult", type=float, default=1.2)
    p.add_argument("--atr-floor-pct", type=float, default=0.003)
    p.add_argument("--symbol", default="047810")
    p.add_argument("--model-kind", default="logistic", choices=["logistic", "gboost"])
    p.add_argument("--feature-mode", default="trio", choices=["all", "trio"])
    p.add_argument("--fee-roundtrip", type=float, default=0.001)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--thr-start", type=float, default=0.50)
    p.add_argument("--thr-end", type=float, default=0.95)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--model-out", default="data/ml/047810/047810_model.pkl")
    p.add_argument("--report-out", default="data/ml/047810/047810_train_report.json")
    return p.parse_args()


def infer_feature_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise RuntimeError(f"no header: {path}")
        ignore = {"date", "label", "fwd_close_ret"}
        return [c for c in rdr.fieldnames if c not in ignore]


def load_split(path: Path, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    fwd: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                X.append([float(r[c]) for c in feature_columns])
                y.append(int(r["label"]))
                fwd.append(float(r["fwd_close_ret"]))
            except Exception:
                continue
    if not X:
        raise RuntimeError(f"empty split: {path}")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(fwd, dtype=float)


def feature_columns_from_rows(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        raise RuntimeError("no rows for feature inference")
    ignore = {"date", "label", "fwd_close_ret"}
    return [k for k in rows[0].keys() if k not in ignore]


def rows_to_arrays(rows: List[Dict[str, str]], feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    fwd: List[float] = []
    for r in rows:
        try:
            X.append([float(r[c]) for c in feature_columns])
            y.append(int(r["label"]))
            fwd.append(float(r["fwd_close_ret"]))
        except Exception:
            continue
    if not X:
        raise RuntimeError("empty rows")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(fwd, dtype=float)


def select_feature_columns(feature_columns: List[str], feature_mode: str) -> List[str]:
    if feature_mode == "all":
        return feature_columns
    selected = [c for c in TRIO_FEATURE_COLUMNS if c in feature_columns]
    if len(selected) != len(TRIO_FEATURE_COLUMNS):
        missing = [c for c in TRIO_FEATURE_COLUMNS if c not in feature_columns]
        raise RuntimeError(f"missing trio features in dataset: {missing}")
    return selected


def class_weights(y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


def eval_threshold(prob: np.ndarray, y: np.ndarray, fwd: np.ndarray, thr: float, fee: float) -> Dict[str, float]:
    sel = prob >= thr
    trades = int(np.sum(sel))
    if trades == 0:
        return {
            "threshold": thr,
            "trades": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "avg_fwd_ret": 0.0,
            "expected_net_ret": -fee,
        }
    y_sel = y[sel]
    f_sel = fwd[sel]
    precision = float(np.mean(y_sel))
    recall = float(np.sum(y_sel == 1) / max(1, np.sum(y == 1)))
    avg_fwd_ret = float(np.mean(f_sel))
    expected_net_ret = avg_fwd_ret - fee
    return {
        "threshold": thr,
        "trades": float(trades),
        "precision": precision,
        "recall": recall,
        "avg_fwd_ret": avg_fwd_ret,
        "expected_net_ret": expected_net_ret,
    }


def select_threshold(
    prob: np.ndarray,
    y: np.ndarray,
    fwd: np.ndarray,
    fee: float,
    min_trades: int,
    start: float,
    end: float,
    step: float,
) -> Dict[str, float]:
    best: Dict[str, float] | None = None
    thr = start
    while thr <= end + 1e-12:
        m = eval_threshold(prob, y, fwd, thr, fee)
        if m["trades"] >= float(min_trades):
            if best is None or m["expected_net_ret"] > best["expected_net_ret"]:
                best = m
        thr += step
    if best is None:
        # Fallback: choose best available regardless of min_trades.
        thr = start
        while thr <= end + 1e-12:
            m = eval_threshold(prob, y, fwd, thr, fee)
            if best is None or m["expected_net_ret"] > best["expected_net_ret"]:
                best = m
            thr += step
    assert best is not None
    return best


def summary_metrics(prob: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (prob >= thr).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "positive_rate": float(np.mean(pred)),
    }


def extract_formula_if_logistic(model: object, feature_columns: List[str]) -> Dict[str, object] | None:
    if not isinstance(model, Pipeline):
        return None
    if "clf" not in model.named_steps or "scaler" not in model.named_steps:
        return None
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]
    if not isinstance(clf, LogisticRegression) or not isinstance(scaler, StandardScaler):
        return None
    coefs = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])
    terms: List[Dict[str, float | str]] = []
    for name, w, mu, sigma in zip(feature_columns, coefs, scaler.mean_.tolist(), scaler.scale_.tolist()):
        terms.append(
            {
                "feature": name,
                "coef_on_z": float(w),
                "mean": float(mu),
                "std": float(sigma if sigma > 0 else 1.0),
            }
        )
    return {
        "kind": "logistic_on_standardized_features",
        "score_definition": "score = intercept + sum(coef_on_z * ((x-mean)/std)); prob = sigmoid(score)",
        "intercept": intercept,
        "terms": terms,
    }


def main() -> None:
    args = parse_args()
    raw_csv = str(args.raw_csv).strip()
    if raw_csv:
        raw = load_raw_split(Path(raw_csv))
        rows_all = build_rows(
            data=raw,
            horizon=args.horizon_bars,
            label_mode=args.label_mode,
            up_threshold=args.up_threshold,
            down_threshold=args.down_threshold,
            atr_up_mult=args.atr_up_mult,
            atr_down_mult=args.atr_down_mult,
            atr_floor_pct=args.atr_floor_pct,
        )
        if len(rows_all) < 1000:
            raise RuntimeError(f"too few rows from raw csv: {len(rows_all)}")
        n = len(rows_all)
        train_ratio = max(0.05, min(0.95, float(args.train_ratio)))
        val_ratio = max(0.0, min(0.90, float(args.val_ratio)))
        if train_ratio + val_ratio >= 0.98:
            val_ratio = max(0.0, 0.98 - train_ratio)
        cut1 = max(1, int(round(n * train_ratio)))
        cut2 = max(cut1 + 1, int(round(n * (train_ratio + val_ratio))))
        cut2 = min(cut2, n - 1)
        train_rows = rows_all[:cut1]
        val_rows = rows_all[cut1:cut2]
        test_rows = rows_all[cut2:]
        feature_columns = feature_columns_from_rows(rows_all)
        feature_columns = select_feature_columns(feature_columns, args.feature_mode)
        X_train, y_train, f_train = rows_to_arrays(train_rows, feature_columns)
        X_val, y_val, f_val = rows_to_arrays(val_rows, feature_columns)
        X_test, y_test, f_test = rows_to_arrays(test_rows, feature_columns)
    else:
        root = Path(args.dataset_dir)
        train_path = root / f"{args.symbol}_train_ml.csv"
        val_path = root / f"{args.symbol}_val_ml.csv"
        test_path = root / f"{args.symbol}_test_ml.csv"

        feature_columns = infer_feature_columns(train_path)
        feature_columns = select_feature_columns(feature_columns, args.feature_mode)
        if not feature_columns:
            raise RuntimeError(f"no feature columns inferred from {train_path}")

        X_train, y_train, f_train = load_split(train_path, feature_columns)
        X_val, y_val, f_val = load_split(val_path, feature_columns)
        X_test, y_test, f_test = load_split(test_path, feature_columns)

    if args.model_kind == "logistic":
        model: object = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=0.5,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
    else:
        model = GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=80,
            subsample=0.8,
            random_state=42,
        )
    sw = class_weights(y_train)
    if args.model_kind == "gboost":
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)

    p_val = model.predict_proba(X_val)[:, 1]  # type: ignore[attr-defined]
    best_thr = select_threshold(
        prob=p_val,
        y=y_val,
        fwd=f_val,
        fee=args.fee_roundtrip,
        min_trades=args.min_trades,
        start=args.thr_start,
        end=args.thr_end,
        step=args.thr_step,
    )
    thr = float(best_thr["threshold"])

    p_train = model.predict_proba(X_train)[:, 1]  # type: ignore[attr-defined]
    p_test = model.predict_proba(X_test)[:, 1]  # type: ignore[attr-defined]
    val_eval = eval_threshold(p_val, y_val, f_val, thr, args.fee_roundtrip)
    test_eval = eval_threshold(p_test, y_test, f_test, thr, args.fee_roundtrip)

    report = {
        "symbol": args.symbol,
        "model_kind": args.model_kind,
        "feature_columns": feature_columns,
        "threshold_selected_on_val": thr,
        "threshold_search_best": best_thr,
        "learned_formula": extract_formula_if_logistic(model, feature_columns),
        "train_class_balance": {
            "rows": int(y_train.shape[0]),
            "pos": int(np.sum(y_train == 1)),
            "neg": int(np.sum(y_train == 0)),
        },
        "metrics": {
            "train": {**summary_metrics(p_train, y_train, thr), **eval_threshold(p_train, y_train, f_train, thr, args.fee_roundtrip)},
            "val": {**summary_metrics(p_val, y_val, thr), **val_eval},
            "test": {**summary_metrics(p_test, y_test, thr), **test_eval},
        },
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "model_kind": args.model_kind,
                "feature_columns": feature_columns,
                "threshold": thr,
                "fee_roundtrip": args.fee_roundtrip,
            },
            f,
        )

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model_saved={model_out}")
    print(f"report_saved={report_out}")
    print(
        "selected_threshold="
        f"{thr:.2f} val_expected_net_ret={val_eval['expected_net_ret']*100:.4f}% "
        f"test_expected_net_ret={test_eval['expected_net_ret']*100:.4f}%"
    )


if __name__ == "__main__":
    main()
