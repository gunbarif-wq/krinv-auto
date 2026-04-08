from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None  # type: ignore[assignment]

from ml_backtest_common import run_policy
from ml_signal_common import PolicyConfig, load_json
from ml_trade_common import build_policy_config
from build_ml_dataset import build_rows, load_split
from train_ml_signal import class_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward ML training/validation/test with fixed policy parameters")
    p.add_argument("--data-root", default="data/backtest_sets_225190_1y")
    p.add_argument("--symbol", default="225190")
    p.add_argument("--model-kind", default="catboost", choices=["catboost", "logistic", "gboost", "hgb", "rf", "et", "auto"])
    p.add_argument("--model-top-k", type=int, default=2, help="use top-K quick-scored models for policy search")
    p.add_argument("--max-model-candidates", type=int, default=12, help="applies when model-kind=auto")
    p.add_argument("--horizon-bars", type=int, default=30)
    p.add_argument("--signal-mode", default="alpha", choices=["prob", "alpha"])
    p.add_argument("--alpha-ret-scale", type=float, default=0.004, help="sigmoid scale for expected return in alpha mode")
    p.add_argument("--alpha-rank-window", type=int, default=120, help="rolling rank window in alpha mode")
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr"])
    p.add_argument("--up-threshold", type=float, default=0.012)
    p.add_argument("--down-threshold", type=float, default=0.007)
    p.add_argument("--atr-up-mult", type=float, default=2.0)
    p.add_argument("--atr-down-mult", type=float, default=1.2)
    p.add_argument("--atr-floor-pct", type=float, default=0.003)
    p.add_argument("--min-history-bars", type=int, default=30, help="minimum same-day history bars required before emitting a row")
    p.add_argument("--fee-roundtrip", type=float, default=0.001)
    p.add_argument("--policy-path", default="data/ml/225190_1y/225190_fast_policy.json")
    p.add_argument("--wf-train-days", type=int, default=60)
    p.add_argument("--wf-val-days", type=int, default=15)
    p.add_argument("--wf-test-days", type=int, default=10)
    p.add_argument("--wf-step-days", type=int, default=10)
    p.add_argument("--wf-max-folds", type=int, default=0, help="0 means no fold limit")
    p.add_argument("--min-thr-trades", type=int, default=20)
    p.add_argument("--thr-start", type=float, default=0.50)
    p.add_argument("--thr-end", type=float, default=0.95)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--mdd-penalty", type=float, default=0.12)
    p.add_argument("--objective-mode", default="profit_max", choices=["profit_max", "risk_adjusted"])
    p.add_argument("--max-mdd-allowed", type=float, default=12.0, help="hard penalty when val MDD is worse than this (abs, pct)")
    p.add_argument("--min-profit-factor", type=float, default=1.0)
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--max-trades", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--min-score-delta", type=float, default=0.0, help="entry requires score_t - score_t-1 >= this value")
    p.add_argument("--hold-bars", type=int, default=10)
    p.add_argument("--min-hold-bars", type=int, default=2)
    p.add_argument("--exit-threshold", type=float, default=0.62, help="score-drop exit threshold (<=0 disables)")
    p.add_argument("--skip-open-min", type=int, default=10)
    p.add_argument("--skip-close-min", type=int, default=10)
    p.add_argument("--loss-streak-for-cooldown", type=int, default=0)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--take-profit-pct", type=float, default=0.0)
    p.add_argument("--stop-loss-pct", type=float, default=0.0)
    p.add_argument("--trailing-stop-pct", type=float, default=0.0)
    p.add_argument("--max-concurrent-positions", type=int, default=3)
    p.add_argument("--position-size-pct", type=float, default=1.0)
    p.add_argument("--min-entry-gap-bars", type=int, default=0)
    p.add_argument("--entry-start-hhmm", type=int, default=900)
    p.add_argument("--entry-end-hhmm", type=int, default=1530)
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--out-dir", default="data/ml")
    return p.parse_args()


def feature_columns_from_rows(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        raise RuntimeError("no rows for feature inference")
    ignore = {"date", "label", "fwd_close_ret"}
    return [k for k in rows[0].keys() if k not in ignore]


def rows_to_arrays(
    rows: List[Dict[str, str]], feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X: List[List[float]] = []
    y: List[int] = []
    open_: List[float] = []
    close: List[float] = []
    fwd: List[float] = []
    vwap_gap_day: List[float] = []
    dates: List[str] = []
    for r in rows:
        try:
            X.append([float(r[c]) for c in feature_cols])
            y.append(int(r["label"]))
            open_.append(float(r["open"]))
            close.append(float(r["close"]))
            fwd.append(float(r["fwd_close_ret"]))
            vwap_gap_day.append(float(r.get("vwap_gap_day", 0.0)))
            dates.append(r["date"])
        except Exception:
            continue
    if not X:
        return (
            np.empty((0, len(feature_cols))),
            np.empty((0,), dtype=int),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,)),
            [],
        )
    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(open_, dtype=float),
        np.asarray(close, dtype=float),
        np.asarray(fwd, dtype=float),
        np.asarray(vwap_gap_day, dtype=float),
        dates,
    )


def unique_days(rows: List[Dict[str, str]]) -> List[str]:
    return sorted({r["date"][:10] for r in rows if len(r.get("date", "")) >= 10})


def build_model(kind: str, y_train: np.ndarray) -> object:
    if kind == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed. Run: pip install catboost")
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            depth=6,
            learning_rate=0.03,
            n_estimators=700,
            l2_leaf_reg=5.0,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    if kind == "logistic":
        return Pipeline(
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
    if kind == "gboost":
        return GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=80,
            subsample=0.8,
            random_state=42,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=450,
            max_depth=4,
            min_samples_leaf=60,
            l2_regularization=0.5,
            random_state=42,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=42,
        )
    raise RuntimeError(f"unknown model kind: {kind}")


@dataclass
class ModelEval:
    name: str
    kind: str
    quick_score: float
    auc: float
    ap: float
    precision_min_trades: float
    signal_val: np.ndarray
    signal_test: np.ndarray


def sigmoid_arr(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def rolling_rank_01(arr: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    n = x.shape[0]
    w = max(10, int(window))
    out = np.full(n, 0.5, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        seg = x[s : i + 1]
        if seg.size <= 1:
            out[i] = 0.5
            continue
        out[i] = float(np.sum(seg <= x[i])) / float(seg.size)
    return np.clip(out, 0.0, 1.0)


def best_precision_with_min_trades(
    prob: np.ndarray, y: np.ndarray, min_trades: int, start: float, end: float, step: float
) -> float:
    best = 0.0
    thr = start
    while thr <= end + 1e-12:
        sel = prob >= thr
        trades = int(np.sum(sel))
        if trades >= min_trades:
            prec = float(np.mean(y[sel])) if trades > 0 else 0.0
            if prec > best:
                best = prec
        thr += step
    return best


def predict_prob(model: object, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]  # type: ignore[attr-defined]


def list_model_kinds(args: argparse.Namespace) -> List[str]:
    if args.model_kind != "auto":
        return [args.model_kind]
    kinds = ["catboost", "logistic", "gboost", "hgb", "rf", "et"]
    rng = random.Random(args.seed)
    if args.max_model_candidates > 0 and len(kinds) > args.max_model_candidates:
        kinds = rng.sample(kinds, k=args.max_model_candidates)
    return kinds


def evaluate_models(
    args: argparse.Namespace,
    y_train: np.ndarray,
    fwd_train: np.ndarray,
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> List[ModelEval]:
    out: List[ModelEval] = []
    for kind in list_model_kinds(args):
        try:
            model = build_model(kind, y_train)
            if kind in {"catboost", "gboost", "hgb", "rf", "et"}:
                model.fit(x_train, y_train, sample_weight=class_weights(y_train))
            else:
                model.fit(x_train, y_train)
            p_val = predict_prob(model, x_val)
            p_test = predict_prob(model, x_test)
            signal_val = p_val
            signal_test = p_test
            if args.signal_mode == "alpha":
                # Regression head learns expected forward return (local opportunity), then we combine it with win-prob.
                y_reg = np.clip(fwd_train, -0.03, 0.03)
                reg = GradientBoostingRegressor(
                    learning_rate=0.05,
                    n_estimators=300,
                    max_depth=3,
                    min_samples_leaf=80,
                    subsample=0.8,
                    random_state=42,
                )
                reg.fit(x_train, y_reg)
                r_val = reg.predict(x_val)
                r_test = reg.predict(x_test)
                ret_score_val = sigmoid_arr(r_val / max(1e-6, float(args.alpha_ret_scale)))
                ret_score_test = sigmoid_arr(r_test / max(1e-6, float(args.alpha_ret_scale)))
                alpha_raw_val = p_val * ret_score_val
                alpha_raw_test = p_test * ret_score_test
                signal_val = rolling_rank_01(alpha_raw_val, int(args.alpha_rank_window))
                signal_test = rolling_rank_01(alpha_raw_test, int(args.alpha_rank_window))

            auc = float(roc_auc_score(y_val, signal_val)) if len(np.unique(y_val)) > 1 else 0.5
            ap = float(average_precision_score(y_val, signal_val))
            prec = best_precision_with_min_trades(
                prob=signal_val,
                y=y_val,
                min_trades=args.min_thr_trades,
                start=args.thr_start,
                end=args.thr_end,
                step=args.thr_step,
            )
            quick = ap * 100.0 + auc * 20.0 + prec * 20.0
            out.append(
                ModelEval(
                    name=kind,
                    kind=kind,
                    quick_score=quick,
                    auc=auc,
                    ap=ap,
                    precision_min_trades=prec,
                    signal_val=signal_val,
                    signal_test=signal_test,
                )
            )
        except Exception as e:
            print(f"[model-skip] kind={kind} reason={e}")
            continue
    out.sort(key=lambda x: x.quick_score, reverse=True)
    return out


def policy_score(
    result: Dict[str, float | int | str | None],
    min_trades: int,
    max_trades: int,
    mdd_penalty: float,
    objective_mode: str,
    max_mdd_allowed: float,
    min_profit_factor: float,
) -> float:
    ret = float(result["total_return_pct"])  # type: ignore[arg-type]
    mdd = float(result["max_drawdown_pct"])  # negative
    trades = int(result["trades"])  # type: ignore[arg-type]
    avg_trade = float(result.get("avg_trade_return_pct", 0.0) or 0.0)
    pf = float(result.get("profit_factor", 0.0) or 0.0)
    if objective_mode == "profit_max":
        # Reward total edge first, then prefer better per-trade quality.
        s = ret + 0.15 * avg_trade * max(1.0, min(100.0, float(trades)))
    else:
        s = ret - mdd_penalty * abs(min(0.0, mdd))
    s -= mdd_penalty * abs(min(0.0, mdd))
    if abs(min(0.0, mdd)) > max_mdd_allowed:
        s -= (abs(min(0.0, mdd)) - max_mdd_allowed) * 2.0
    if pf < min_profit_factor:
        s -= (min_profit_factor - pf) * 3.0
    if trades < min_trades:
        s -= (min_trades - trades) * 0.05
    if trades > max_trades:
        s -= (trades - max_trades) * 0.02
    return s


def main() -> None:
    args = parse_args()

    full_csv = Path(args.data_root) / "full_1m" / f"{args.symbol}_1m_full.csv"
    if not full_csv.exists():
        raise RuntimeError(f"missing full data csv: {full_csv}")

    raw = load_split(full_csv)
    rows_all = build_rows(
        data=raw,
        horizon=args.horizon_bars,
        min_history_bars=args.min_history_bars,
        label_mode=args.label_mode,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        atr_up_mult=args.atr_up_mult,
        atr_down_mult=args.atr_down_mult,
        atr_floor_pct=args.atr_floor_pct,
    )
    if len(rows_all) < 1000:
        raise RuntimeError(f"too few ML rows after feature build: {len(rows_all)}")

    feature_cols = feature_columns_from_rows(rows_all)
    all_days = unique_days(rows_all)
    need = args.wf_train_days + args.wf_val_days + args.wf_test_days
    if len(all_days) < need:
        raise RuntimeError(f"not enough days for walk-forward: have={len(all_days)} need={need}")

    policy = load_json(Path(args.policy_path))
    cfg_fixed = build_policy_config(
        policy,
        threshold=float(policy.get("threshold", args.threshold)),
        fee_roundtrip=float(policy.get("fee_roundtrip", args.fee_roundtrip)),
        min_hold_bars=max(1, int(policy.get("min_hold_bars", args.min_hold_bars))),
        entry_start_hhmm=int(policy.get("entry_start_hhmm", args.entry_start_hhmm)),
        entry_end_hhmm=int(policy.get("entry_end_hhmm", args.entry_end_hhmm)),
        skip_open_min=max(0, int(policy.get("skip_open_min", args.skip_open_min))),
        skip_close_min=max(0, int(policy.get("skip_close_min", args.skip_close_min))),
        loss_streak_for_cooldown=max(0, int(policy.get("loss_streak_for_cooldown", args.loss_streak_for_cooldown))),
        cooldown_bars=max(0, int(policy.get("cooldown_bars", args.cooldown_bars))),
        exit_threshold=max(0.0, float(policy.get("exit_threshold", args.exit_threshold))),
        trailing_stop_pct=max(0.0, float(policy.get("trailing_stop_pct", args.trailing_stop_pct))),
        trailing_activate_pct=max(0.0, float(policy.get("trailing_activate_pct", 0.0))),
        vwap_exit_min_hold_bars=max(0, int(policy.get("vwap_exit_min_hold_bars", 0))),
        vwap_exit_max_profit_pct=float(policy.get("vwap_exit_max_profit_pct", 0.0)),
        max_concurrent_positions=max(1, int(policy.get("max_concurrent_positions", args.max_concurrent_positions))),
        position_size_pct=min(1.0, max(0.01, float(policy.get("position_size_pct", args.position_size_pct)))),
        min_entry_gap_bars=max(0, int(policy.get("min_entry_gap_bars", args.min_entry_gap_bars))),
    )

    out_dir = Path(args.out_dir) / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{args.symbol}_ml_wf_{stamp}.csv"
    summary_path = out_dir / f"{args.symbol}_ml_wf_summary_{stamp}.json"

    fieldnames = [
        "fold",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
        "model",
        "model_quick_score",
        "model_auc",
        "model_ap",
        "model_prec_min_trades",
        "thr_selected",
        "min_score_delta",
        "hold_bars",
        "min_hold_bars",
        "exit_threshold",
        "skip_open_min",
        "skip_close_min",
        "loss_streak",
        "cooldown_bars",
        "take_profit_pct",
        "stop_loss_pct",
        "trailing_stop_pct",
        "max_concurrent_positions",
        "position_size_pct",
        "min_entry_gap_bars",
        "val_score",
        "val_return_pct",
        "val_mdd_pct",
        "val_trades",
        "test_return_pct",
        "test_mdd_pct",
        "test_trades",
    ]

    fold_results: List[Dict[str, object]] = []
    start = 0
    fold_idx = 1
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        while start + need <= len(all_days):
            train_days = all_days[start : start + args.wf_train_days]
            val_days = all_days[start + args.wf_train_days : start + args.wf_train_days + args.wf_val_days]
            test_days = all_days[
                start + args.wf_train_days + args.wf_val_days : start + args.wf_train_days + args.wf_val_days + args.wf_test_days
            ]
            train_set = set(train_days)
            val_set = set(val_days)
            test_set = set(test_days)

            train_rows = [r for r in rows_all if r["date"][:10] in train_set]
            val_rows = [r for r in rows_all if r["date"][:10] in val_set]
            test_rows = [r for r in rows_all if r["date"][:10] in test_set]

            X_train, y_train, _open_train, _close_train, fwd_train, _vwap_train, _dates_train = rows_to_arrays(train_rows, feature_cols)
            X_val, y_val, open_val, close_val, _fwd_val, vwap_val, dates_val = rows_to_arrays(val_rows, feature_cols)
            X_test, y_test, open_test, close_test, _fwd_test, vwap_test, dates_test = rows_to_arrays(test_rows, feature_cols)

            if min(X_train.shape[0], X_val.shape[0], X_test.shape[0]) < 100:
                start += args.wf_step_days
                fold_idx += 1
                continue

            model_evals = evaluate_models(
                args=args,
                y_train=y_train,
                fwd_train=fwd_train,
                x_train=X_train,
                x_val=X_val,
                y_val=y_val,
                x_test=X_test,
            )
            if not model_evals:
                start += args.wf_step_days
                fold_idx += 1
                continue
            top_k = max(1, int(args.model_top_k))
            model_evals = model_evals[:top_k]

            best_score = -10**18
            best_policy: PolicyConfig | None = None
            best_val: Dict[str, float | int | str | None] | None = None
            best_test: Dict[str, float | int | str | None] | None = None
            best_model: ModelEval | None = None

            for m in model_evals:
                val_res = run_policy(
                    prob=m.signal_val,
                    open_=open_val,
                    close=close_val,
                    vwap_gap_day=vwap_val,
                    dates=dates_val,
                    cfg=cfg_fixed,
                    initial_cash=float(args.initial_cash),
                )
                s = policy_score(
                    result=val_res,
                    min_trades=args.min_trades,
                    max_trades=args.max_trades,
                    mdd_penalty=args.mdd_penalty,
                    objective_mode=args.objective_mode,
                    max_mdd_allowed=args.max_mdd_allowed,
                    min_profit_factor=args.min_profit_factor,
                )
                if s > best_score:
                    best_score = s
                    best_policy = cfg_fixed
                    best_val = val_res
                    best_model = m
                    best_test = run_policy(
                        prob=m.signal_test,
                        open_=open_test,
                        close=close_test,
                        vwap_gap_day=vwap_test,
                        dates=dates_test,
                        cfg=cfg_fixed,
                        initial_cash=float(args.initial_cash),
                    )
            if best_policy is None or best_val is None or best_test is None or best_model is None:
                start += args.wf_step_days
                fold_idx += 1
                continue

            row = {
                "fold": fold_idx,
                "train_start": train_days[0],
                "train_end": train_days[-1],
                "val_start": val_days[0],
                "val_end": val_days[-1],
                "test_start": test_days[0],
                "test_end": test_days[-1],
                "model": best_model.name,
                "model_quick_score": best_model.quick_score,
                "model_auc": best_model.auc,
                "model_ap": best_model.ap,
                "model_prec_min_trades": best_model.precision_min_trades,
                "thr_selected": best_policy.threshold,
                "min_score_delta": best_policy.min_score_delta,
                "hold_bars": best_policy.hold_bars,
                "min_hold_bars": best_policy.min_hold_bars,
                "exit_threshold": best_policy.exit_threshold,
                "skip_open_min": best_policy.skip_open_min,
                "skip_close_min": best_policy.skip_close_min,
                "loss_streak": best_policy.loss_streak_for_cooldown,
                "cooldown_bars": best_policy.cooldown_bars,
                "take_profit_pct": best_policy.take_profit_pct,
                "stop_loss_pct": best_policy.stop_loss_pct,
                "trailing_stop_pct": best_policy.trailing_stop_pct,
                "max_concurrent_positions": best_policy.max_concurrent_positions,
                "position_size_pct": best_policy.position_size_pct,
                "min_entry_gap_bars": best_policy.min_entry_gap_bars,
                "val_score": best_score,
                "val_return_pct": float(best_val["total_return_pct"]),
                "val_mdd_pct": float(best_val["max_drawdown_pct"]),
                "val_trades": int(best_val["trades"]),
                "test_return_pct": float(best_test["total_return_pct"]),
                "test_mdd_pct": float(best_test["max_drawdown_pct"]),
                "test_trades": int(best_test["trades"]),
            }
            w.writerow(row)
            f.flush()
            print(
                f"[fold {fold_idx}] model={row['model']} test_ret={row['test_return_pct']:.2f}% "
                f"test_mdd={row['test_mdd_pct']:.2f}% trades={row['test_trades']} "
                f"thr={row['thr_selected']:.2f} hold={row['hold_bars']} tp={row['take_profit_pct']:.3f} "
                f"sl={row['stop_loss_pct']:.3f} pos={row['max_concurrent_positions']} size={row['position_size_pct']:.2f}"
            )
            fold_results.append(
                {
                    **row,
                    "policy": {
                        "threshold": best_policy.threshold,
                        "min_score_delta": best_policy.min_score_delta,
                        "hold_bars": best_policy.hold_bars,
                        "min_hold_bars": best_policy.min_hold_bars,
                        "exit_threshold": best_policy.exit_threshold,
                        "entry_start_hhmm": best_policy.entry_start_hhmm,
                        "entry_end_hhmm": best_policy.entry_end_hhmm,
                        "skip_open_min": best_policy.skip_open_min,
                        "skip_close_min": best_policy.skip_close_min,
        "loss_streak_for_cooldown": best_policy.loss_streak_for_cooldown,
        "cooldown_bars": best_policy.cooldown_bars,
        "take_profit_pct": best_policy.take_profit_pct,
        "stop_loss_pct": best_policy.stop_loss_pct,
        "trailing_stop_pct": best_policy.trailing_stop_pct,
        "trailing_activate_pct": best_policy.trailing_activate_pct,
        "vwap_exit_min_hold_bars": best_policy.vwap_exit_min_hold_bars,
        "vwap_exit_max_profit_pct": best_policy.vwap_exit_max_profit_pct,
        "max_concurrent_positions": best_policy.max_concurrent_positions,
        "position_size_pct": best_policy.position_size_pct,
                        "min_entry_gap_bars": best_policy.min_entry_gap_bars,
                        "fee_roundtrip": best_policy.fee_roundtrip,
                    },
                }
            )

            start += args.wf_step_days
            fold_idx += 1
            if args.wf_max_folds > 0 and len(fold_results) >= args.wf_max_folds:
                break

    if not fold_results:
        raise RuntimeError("no valid fold result produced")

    test_rets = [float(x["test_return_pct"]) for x in fold_results]
    test_mdds = [float(x["test_mdd_pct"]) for x in fold_results]
    test_trades = [int(x["test_trades"]) for x in fold_results]
    model_counts: Dict[str, int] = {}
    for x in fold_results:
        k = str(x["model"])
        model_counts[k] = model_counts.get(k, 0) + 1
    aggregate = {
        "fold_count": len(fold_results),
        "test_return_mean_pct": float(statistics.mean(test_rets)),
        "test_return_median_pct": float(statistics.median(test_rets)),
        "test_return_min_pct": float(min(test_rets)),
        "test_return_max_pct": float(max(test_rets)),
        "test_mdd_median_pct": float(statistics.median(test_mdds)),
        "test_trades_sum": int(sum(test_trades)),
        "model_usage": model_counts,
    }
    summary = {
        "args": vars(args),
        "source_days": {"start": all_days[0], "end": all_days[-1], "count": len(all_days)},
        "feature_columns": feature_cols,
        "aggregate_oos": aggregate,
        "folds": fold_results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_csv={csv_path}")
    print(f"saved_summary={summary_path}")
    print(
        f"oos_summary: mean_ret={aggregate['test_return_mean_pct']:.2f}% "
        f"median_ret={aggregate['test_return_median_pct']:.2f}% "
        f"median_mdd={aggregate['test_mdd_median_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
