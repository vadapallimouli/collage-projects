"""
external_evaluation.py
======================

Batch evaluation of an independent synthetic test set (see
``external_signal_generator.py``) with the EXISTING prediction pipeline.

* Every signal is predicted with ``predict.SignalPredictor.predict(path)`` -
  the same preprocessing, 12-feature extraction and saved best model used for
  single-signal analysis.  Nothing is re-implemented here.
* Ground truth is read from the run's ``metadata.csv`` only; it is never given
  to the predictor, and file names/folders carry no class information.
* Failed predictions are logged, reported and NEVER counted as correct.
  Metrics are computed on successful predictions; a strict accuracy that
  counts failures as wrong is reported alongside.
* Nothing is fitted, tuned or saved to ``trained_models/``,
  ``dataset/`` or ``results/evaluation_summary.json``.

Outputs (a NEW folder per evaluation, earlier ones are preserved)::

    results/external_test/eval_<timestamp>_<run_id>/
        predictions.csv            one row per signal (label, prediction, probabilities, status)
        summary.json               full machine-readable summary
        summary.csv                one-row headline metrics
        classification_report.csv  per-class precision / recall / F1 / support
        confusion_matrix.csv       rows = true class, columns = predicted class
        errors.log                 only when some predictions failed

CLI
---
    python external_evaluation.py              # latest generated test set
    python external_evaluation.py --run 20261020_221500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from external_signal_generator import (
    EXTERNAL_SIGNALS_ROOT,
    PROJECT_ROOT,
    list_test_runs,
    load_generation_summary,
    load_run_metadata,
    run_signal_path,
)
from predict import CLASS_ORDER

logger = logging.getLogger("external_evaluation")

EXTERNAL_RESULTS_ROOT: Path = PROJECT_ROOT / "results" / "external_test"
PROTOCOL_TEXT: str = (
    "Independent synthetic test set: newly generated signals (separate seed, newly sampled parameters, phases and "
    "noise) that were never used for training, tuning or model selection."
)
LIMITATIONS: List[str] = [
    "Signals come from the same generator assumptions (equations, parameter ranges, AWGN) as the training data, "
    "so this measures robustness to new realisations, not real-world generalisation.",
    "Only the saved best model (the one SignalPredictor loads) is evaluated.",
    "Recorded real signals, if obtained later, must be evaluated separately.",
]

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


class ExternalEvaluationError(Exception):
    """Raised when an evaluation cannot be performed or saved."""


class EvaluationCancelled(Exception):
    """Raised when the caller requests cancellation (nothing is saved)."""

    is_cancellation = True


@dataclass
class BatchEvaluationResult:
    """Return value of :func:`evaluate_test_run`."""

    run_id: str
    output_dir: Path
    summary: Dict[str, Any]
    predictions: pd.DataFrame


# ==============================================================================
# HELPERS
# ==============================================================================

def _assert_safe_output(path: Path) -> None:
    resolved = path.resolve()
    for name in ("dataset", "trained_models"):
        protected = (PROJECT_ROOT / name).resolve()
        if resolved == protected or protected in resolved.parents:
            raise ExternalEvaluationError(f"Refusing to write evaluation output inside '{name}/'.")


def _unique_dir(root: Path, name: str) -> Path:
    candidate, counter = root / name, 2
    while candidate.exists():
        candidate = root / f"{name}_{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")


def _clean(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_test_run(predictor: Any, run_dir: Path,
                      results_root: Path = EXTERNAL_RESULTS_ROOT,
                      progress_cb: Optional[ProgressCallback] = None,
                      should_cancel: Optional[CancelCallback] = None) -> BatchEvaluationResult:
    """Predicts every signal of a generated run and saves the evaluation.

    Args:
        predictor: A ``predict.SignalPredictor`` (already loaded; never refit).
        run_dir: ``external_test_signals/<run_id>`` folder.
        results_root: Parent folder for the new evaluation folder.
        progress_cb: Optional ``(done, total, message)`` callback.
        should_cancel: Optional cancellation check.

    Raises:
        EvaluationCancelled: if cancelled (no output is written).
        ExternalEvaluationError: for unusable input or output problems.
    """
    run_dir = Path(run_dir)
    results_root = Path(results_root)
    _assert_safe_output(results_root)

    metadata = load_run_metadata(run_dir)
    run_id = run_dir.name
    records = metadata.to_dict("records")
    total = len(records)
    labels = list(CLASS_ORDER)
    prob_columns = [f"prob_{c}" for c in labels]

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    model_name = "unknown"

    for index, record in enumerate(records, start=1):
        if should_cancel is not None and should_cancel():
            raise EvaluationCancelled("Evaluation cancelled; nothing was saved.")
        entry: Dict[str, Any] = {
            "signal_id": str(record["signal_id"]), "filename": str(record["filename"]),
            "true_label": str(record["modulation"]), "predicted_label": "", "correct": None,
            "confidence": np.nan, **{c: np.nan for c in prob_columns},
            "snr_db": float(record["snr_db"]), "status": "ok", "error": "",
        }
        try:
            result = predictor.predict(run_signal_path(run_dir, entry["filename"]))
            entry["predicted_label"] = str(result.predicted_modulation)
            entry["correct"] = entry["predicted_label"] == entry["true_label"]
            if result.confidence is not None:
                entry["confidence"] = float(result.confidence)
            if result.class_probabilities:
                for cls in labels:
                    entry[f"prob_{cls}"] = float(result.class_probabilities.get(cls, np.nan))
            model_name = str(result.model_name)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded, never counted as correct
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failures.append({"signal_id": entry["signal_id"], "error": entry["error"]})
            logger.warning("Prediction failed for %s: %s", entry["signal_id"], entry["error"])
        rows.append(entry)
        if progress_cb is not None:
            progress_cb(index, total, f"Evaluating signal {index} of {total}")

    predictions = pd.DataFrame(rows)
    ok = predictions[predictions["status"] == "ok"]
    n_ok, n_failed = int(len(ok)), int(len(predictions) - len(ok))

    metrics: Dict[str, Optional[float]] = {
        "accuracy": None, "precision_weighted": None, "recall_weighted": None, "f1_weighted": None,
        "accuracy_failures_as_incorrect": float(predictions["correct"].fillna(False).astype(bool).sum() / total),
    }
    matrix: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    if n_ok:
        y_true, y_pred = ok["true_label"].to_numpy(), ok["predicted_label"].to_numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        metrics.update({
            "accuracy": float(accuracy_score(y_true, y_pred)), "precision_weighted": float(precision),
            "recall_weighted": float(recall), "f1_weighted": float(f1),
        })
        matrix = {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
        report = classification_report(y_true, y_pred, labels=labels, target_names=labels,
                                       output_dict=True, zero_division=0)

    by_snr: List[Dict[str, Any]] = []
    for snr in sorted(predictions["snr_db"].unique()):
        subset = predictions[predictions["snr_db"] == snr]
        subset_ok = subset[subset["status"] == "ok"]
        by_snr.append({
            "snr_db": float(snr), "n_signals": int(len(subset)), "n_successful": int(len(subset_ok)),
            "accuracy": _clean(float(subset_ok["correct"].astype(bool).mean())) if len(subset_ok) else None,
        })

    seed = load_generation_summary(run_dir).get("random_state")
    out_dir = _unique_dir(results_root, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}")
    files = {
        "predictions_csv": "predictions.csv", "summary_json": "summary.json", "summary_csv": "summary.csv",
        "classification_report_csv": "classification_report.csv", "confusion_matrix_csv": "confusion_matrix.csv",
    }
    if failures:
        files["errors_log"] = "errors.log"

    summary: Dict[str, Any] = {
        "evaluation": {
            "run_id": run_id, "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name, "model_artifact": "trained_models/best_model.joblib",
            "protocol": PROTOCOL_TEXT, "generation_seed": seed, "output_dir": str(out_dir),
        },
        "counts": {
            "total": total, "successful": n_ok, "failed": n_failed,
            "ground_truth_distribution": {c: int((predictions["true_label"] == c).sum()) for c in labels},
            "successful_distribution": {c: int((ok["true_label"] == c).sum()) for c in labels},
        },
        "metrics": metrics, "labels": labels, "confusion_matrix": matrix, "classification_report": report,
        "accuracy_by_snr": by_snr, "failures": failures[:100], "output_files": files, "limitations": LIMITATIONS,
    }

    predictions.to_csv(out_dir / files["predictions_csv"], index=False)
    with open(out_dir / files["summary_json"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    pd.DataFrame([{
        "run_id": run_id, "model": model_name, "total": total, "successful": n_ok, "failed": n_failed,
        **metrics, "evaluated_at": summary["evaluation"]["evaluated_at"],
    }]).to_csv(out_dir / files["summary_csv"], index=False)
    if report is not None:
        pd.DataFrame(report).T.to_csv(out_dir / files["classification_report_csv"])
        pd.DataFrame(matrix["matrix"], index=[f"true_{c}" for c in labels],
                     columns=[f"pred_{c}" for c in labels]).to_csv(out_dir / files["confusion_matrix_csv"])
    else:
        pd.DataFrame().to_csv(out_dir / files["classification_report_csv"], index=False)
        pd.DataFrame().to_csv(out_dir / files["confusion_matrix_csv"], index=False)
    if failures:
        with open(out_dir / files["errors_log"], "w", encoding="utf-8") as fh:
            fh.writelines(f"{f['signal_id']}\t{f['error']}\n" for f in failures)

    logger.info("Evaluation saved to '%s' (%d ok, %d failed).", out_dir, n_ok, n_failed)
    return BatchEvaluationResult(run_id, out_dir, summary, predictions)


# ==============================================================================
# SAVED EVALUATIONS (for the GUI)
# ==============================================================================

def list_evaluations(results_root: Path = EXTERNAL_RESULTS_ROOT) -> List[Path]:
    """Evaluation folders that contain a ``summary.json``, newest first."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and (p / "summary.json").is_file()),
                  key=lambda p: p.name, reverse=True)


def load_evaluation_summary(eval_dir: Path) -> Dict[str, Any]:
    """Loads ``summary.json`` of a saved evaluation folder."""
    path = Path(eval_dir) / "summary.json"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalEvaluationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict) or "metrics" not in data or "counts" not in data:
        raise ExternalEvaluationError(f"{path} does not look like an external-test summary.")
    return data


# ==============================================================================
# CLI
# ==============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    from predict import PredictConfig, SignalPredictor

    parser = argparse.ArgumentParser(description="Evaluate an independent synthetic test set.")
    parser.add_argument("--run", default=None, help="run id under external_test_signals/ (default: newest)")
    args = parser.parse_args(argv)

    if args.run:
        run_dir = EXTERNAL_SIGNALS_ROOT / args.run
    else:
        runs = list_test_runs()
        if not runs:
            print("No generated test sets found. Run external_signal_generator.py first.", file=sys.stderr)
            return 2
        run_dir = runs[0].run_dir

    predictor = SignalPredictor(PredictConfig(models_dir=PROJECT_ROOT / "trained_models"))
    result = evaluate_test_run(predictor, run_dir,
                               progress_cb=lambda d, t, m: print(f"\r{m}", end="", flush=True) if d % 25 == 0 or d == t else None)
    s, m, c = result.summary, result.summary["metrics"], result.summary["counts"]
    print(f"\n\nINDEPENDENT SYNTHETIC TEST SET - run {result.run_id}")
    print(f"Model      : {s['evaluation']['model_name']}")
    print(f"Signals    : {c['total']} total, {c['successful']} evaluated, {c['failed']} failed")
    print(f"Classes    : {c['ground_truth_distribution']}")
    if m["accuracy"] is not None:
        print(f"Accuracy   : {m['accuracy']:.4f}  (failures counted wrong: {m['accuracy_failures_as_incorrect']:.4f})")
        print(f"Precision  : {m['precision_weighted']:.4f} (weighted)")
        print(f"Recall     : {m['recall_weighted']:.4f} (weighted)")
        print(f"F1         : {m['f1_weighted']:.4f} (weighted)")
    print(f"Saved to   : {result.output_dir}")
    print("Note       : synthetic test set - not proof of real-world generalisation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())