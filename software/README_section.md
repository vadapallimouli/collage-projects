## Desktop GUI and Independent Signal Testing

### Running the GUI

```powershell
python main.py
```

The window has three tabs:

| Tab | Purpose |
|---|---|
| **Signal Analysis** | Upload or drag-and-drop a `.npy` signal, view its waveform and single-sided spectrum (raw or processed), and classify it with the saved best model. A second sub-tab, **Independent Signal Testing**, generates and evaluates new synthetic signals. |
| **Feature Analysis** | The 12 features returned by the existing `FeatureExtractor`, in training order, as cards or a sortable table. |
| **Model Performance** | Read-only view of `results/evaluation_summary.json`, `trained_models/training_summary.json` and the saved evaluation plots. Opening the GUI never retrains anything. |

All predictions use `predict.SignalPredictor`; preprocessing and feature extraction are not re-implemented in the GUI.

### Independent synthetic test set

The original models were trained on 4,000 signals (`dataset/`, seed 42) and scored on a stratified 80:20 held-out split. To check the saved model on signals it has never seen, the project can generate a separate test set:

```powershell
python external_signal_generator.py --per-class 250 --snr 30 20 10 5 --seed 20261020
python external_evaluation.py
```

(or use the GUI: *Signal Analysis -> Independent Signal Testing*).

* Uses the modulation equations, parameter ranges and AWGN model of `signal_generator.py` (SSB = upper sideband via Hilbert transform), with a **separate seed** (default `20261020`; the training seed `42` is rejected), freshly sampled parameters, phases and noise.
* Balanced across AM, FM, DSB-SC and SSB; SNR chosen by the user (default: the four training SNR values, spread evenly).
* Signals are saved as `EXT-<run_id>-<index>.npy` in shuffled order; the class appears **only** in `metadata.csv`. File names and folders are never used for prediction.
* These signals are never used to train, tune, select a model, change `dataset/features.csv`, or update `trained_models/`.

| Location | Contents |
|---|---|
| `external_test_signals/<run_id>/signals/` | generated `.npy` signals |
| `external_test_signals/<run_id>/metadata.csv` | ground truth, parameters, SNR per signal |
| `external_test_signals/<run_id>/generation_summary.json` | seed, SNR set, class counts, parameter ranges |
| `results/external_test/eval_<timestamp>_<run_id>/` | `predictions.csv`, `summary.json`, `summary.csv`, `classification_report.csv`, `confusion_matrix.csv`, `errors.log` (only if failures) |

Each generation and each evaluation creates a **new** folder; earlier runs are preserved.

Failed predictions are logged and reported separately. They are never counted as correct; the summary also reports an accuracy that counts failures as wrong.

### Limitation

This is an **independent synthetic test set**, not proof of real-world generalization. The signals share the training data's generation assumptions (same equations, parameter ranges, noise model). Real recorded signals, if obtained later, should be evaluated separately. Also note that the best model was selected using the original 20% test split, so that split's scores are optimistic.

### Dependencies

No new packages beyond those the project already uses: `PySide6`, `pyqtgraph`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, `tqdm`, `matplotlib`.
