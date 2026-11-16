"""Starved lubrication model (Part 2).

Each signal is described by four whole-signal features (``src.features.signal_features``: ``rms``, ``crest``, ``kurt``,
``skew``) and scored by an RBF SVM with Platt calibration. The score is the calibrated
probability of starved lubrication. The asset alert is the scaffold's fixed rule (75th percentile of the scores > 0.75).
The choice of model and features, and the asset-grouped validation, are documented in ``notebooks/eda_part2.ipynb``
(sections 10 and 11).

The SVM is trained on the ``predict`` signals of the labelled assets, stored in ``models/lubrication_training_signals.npz``
(plain arrays, no pickle). To rebuild it from ``data/part2``::

    uv run python -m src.lubrication_model
"""

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import signal as sps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.carpet_detector import CarpetDetector
from src.features import SIGNAL_F_MAX_HZ, SIGNAL_FEATURES, WELCH_DF_HZ, nperseg_for, signal_features, wave_to_array
from src.interface import AssetData, Wave

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _PROJECT_ROOT / "models" / "lubrication_training_signals.npz"

_ALERT_PERCENTILE = 75.0  # mirrors the fixed rule in LubricationModel.predict (used only in plots)
_ALERT_THRESHOLD = 0.75
_PSD_GRID_HZ = np.arange(0.0, SIGNAL_F_MAX_HZ + 1, 8.0)
_CLASSIFIER_CACHE: Dict[Tuple[str, float], Pipeline] = {}

_COLOR_PREDICT = "#2a78d6"
_COLOR_ALERT = "#eb6834"
_COLOR_REFERENCE = "#8a8a85"


def _wave_features(wave: Wave) -> dict:
    x, fs = wave_to_array(wave)
    return signal_features(x, fs)


def _wave_crest(wave: Wave) -> float:
    x, _ = wave_to_array(wave)
    rms = np.sqrt(np.mean(x**2))
    return float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0


def _design_matrix(features: pd.DataFrame) -> np.ndarray:
    return features[SIGNAL_FEATURES].astype(float).to_numpy()


def _build_classifier() -> Pipeline:
    svm = SVC(C=1.0, gamma="scale", class_weight="balanced")
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False))


def _load_classifier(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"Lubrication model training table not found: {path}\n"
            "Build it with: uv run python -m src.lubrication_model"
        )
    key = (str(path), path.stat().st_mtime)
    if key not in _CLASSIFIER_CACHE:
        npz = np.load(path, allow_pickle=False)
        names = [str(n) for n in npz["feature_names"]]
        if names != SIGNAL_FEATURES:
            raise ValueError(
                f"Training table features {names} do not match SIGNAL_FEATURES {SIGNAL_FEATURES}.\n"
                "Rebuild it with: uv run python -m src.lubrication_model"
            )
        features = pd.DataFrame(npz["features"], columns=names)
        start = time.perf_counter()
        _CLASSIFIER_CACHE[key] = _build_classifier().fit(_design_matrix(features), npz["labels"].astype(bool))
        logger.info("Lubrication model: SVM fitted on %d signals from %s in %.1fs", len(features), path.name,
                    time.perf_counter() - start)
    return _CLASSIFIER_CACHE[key]


def _log_psd_on_grid(wave: Wave) -> np.ndarray:
    x, fs = wave_to_array(wave)
    f, p = sps.welch(x, fs, nperseg=nperseg_for(fs, len(x), WELCH_DF_HZ))
    return np.interp(_PSD_GRID_HZ, f, 10 * np.log10(p + 1e-20))


def build_training_table(
    assets: Mapping[str, AssetData], labels: Mapping[str, bool]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Features of every ``predict`` signal of the labelled assets; each signal inherits its asset label.

    Assets without a label (e.g. the two ``test …`` folders under ``assets/``) are skipped.
    """
    rows, targets = [], []
    for asset_id, starved in labels.items():
        if asset_id not in assets:
            continue
        for wave in assets[asset_id].predict:
            rows.append(_wave_features(wave))
            targets.append(bool(starved))
    return pd.DataFrame(rows, columns=SIGNAL_FEATURES), np.array(targets, dtype=bool)


class LubricationModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        # Kept from the scaffold. The score below does not use it: in notebooks/eda_part2.ipynb the whole-signal
        # features already reach 0.988 asset accuracy with the SVM, and the carpet features were left out of scope.
        self.carpet_detector = CarpetDetector()
        self.model_path = Path(model_path)
        self._classifier: Optional[Pipeline] = None
        self._training: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
        self._reference_crest: Optional[np.ndarray] = None
        self._reference_psd: Optional[np.ndarray] = None
        self._last: Dict[str, Tuple[float, dict]] = {}

    def fit(self, data: List[Wave]) -> None:
        """Store a summary of the reference (healthy) signals for the plots.

        The score does not depend on the reference: features relative to the baseline brought no gain in
        ``notebooks/eda_part2.ipynb`` (section 10).
        """
        if not data:
            return
        self._reference_crest = np.array([_wave_crest(w) for w in data])
        self._reference_psd = np.median(np.vstack([_log_psd_on_grid(w) for w in data]), axis=0)

    def train(self, assets: Mapping[str, AssetData], labels: Mapping[str, bool]) -> "LubricationModel":
        """Train the signal classifier on labelled assets (replaces the stored training table in memory)."""
        features, targets = build_training_table(assets, labels)
        self._training = (features, targets)
        self._classifier = _build_classifier().fit(_design_matrix(features), targets)
        return self

    def save(self, path: Optional[Path] = None) -> Path:
        """Store the training table of the last ``train`` as ``.npz`` (no pickle)."""
        if self._training is None:
            raise RuntimeError("Nothing to save: call train() first.")
        path = Path(path or self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        features, targets = self._training
        np.savez_compressed(
            path,
            features=_design_matrix(features),
            labels=targets.astype(bool),
            feature_names=np.array(SIGNAL_FEATURES),
        )
        return path

    def _get_classifier(self) -> Pipeline:
        if self._classifier is None:
            self._classifier = _load_classifier(self.model_path)
        return self._classifier

    def predict_sample(self, wave: Wave) -> float:
        """Score a single waveform in [0, 1] (0 = healthy, 1 = starved lubrication)."""
        feats = _wave_features(wave)
        score = float(self._get_classifier().predict_proba(_design_matrix(pd.DataFrame([feats])))[0, 1])
        self._last[wave.sample_id] = (score, feats)
        return score

    def predict(self, data: List[Wave]) -> bool:
        """Aggregate per-sample scores; alert if 75th percentile > 0.75."""
        scores = [self.predict_sample(wave) for wave in data]
        return float(np.percentile(scores, 75)) > 0.75

    def _scores_and_features(self, data: List[Wave]) -> Tuple[np.ndarray, pd.DataFrame]:
        for wave in data:
            if wave.sample_id not in self._last:
                self.predict_sample(wave)
        scores = np.array([self._last[w.sample_id][0] for w in data])
        features = pd.DataFrame([self._last[w.sample_id][1] for w in data], columns=SIGNAL_FEATURES)
        return scores, features

    def plot_results(self, data: List[Wave], output_dir: Path) -> None:
        """Save PNG plots about the asset's condition to ``output_dir``."""
        scores, features = self._scores_and_features(data)
        p75 = float(np.percentile(scores, _ALERT_PERCENTILE))
        alert = p75 > _ALERT_THRESHOLD

        fig = Figure(figsize=(15, 4.2))
        ax_score, ax_psd, ax_crest = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1.6, 0.8]})

        # 1) Score of each signal, sorted, with the alert rule
        order = np.sort(scores)
        q = np.linspace(0, 1, len(order))
        colors = np.where(order > _ALERT_THRESHOLD, _COLOR_ALERT, _COLOR_PREDICT)
        ax_score.scatter(q, order, c=colors, s=18, linewidths=0)
        ax_score.axhline(_ALERT_THRESHOLD, color="#555555", linestyle=":", linewidth=1)
        ax_score.axvline(_ALERT_PERCENTILE / 100, color="#555555", linestyle=":", linewidth=1)
        ax_score.plot([_ALERT_PERCENTILE / 100], [p75], marker="D", color="#222222", markersize=6)
        ax_score.annotate(f"p75 = {p75:.2f}", (_ALERT_PERCENTILE / 100, p75), xytext=(-8, 8),
                          textcoords="offset points", ha="right", fontsize=9)
        ax_score.set(xlabel="Quantile of the signals (sorted)", ylabel="Starved lubrication score",
                     ylim=(-0.05, 1.05), title="Score per signal")

        # 2) Median spectrum: predicted signals against the reference
        if self._reference_psd is not None:
            ax_psd.plot(_PSD_GRID_HZ, self._reference_psd, color=_COLOR_REFERENCE, linewidth=1.2, label="reference (fit)")
        psd = np.median(np.vstack([_log_psd_on_grid(w) for w in data]), axis=0)
        ax_psd.plot(_PSD_GRID_HZ, psd, color=_COLOR_ALERT if alert else _COLOR_PREDICT, linewidth=1.2,
                    label="signals to classify")
        ax_psd.set(xlabel="Frequency (Hz)", ylabel="Median PSD (dB re 1 g²/Hz)", title="Median spectrum")
        ax_psd.legend(frameon=False, loc="upper right")

        # 3) Crest factor (most important feature) against the reference
        groups = [("signals", features["crest"].to_numpy(), _COLOR_ALERT if alert else _COLOR_PREDICT)]
        if self._reference_crest is not None:
            groups.insert(0, ("reference", self._reference_crest, _COLOR_REFERENCE))
        rng = np.random.default_rng(0)
        for k, (_, values, color) in enumerate(groups):
            ax_crest.boxplot(values, positions=[k], widths=0.5, showfliers=False,
                             medianprops={"color": "#222222"}, boxprops={"color": "#555555"},
                             whiskerprops={"color": "#555555"}, capprops={"color": "#555555"})
            ax_crest.scatter(k + rng.uniform(-0.15, 0.15, len(values)), values, s=10, color=color, alpha=0.7,
                             linewidths=0)
        ax_crest.set_xticks(range(len(groups)))
        ax_crest.set_xticklabels([g[0] for g in groups])
        ax_crest.set(ylabel="Crest factor", title="Crest factor")

        for ax in (ax_score, ax_psd, ax_crest):
            ax.grid(alpha=0.25, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        verdict = "ALERT: starved lubrication" if alert else "no alert: healthy"
        fig.suptitle(f"{verdict} (p75 of {len(scores)} scores = {p75:.2f}, threshold {_ALERT_THRESHOLD})", fontsize=12)
        fig.tight_layout()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "condition.png", dpi=110)


def train_default_model(path: Path = MODEL_PATH) -> Path:
    """Build the training table from ``data/part2`` (labelled assets) and store it at ``path``."""
    from src.data_loader import load_part2

    assets, labels, _ = load_part2()
    return LubricationModel(model_path=path).train(assets, labels).save(path)


if __name__ == "__main__":
    out = train_default_model()
    print(f"Lubrication model training table -> {out}")
