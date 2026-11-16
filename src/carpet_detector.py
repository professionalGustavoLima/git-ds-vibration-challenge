"""Carpet detector (Part 1).

Each sliding band of the spectrum (250 Hz wide, 125 Hz step, above 1000 Hz) is described by the features in
``src.features`` and classified by k-nearest neighbours (k = 25, standardised features); the carpet probability is
the fraction of carpet bands among the neighbours. Accepted bands are merged into regions.
The choice of model and the validation (signal-grouped CV, IoU in Hz) are documented in
``notebooks/eda_part1.ipynb``, section 8.

The k-NN is trained on the band table of the Part 1 labels, stored in ``models/carpet_training_bands.npz``
(plain arrays, no pickle). To rebuild it from ``data/part1``::

    uv run python -m src.carpet_detector
"""

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (
    BAND_STEP_HZ,
    BAND_W_HZ,
    F_MIN_HZ,
    MODEL_FEATURES,
    amp_spectrum,
    band_overlap,
    band_table,
    wave_to_array,
)
from src.interface import CarpetRegion, Wave
from src.metrics import evaluate_part1

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _PROJECT_ROOT / "models" / "carpet_training_bands.npz"

_LOG_FEATURES = ("env_peak_ratio", "kurt", "energy_cv")  # long tails: log10 before scaling
_START_OFFSET_HZ = 0.5  # regions start strictly above F_MIN_HZ (labels have 1 Hz precision)
_N_NEIGHBORS = 25  # best model of notebooks/eda_part1.ipynb, section 8
_CLASSIFIER_CACHE: Dict[Tuple[str, float], Pipeline] = {}

_COLOR_SPECTRUM = "#8a8a85"
_COLOR_TRUTH = "#eb6834"  # same palette as the notebooks: labelled carpet = orange
_COLOR_DETECTED = "#2a78d6"  # detected carpet = blue


def _design_matrix(features: pd.DataFrame) -> np.ndarray:
    X = features[MODEL_FEATURES].astype(float).copy()
    cols = list(_LOG_FEATURES)
    X[cols] = np.log10(X[cols].clip(lower=1e-9))
    return X.to_numpy()


def _build_classifier() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=_N_NEIGHBORS),
    )


def _load_classifier(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(
            f"Carpet detector training table not found: {path}\n"
            "Build it with: uv run python -m src.carpet_detector"
        )
    key = (str(path), path.stat().st_mtime)
    if key not in _CLASSIFIER_CACHE:
        npz = np.load(path, allow_pickle=False)
        names = [str(n) for n in npz["feature_names"]]
        if names != MODEL_FEATURES:
            raise ValueError(
                f"Training table features {names} do not match MODEL_FEATURES {MODEL_FEATURES}.\n"
                "Rebuild it with: uv run python -m src.carpet_detector"
            )
        features = pd.DataFrame(npz["features"], columns=names)
        start = time.perf_counter()
        _CLASSIFIER_CACHE[key] = _build_classifier().fit(_design_matrix(features), npz["labels"].astype(bool))
        logger.info("Carpet detector: k-NN fitted on %d bands from %s in %.1fs", len(features), path.name,
                    time.perf_counter() - start)
    return _CLASSIFIER_CACHE[key]


def _bands_to_regions(
    lo: np.ndarray, accepted: np.ndarray, max_gap_hz: float, min_width_hz: float
) -> List[Tuple[float, float]]:
    """Merge accepted bands into sorted, disjoint regions starting strictly above ``F_MIN_HZ``."""
    merged: List[List[float]] = []
    for band_lo in np.sort(lo[accepted]):
        centre = band_lo + BAND_W_HZ / 2
        start = F_MIN_HZ + _START_OFFSET_HZ if band_lo <= F_MIN_HZ else centre - BAND_STEP_HZ / 2
        end = centre + BAND_STEP_HZ / 2
        if merged and start - merged[-1][1] <= max_gap_hz:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged if b - a >= min_width_hz]


def build_training_table(
    waves: Mapping[str, Wave], labels: Mapping[str, List[CarpetRegion]]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Band features and carpet labels for every unambiguous band of the labelled signals.

    A band is carpet when at least half of it lies inside a labelled region and background when it does not
    touch any region; bands in between are left out.
    """
    tables, targets = [], []
    for sample_id, wave in waves.items():
        x, fs = wave_to_array(wave)
        table = band_table(x, fs)
        overlap = band_overlap(table, [(r.start_hz, r.end_hz) for r in labels.get(sample_id, [])])
        clear = (overlap == 0) | (overlap >= 0.5)
        tables.append(table.loc[clear, MODEL_FEATURES])
        targets.append(overlap[clear] >= 0.5)
    return pd.concat(tables, ignore_index=True), np.concatenate(targets)


class CarpetDetector:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold: float = 0.5,
        max_gap_hz: float = 250.0,
        min_width_hz: float = 500.0,
    ) -> None:
        """
        Args:
            model_path: band table used to train the k-NN (see ``build_training_table``).
            threshold: minimum carpet probability (fraction of carpet neighbours) for a band to be accepted as carpet.
            max_gap_hz: accepted bands closer than this are merged into one region.
            min_width_hz: regions narrower than this are discarded.
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.max_gap_hz = max_gap_hz
        self.min_width_hz = min_width_hz
        self._classifier: Optional[Pipeline] = None
        self._training: Optional[Tuple[pd.DataFrame, np.ndarray]] = None

    def fit(
        self, waves: Mapping[str, Wave], labels: Mapping[str, List[CarpetRegion]]
    ) -> "CarpetDetector":
        """Train the band classifier on labelled waves (replaces the stored training table in memory)."""
        features, targets = build_training_table(waves, labels)
        self._training = (features, targets)
        self._classifier = _build_classifier().fit(_design_matrix(features), targets)
        return self

    def save(self, path: Optional[Path] = None) -> Path:
        """Store the training table of the last ``fit`` as ``.npz`` (no pickle)."""
        if self._training is None:
            raise RuntimeError("Nothing to save: call fit() first.")
        path = Path(path or self.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        features, targets = self._training
        np.savez_compressed(
            path,
            features=features[MODEL_FEATURES].to_numpy(float),
            labels=targets.astype(bool),
            feature_names=np.array(MODEL_FEATURES),
        )
        return path

    def _get_classifier(self) -> Pipeline:
        if self._classifier is None:
            self._classifier = _load_classifier(self.model_path)
        return self._classifier

    def band_probabilities(self, wave: Wave) -> pd.DataFrame:
        """Band table of ``wave`` with the carpet probability in column ``p_carpet``."""
        x, fs = wave_to_array(wave)
        table = band_table(x, fs)
        if table.empty:
            return table.assign(p_carpet=pd.Series(dtype=float))
        return table.assign(p_carpet=self._get_classifier().predict_proba(_design_matrix(table))[:, 1])

    def predict(self, wave: Wave) -> List[CarpetRegion]:
        """Return detected carpet regions, or an empty list if none."""
        table = self.band_probabilities(wave)
        if table.empty:
            return []
        accepted = (table["p_carpet"] > self.threshold).to_numpy()
        regions = _bands_to_regions(table["lo"].to_numpy(), accepted, self.max_gap_hz, self.min_width_hz)
        return [CarpetRegion(start_hz=a, end_hz=b) for a, b in regions]

    def plot_results(
        self,
        wave: Wave,
        regions: List[CarpetRegion],
        output_dir: Path,
        sample_id: str = "sample",
        true_regions: Optional[List[CarpetRegion]] = None,
    ) -> None:
        """Plot frequency spectrum with regions highlighted; save to ``output_dir / f"{sample_id}.png"``.

        When ``true_regions`` is given, a track above the spectrum compares the labelled carpet with the detected
        one and the title shows the IoU of this signal.
        """
        x, fs = wave_to_array(wave)
        f, amp = amp_spectrum(x, fs)

        tracks = [("Detected", regions, _COLOR_DETECTED)]
        if true_regions is not None:
            tracks.insert(0, ("Ground truth", true_regions, _COLOR_TRUTH))

        fig = Figure(figsize=(12, 4.8))
        ax_tracks, ax = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [0.45 * len(tracks), 3]})

        for row, (name, rr, color) in enumerate(tracks):
            y = len(tracks) - 1 - row
            for r in rr:
                ax_tracks.plot([r.start_hz, r.end_hz], [y, y], color=color, linewidth=7, solid_capstyle="butt")
        ax_tracks.set_yticks(range(len(tracks)))
        ax_tracks.set_yticklabels([name for name, _, _ in tracks][::-1], fontsize=9)
        ax_tracks.set_ylim(-0.7, len(tracks) - 0.3)
        ax_tracks.tick_params(axis="x", length=0)
        ax_tracks.grid(axis="x", alpha=0.25, linewidth=0.6)

        ax.plot(f, 20 * np.log10(amp + 1e-12), color=_COLOR_SPECTRUM, linewidth=0.4)
        for r in regions:
            ax.axvspan(r.start_hz, r.end_hz, color=_COLOR_DETECTED, alpha=0.18, linewidth=0)
        ax.set(xlabel="Frequency (Hz)", ylabel="Amplitude (dB re 1 g)", xlim=(0, fs / 2))
        ax.grid(alpha=0.25, linewidth=0.6)

        for a in (ax_tracks, ax):
            a.axvline(F_MIN_HZ, color="#555555", linestyle=":", linewidth=1)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)

        title = f"{sample_id} — {len(regions)} carpet region(s) detected"
        if true_regions is not None:
            iou = evaluate_part1({sample_id: regions}, {sample_id: true_regions})["mean_iou"]
            title += f" | {len(true_regions)} labelled | IoU = {iou:.2f}"
        ax_tracks.set_title(title, fontsize=11)
        fig.tight_layout()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{sample_id}.png", dpi=110)


def train_default_model(path: Path = MODEL_PATH) -> Path:
    """Build the training table from ``data/part1`` and store it at ``path``."""
    from src.data_loader import load_part1

    waves, labels = load_part1()
    return CarpetDetector(model_path=path).fit(waves, labels).save(path)


if __name__ == "__main__":
    out = train_default_model()
    print(f"Carpet detector training table -> {out}")
