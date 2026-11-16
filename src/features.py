"""Band-wise spectral features of a vibration signal.

The definitions follow ``notebooks/eda_part1.ipynb`` (section 7). Every spectral parameter is set in Hz and
converted to samples per signal, because the signals have different sampling rates and lengths.
"""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

from src.interface import Wave

F_MIN_HZ = 1000.0  # carpet regions must start above this frequency
BAND_W_HZ = 250.0  # width of each sliding band
BAND_STEP_HZ = 125.0  # step between consecutive bands
WELCH_DF_HZ = 8.0  # target resolution of the Welch PSD
STFT_DF_HZ = 62.5  # target resolution of the STFT (spectral kurtosis, spectrogram)
ENV_FMAX_HZ = 500.0  # upper limit of the envelope spectrum

MODEL_FEATURES: List[str] = [
    "level_db",
    "flatness",
    "env_peak_ratio",
    "kurt",
    "sk_mean",
    "energy_cv",
]


def wave_to_array(wave: Wave) -> Tuple[np.ndarray, float]:
    """Return the zero-mean signal and its sampling rate in Hz."""
    t = np.asarray(wave.time, dtype=float)
    x = np.asarray(wave.signal, dtype=float)
    return x - x.mean(), float(1.0 / np.median(np.diff(t)))


def nperseg_for(fs: float, n: int, target_df_hz: float, min_segments: int = 4) -> int:
    """Power-of-two segment length giving ``target_df_hz`` resolution, with at least ``min_segments`` segments."""
    nper = 2 ** int(np.round(np.log2(fs / target_df_hz)))
    return int(min(nper, 2 ** int(np.floor(np.log2(n / min_segments)))))


def amp_spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum (g) with a Hann window, at full resolution."""
    w = sps.windows.hann(len(x))
    spec = np.fft.rfft(x * w)
    return np.fft.rfftfreq(len(x), 1 / fs), 2 * np.abs(spec) / w.sum()


def welch_db(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD in dB (g²/Hz) at ``WELCH_DF_HZ`` resolution."""
    f, p = sps.welch(x, fs, nperseg=nperseg_for(fs, len(x), WELCH_DF_HZ))
    return f, 10 * np.log10(p + 1e-20)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    sos = sps.butter(order, [lo, min(hi, 0.49 * fs)], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _spectral_kurtosis(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = nperseg_for(fs, len(x), STFT_DF_HZ)
    f, _, z = sps.stft(x, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    p = np.abs(z) ** 2
    return f, np.mean(p**2, axis=1) / (np.mean(p, axis=1) ** 2 + 1e-30) - 2


def band_table(x: np.ndarray, fs: float) -> pd.DataFrame:
    """Features of every sliding band between ``F_MIN_HZ`` and 0.95 * Nyquist.

    Columns: ``lo``, ``hi`` and the names in ``MODEL_FEATURES``. Empty if the signal has no band above ``F_MIN_HZ``.
    """
    fw, pw_db = welch_db(x, fs)
    pw = 10 ** (pw_db / 10)
    floor_db = np.median(pw_db[(fw >= F_MIN_HZ) & (fw <= 0.95 * fs / 2)]) if fs / 2 > F_MIN_HZ else 0.0
    f_sk, sk = _spectral_kurtosis(x, fs)
    nper_sg = nperseg_for(fs, len(x), STFT_DF_HZ)
    f_sg, _, sg = sps.spectrogram(x, fs, nperseg=nper_sg, noverlap=nper_sg // 2)

    rows = []
    for lo in np.arange(F_MIN_HZ, 0.95 * fs / 2 - BAND_W_HZ, BAND_STEP_HZ):
        hi = lo + BAND_W_HZ
        y = bandpass(x, fs, lo, hi)
        env = np.abs(sps.hilbert(y))
        f_env, e_env = amp_spectrum(env - env.mean(), fs)
        m_env = (f_env > 2) & (f_env <= ENV_FMAX_HZ)
        p_band = pw[(fw >= lo) & (fw < hi)]
        energy_t = sg[(f_sg >= lo) & (f_sg < hi)].sum(axis=0)
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "level_db": pw_db[(fw >= lo) & (fw < hi)].mean() - floor_db,
                "flatness": np.exp(np.mean(np.log(p_band + 1e-30))) / np.mean(p_band),
                "env_peak_ratio": e_env[m_env].max() / (np.median(e_env[m_env]) + 1e-20),
                "kurt": stats.kurtosis(y, fisher=False),
                "sk_mean": sk[(f_sk >= lo) & (f_sk < hi)].mean(),
                "energy_cv": energy_t.std() / (energy_t.mean() + 1e-30),
            }
        )
    return pd.DataFrame(rows, columns=["lo", "hi", *MODEL_FEATURES])


SIGNAL_F_MAX_HZ = 7800.0  # upper limit of the Part 2 spectral plots (below the Part 2 Nyquist)

SIGNAL_FEATURES: List[str] = ["rms", "crest", "kurt", "skew"]


def signal_features(x: np.ndarray, fs: float) -> dict:
    """Whole-signal features used by the lubrication model (definitions of ``notebooks/eda_part2.ipynb``, section 2).

    ``rms``, ``crest``, ``kurt`` and ``skew`` are time-domain statistics of the zero-mean signal (Kannan et al. 2024,
    Eq. 5-8). ``fs`` is kept in the signature so every feature function takes the same arguments.
    """
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x**2)))
    return {
        "rms": rms,
        "crest": float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0,
        "kurt": float(stats.kurtosis(x, fisher=False)),
        "skew": float(stats.skew(x)),
    }


def band_overlap(table: pd.DataFrame, regions: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Fraction of each band covered by the given regions (0 = outside, 1 = fully inside)."""
    lo, hi = table["lo"].to_numpy(), table["hi"].to_numpy()
    covered = np.zeros(len(table))
    for a, b in regions:
        covered += np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
    return np.minimum(covered / BAND_W_HZ, 1.0)
