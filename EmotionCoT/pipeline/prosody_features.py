"""Signal-level prosodic feature extraction for the EmotionCoT annotation pipeline.

Everything in this module is pure DSP and depends only on numpy/scipy/librosa, so
it can be run in a process pool without touching a GPU.

Design rules that the earlier version of this pipeline violated, and that the
callers below rely on:

* Pitch means fundamental frequency. It is tracked with pYIN over a range that is
  plausible for speech, never with a spectral peak tracker whose output is a
  mixture of harmonics and therefore covaries with timbre and vocal effort.
* Every magnitude is reported as a continuous physical quantity in an
  interpretable unit (Hz, semitones, dBFS, phonemes per second). Mapping to
  low/normal/high is the job of ``calibration.py``, against corpus percentiles.
* Silence is excluded before any average is taken, so that leading/trailing pad
  and pause structure cannot move a level label.
* Unvoiced frames are never filled with zeros. They are simply absent, and every
  statistic is computed over the voiced frames with their true timestamps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.ndimage import median_filter

SAMPLE_RATE = 16_000
HOP_LENGTH = 160  # 10 ms
FRAME_LENGTH = 1024  # 64 ms, which spans >= 2 periods at F0_FLOOR_HZ
# Energy and voice activity use a shorter window than pitch tracking does. A
# 64 ms window smears the speech/silence boundary by half its length in each
# direction, which inflates measured speech duration by tens of milliseconds.
ENERGY_FRAME_LENGTH = 400  # 25 ms

# A ceiling of 500 Hz comfortably covers expressive female and child speech. The
# previous C7 (2093 Hz) ceiling mostly served to admit upward octave errors.
F0_FLOOR_HZ = 55.0
F0_CEIL_HZ = 500.0

# An octave correction is only applied when it moves a frame at least this much
# closer to its local neighbourhood. Being conservative here matters: an
# over-eager correction flattens genuine pitch excursions, which is a worse
# failure than leaving a stray frame alone.
OCTAVE_REPAIR_MIN_GAIN_ST = 6.0
OCTAVE_REPAIR_KERNEL = 9

# A second pYIN pass runs over a search range centred on the speaker's own median
# from the first pass. Halving errors, which arrive in sustained runs that a
# local median filter follows rather than corrects, then fall outside the search
# range and cannot be produced at all. The window is deliberately asymmetric:
# large upward excursions are ordinary in expressive speech, whereas most of what
# appears far below the median is either a subharmonic or creak.
ADAPTIVE_RANGE_LOW = 0.60  # -8.8 semitones
ADAPTIVE_RANGE_HIGH = 2.00  # +12 semitones

# Frames this far from the utterance median that land within GLOBAL_REPAIR_TOL_ST
# of it after an octave shift are treated as octave errors.
GLOBAL_REPAIR_MIN_DEVIATION_ST = 9.0
GLOBAL_REPAIR_TOL_ST = 6.0

# Below this share of speech frames carrying a usable F0, the pitch statistics
# rest on too little material to be relied on.
MIN_VOICED_SHARE_OF_SPEECH = 0.20

VAD_TOP_DB = 35.0  # a frame is speech if it is within this many dB of the robust peak
# On a recording whose noise floor sits only ~30 dB below the speech, a purely
# peak-relative threshold marks the entire file as speech, which inflates the
# speech duration and so deflates the articulation rate. The threshold is
# therefore raised towards the noise floor as well -- but only when the file has
# enough dynamic range for its 5th percentile to plausibly be a floor rather
# than more speech, and never so far that frames near the peak get excluded.
VAD_MARGIN_OVER_NOISE_DB = 6.0
VAD_MIN_DYNAMIC_RANGE_DB = 20.0
VAD_MIN_TOP_DB = 15.0
VAD_MIN_SPEECH_S = 0.05
VAD_MIN_GAP_S = 0.06

EPS = 1e-10

_ARPABET = re.compile(r"^[A-Z]{1,3}[0-2]?$")
_NON_LATIN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

_G2P = None


# --------------------------------------------------------------------------- #
# audio loading
# --------------------------------------------------------------------------- #
def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load as mono float32 at ``sr``.

    ``mono=True`` averages channels, which the previous ``waveform.squeeze()``
    did not do: squeeze leaves a (2, N) stereo tensor two-dimensional, so every
    stereo file (all of Expresso's improvised dialogues, for one) either crashed
    downstream or was silently mis-analysed.
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size:
        y = y - float(np.mean(y))  # remove DC so that RMS reflects acoustic energy
    return y


# --------------------------------------------------------------------------- #
# voice activity detection
# --------------------------------------------------------------------------- #
def _runs(mask: np.ndarray) -> List[Tuple[int, int, bool]]:
    if mask.size == 0:
        return []
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], changes, [mask.size]))
    return [
        (int(bounds[i]), int(bounds[i + 1]), bool(mask[bounds[i]]))
        for i in range(len(bounds) - 1)
    ]


def _bridge_and_prune(mask: np.ndarray, min_gap: int, min_run: int) -> np.ndarray:
    out = mask.copy()
    for start, stop, value in _runs(out):
        if not value and (stop - start) < min_gap and start > 0 and stop < out.size:
            out[start:stop] = True
    for start, stop, value in _runs(out):
        if value and (stop - start) < min_run:
            out[start:stop] = False
    return out


def frame_rms(y: np.ndarray) -> np.ndarray:
    return librosa.feature.rms(
        y=y, frame_length=ENERGY_FRAME_LENGTH, hop_length=HOP_LENGTH, center=True
    )[0]


def detect_speech_frames(rms: np.ndarray, top_db: float = VAD_TOP_DB) -> np.ndarray:
    """Energy VAD referenced to a robust (95th percentile) peak.

    Referencing the 95th percentile rather than the maximum keeps a single click
    or a door slam from raising the floor and swallowing the whole utterance.
    """
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    db = 20.0 * np.log10(np.maximum(rms, EPS))
    ref_db = float(np.percentile(db, 95))
    floor_db = float(np.percentile(db, 5))
    threshold = ref_db - top_db
    if ref_db - floor_db > VAD_MIN_DYNAMIC_RANGE_DB:
        threshold = max(threshold, floor_db + VAD_MARGIN_OVER_NOISE_DB)
    threshold = min(threshold, ref_db - VAD_MIN_TOP_DB)
    mask = db > threshold
    return _bridge_and_prune(
        mask,
        min_gap=max(1, int(round(VAD_MIN_GAP_S * SAMPLE_RATE / HOP_LENGTH))),
        min_run=max(1, int(round(VAD_MIN_SPEECH_S * SAMPLE_RATE / HOP_LENGTH))),
    )


def speech_duration_s(speech: np.ndarray) -> float:
    return float(np.count_nonzero(speech) * HOP_LENGTH / SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# fundamental frequency
# --------------------------------------------------------------------------- #
def _repair_octave_errors(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Snap isolated octave jumps back towards their local neighbourhood.

    The reference is a running median over nearby voiced frames rather than the
    global median, so a genuine wide excursion (whose neighbourhood moves with
    it) is left untouched while a one-off doubling is pulled back.
    """
    out = np.array(f0, dtype=float)
    idx = np.flatnonzero(voiced)
    if idx.size < 3:
        return out

    log2f = np.log2(out[idx])
    kernel = min(OCTAVE_REPAIR_KERNEL, idx.size if idx.size % 2 else idx.size - 1)
    kernel = max(3, kernel if kernel % 2 else kernel - 1)
    min_gain = OCTAVE_REPAIR_MIN_GAIN_ST / 12.0

    for _ in range(2):
        local = median_filter(log2f, size=kernel, mode="nearest")
        for shift in (-1.0, 1.0):
            candidate = log2f + shift
            gain = np.abs(log2f - local) - np.abs(candidate - local)
            in_range = (candidate >= np.log2(F0_FLOOR_HZ)) & (
                candidate <= np.log2(F0_CEIL_HZ)
            )
            log2f = np.where((gain > min_gain) & in_range, candidate, log2f)

    out[idx] = np.exp2(log2f)
    return out


def _repair_octave_errors_global(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Catch sustained octave errors that the local median filter follows.

    A run of halved frames longer than the filter kernel drags the local
    reference down with it, so it also needs checking against the utterance as a
    whole. Only frames that are close to a whole octave away are touched, which
    leaves genuine excursions of up to nine semitones intact.
    """
    out = np.array(f0, dtype=float)
    idx = np.flatnonzero(voiced)
    if idx.size < 5:
        return out

    log2f = np.log2(out[idx])
    reference = float(np.median(log2f))
    deviation = np.abs(log2f - reference) * 12.0
    for shift in (-1.0, 1.0):
        candidate = log2f + shift
        take = (deviation > GLOBAL_REPAIR_MIN_DEVIATION_ST) & (
            np.abs(candidate - reference) * 12.0 <= GLOBAL_REPAIR_TOL_ST
        )
        log2f = np.where(take, candidate, log2f)

    out[idx] = np.exp2(log2f)
    return out


def _run_pyin(y: np.ndarray, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=SAMPLE_RATE,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
    )
    return f0, np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0)


def extract_f0(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (f0 in Hz with NaN where unvoiced, boolean voiced mask)."""
    if y.size < FRAME_LENGTH:
        n = 1 + y.size // HOP_LENGTH
        return np.full(n, np.nan), np.zeros(n, dtype=bool)

    f0, voiced = _run_pyin(y, F0_FLOOR_HZ, F0_CEIL_HZ)

    if np.count_nonzero(voiced) >= 10:
        centre = float(np.median(f0[voiced]))
        fmin = max(F0_FLOOR_HZ, ADAPTIVE_RANGE_LOW * centre)
        fmax = min(F0_CEIL_HZ, ADAPTIVE_RANGE_HIGH * centre)
        if fmax / fmin >= 2.0:  # pYIN needs at least an octave to search
            f0, voiced = _run_pyin(y, fmin, fmax)

    f0 = _repair_octave_errors(f0, voiced)
    f0 = _repair_octave_errors_global(f0, voiced)
    with np.errstate(invalid="ignore"):
        voiced &= (f0 >= F0_FLOOR_HZ) & (f0 <= F0_CEIL_HZ)
    f0 = np.where(voiced, f0, np.nan)
    return f0, voiced


def restrict_to_speech(voiced: np.ndarray, speech: np.ndarray) -> np.ndarray:
    """Drop voiced frames the VAD places outside speech.

    pYIN will happily report a periodicity in hum or room tone; those frames
    should not contribute to a speaker's pitch statistics.
    """
    n = min(voiced.size, speech.size)
    out = np.zeros(voiced.size, dtype=bool)
    out[:n] = voiced[:n] & speech[:n]
    return out


def f0_statistics(
    f0: np.ndarray, voiced: np.ndarray, n_speech_frames: Optional[int] = None
) -> Dict[str, Any]:
    values = f0[voiced]
    if values.size == 0:
        return {
            "f0_median_hz": None,
            "f0_p5_hz": None,
            "f0_p95_hz": None,
            "f0_range_st": None,
            "f0_std_st": None,
            "n_voiced_frames": 0,
            "voiced_ratio": 0.0,
            "voiced_share_of_speech": 0.0,
            "f0_reliable": False,
        }
    median = float(np.median(values))
    p5, p95 = (float(v) for v in np.percentile(values, [5, 95]))
    semitones = 12.0 * np.log2(values / median)
    share = float(values.size / n_speech_frames) if n_speech_frames else 0.0
    return {
        "f0_median_hz": round(median, 2),
        "f0_p5_hz": round(p5, 2),
        "f0_p95_hz": round(p95, 2),
        # A 5th-to-95th percentile span is used instead of max-minus-min so that
        # one surviving tracking error cannot define the range.
        "f0_range_st": round(float(12.0 * np.log2(p95 / p5)), 2),
        "f0_std_st": round(float(np.std(semitones)), 2),
        "n_voiced_frames": int(values.size),
        "voiced_ratio": round(float(values.size / max(1, f0.size)), 3),
        "voiced_share_of_speech": round(share, 3),
        # Noisy or heavily degraded recordings yield a usable F0 on only a
        # fraction of their speech frames; the resulting statistics are reported
        # but marked, so that they can be filtered rather than quietly believed.
        "f0_reliable": bool(
            values.size >= 25 and share >= MIN_VOICED_SHARE_OF_SPEECH
        ),
    }


# --------------------------------------------------------------------------- #
# energy
# --------------------------------------------------------------------------- #
def active_speech_level_dbfs(rms: np.ndarray, speech: np.ndarray) -> Optional[float]:
    """RMS over speech frames only, in dBFS.

    Averaging over the whole file instead makes the value a function of how much
    pad the segmentation happened to leave: two seconds of leading and trailing
    silence is enough to move a level label across a threshold.
    """
    if rms.size == 0 or not speech.any():
        return None
    level = float(np.sqrt(np.mean(np.square(rms[speech]))))
    return round(20.0 * float(np.log10(max(level, EPS))), 2)


def noise_floor_dbfs(rms: np.ndarray) -> Optional[float]:
    """Rough noise floor, taken as the 5th percentile frame level."""
    if rms.size == 0:
        return None
    return round(20.0 * float(np.log10(max(float(np.percentile(rms, 5)), EPS))), 2)


# --------------------------------------------------------------------------- #
# speaking rate
# --------------------------------------------------------------------------- #
def _ensure_nltk_data() -> None:
    """g2p_en needs a POS tagger and CMUdict; fetch them once if absent."""
    import nltk

    for resource, path in (
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("cmudict", "corpora/cmudict"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def _g2p():
    global _G2P
    if _G2P is None:
        _ensure_nltk_data()
        from g2p_en import G2p

        _G2P = G2p()
    return _G2P


def count_phonemes(text: str) -> Optional[int]:
    """Number of ARPAbet phones the transcript is expected to realise.

    Returns ``None`` for text this grapheme-to-phoneme model cannot handle, so
    that unsupported languages surface as a missing value rather than as a
    confident wrong number. ``g2p_en`` is English-only.
    """
    if not text or not text.strip():
        return None
    if _NON_LATIN.search(text):
        return None
    phones = [p for p in _g2p()(text) if _ARPABET.match(p)]
    return len(phones) or None


def speaking_rates(
    n_phonemes: Optional[int], total_duration: float, speech_duration: float
) -> Dict[str, Any]:
    """Both rates, in phonemes per second.

    ``articulation_rate`` divides by speech time only and is the one that gets
    calibrated; ``speaking_rate`` keeps the pauses in and is reported alongside
    because the difference between the two is itself informative (a speaker who
    is fast but hesitant looks different from one who is uniformly fast).
    """
    if not n_phonemes:
        return {
            "n_phonemes": None,
            "speaking_rate_pps": None,
            "articulation_rate_pps": None,
        }
    return {
        "n_phonemes": int(n_phonemes),
        "speaking_rate_pps": (
            round(n_phonemes / total_duration, 3) if total_duration > 0 else None
        ),
        "articulation_rate_pps": (
            round(n_phonemes / speech_duration, 3) if speech_duration > 0 else None
        ),
    }


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def analyse_audio(
    audio: np.ndarray, text: Optional[str] = None, with_contour: bool = True
) -> Dict[str, Any]:
    """Continuous prosodic features for one utterance. No level labels here."""
    # local imports keep the modules decoupled
    from intonation import collapse_to_intonation, describe_contour

    total_duration = float(audio.size / SAMPLE_RATE)
    rms = frame_rms(audio)
    speech = detect_speech_frames(rms)
    spoken = speech_duration_s(speech)

    f0, voiced = extract_f0(audio)
    voiced = restrict_to_speech(voiced, speech)
    f0 = np.where(voiced, f0, np.nan)

    level = active_speech_level_dbfs(rms, speech)
    floor = noise_floor_dbfs(rms)

    features: Dict[str, Any] = {
        "duration_s": round(total_duration, 3),
        "speech_duration_s": round(spoken, 3),
        "speech_ratio": round(spoken / total_duration, 3) if total_duration else 0.0,
        "energy_dbfs": level,
        "noise_floor_dbfs": floor,
        "est_snr_db": (
            round(level - floor, 2) if level is not None and floor is not None else None
        ),
    }
    features.update(f0_statistics(f0, voiced, int(np.count_nonzero(speech))))
    features.update(speaking_rates(count_phonemes(text or ""), total_duration, spoken))
    if with_contour:
        contour = describe_contour(f0, voiced)
        if not features["f0_reliable"]:
            # A contour drawn through a handful of scattered voiced frames is not
            # a description of the utterance's intonation.
            contour.update(
                contour_shape="insufficient_voiced",
                contour_dynamics="insufficient_voiced",
                contour_confidence=0.0,
            )
        contour["intonation"] = collapse_to_intonation(contour)
        features.update(contour)
    return features


def analyse_file(
    path: str, text: Optional[str] = None, with_contour: bool = True
) -> Dict[str, Any]:
    return analyse_audio(load_audio(path), text=text, with_contour=with_contour)
