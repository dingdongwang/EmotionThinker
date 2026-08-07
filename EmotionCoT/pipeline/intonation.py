"""Intonation contour description from a pYIN F0 track.

The analysis runs on voiced frames only, carrying their real timestamps. It never
substitutes zeros for unvoiced frames: doing so turns every consonant and every
pause into an artificial pitch valley, which makes a mid-utterance pause look
like a falling-rising contour and makes the normalised F0 range approach 1.0 for
essentially every utterance.

Working in semitones rather than Hz matters too, because pitch movement is
perceived on a log scale: a 40 Hz rise starting from 90 Hz is a large movement,
the same rise starting from 300 Hz is barely audible.

The output is split into three independent fields:

``contour_shape``      rising / falling / rising-falling / falling-rising / level
``contour_dynamics``   flat / moderate / expressive
``contour_confidence`` heuristic 0..1, see ``_confidence``

Both label fields can also take ``insufficient_voiced`` when there is not enough
voiced speech to support a judgement.

``collapse_to_intonation`` folds the three back into the single ``intonation``
string that the README documents and the CoT prompt consumes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from prosody_features import HOP_LENGTH, SAMPLE_RATE

# A semitone is a ~6% change in F0. Neutral read speech typically spans 4-8
# semitones; below ~3 it is heard as monotone, and sustained spans beyond ~9 are
# characteristic of expressive or emotional delivery.
FLAT_RANGE_ST = 3.0
EXPRESSIVE_RANGE_ST = 9.0

# Direction of pitch movement in connected speech becomes reliably audible at
# somewhat over one semitone; 2 leaves margin above the tracker's own jitter.
RISE_FALL_DELTA_ST = 2.0
TURN_DEPTH_ST = 2.0

# A turning point only counts as one if it sits away from the edges, otherwise
# any monotone contour has a "turn" at one end.
TURN_POSITION_RANGE = (0.25, 0.75)
# ...and the quadratic has to explain meaningfully more variance than a straight
# line, so that noise alone cannot manufacture curvature.
TURN_MIN_R2_GAIN = 0.05

DYNAMICS_MIN_VOICED_S = 0.25
SHAPE_MIN_VOICED_S = 0.50
LONG_PAUSE_S = 0.30

INSUFFICIENT = "insufficient_voiced"

DIRECTIONAL_SHAPES = ("rising", "falling", "rising-falling", "falling-rising")
# Below this the deciding statistic sits so close to its threshold that the
# label would be a coin flip; the prompt template treats "uncertain" as a signal
# to leave intonation out of the reasoning altogether, which is the right
# outcome here.
MIN_REPORTABLE_CONFIDENCE = 0.15


def _smooth(values: np.ndarray) -> np.ndarray:
    if values.size >= 5:
        values = median_filter(values, size=5, mode="nearest")
    if values.size >= 9:
        values = savgol_filter(values, window_length=9, polyorder=2)
    return values


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot


def _confidence(
    shape: str,
    delta_st: float,
    turn_depth_st: Optional[float],
    voiced_s: float,
    n_long_pauses: int,
) -> float:
    """Heuristic reliability score, not a calibrated probability.

    It combines how far the deciding statistic sits past its threshold, how much
    voiced material the decision rests on, and whether the utterance is split by
    pauses (a single contour fitted across two prosodic phrases describes neither
    of them well).
    """
    if shape in ("rising", "falling"):
        margin = abs(delta_st) / RISE_FALL_DELTA_ST - 1.0
    elif shape in ("rising-falling", "falling-rising"):
        margin = abs(turn_depth_st or 0.0) / TURN_DEPTH_ST - 1.0
    elif shape == "level":
        margin = 1.0 - abs(delta_st) / RISE_FALL_DELTA_ST
    else:
        return 0.0

    score = float(np.clip(margin, 0.0, 1.0))
    score *= float(np.clip(voiced_s / (2.0 * SHAPE_MIN_VOICED_S), 0.0, 1.0))
    if n_long_pauses:
        score *= 0.7**n_long_pauses
    return round(score, 3)


def collapse_to_intonation(contour: Dict[str, Any]) -> str:
    """One string over the vocabulary the README and the CoT prompt use.

    The paper describes contours at two levels, a coarse style (expressive vs
    flat) and a fine-grained pattern, but the annotation carries a single field,
    so the fine-grained pattern wins whenever there is one and the coarse style
    fills in otherwise.
    """
    shape = contour.get("contour_shape")
    if shape in (None, INSUFFICIENT):
        return "too_short"
    if float(contour.get("contour_confidence") or 0.0) < MIN_REPORTABLE_CONFIDENCE:
        return "uncertain"
    if shape in DIRECTIONAL_SHAPES:
        return shape
    return "expressive" if contour.get("contour_dynamics") == "expressive" else "flat"


def describe_contour(f0_hz: np.ndarray, voiced: np.ndarray) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "contour_shape": INSUFFICIENT,
        "contour_dynamics": INSUFFICIENT,
        "contour_confidence": 0.0,
        "contour_delta_st": None,
        "contour_slope_st": None,
        "contour_turn_depth_st": None,
        "contour_turn_position": None,
        "contour_n_long_pauses": 0,
    }

    idx = np.flatnonzero(voiced)
    if idx.size < 2:
        return result

    voiced_s = idx.size * HOP_LENGTH / SAMPLE_RATE
    values = f0_hz[idx]
    semitones = _smooth(12.0 * np.log2(values / np.median(values)))

    gaps = np.diff(idx) * HOP_LENGTH / SAMPLE_RATE
    n_long_pauses = int(np.count_nonzero(gaps > LONG_PAUSE_S))
    result["contour_n_long_pauses"] = n_long_pauses

    p5, p95 = np.percentile(semitones, [5, 95])
    range_st = float(p95 - p5)

    if voiced_s < DYNAMICS_MIN_VOICED_S:
        return result
    if range_st < FLAT_RANGE_ST:
        result["contour_dynamics"] = "flat"
    elif range_st > EXPRESSIVE_RANGE_ST:
        result["contour_dynamics"] = "expressive"
    else:
        result["contour_dynamics"] = "moderate"

    if voiced_s < SHAPE_MIN_VOICED_S:
        return result

    time = (idx - idx[0]) / max(1, idx[-1] - idx[0])
    head, tail = time <= 0.25, time >= 0.75
    delta_st = float(np.median(semitones[tail]) - np.median(semitones[head]))

    lin = np.polyfit(time, semitones, 1)
    quad = np.polyfit(time, semitones, 2)
    r2_gain = _r2(semitones, np.polyval(quad, time)) - _r2(
        semitones, np.polyval(lin, time)
    )

    a, b, _ = quad
    turn_position: Optional[float] = None
    turn_depth: Optional[float] = None
    if abs(a) > 1e-9:
        turn_position = float(-b / (2.0 * a))
        chord_midpoint = 0.5 * (np.polyval(quad, 0.0) + np.polyval(quad, 1.0))
        turn_depth = float(np.polyval(quad, turn_position) - chord_midpoint)

    result["contour_delta_st"] = round(delta_st, 2)
    result["contour_slope_st"] = round(float(lin[0]), 2)
    result["contour_turn_depth_st"] = (
        round(turn_depth, 2) if turn_depth is not None else None
    )
    result["contour_turn_position"] = (
        round(turn_position, 3) if turn_position is not None else None
    )

    has_turn = (
        turn_position is not None
        and TURN_POSITION_RANGE[0] <= turn_position <= TURN_POSITION_RANGE[1]
        and abs(turn_depth) >= TURN_DEPTH_ST
        and r2_gain >= TURN_MIN_R2_GAIN
    )

    if has_turn:
        shape = "rising-falling" if turn_depth > 0 else "falling-rising"
    elif delta_st >= RISE_FALL_DELTA_ST:
        shape = "rising"
    elif delta_st <= -RISE_FALL_DELTA_ST:
        shape = "falling"
    else:
        shape = "level"

    result["contour_shape"] = shape
    result["contour_confidence"] = _confidence(
        shape, delta_st, turn_depth, voiced_s, n_long_pauses
    )
    return result
