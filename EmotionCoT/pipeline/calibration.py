"""Turn continuous prosodic measurements into low / normal / high labels.

Absolute thresholds cannot work here. Recording gain differs by tens of dB
between corpora (an anechoic 48 kHz studio recording against a television
soundtrack), and absolute F0 is dominated by speaker identity, so a fixed cut
point mostly encodes which corpus and which speaker a clip came from. Instead
the cut points are percentiles of the distribution the clip actually belongs to:
per corpus for energy and rate, and per corpus and gender for pitch.

Default cuts are tertiles, which produce roughly balanced three-way classes.
That is what the prosodic attribute classification task in the paper wants; if
the labels are meant to read as descriptions rather than as balanced classes,
widen the middle band with --percentiles (for example 25 75).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_PERCENTILES = (100.0 / 3.0, 200.0 / 3.0)
DEFAULT_MIN_GROUP_SIZE = 50

# value field -> (output field, labels ordered from low to high, extra grouping key)
FEATURE_SPECS: Dict[str, Tuple[str, Tuple[str, str, str], Optional[str]]] = {
    "f0_median_hz": ("pitch_level", ("low", "normal", "high"), "gender"),
    "energy_dbfs": ("energy_level", ("low", "normal", "high"), None),
    "articulation_rate_pps": ("speed_level", ("slow", "normal", "fast"), None),
}

GLOBAL_GROUP = "__all__"


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (q / 100.0) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def group_key(record: Dict[str, Any], group_field: Optional[str], extra: Optional[str]) -> str:
    parts = []
    if group_field:
        parts.append(str(record.get(group_field) or GLOBAL_GROUP))
    else:
        parts.append(GLOBAL_GROUP)
    if extra:
        parts.append(str(record.get(extra) or "unknown"))
    return "|".join(parts)


def fit_thresholds(
    records: Iterable[Dict[str, Any]],
    group_field: Optional[str] = "dataset",
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> Dict[str, Any]:
    records = list(records)
    thresholds: Dict[str, Any] = {
        "percentiles": list(percentiles),
        "min_group_size": min_group_size,
        "group_field": group_field,
        "features": {},
    }

    for value_field, (output_field, labels, extra) in FEATURE_SPECS.items():
        buckets: Dict[str, List[float]] = defaultdict(list)
        pooled: Dict[str, List[float]] = defaultdict(list)
        for record in records:
            value = record.get(value_field)
            if value is None:
                continue
            buckets[group_key(record, group_field, extra)].append(float(value))
            # The pooled fallback still conditions on gender for pitch, because
            # pooling male and female F0 would put most women above the upper cut.
            pooled[group_key(record, None, extra)].append(float(value))

        entry: Dict[str, Any] = {
            "value_field": value_field,
            "output_field": output_field,
            "labels": list(labels),
            "groups": {},
            "fallback": {},
        }
        for key, values in sorted(pooled.items()):
            if len(values) >= min_group_size:
                entry["fallback"][key] = {
                    "n": len(values),
                    "cuts": [round(_percentile(values, q), 4) for q in percentiles],
                }
        for key, values in sorted(buckets.items()):
            entry["groups"][key] = {
                "n": len(values),
                "cuts": (
                    [round(_percentile(values, q), 4) for q in percentiles]
                    if len(values) >= min_group_size
                    else None
                ),
            }
        thresholds["features"][value_field] = entry

    return thresholds


def _lookup_cuts(
    thresholds: Dict[str, Any], value_field: str, record: Dict[str, Any]
) -> Optional[List[float]]:
    entry = thresholds["features"].get(value_field)
    if not entry:
        return None
    extra = FEATURE_SPECS[value_field][2]
    key = group_key(record, thresholds.get("group_field"), extra)
    group = entry["groups"].get(key)
    if group and group.get("cuts"):
        return group["cuts"]
    fallback = entry["fallback"].get(group_key(record, None, extra))
    return fallback["cuts"] if fallback else None


def apply_thresholds(record: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, Any]:
    for value_field, (output_field, labels, _) in FEATURE_SPECS.items():
        value = record.get(value_field)
        cuts = _lookup_cuts(thresholds, value_field, record)
        if value is None or cuts is None:
            record[output_field] = None
            continue
        value = float(value)
        record[output_field] = (
            labels[0] if value <= cuts[0] else labels[1] if value <= cuts[1] else labels[2]
        )
    return record


def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Label counts plus the contour statistics whose thresholds stay fixed.

    ``contour_dynamics`` is cut at perceptually motivated semitone values rather
    than percentiles, so the observed spread of ``f0_range_st`` is reported here
    to make those two constants auditable against real data.
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        for field in ("pitch_level", "energy_level", "speed_level", "intonation",
                      "contour_shape", "contour_dynamics", "gender", "age_level"):
            counts[field][str(record.get(field))] += 1

    ranges = sorted(
        float(r["f0_range_st"]) for r in records if r.get("f0_range_st") is not None
    )
    spread = {}
    if ranges:
        spread = {
            f"p{q:g}": round(_percentile(ranges, q), 2)
            for q in (5, 25, 50, 75, 95)
        }

    # Pitch is calibrated within each predicted gender, so a gender classifier
    # going wrong at scale would quietly distort the pitch labels. Reporting the
    # F0 distribution per predicted gender makes that visible: the male and
    # female medians should sit roughly an octave apart.
    by_gender: Dict[str, Any] = {}
    for gender in {str(r.get("gender")) for r in records}:
        values = sorted(
            float(r["f0_median_hz"])
            for r in records
            if str(r.get("gender")) == gender and r.get("f0_median_hz") is not None
        )
        if values:
            by_gender[gender] = {
                "n": len(values),
                "f0_p10": round(_percentile(values, 10), 1),
                "f0_median": round(_percentile(values, 50), 1),
                "f0_p90": round(_percentile(values, 90), 1),
            }

    return {
        "n_records": len(records),
        "n_f0_unreliable": sum(1 for r in records if r.get("f0_reliable") is False),
        "label_counts": {k: dict(v) for k, v in counts.items()},
        "f0_range_st_percentiles": spread,
        "f0_by_predicted_gender": by_gender,
    }
