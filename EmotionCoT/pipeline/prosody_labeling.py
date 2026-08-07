#!/usr/bin/env python3
"""Automatic prosody annotation pipeline for EmotionCoT.

Annotates each utterance with speaker traits, pitch, energy, speaking rate,
word-level stress and intonation contour, which are then supplied to the CoT
generation prompt as grounded acoustic facts.

Usage
-----
    python prosody_labeling.py --input_path in.jsonl --output_path labelled.jsonl

The input is JSON Lines with at least ``audio_path`` and a transcript in
``transcription`` (``text`` is also accepted). An optional ``dataset`` field
defines the calibration groups; without it the whole input is treated as one
corpus. Any other fields, ``emotion`` in particular, are carried through to the
output untouched.

Internally this is two passes, because the low/normal/high labels are defined
relative to the corpus rather than against absolute constants:

    extract    per-utterance measurement (continuous values only)
    calibrate  percentile cut points per corpus, then label assignment

Both run by default. On a large corpus it is worth splitting them with --stage,
so that a re-labelling with different cut points does not repeat the expensive
measurement pass:

    python prosody_labeling.py --stage extract \\
        --input_path in.jsonl --output_path features.jsonl
    python prosody_labeling.py --stage calibrate \\
        --input_path features.jsonl --output_path labelled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration import (  # noqa: E402
    DEFAULT_MIN_GROUP_SIZE,
    DEFAULT_PERCENTILES,
    apply_thresholds,
    fit_thresholds,
    summarise,
)

TEXT_FIELDS = ("transcription", "text", "transcript")


def _progress(iterable, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        def generator():
            for i, item in enumerate(iterable, 1):
                if i % 200 == 0 or i == total:
                    print(f"  {desc}: {i}/{total}", file=sys.stderr, flush=True)
                yield item

        return generator()


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_text(record: Dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if value:
            try:
                from ftfy import fix_text

                return fix_text(str(value))
            except ImportError:
                return str(value)
    return ""


# --------------------------------------------------------------------------- #
# stage 1a: signal features (CPU, parallel)
# --------------------------------------------------------------------------- #
def _dsp_worker(payload: Tuple[int, str, str]) -> Tuple[int, Dict[str, Any], Optional[str]]:
    index, audio_path, text = payload
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from prosody_features import analyse_file

        return index, analyse_file(audio_path, text=text), None
    except Exception:
        return index, {}, traceback.format_exc(limit=3)


def run_dsp(records: List[Dict[str, Any]], workers: int) -> List[Optional[str]]:
    payloads = [(i, r["audio_path"], get_text(r)) for i, r in enumerate(records)]
    errors: List[Optional[str]] = [None] * len(records)

    if workers <= 1:
        results = _progress(map(_dsp_worker, payloads), len(payloads), "signal features")
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = _progress(
            pool.map(_dsp_worker, payloads, chunksize=8), len(payloads), "signal features"
        )

    for index, features, error in results:
        if error:
            errors[index] = error
        else:
            records[index].update(features)
    return errors


# --------------------------------------------------------------------------- #
# stage 1b: model-based attributes (GPU, sequential)
# --------------------------------------------------------------------------- #
def run_models(
    records: List[Dict[str, Any]],
    errors: List[Optional[str]],
    device: str,
    whistress_dir: Optional[str],
) -> None:
    from prosody_features import load_audio
    from speaker_attrs import SpeakerAttributeTagger
    from stress_labeling import StressTagger

    speaker_tagger = SpeakerAttributeTagger(device=device)
    stress_tagger = StressTagger(device=device, whistress_dir=whistress_dir)

    indices = [i for i, error in enumerate(errors) if error is None]
    for index in _progress(indices, len(indices), "speaker + stress"):
        record = records[index]
        try:
            audio = load_audio(record["audio_path"])
            record.update(speaker_tagger.predict(audio))
            record.update(stress_tagger.predict(audio, get_text(record)))
        except Exception:
            errors[index] = traceback.format_exc(limit=3)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_extract(args) -> List[Dict[str, Any]]:
    records = read_jsonl(args.input, args.limit)
    print(f"[extract] {len(records)} utterances from {args.input}", file=sys.stderr)

    errors = run_dsp(records, args.workers)
    if not args.skip_models:
        run_models(records, errors, args.device, args.whistress_dir)

    failed = [
        {"audio_path": records[i].get("audio_path"), "error": error}
        for i, error in enumerate(errors)
        if error
    ]
    kept = [r for r, error in zip(records, errors) if error is None]

    if failed:
        error_path = args.errors or (os.path.splitext(args.output)[0] + ".errors.jsonl")
        write_jsonl(error_path, failed)
        reasons = Counter(e["error"].strip().splitlines()[-1][:120] for e in failed)
        print(
            f"[extract] {len(failed)}/{len(records)} failed, written to {error_path}",
            file=sys.stderr,
        )
        for reason, count in reasons.most_common(5):
            print(f"           {count:>6}  {reason}", file=sys.stderr)
    else:
        print("[extract] no failures", file=sys.stderr)

    write_jsonl(args.output, kept)
    print(f"[extract] wrote {len(kept)} records to {args.output}", file=sys.stderr)
    return kept


def stage_calibrate(args, records: Optional[List[Dict[str, Any]]] = None) -> None:
    if records is None:
        records = read_jsonl(args.input)
    print(f"[calibrate] {len(records)} utterances", file=sys.stderr)

    thresholds = fit_thresholds(
        records,
        group_field=args.group_by,
        percentiles=tuple(args.percentiles),
        min_group_size=args.min_group_size,
    )
    for record in records:
        apply_thresholds(record, thresholds)

    threshold_path = args.thresholds or (
        os.path.splitext(args.output)[0] + ".thresholds.json"
    )
    report = summarise(records)
    with open(threshold_path, "w", encoding="utf-8") as handle:
        json.dump({"thresholds": thresholds, "report": report}, handle, indent=2)

    write_jsonl(args.output, records)
    print(f"[calibrate] wrote {len(records)} records to {args.output}", file=sys.stderr)
    print(f"[calibrate] thresholds and report in {threshold_path}", file=sys.stderr)
    for field in ("pitch_level", "energy_level", "speed_level", "intonation",
                  "gender", "age_level"):
        counts = report["label_counts"].get(field, {})
        rendered = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"           {field:<18} {rendered}", file=sys.stderr)
    print(
        f"           f0_range_st percentiles {report['f0_range_st_percentiles']}",
        file=sys.stderr,
    )
    for gender, stats in sorted(report["f0_by_predicted_gender"].items()):
        print(
            f"           F0 for predicted {gender:<8} n={stats['n']:<6} "
            f"p10/median/p90 = {stats['f0_p10']}/{stats['f0_median']}/{stats['f0_p90']} Hz",
            file=sys.stderr,
        )
    print(
        f"           {report['n_f0_unreliable']} utterances have an unreliable F0 track",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_path", "--input", dest="input", required=True,
                        help="input JSON Lines file")
    parser.add_argument("--output_path", "--output", dest="output", required=True,
                        help="where to write the annotated JSON Lines file")
    parser.add_argument("--stage", choices=("all", "extract", "calibrate"), default="all")

    measurement = parser.add_argument_group("measurement (--stage all, extract)")
    measurement.add_argument("--workers", type=int,
                             default=max(1, (os.cpu_count() or 2) // 2))
    measurement.add_argument("--device", default="cuda")
    measurement.add_argument("--whistress-dir", default=None)
    measurement.add_argument("--skip-models", action="store_true",
                             help="signal features only, no speaker or stress model")
    measurement.add_argument("--errors", default=None,
                             help="where to write records that failed (default: "
                                  "alongside the output)")
    measurement.add_argument("--limit", type=int, default=None)
    measurement.add_argument("--features", default=None,
                             help="where to keep the intermediate measurements")

    labelling = parser.add_argument_group("labelling (--stage all, calibrate)")
    labelling.add_argument("--group-by", default="dataset",
                           help="record field defining a calibration group; "
                                "'' for one group")
    labelling.add_argument("--percentiles", type=float, nargs=2,
                           default=list(DEFAULT_PERCENTILES))
    labelling.add_argument("--min-group-size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    labelling.add_argument("--thresholds", default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.group_by == "":
        args.group_by = None

    if args.stage == "extract":
        stage_extract(args)
    elif args.stage == "calibrate":
        stage_calibrate(args)
    else:
        final_output = args.output
        args.output = args.features or (os.path.splitext(final_output)[0] + ".features.jsonl")
        records = stage_extract(args)
        args.output = final_output
        stage_calibrate(args, records)


if __name__ == "__main__":
    main()
