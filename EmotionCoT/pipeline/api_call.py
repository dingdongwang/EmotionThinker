#!/usr/bin/env python3
"""EmotionCoT-style reasoning augmentation with the OpenAI API.

Takes the prosody annotations produced by ``prosody_labeling.py`` and asks
GPT-4o to write the reasoning trace that turns them into an EmotionCoT record.
The prompt is the template from Appendix B.3 (Figure 5) of the paper, reproduced
verbatim.

Usage
-----
    python api_call.py \\
        --input_path /path/to/prosody_labeling.jsonl \\
        --output_path /path/to/emotioncot_augmented.jsonl

Set your credentials in ``OPENAI_API_KEY`` below, or export it in the
environment, or pass ``--api-key``.

Output is appended as it arrives and keyed on ``audio_path``, so re-running the
same command resumes where an interrupted run stopped. Records the API rejected
are written to a sibling ``.errors.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# credentials: fill these in, or use the environment / command line
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = ""
# Leave empty for the public OpenAI endpoint. Set it for Azure OpenAI or any
# other OpenAI-compatible gateway.
OPENAI_BASE_URL = ""
# Only used by Azure-style deployments that authenticate per organisation.
OPENAI_ORGANIZATION = ""

DEFAULT_MODEL = "gpt-4o"

# The paper asks for 50-200 words and for stylistic variation across outputs, so
# sampling stays at the API default temperature rather than being made greedy.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 512


# --------------------------------------------------------------------------- #
# Appendix B.3, Figure 5: "Prompt template used to elicit emotion reasoning
# traces from GPT-4o". Reproduced verbatim; only the line wrapping differs.
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = """You are an expert in emotional speech and vocal analysis. You have listened to an audio sample with the following cues:
- `groundtruth_emotion`: "{groundtruth_emotion}"
- `transcription`: "{transcription}"
- `speaker_gender`: "{speaker_gender}"
- `speaker_age`: "{speaker_age}"
- `pitch_level`: "{pitch_level}"
- `energy_level`: "{energy_level}"
- `speed_level`: "{speed_level}"
- `intonation_contour`: "{intonation_contour}"
- `stressed_word`: "{stressed_word}"

Your task is to provide a detailed reasoning explanation of **why** the speaker is likely expressing the emotion labeled as `{groundtruth_emotion}`.

**Important:** Assume you do **not** know the ground-truth label. Like a human listener, infer the most likely emotion through step-by-step reasoning. Let the conclusion arise naturally from your reasoning process. The reasoning process focus should be on **acoustic and prosodic cues**, while semantic content may serve as a secondary reference.**

**Requirements:**
- Explicitly reference and quoting the **transcription** in your reasoning
- Naturally incorporate the speaker profile into your writing (e.g., "a middle-aged female speaker").
- Discuss key acoustic/prosodic features you hear, such as pitch, energy, speech rate, intonation, stress, etc. Focus on features that you believe strongly support the predicted emotion, and you may omit or downplay others that seem unimportant or irrelevant.
- Aim for stylistic variation across outputs by using diverse sentence structures and distributing your reasoning across multiple, well-structured sentences.
- When analyzing the semantic content, consider whether it aligns with the labeled emotion. If it does, point out specific cues (e.g., frustration, concern); if not, explain how the emotion may still be conveyed prosodically despite the content.
- If the original given `intonation_contour` is "uncertain" or "too_short", you can ignore this part analysis.
- You may briefly mention stress patterns if they provide insight, but you are encouraged to omit details that seem incidental or not emotionally meaningful.
- Length: 50 – 200 words. Do **not** exceed 200 words.
- Do **not** begin your explanation with the emotion label. Instead, simulate a human reasoning process where the emotional interpretation emerges gradually through observation and analysis.

Only return the emotional reasoning content."""


# Placeholder -> the record fields it may be stored under, best first. The
# aliases let the same script read this pipeline's output and the field names
# used in the released EmotionCoT files.
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "groundtruth_emotion": ("emotion", "gt", "groundtruth_emotion", "label"),
    "transcription": ("transcription", "text", "transcript"),
    "speaker_gender": ("gender", "gender_new", "speaker_gender"),
    "speaker_age": ("age_level", "age_group", "speaker_age"),
    "pitch_level": ("pitch_level",),
    "energy_level": ("energy_level",),
    "speed_level": ("speed_level",),
    "intonation_contour": ("intonation", "intonation_contour", "contour_shape"),
    "stressed_word": ("stressed_words", "stress", "stress_words"),
}

REQUIRED_PLACEHOLDERS = ("groundtruth_emotion", "transcription")

# An absent contour is exactly the case the template already knows how to skip.
MISSING_DEFAULTS = {"intonation_contour": "uncertain"}
MISSING_DEFAULT = "unknown"


def _render_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def build_prompt(record: Dict[str, Any]) -> str:
    """Fill the paper's template from one annotated record.

    Raises ``ValueError`` when the emotion label or the transcript is missing,
    since neither has a sensible stand-in: the template quotes the transcript
    and reasons towards the label.
    """
    values: Dict[str, str] = {}
    for placeholder, aliases in FIELD_ALIASES.items():
        value = next(
            (record[a] for a in aliases if record.get(a) not in (None, "", [])), None
        )
        if value is None:
            if placeholder in REQUIRED_PLACEHOLDERS:
                raise ValueError(f"record has no {' / '.join(aliases)} field")
            value = MISSING_DEFAULTS.get(placeholder, MISSING_DEFAULT)
        values[placeholder] = _render_value(value)
    return PROMPT_TEMPLATE.format(**values)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def make_client(args):
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")

    key = args.api_key or OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit(
            "No API key. Set OPENAI_API_KEY at the top of this file, export it in "
            "the environment, or pass --api-key."
        )

    base_url = args.base_url or OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL", "")
    organization = OPENAI_ORGANIZATION or os.environ.get("OPENAI_ORG_ID", "")
    return OpenAI(
        api_key=key,
        base_url=base_url or None,
        organization=organization or None,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )


def request_reasoning(client, record: Dict[str, Any], args) -> str:
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": build_prompt(record)}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def record_key(record: Dict[str, Any], index: int) -> str:
    return str(record.get("audio_path") or record.get("id") or f"#{index}")


def already_done(path: str) -> set:
    """Keys present in a previous, possibly interrupted, run of the same command."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(record_key(json.loads(line), index))
            except json.JSONDecodeError:
                continue  # truncated final line from a killed run
    return done


class JsonlAppender:
    """Append-as-you-go, so an interrupted run keeps everything it paid for."""

    def __init__(self, path: str):
        self.handle = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        with self.lock:
            self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _progress(iterable: Iterable, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        def generator():
            for i, item in enumerate(iterable, 1):
                if i % 50 == 0 or i == total:
                    print(f"  {desc}: {i}/{total}", file=sys.stderr, flush=True)
                yield item

        return generator()


# --------------------------------------------------------------------------- #
def run(args) -> int:
    records = read_jsonl(args.input, args.limit)
    print(f"[api] {len(records)} records from {args.input}", file=sys.stderr)

    if args.dry_run:
        for index, record in enumerate(records[: args.dry_run]):
            print(f"\n----- prompt {index + 1} / {record_key(record, index)} -----")
            try:
                print(build_prompt(record))
            except ValueError as exc:
                print(f"[skipped] {exc}")
        return 0

    done = already_done(args.output) if args.resume else set()
    pending = [
        (i, r) for i, r in enumerate(records) if record_key(r, i) not in done
    ]
    if done:
        print(f"[api] {len(done)} already present, {len(pending)} to go", file=sys.stderr)

    client = make_client(args)
    writer = JsonlAppender(args.output)
    error_path = args.errors or (os.path.splitext(args.output)[0] + ".errors.jsonl")
    errors = JsonlAppender(error_path)
    n_failed = 0

    def handle(item: Tuple[int, Dict[str, Any]]) -> bool:
        index, record = item
        try:
            reasoning = request_reasoning(client, record, args)
        except Exception as exc:
            errors.write(
                {
                    "audio_path": record.get("audio_path"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return False
        writer.write({**record, args.field: reasoning})
        return True

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for ok in _progress(pool.map(handle, pending), len(pending), "reasoning"):
                if not ok:
                    n_failed += 1
    finally:
        writer.close()
        errors.close()

    print(f"[api] wrote {len(pending) - n_failed} records to {args.output}", file=sys.stderr)
    if n_failed:
        print(f"[api] {n_failed} failed, see {error_path}", file=sys.stderr)
    elif os.path.getsize(error_path) == 0:
        os.remove(error_path)
    return 1 if n_failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input_path", "--input", dest="input", required=True,
                        help="JSON Lines file from prosody_labeling.py")
    parser.add_argument("--output_path", "--output", dest="output", required=True,
                        help="where to write the augmented JSON Lines file")
    parser.add_argument("--field", default="reasoning",
                        help="record field to store the generated trace in")

    api = parser.add_argument_group("API")
    api.add_argument("--api-key", default=None, help="overrides OPENAI_API_KEY")
    api.add_argument("--base-url", default=None,
                     help="OpenAI-compatible endpoint, for Azure or a gateway")
    api.add_argument("--model", default=DEFAULT_MODEL)
    api.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    api.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    api.add_argument("--max-retries", type=int, default=5)
    api.add_argument("--timeout", type=float, default=60.0)

    run_opts = parser.add_argument_group("run")
    run_opts.add_argument("--workers", type=int, default=8,
                          help="concurrent requests; lower this if you are rate limited")
    run_opts.add_argument("--limit", type=int, default=None)
    run_opts.add_argument("--errors", default=None)
    run_opts.add_argument("--no-resume", dest="resume", action="store_false",
                          help="regenerate records already present in the output")
    run_opts.add_argument("--dry-run", type=int, nargs="?", const=1, default=0,
                          metavar="N",
                          help="print N filled-in prompts and exit without calling the API")

    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
