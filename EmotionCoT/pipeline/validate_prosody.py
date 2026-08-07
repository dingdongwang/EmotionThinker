#!/usr/bin/env python3
"""Correctness checks for the prosody annotation pipeline.

Every check is a property that must hold for the feature to mean what its name
says. Most of them are regressions against concrete failures of the earlier
implementation, noted inline.

    python validate_prosody.py                  # signal-level checks
    python validate_prosody.py --with-models    # also exercise the neural taggers
    python validate_prosody.py --audio-dir DIR  # print labels for real recordings
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intonation import describe_contour  # noqa: E402
from prosody_features import (  # noqa: E402
    SAMPLE_RATE,
    active_speech_level_dbfs,
    analyse_audio,
    analyse_file,
    detect_speech_frames,
    extract_f0,
    f0_statistics,
    frame_rms,
    load_audio,
)

SR = SAMPLE_RATE
RESULTS: List[Tuple[bool, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((bool(condition), name, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
# synthetic signals
# --------------------------------------------------------------------------- #
def harmonic(f0: float, dur: float = 3.0, n_harm: int = 25, decay: float = 0.85,
             amp: float = 0.3, first_harm: int = 1) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    y = sum((decay**k) * np.sin(2 * np.pi * f0 * k * t) for k in range(first_harm, n_harm + 1))
    return (y / np.max(np.abs(y)) * amp).astype(np.float32)


def sweep(f_start: float, f_end: float, dur: float = 3.0, n_harm: int = 25,
          decay: float = 0.85, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    phase = 2 * np.pi * np.cumsum(np.linspace(f_start, f_end, len(t))) / SR
    y = sum((decay**k) * np.sin(k * phase) for k in range(1, n_harm + 1))
    return (y / np.max(np.abs(y)) * amp).astype(np.float32)


def with_unvoiced_gaps(y: np.ndarray, n_gaps: int, gap_s: float = 0.12) -> np.ndarray:
    chunks = np.array_split(y, n_gaps + 1)
    gap = np.zeros(int(gap_s * SR), np.float32)
    out: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        out.append(chunk)
        if i < n_gaps:
            out.append(gap)
    return np.concatenate(out)


def measured_f0(y: np.ndarray) -> float:
    f0, voiced = extract_f0(y)
    return f0_statistics(f0, voiced)["f0_median_hz"]


def measured_energy(y: np.ndarray) -> float:
    rms = frame_rms(y)
    return active_speech_level_dbfs(rms, detect_speech_frames(rms))


def contour_of(y: np.ndarray) -> dict:
    f0, voiced = extract_f0(y)
    return describe_contour(f0, voiced)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_pitch() -> None:
    print("\nPitch is fundamental frequency")
    worst = 0.0
    for true_f0 in (80, 110, 130, 180, 250, 330):
        got = measured_f0(harmonic(true_f0))
        worst = max(worst, abs(got - true_f0) / true_f0)
    check("F0 recovered within 2% over 80-330 Hz", worst < 0.02, f"max error {worst:.2%}")

    # The previous implementation averaged every spectral peak, so the reported
    # value moved by a factor of 4.5 across this range at constant F0.
    values = [measured_f0(harmonic(130, decay=d)) for d in (0.95, 0.85, 0.70, 0.50)]
    spread = (max(values) - min(values)) / np.mean(values)
    check("F0 independent of spectral tilt at fixed pitch", spread < 0.01,
          f"spread {spread:.3%} over {values}")

    # A telephone-band or small-speaker recording has little energy at F0 itself.
    got = measured_f0(harmonic(130, first_harm=2))
    check("F0 correct with a missing fundamental", abs(got - 130) / 130 < 0.03,
          f"{got} Hz")

    low = measured_f0(harmonic(95))
    check("male-range F0 is not floored out", low is not None and abs(low - 95) / 95 < 0.02,
          f"{low} Hz")


def check_energy() -> None:
    print("\nEnergy measures the speech, not the padding")
    base = harmonic(130, dur=3.0, amp=0.3)
    reference = measured_energy(base)

    # Two seconds of pad used to move the level by 3.6 dB, across a threshold.
    padded = [
        measured_energy(np.concatenate([
            np.zeros(int(p * SR), np.float32), base, np.zeros(int(p * SR), np.float32)
        ]))
        for p in (0.5, 1.0, 2.0)
    ]
    drift = max(abs(v - reference) for v in padded)
    check("level unchanged by silence padding", drift < 0.5, f"max drift {drift:.2f} dB")

    for gain_db in (-12.0, -6.0, 6.0):
        got = measured_energy(base * (10 ** (gain_db / 20)))
        expected = reference + gain_db
        check(f"level tracks a {gain_db:+.0f} dB gain change",
              abs(got - expected) < 0.2, f"{got:.2f} vs {expected:.2f} dBFS")


def check_rate() -> None:
    print("\nSpeaking rate excludes pauses")
    from prosody_features import count_phonemes

    n = count_phonemes("the quick brown fox jumps over the lazy dog")
    check("phoneme counting returns a plausible count", n is not None and 25 < n < 45, f"{n} phones")
    check("non-English text yields no count, rather than a wrong one",
          count_phonemes("这是一句中文") is None)

    base = harmonic(130, dur=3.0)
    text = "the quick brown fox jumps over the lazy dog"
    padded = np.concatenate([
        np.zeros(int(1.5 * SR), np.float32), base, np.zeros(int(1.5 * SR), np.float32)
    ])
    plain = analyse_audio(base, text=text, with_contour=False)
    with_pad = analyse_audio(padded, text=text, with_contour=False)

    drift = abs(with_pad["articulation_rate_pps"] - plain["articulation_rate_pps"])
    check("articulation rate unchanged by silence padding", drift < 0.15,
          f"{plain['articulation_rate_pps']} vs {with_pad['articulation_rate_pps']} ph/s")
    check("speaking rate does react to padding, and is reported separately",
          with_pad["speaking_rate_pps"] < plain["speaking_rate_pps"] * 0.75,
          f"{plain['speaking_rate_pps']} vs {with_pad['speaking_rate_pps']} ph/s")


def check_contour() -> None:
    print("\nIntonation contour")
    cases = [
        ("rising 100->200 Hz", sweep(100, 200), "rising"),
        ("rising 100->260 Hz", sweep(100, 260), "rising"),
        ("falling 260->100 Hz", sweep(260, 100), "falling"),
        ("steady 130 Hz", harmonic(130), "level"),
        ("rise then fall", np.concatenate([sweep(100, 240, 1.5), sweep(240, 100, 1.5)]),
         "rising-falling"),
        ("fall then rise", np.concatenate([sweep(240, 100, 1.5), sweep(100, 240, 1.5)]),
         "falling-rising"),
    ]
    for name, signal, expected in cases:
        got = contour_of(signal)["contour_shape"]
        check(f"{name} -> {expected}", got == expected, f"got {got}")

    # A pause is not a pitch valley. Zero-filling unvoiced frames used to label
    # this constant-pitch signal falling-rising.
    steady = harmonic(130, dur=1.2)
    for pause in (0.3, 0.6):
        signal = np.concatenate([steady, np.zeros(int(pause * SR), np.float32), steady])
        got = contour_of(signal)["contour_shape"]
        check(f"steady pitch with a {pause}s pause stays level", got == "level", f"got {got}")

    # Whether an utterance contains consonants used to decide whether it could be
    # called rising at all.
    clean = contour_of(sweep(100, 260))["contour_shape"]
    gapped = [contour_of(with_unvoiced_gaps(sweep(100, 260), n))["contour_shape"]
              for n in (1, 3, 6)]
    check("label independent of how many unvoiced gaps the signal has",
          all(g == clean for g in gapped), f"{clean} vs {gapped}")

    # The label space used to change with duration: the same rise was too_short,
    # then flat, then uncertain.
    full = sweep(100, 260, dur=6.0)
    durations = [contour_of(full[: int(d * SR)])["contour_shape"] for d in (1.0, 2.0, 3.0, 6.0)]
    check("label stable across durations of the same contour",
          len(set(durations)) == 1 and durations[0] == "rising", f"{durations}")

    dynamics = contour_of(harmonic(130))["contour_dynamics"]
    check("monotone signal is flat", dynamics == "flat", f"got {dynamics}")
    dynamics = contour_of(sweep(100, 300))["contour_dynamics"]
    check("wide excursion is expressive", dynamics == "expressive", f"got {dynamics}")

    confidence = contour_of(np.concatenate([
        harmonic(130, dur=1.2), np.zeros(int(0.5 * SR), np.float32), harmonic(130, dur=1.2)
    ]))["contour_confidence"]
    check("multi-phrase utterance is flagged with lower confidence", confidence < 1.0,
          f"confidence {confidence}")

    # The three fields above are collapsed into one for the README and for the
    # CoT prompt, which only understands this vocabulary.
    from intonation import collapse_to_intonation

    documented = {"rising", "falling", "rising-falling", "falling-rising", "flat",
                  "expressive", "uncertain", "too_short"}
    collapsed = {name: collapse_to_intonation(contour_of(signal))
                 for name, signal, _ in cases}
    check("collapsed intonation stays inside the documented vocabulary",
          set(collapsed.values()) <= documented, f"{sorted(set(collapsed.values()))}")
    check("a clear rise collapses to rising", collapsed["rising 100->260 Hz"] == "rising",
          collapsed["rising 100->260 Hz"])
    check("a monotone signal collapses to flat", collapsed["steady 130 Hz"] == "flat",
          collapsed["steady 130 Hz"])


def check_stereo() -> None:
    print("\nChannel handling")
    import soundfile as sf

    mono = sweep(120, 220, dur=3.0)
    with tempfile.TemporaryDirectory() as tmp:
        mono_path = os.path.join(tmp, "mono.wav")
        stereo_path = os.path.join(tmp, "stereo.wav")
        sf.write(mono_path, mono, SR)
        sf.write(stereo_path, np.stack([mono, mono], axis=1), SR)

        # `waveform.squeeze()` leaves a (2, N) tensor two-dimensional, so stereo
        # files previously crashed into the pipeline's bare except and vanished.
        loaded = load_audio(stereo_path)
        check("stereo file loads as 1-D mono", loaded.ndim == 1, f"shape {loaded.shape}")

        a = analyse_file(mono_path)
        b = analyse_file(stereo_path)
        check("stereo and mono give the same F0",
              abs(a["f0_median_hz"] - b["f0_median_hz"]) < 1.0,
              f"{a['f0_median_hz']} vs {b['f0_median_hz']} Hz")
        check("stereo and mono give the same contour",
              a["contour_shape"] == b["contour_shape"],
              f"{a['contour_shape']} vs {b['contour_shape']}")


def check_degenerate() -> None:
    print("\nDegenerate input is reported, not guessed")
    silence = np.zeros(int(2.0 * SR), np.float32)
    result = analyse_audio(silence, text="hello world")
    check("silence yields no pitch", result["f0_median_hz"] is None)
    check("silence yields an insufficient-voiced contour",
          result["contour_shape"] == "insufficient_voiced", result["contour_shape"])

    check("silence collapses to too_short rather than to a contour name",
          result["intonation"] == "too_short", result["intonation"])

    result = analyse_audio(harmonic(150, dur=0.2), text="hi")
    check("a 0.2 s clip yields no contour shape",
          result["contour_shape"] == "insufficient_voiced", result["contour_shape"])


def check_models(device: str, whistress_dir: str = None) -> None:
    print("\nSpeaker attributes")
    from speaker_attrs import GENDER_LABELS, SpeakerAttributeTagger, age_to_level

    check("age bins are ordered and cover the range",
          [age_to_level(a) for a in (5, 16, 25, 45, 70)]
          == ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"])

    tagger = SpeakerAttributeTagger(device=device)
    out = tagger.predict(harmonic(120, dur=2.0))
    check("gender label comes from the documented [female, male, child] order",
          out["gender"] in GENDER_LABELS, f"{out['gender']}")
    check("age is produced at all", out["age_years"] is not None, f"{out['age_years']} years")
    check("gender probabilities sum to one",
          abs(sum(out["gender_probs"].values()) - 1.0) < 0.01)

    print("\nWord-level stress")
    from stress_labeling import StressTagger

    stress = StressTagger(device=device, whistress_dir=whistress_dir)
    check("decoder capacity is used instead of a 30-token cap", stress.max_tokens > 100,
          f"max_tokens={stress.max_tokens}")

    long_text = (
        "I did not say he stole the money but I certainly thought about it for a very "
        "long time and I still believe that somebody in this room knows exactly what "
        "happened on that particular evening"
    )
    audio = harmonic(150, dur=6.0)
    result = stress.predict(audio, long_text)
    covered = len(result["stress_pairs"])
    expected = len(long_text.split())
    check("stress covers the whole transcript", covered >= expected - 2,
          f"{covered} of {expected} words")
    check("truncation is reported rather than silent",
          result["stress_truncated"] is False, f"{result['stress_truncated']}")

    # Side by side with the helper this module replaces, on the same input.
    try:
        from whistress.inference_client.utils import (
            inference_from_audio_and_transcription,
        )

        upstream_pairs = inference_from_audio_and_transcription(
            stress._peak_normalise(audio), long_text, stress.model, device
        )
        upstream_words = len([w for w, _ in stress._merge(upstream_pairs) if w.strip()])
        check("more words covered than the upstream 30-token helper",
              covered > upstream_words,
              f"{covered} words here vs {upstream_words} upstream, out of {expected}")
    except Exception as exc:  # upstream layout differs, or transformers moved on
        print(f"  [skip] upstream comparison unavailable ({type(exc).__name__})")


def report_real_audio(audio_dir: str, device: str, with_models: bool,
                      whistress_dir: str = None) -> None:
    paths = sorted(
        os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".flac", ".mp3"))
    )
    if not paths:
        print(f"\nNo audio found in {audio_dir}")
        return

    print(f"\nMeasurements on real recordings in {audio_dir}")
    tagger = None
    if with_models:
        from speaker_attrs import SpeakerAttributeTagger

        tagger = SpeakerAttributeTagger(device=device)

    for path in paths:
        audio = load_audio(path)
        f = analyse_audio(audio)
        line = (
            f"  {os.path.basename(path):<14} "
            f"F0={f['f0_median_hz']:>6} Hz  range={f['f0_range_st']:>5} st  "
            f"energy={f['energy_dbfs']:>7} dBFS  speech={f['speech_ratio']:.2f}  "
            f"{f['contour_shape']:<15} {f['contour_dynamics']:<10} conf={f['contour_confidence']}"
        )
        if tagger is not None:
            attrs = tagger.predict(audio)
            line += f"  {attrs['gender']}/{attrs['age_level']}({attrs['age_years']:.0f}y)"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-models", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--whistress-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    args = parser.parse_args()

    check_pitch()
    check_energy()
    check_rate()
    check_contour()
    check_stereo()
    check_degenerate()
    if args.with_models:
        check_models(args.device, args.whistress_dir)
    if args.audio_dir:
        report_real_audio(args.audio_dir, args.device, args.with_models, args.whistress_dir)

    failures = [name for ok, name, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} checks passed")
    for name in failures:
        print(f"  FAILED: {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
