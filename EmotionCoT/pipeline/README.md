# Prosody annotation pipeline

Automatic acoustic annotation for EmotionCoT. Each utterance is described by
speaker traits, pitch, energy, speaking rate, word-level stress and intonation
contour; those measurements are then supplied to the CoT generation prompt as
grounded facts about the audio.

## Install

```bash
pip install -r requirements.txt
git clone https://github.com/slp-rl/WhiStress && python WhiStress/download_weights.py
export WHISTRESS_DIR=$PWD/WhiStress
```

## Run

```bash
python prosody_labeling.py --input_path in.jsonl --output_path labelled.jsonl
```

Input is JSON Lines with at least `audio_path` and a transcript in
`transcription` (`text` is also accepted). An optional `dataset` field defines
the calibration groups. Any other fields, `emotion` in particular, are carried
through untouched.

The two stages can also be run separately, which is useful when the corpus is
large enough that the measurement pass wants a different machine than the
calibration pass, or when relabelling with different cut points should not
repeat the expensive measurement:

```bash
python prosody_labeling.py --stage extract   --input_path in.jsonl       --output_path features.jsonl --workers 32
python prosody_labeling.py --stage calibrate --input_path features.jsonl --output_path labelled.jsonl
```

`extract` parallelises the signal analysis across processes and runs the two
neural taggers sequentially on the GPU. Utterances that fail are written to a
sibling `.errors.jsonl` with their traceback and counted in the log, rather than
being dropped silently.

## Why two passes

Low / normal / high cannot be defined by absolute constants. Recording gain
differs by tens of dB across the source corpora — EARS is anechoic 48 kHz,
MELD is a television soundtrack — and absolute F0 is dominated by speaker
identity. A fixed cut point mostly encodes which corpus and which speaker a clip
came from. So `extract` only measures, in physical units, and `calibrate`
derives cut points from percentiles of the distribution each clip belongs to:
per corpus for energy and rate, per corpus and gender for pitch.

Defaults are tertiles, giving roughly balanced three-way classes, which is what
the prosodic attribute classification task wants. Use `--percentiles 25 75` for
a wider "normal" band if the labels are meant to read as descriptions.

Thresholds and a distribution report are written next to the output, including
the F0 distribution per predicted gender — if the gender classifier degrades on
some corpus, the male and female medians will stop being about an octave apart.

## Fields produced

| Field | Meaning |
| --- | --- |
| `f0_median_hz`, `f0_p5_hz`, `f0_p95_hz` | fundamental frequency over voiced speech frames |
| `f0_range_st`, `f0_std_st` | pitch span and variability in semitones |
| `f0_reliable`, `voiced_share_of_speech` | whether enough speech frames carried a usable F0 |
| `energy_dbfs`, `noise_floor_dbfs`, `est_snr_db` | active speech level and a recording quality estimate |
| `n_phonemes`, `articulation_rate_pps`, `speaking_rate_pps` | rate excluding and including pauses |
| `intonation` | rising / falling / rising-falling / falling-rising / flat / expressive, or `uncertain` / `too_short` |
| `contour_shape` | rising / falling / rising-falling / falling-rising / level |
| `contour_dynamics` | flat / moderate / expressive |
| `contour_confidence` | heuristic 0..1 reliability of the shape decision |
| `contour_delta_st`, `contour_slope_st`, `contour_turn_depth_st`, `contour_turn_position` | the statistics the shape decision rests on |
| `gender`, `gender_confidence`, `age_years`, `age_level` | speaker traits |
| `stressed_words`, `stress_pairs`, `stress_truncated` | word-level sentence stress |
| `pitch_level`, `energy_level`, `speed_level` | added by `calibrate` |

`intonation` is the single field the top-level README documents and the CoT
prompt reads. The `contour_*` fields behind it are kept because the decision is
really two independent ones — a direction and a size of excursion — and because
the statistics let a doubtful label be re-examined without rerunning the audio.

Label fields are `null`, or `insufficient_voiced`, when the measurement does not
support a judgement. Nothing is guessed to fill a column.

## Methods

**Pitch.** pYIN over 55-500 Hz, then a second pass over a range centred on the
speaker's own median from the first pass. Sustained halving errors, which a
local median filter follows rather than corrects, then fall outside the search
range entirely. Isolated octave jumps are repaired against a local reference,
and frames close to a whole octave from the utterance median are repaired
against the global one. Statistics use only voiced frames that the VAD also
places inside speech.

**Energy.** RMS over speech frames in dBFS. Silence is excluded so that pad and
pause structure cannot move the value.

**Rate.** Phones from `g2p_en` over the transcript, divided by VAD speech
duration. `speaking_rate_pps` keeps the pauses in and is reported alongside,
because the gap between the two distinguishes a fast speaker from a hesitant
one. English only; other languages return `null` rather than a wrong number.

**Intonation.** Voiced frames converted to semitones relative to the utterance
median, carrying their real timestamps, with no interpolation over unvoiced
regions. Direction comes from the difference between the last and first quarter;
a turning point is accepted when a quadratic fit places its vertex away from the
edges, deep enough to hear, and explains meaningfully more variance than a
straight line. Dynamics are cut at 3 and 9 semitones of range, which are
perceptual rather than percentile thresholds — the calibration report prints the
observed spread of `f0_range_st` so those two constants can be checked against
the corpus.

**Stress.** WhiStress teacher-forced on the reference transcript. Only the
tokenisation is reimplemented, because the upstream helper caps transcripts at
30 tokens and silently drops everything past roughly twenty words.

**Speaker traits.** `audeering/wav2vec2-large-robust-24-ft-age-gender`, one
encoder with an age regression head and a three-way gender head. Weights are
loaded explicitly and the load is verified, so a checkpoint that fails to supply
the heads raises instead of predicting from random weights.

## Reasoning augmentation

`api_call.py` turns the annotations into EmotionCoT records by asking GPT-4o for
the reasoning trace, using the prompt template from Appendix B.3 of the paper
verbatim. Put your key in `OPENAI_API_KEY` at the top of the script, or export
it, or pass `--api-key`.

```bash
python api_call.py --input_path labelled.jsonl --output_path emotioncot_augmented.jsonl
```

`--dry-run N` prints N filled-in prompts and exits without spending anything,
which is the fastest way to confirm the annotations reach the template intact.
Output is appended as it arrives and keyed on `audio_path`, so rerunning the
same command resumes an interrupted run instead of paying twice.

## Validation

```bash
python validate_prosody.py --with-models --audio-dir /path/to/wavs
```

Checks that F0 tracks the fundamental and not the spectral envelope, that energy
and rate are invariant to silence padding, that contour labels are invariant to
unvoiced gaps and to duration, that a pause is not read as a pitch valley, that
stereo and mono agree, and that degenerate input is reported rather than
guessed. Each check corresponds to a way the measurement can stop meaning what
its name says.
