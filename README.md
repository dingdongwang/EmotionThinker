<div align="center">
    <h1>
    EmotionThinker
    </h1>
Official repository of the ICLR 2026 (Oral) paper  
<b><em>"EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning".</em></b> 
For details, please refer to our  
<a href="https://arxiv.org/pdf/2601.15668">Paper</a>.
<br><br>
<img src="assets/intro.png" alt="EmotionThinker Overview" width="65%">
<br><br>

<a href="https://arxiv.org/pdf/2601.15668">
<img src="https://img.shields.io/badge/Paper-arXiv-red">
</a>

<a href="https://huggingface.co/ddwang2000/EmotionThinker">
<img src="https://img.shields.io/badge/Models-HuggingFace-yellow">
</a>

<a href="https://huggingface.co/datasets/ddwang2000/EmotionCoT">
<img src="https://img.shields.io/badge/Dataset-EmotionCoT-blue">
</a>

</div>


## Introduction

**EmotionThinker** is the first RL–enhanced SpeechLLM framework for interpretable speech emotion reasoning.

Unlike conventional speech emotion recognition (SER) systems that treat emotion as a flat classification problem, EmotionThinker reframes SER as a **deep reasoning** problem, enabling models to jointly produce accurate emotion labels and **structured, human-aligned explanations**.

EmotionThinker offers the following advantages: 

- Higher emotion recognition accuracy compared to existing SpeechLLMs; 
- Deep reasoning ability to integrate emotion-related cues for justification; 
- Fine-grained audio caption covering speaker traits, prosodic cues and semantic information. 


## News

- [Feb. 12, 2026] We open-source the **EmotionThinker** model on [Hugging Face](https://huggingface.co/ddwang2000/EmotionThinker).

- [Feb. 12, 2026] We release the **EmotionCoT dataset** on [Hugging Face](https://huggingface.co/datasets/ddwang2000/EmotionCoT).

- [Feb. 05, 2026] 🎉 EmotionThinker is selected for **Oral Presentation** at ICLR 2026.

- [Jan. 26, 2026] 🎉 EmotionThinker is accepted to **ICLR 2026**. See you in Brazil! 🇧🇷

## Inference with EmotionThinker

**Step 0: Prepare invironment**

```bash
conda create -n emotionthinker python=3.10
conda activate emotionthinker
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Step 1: Download EmotionThinker Model**

Download the pretrained EmotionThinker model from [Hugging Face](https://huggingface.co/ddwang2000/EmotionThinker). Set the local model path accordingly.

**Step 2: Run Inference Code**

```
python scripts/emotionthinker_infer.py
```

## EmotionCoT 

The **EmotionCoT** section provide structured prosody labeling and Chain-of-Thought (CoT) emotion reasoning annotations for speech emotion understanding, and related automatic labeling pipeline.

### EmotionCoT Datasets

> **Pre-request:** EmotionCoT does not redistribute audio files. Please download the original datasets from the official sources:
- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [MELD](https://github.com/declare-lab/MELD)
- [Expresso](https://speechbot.github.io/expresso/)
- [EARS](https://github.com/facebookresearch/ears_dataset)
- [MSP-Podcast](https://www.lab-msp.com/MSP/MSP-Podcast.html)

**EmotionCoT Annotations:** We provide prosody labeling and CoT-style emotion reasoning annotations for: IEMOCAP, MELD, Expresso, EARS, MSP-Podcast (partial). Please download the EmotionCoT dataset on [Hugging Face](https://huggingface.co/datasets/ddwang2000/EmotionCoT)

### Automatic Labeling Pipeline (Comming Soon)

To facilitate large-scale labeling and data augmentation, we provide an automated prosody labeling pipeline for EmotionCoT.

**Step 0: Prepare invironment**

> Note: If you have already prepared the environment during EmotionThinker inference stage, you may skip this step.

```bash
conda create -n emotionthinker python=3.10
conda activate emotionthinker
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Step 1: Download Required Models**

The speaker age and gender model (`audeering/wav2vec2-large-robust-24-ft-age-gender`) is fetched from Hugging Face on first use, and pitch, energy and speaking rate are computed with signal processing only, so neither needs setting up.

Word-level stress uses [WhiStress](https://github.com/slp-rl/WhiStress), which is not on PyPI and has to be checked out separately:

```bash
git clone https://github.com/slp-rl/WhiStress && python WhiStress/download_weights.py
export WHISTRESS_DIR=$PWD/WhiStress
```

`WHISTRESS_DIR` can also be passed as `--whistress-dir`. If you do not need stress labels, run the pipeline with `--skip-models`, which also skips gender and age.

**Step 2: Prepare Input JSONL**

Your input file must follow this format:
```json
{
  "audio_path": "path/to/audio.wav",
  "transcription": "text transcription",
  "emotion": "emotion_label"
}
```
**Step 3: Extract Prosody Labeling**

```bash
python EmotionCoT/pipeline/prosody_labeling.py \
    --input_path /path/to/input.jsonl \
    --output_path /path/to/prosody_labeling.jsonl
```
The script will automatically extract and label:
- `pitch_level`: low / normal / high
- `energy_level`: low / normal / high
- `speed_level`: slow / normal / fast
- `stressed_words`: stressed words from transcription
- `intonation`: rising / falling / rising-falling / falling-rising / flat / expressive
- `gender`: Male / Female 
- `age_level`: Child / Teenager / Young Adult / Middle-aged / Elderly

The output will be saved as a JSONL file with enriched prosody annotations. Any
other fields in the input, such as `emotion`, are carried through untouched.

`pitch_level`, `energy_level` and `speed_level` are relative to the corpus you
pass in, since recording gain and speaker identity make absolute cut points
meaningless across datasets. `intonation` is `uncertain` or `too_short` when the
utterance does not carry enough voiced speech to support a contour judgement;
the CoT prompt in the next step knows to skip those. The measurements the labels
were derived from (F0 in Hz, level in dBFS, phonemes per second, and so on) are
kept in the output as well — see `EmotionCoT/pipeline/README.md`.

### 4. (Optional) EmotionCoT-style Reasoning Augmentation

If you are interested in augmenting your dataset with EmotionCoT-like reasoning format, you can use the provided `api_call.py` script. It uses the prompt template from Appendix B.3 of the paper.

Configure your OpenAI API token in the script, then run:
```bash
python EmotionCoT/pipeline/api_call.py \
    --input_path /path/to/prosody_labeling.jsonl \
    --output_path /path/to/emotioncot_augmented.jsonl
```
This will generate structured emotion reasoning chains aligned with the EmotionCoT format. Add `--dry-run 3` to print the filled-in prompts without calling the API.



## Citation

If you find this work useful in your research, please kindly cite:
```
@inproceedings{wang2026emotionthinker,
  title={EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning},
  author={Wang, Dingdong and Liu, Shujie and Zhang, Tianhua and Chen, Youjun and Li, Jinyu and Meng, Helen},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
