"""Speaker age and gender from a single wav2vec 2.0 model.

Both attributes come from ``audeering/wav2vec2-large-robust-24-ft-age-gender``,
which has an age regression head and a three-way gender head sharing one
encoder. Using one model for both avoids the failure mode of the earlier code,
which fed audio through the feature extractor of *this* checkpoint into a
*different* gender classifier, and which never instantiated an age model at all
despite the pipeline advertising an age label.

The gender output order is ``[female, male, child]``, taken from the worked
example on the model card rather than assumed. The emitted ``gender`` and
``age_level`` strings use the vocabulary the top-level README documents, since
they are copied verbatim into the CoT generation prompt.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MODEL_NAME = "audeering/wav2vec2-large-robust-24-ft-age-gender"

# Order of the classifier's output units, taken from the model card.
GENDER_UNITS = ("female", "male", "child")
# The written labels are the ones the README documents and the CoT prompt reads.
GENDER_LABELS = ("Female", "Male", "Child")

# The head regresses age onto 0..1 over a 0..100 year span.
AGE_SCALE_YEARS = 100.0

AGE_GROUP_BOUNDS = (
    (13.0, "Child"),
    (20.0, "Teenager"),
    (35.0, "Young Adult"),
    (60.0, "Middle-aged"),
    (float("inf"), "Elderly"),
)

# Anything longer is trimmed before the encoder; attention is quadratic in length
# and speaker identity does not need more than this.
MAX_SECONDS = 15.0


class ModelHead(nn.Module):
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Architecture must match the published checkpoint exactly to load weights."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        try:
            self.init_weights()
        except AttributeError:
            # init_weights reaches into weight-tying machinery that transformers 5
            # only sets up for models defined inside the library. This
            # architecture ties nothing and every parameter is overwritten from
            # the checkpoint, so random initialisation can simply be skipped.
            pass

    def forward(self, input_values):
        hidden_states = torch.mean(self.wav2vec2(input_values)[0], dim=1)
        return (
            hidden_states,
            self.age(hidden_states),
            torch.softmax(self.gender(hidden_states), dim=1),
        )


def age_to_level(years: float) -> str:
    for upper, name in AGE_GROUP_BOUNDS:
        if years < upper:
            return name
    return AGE_GROUP_BOUNDS[-1][1]


def load_age_gender_model(model_name: str = MODEL_NAME) -> AgeGenderModel:
    """Build the model and load the checkpoint explicitly.

    ``from_pretrained`` on a hand-written ``PreTrainedModel`` subclass is fragile
    across transformers major versions (it raises on 5.x for this class). Loading
    the state dict directly works everywhere, and lets us fail loudly if either
    prediction head came back uninitialised instead of returning a model that
    silently emits predictions from random weights.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    model = AgeGenderModel(config)
    state = load_file(hf_hub_download(model_name, "model.safetensors"))

    incompatible = model.load_state_dict(state, strict=False)
    unfilled = [
        key
        for key in incompatible.missing_keys
        if key.startswith(("age.", "gender.", "wav2vec2.encoder.", "wav2vec2.feature_extractor."))
    ]
    if unfilled:
        raise RuntimeError(
            f"{model_name} did not supply weights for {len(unfilled)} parameters, "
            f"starting with {unfilled[:5]}"
        )
    return model


class SpeakerAttributeTagger:
    def __init__(self, device: str = "cpu", model_name: str = MODEL_NAME):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = load_age_gender_model(model_name).to(device).eval()

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sampling_rate: int = 16_000) -> Dict[str, Any]:
        limit = int(MAX_SECONDS * sampling_rate)
        if audio.size > limit:
            audio = audio[:limit]
        if audio.size == 0:
            return {
                "gender": None,
                "gender_confidence": None,
                "age_years": None,
                "age_level": None,
            }

        inputs = self.processor(
            audio, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_values.to(self.device)
        _, age_logits, gender_probs = self.model(inputs)

        probs = gender_probs[0].cpu().numpy()
        best = int(np.argmax(probs))
        years = float(age_logits[0, 0].cpu()) * AGE_SCALE_YEARS
        years = float(np.clip(years, 0.0, AGE_SCALE_YEARS))

        return {
            "gender": GENDER_LABELS[best],
            "gender_confidence": round(float(probs[best]), 3),
            "gender_probs": {
                unit: round(float(p), 3) for unit, p in zip(GENDER_UNITS, probs)
            },
            "age_years": round(years, 1),
            "age_level": age_to_level(years),
        }
