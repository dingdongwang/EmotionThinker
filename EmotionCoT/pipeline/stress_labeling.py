"""Word-level sentence stress via WhiStress, with the transcript truncation fixed.

The upstream helper ``inference_from_audio_and_transcription`` tokenises the
reference transcript with ``padding="max_length", max_length=30``. Anything past
roughly twenty words is silently dropped, so long utterances came back with
stress annotated only over their opening clause and no indication that the rest
was never looked at. This module runs the same model with the same alignment
convention but tokenises up to the decoder's real capacity, and reports when an
utterance genuinely hits that limit.

Only the tokenisation is reimplemented; token merging and special-token handling
are reused from upstream so the two stay consistent.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _add_whistress_to_path(whistress_dir: Optional[str]) -> None:
    candidates = [
        whistress_dir,
        os.environ.get("WHISTRESS_DIR"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "WhiStress"),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "whistress")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return
    raise FileNotFoundError(
        "Could not locate a WhiStress checkout. Pass --whistress-dir or set "
        "WHISTRESS_DIR to a directory containing a 'whistress' package."
    )


class StressTagger:
    def __init__(self, device: str = "cpu", whistress_dir: Optional[str] = None):
        _add_whistress_to_path(whistress_dir)
        from whistress.inference_client.utils import (  # noqa: E402
            get_loaded_model,
            get_word_emphasis_pairs,
            merge_stressed_tokens,
        )

        self.device = device
        self.model = get_loaded_model(device)
        self._pairs = get_word_emphasis_pairs
        self._merge = merge_stressed_tokens
        self.max_tokens = int(self.model.whisper_model.config.max_target_positions)

    @staticmethod
    def _peak_normalise(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=float)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return audio / peak if peak > 0 else audio

    @torch.no_grad()
    def predict(self, audio: np.ndarray, transcription: str) -> Dict[str, Any]:
        if not transcription or not transcription.strip() or audio.size == 0:
            return {
                "stressed_words": None,
                "stress_pairs": None,
                "stress_truncated": None,
            }

        features = self.model.processor.feature_extractor(
            self._peak_normalise(audio), sampling_rate=16_000, return_tensors="pt"
        )["input_features"]

        encoded = self.model.processor.tokenizer(
            transcription,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        input_ids = encoded["input_ids"]

        output = self.model(
            input_features=features.to(self.device),
            decoder_input_ids=input_ids.to(self.device),
        )
        logits = output.logits
        if logits.dim() == 2:
            # Transformers versions differ in whether a decoder layer returns a
            # tuple or a bare tensor, which turns WhiStress's `[0]` from "first
            # tuple element" into "first batch element" and drops the batch axis.
            # Batch size is always one here, so the values are unaffected.
            logits = logits.unsqueeze(0)

        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        # Upstream's alignment convention: the prediction at position i-1 belongs
        # to token i, so the sequence is rotated by one before pairing.
        preds = torch.cat((preds[:, -1:], preds[:, :-1]), dim=1)

        pairs = self._pairs(
            input_ids[0], preds[0], self.model.processor, filter_special_tokens=True
        )
        merged = [(word.strip(), int(flag)) for word, flag in self._merge(pairs)]
        merged = [(word, flag) for word, flag in merged if word]

        return {
            "stressed_words": [word for word, flag in merged if flag == 1],
            "stress_pairs": [[word, flag] for word, flag in merged],
            "stress_truncated": bool(input_ids.shape[1] >= self.max_tokens),
        }
