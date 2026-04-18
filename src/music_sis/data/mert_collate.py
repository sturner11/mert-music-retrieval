from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from transformers import Wav2Vec2FeatureExtractor


@dataclass
class MertBatch:
    input_values: torch.Tensor
    attention_mask: torch.Tensor
    clip_ids: List[str]
    track_ids: List[str]
    labels: List[str]
    paths: List[str]


def make_mert_collate_fn(
    processor: Wav2Vec2FeatureExtractor,
    max_seconds: Optional[float] = None,
) -> Callable[[List[Dict[str, object]]], MertBatch]:
    """
    Collate function that converts raw clip arrays into MERT input format.
    """
    target_sr = int(processor.sampling_rate)
    max_length = int(max_seconds * target_sr) if max_seconds is not None else None

    def collate_fn(samples: List[Dict[str, object]]) -> MertBatch:
        waveforms = [s["audio"] for s in samples]
        encoded = processor(
            waveforms,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )

        attention_mask = encoded.get(
            "attention_mask",
            torch.ones_like(encoded["input_values"], dtype=torch.long),
        )

        return MertBatch(
            input_values=encoded["input_values"],
            attention_mask=attention_mask,
            clip_ids=[str(s["clip_id"]) for s in samples],
            track_ids=[str(s["track_id"]) for s in samples],
            labels=[str(s["label"]) for s in samples],
            paths=[str(s["path"]) for s in samples],
        )

    return collate_fn
