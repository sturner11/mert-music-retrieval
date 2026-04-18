from typing import Dict, Optional

import pandas as pd
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from music_sis.data.manifest_dataset import ManifestAudioDataset
from music_sis.data.mert_collate import make_mert_collate_fn


def build_split_dataloaders(
    split_frames: Dict[str, pd.DataFrame],
    processor: Wav2Vec2FeatureExtractor,
    batch_size: int = 16,
    num_workers: int = 0,
    max_seconds: Optional[float] = 5.0,
) -> Dict[str, DataLoader]:
    datasets = {
        split: ManifestAudioDataset(df, target_sr=int(processor.sampling_rate))
        for split, df in split_frames.items()
    }
    collate_fn = make_mert_collate_fn(processor=processor, max_seconds=max_seconds)

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        ),
    }
