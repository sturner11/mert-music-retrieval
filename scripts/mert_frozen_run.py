from pathlib import Path
import os
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

import sys

PROJECT_ROOT = Path(
    os.environ.get("MMR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_sis.config import DEFAULT_MERT_NAME
from music_sis.data.loaders import build_split_dataloaders
from music_sis.data.mert_collate import MertBatch
from music_sis.data.split_manifests import load_split_dataframes


def load_frozen_mert(model_name: str = DEFAULT_MERT_NAME) -> AutoModel:
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main() -> None:
    split_frames = load_split_dataframes()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        DEFAULT_MERT_NAME, trust_remote_code=True
    )
    dataloaders = build_split_dataloaders(
        split_frames=split_frames,
        processor=processor,
        batch_size=4,
        max_seconds=5.0,
    )

    # Smoke-test the full preprocess path into MERT input format.
    train_batch: MertBatch = next(iter(dataloaders["train"]))
    print("Train batch preprocess check:")
    print("input_values shape:", tuple(train_batch.input_values.shape))
    print("attention_mask shape:", tuple(train_batch.attention_mask.shape))
    print("first clip_id:", train_batch.clip_ids[0])
    print("first track_id:", train_batch.track_ids[0])
    print("first label:", train_batch.labels[0])


if __name__ == "__main__":
    main()
