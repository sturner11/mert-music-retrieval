from typing import Dict

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset


def load_audio_mono_resampled(path: str, target_sr: int) -> np.ndarray:
    src_sr, y = wavfile.read(path)

    if y.dtype.kind in ("i", "u"):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    else:
        y = y.astype(np.float32)

    if y.ndim == 2:
        y = y.mean(axis=1)

    if src_sr != target_sr:
        gcd = np.gcd(src_sr, target_sr)
        up = target_sr // gcd
        down = src_sr // gcd
        y = signal.resample_poly(y, up, down).astype(np.float32)

    return y


class ManifestAudioDataset(Dataset):
    """
    Dataset wrapper for split manifest DataFrames.
    Required columns: clip_id, track_id, label, path
    """

    def __init__(self, manifest_df: pd.DataFrame, target_sr: int = 24000):
        required = {"clip_id", "track_id", "label", "path"}
        missing = required.difference(manifest_df.columns)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")

        self.df = manifest_df.reset_index(drop=True).copy()
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        audio = load_audio_mono_resampled(row["path"], target_sr=self.target_sr)
        if audio.size == 0:
            raise ValueError(f"empty audio clip: {row['clip_id']} ({row['path']})")

        return {
            "clip_id": row["clip_id"],
            "track_id": row["track_id"],
            "label": row["label"],
            "path": row["path"],
            "audio": audio,
            "sampling_rate": self.target_sr,
        }
