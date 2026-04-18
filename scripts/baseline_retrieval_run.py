from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fftpack import dct

PROJECT_ROOT = Path(
    os.environ.get("MMR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_sis.config import (
    BASELINE_RANKINGS_FILE,
    BASELINE_RESULTS_FILE,
    REQUIRED_K_VALUES,
)
from music_sis.data.manifest_dataset import load_audio_mono_resampled
from music_sis.data.split_manifests import load_split_dataframes
from music_sis.eval.io import finalize_metrics_table, write_retrieval_artifacts
from music_sis.eval.retrieval import (
    build_protocol_eval_context,
    build_rankings_cosine,
    build_rankings_with_scores_df,
    evaluate_recall_at_k_from_rankings,
)


def get_feature_extraction(df: pd.DataFrame, n_mfcc: int = 20, sr: int = 24000) -> pd.DataFrame:
    """
    Build MFCC baseline embeddings for each clip row.
    Output columns: clip_id, track_id, label, embedding
    """
    required_cols = {"clip_id", "track_id", "label", "path"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")

    def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    def _make_mel_filterbank(
        sample_rate: int,
        n_fft: int,
        n_mels: int = 40,
        f_min: float = 0.0,
        f_max: float = None,
    ) -> np.ndarray:
        if f_max is None:
            f_max = sample_rate / 2.0

        mel_min = _hz_to_mel(np.array([f_min], dtype=np.float64))[0]
        mel_max = _hz_to_mel(np.array([f_max], dtype=np.float64))[0]
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = _mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            left = bin_points[m - 1]
            center = bin_points[m]
            right = bin_points[m + 1]

            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1

            for k in range(left, center):
                fb[m - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                fb[m - 1, k] = (right - k) / (right - center)
        return fb

    def _mfcc_scipy(y: np.ndarray, sample_rate: int, num_mfcc: int) -> np.ndarray:
        n_fft = 1024
        hop_length = 256
        win_length = 1024
        n_mels = 40
        eps = 1e-10

        _, _, zxx = signal.stft(
            y,
            fs=sample_rate,
            window="hann",
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        power_spec = (np.abs(zxx) ** 2).astype(np.float32)
        mel_fb = _make_mel_filterbank(sample_rate, n_fft, n_mels=n_mels)
        mel_spec = mel_fb @ power_spec
        log_mel = np.log(np.maximum(mel_spec, eps))
        mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:num_mfcc, :]
        return mfcc.astype(np.float32)

    rows = []
    for row in df.itertuples(index=False):
        y = load_audio_mono_resampled(path=row.path, target_sr=sr)
        if y.size == 0:
            raise ValueError(f"empty audio clip encountered: {row.clip_id} ({row.path})")

        mfcc = _mfcc_scipy(y=y, sample_rate=sr, num_mfcc=n_mfcc)
        if mfcc.ndim != 2 or mfcc.shape[0] != n_mfcc:
            raise ValueError(
                f"unexpected MFCC shape for {row.clip_id}: {mfcc.shape}, expected ({n_mfcc}, n_frames)"
            )

        mean_vec = mfcc.mean(axis=1)
        std_vec = mfcc.std(axis=1)
        emb = np.concatenate([mean_vec, std_vec]).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        rows.append(
            {
                "clip_id": row.clip_id,
                "track_id": row.track_id,
                "label": row.label,
                "embedding": emb,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    split_frames = load_split_dataframes()
    test_df = split_frames["test"]
    eval_context = build_protocol_eval_context(test_df, k_values=REQUIRED_K_VALUES)
    print(
        "Protocol check passed:",
        f"queries={len(eval_context['queries_df'])},",
        f"candidates={len(eval_context['candidates_df'])},",
        f"k={eval_context['k_values']},",
        f"tasks={eval_context['tasks']}",
    )

    print("Extracting MFCC embeddings for test clips...")
    test_embeddings_df = get_feature_extraction(test_df)

    print("Scoring retrieval with cosine nearest neighbors...")
    rankings_by_query = build_rankings_cosine(
        query_embeddings_df=test_embeddings_df,
        candidate_embeddings_df=test_embeddings_df,
        exclude_self_clip=eval_context["exclude_self_clip"],
    )
    rankings_scores_df = build_rankings_with_scores_df(
        query_embeddings_df=test_embeddings_df,
        candidate_embeddings_df=test_embeddings_df,
        exclude_self_clip=eval_context["exclude_self_clip"],
    )

    print("Computing R@1/5/10 metrics...")
    raw_metrics_df = evaluate_recall_at_k_from_rankings(
        rankings_by_query=rankings_by_query,
        eval_context=eval_context,
    )
    metrics_df = finalize_metrics_table(raw_metrics_df, run_name="mfcc_baseline")

    write_retrieval_artifacts(
        metrics_df=metrics_df,
        rankings_df=rankings_scores_df,
        metrics_path=BASELINE_RESULTS_FILE,
        rankings_path=BASELINE_RANKINGS_FILE,
    )
    print(f"Saved baseline metrics to: {BASELINE_RESULTS_FILE}")
    print(f"Saved baseline rankings to: {BASELINE_RANKINGS_FILE}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
