import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import dct

test_file = '/Users/samuelturner/Documents/mert-music-retrieval/artifacts/splits_5s/test.csv'
train_file = '/Users/samuelturner/Documents/mert-music-retrieval/artifacts/splits_5s/train.csv'
val_file = '/Users/samuelturner/Documents/mert-music-retrieval/artifacts/splits_5s/val.csv'
REQUIRED_K_VALUES = (1, 5, 10)
BASELINE_RESULTS_FILE = "/Users/samuelturner/Documents/mert-music-retrieval/artifacts/baseline_results.csv"
BASELINE_RANKINGS_FILE = "/Users/samuelturner/Documents/mert-music-retrieval/artifacts/baseline_rankings.csv"

def load_datasets():
    test_pd = pd.read_csv(test_file)
    train_pd = pd.read_csv(train_file)
    val_pd = pd.read_csv(val_file)
    return test_pd, train_pd, val_pd


def build_protocol_eval_context(test_pd: pd.DataFrame) -> dict:
    """
    Enforce the locked Phase 3 evaluation protocol for retrieval.

    Rules enforced:
    1) All test clips are queries.
    2) Candidate pool is all test clips.
    3) Exact same clip_id is excluded at scoring time.
    4) Metrics are reported at K in {1, 5, 10} for:
       - same_track (track_id match)
       - same_genre (label match)
    """
    required_cols = {"clip_id", "track_id", "label"}
    missing = required_cols.difference(test_pd.columns)
    if missing:
        raise ValueError(f"test manifest missing required columns: {sorted(missing)}")

    if test_pd["clip_id"].duplicated().any():
        dups = test_pd.loc[test_pd["clip_id"].duplicated(), "clip_id"].head(5).tolist()
        raise ValueError(f"clip_id must be unique in test manifest; found duplicates like: {dups}")

    queries_df = test_pd.copy().reset_index(drop=True)
    candidates_df = test_pd.copy().reset_index(drop=True)

    return {
        "queries_df": queries_df,
        "candidates_df": candidates_df,
        "k_values": REQUIRED_K_VALUES,
        "tasks": ("same_track", "same_genre"),
        "exclude_self_clip": True,
    }


def candidate_pool_for_query(candidates_df: pd.DataFrame, query_clip_id: str) -> pd.DataFrame:
    """Return test candidates with the query clip removed."""
    return candidates_df[candidates_df["clip_id"] != query_clip_id]


def evaluate_recall_at_k_from_rankings(
    rankings_by_query: dict,
    eval_context: dict,
) -> pd.DataFrame:
    """
    Evaluate R@1/5/10 for same-track and same-genre from ranked candidate ids.

    rankings_by_query format:
      {
        "<query_clip_id>": ["<candidate_clip_id_1>", "<candidate_clip_id_2>", ...],
        ...
      }
    Candidate lists are expected in descending similarity order.
    """
    queries_df = eval_context["queries_df"]
    candidates_df = eval_context["candidates_df"]
    k_values = eval_context["k_values"]

    candidate_meta = candidates_df.set_index("clip_id")[["track_id", "label"]].to_dict("index")
    query_meta = queries_df.set_index("clip_id")[["track_id", "label"]].to_dict("index")

    hit_counts = {
        "same_track": {k: 0 for k in k_values},
        "same_genre": {k: 0 for k in k_values},
    }

    query_ids = queries_df["clip_id"].tolist()
    for qid in query_ids:
        if qid not in rankings_by_query:
            raise ValueError(f"missing ranking list for query clip_id: {qid}")

        ranked_ids = [cid for cid in rankings_by_query[qid] if cid != qid]
        track_target = query_meta[qid]["track_id"]
        genre_target = query_meta[qid]["label"]

        for k in k_values:
            top_k = ranked_ids[:k]
            same_track_hit = any(
                (cid in candidate_meta) and (candidate_meta[cid]["track_id"] == track_target)
                for cid in top_k
            )
            same_genre_hit = any(
                (cid in candidate_meta) and (candidate_meta[cid]["label"] == genre_target)
                for cid in top_k
            )
            hit_counts["same_track"][k] += int(same_track_hit)
            hit_counts["same_genre"][k] += int(same_genre_hit)

    n_queries = len(query_ids)
    rows = []
    for task in ("same_track", "same_genre"):
        row = {"task": task, "num_queries": n_queries}
        for k in k_values:
            row[f"R@{k}"] = hit_counts[task][k] / n_queries if n_queries else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def build_rankings_cosine(
    query_embeddings_df: pd.DataFrame,
    candidate_embeddings_df: pd.DataFrame,
    exclude_self_clip: bool = True,
) -> dict:
    """
    Build ranked candidate clip_id lists per query using cosine similarity.
    Deterministic tie-break: score descending, then clip_id ascending.
    """
    required_cols = {"clip_id", "embedding"}
    for name, df in [("query", query_embeddings_df), ("candidate", candidate_embeddings_df)]:
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"{name} embeddings missing required columns: {sorted(missing)}")

    query_ids = query_embeddings_df["clip_id"].tolist()
    candidate_ids = candidate_embeddings_df["clip_id"].tolist()
    candidate_id_array = np.array(candidate_ids)

    query_matrix = np.stack(query_embeddings_df["embedding"].to_numpy())
    candidate_matrix = np.stack(candidate_embeddings_df["embedding"].to_numpy())

    rankings_by_query = {}
    for i, qid in enumerate(query_ids):
        q_vec = query_matrix[i]
        scores = candidate_matrix @ q_vec
        sorted_idx = np.lexsort((candidate_id_array, -scores))
        ranked_ids = candidate_id_array[sorted_idx].tolist()

        if exclude_self_clip:
            ranked_ids = [cid for cid in ranked_ids if cid != qid]

        rankings_by_query[qid] = ranked_ids

    return rankings_by_query


def build_rankings_with_scores_df(
    query_embeddings_df: pd.DataFrame,
    candidate_embeddings_df: pd.DataFrame,
    exclude_self_clip: bool = True,
) -> pd.DataFrame:
    """
    Return long-form rankings with similarity scores.
    Columns: query_clip_id, candidate_clip_id, rank, score
    """
    required_cols = {"clip_id", "embedding"}
    for name, df in [("query", query_embeddings_df), ("candidate", candidate_embeddings_df)]:
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"{name} embeddings missing required columns: {sorted(missing)}")

    query_ids = query_embeddings_df["clip_id"].tolist()
    candidate_ids = candidate_embeddings_df["clip_id"].tolist()
    candidate_id_array = np.array(candidate_ids)

    query_matrix = np.stack(query_embeddings_df["embedding"].to_numpy())
    candidate_matrix = np.stack(candidate_embeddings_df["embedding"].to_numpy())

    rows = []
    for i, qid in enumerate(query_ids):
        scores = candidate_matrix @ query_matrix[i]
        sorted_idx = np.lexsort((candidate_id_array, -scores))
        ranked_ids = candidate_id_array[sorted_idx]
        ranked_scores = scores[sorted_idx]

        rank_counter = 1
        for cid, score in zip(ranked_ids.tolist(), ranked_scores.tolist()):
            if exclude_self_clip and cid == qid:
                continue
            rows.append(
                {
                    "query_clip_id": qid,
                    "candidate_clip_id": cid,
                    "rank": rank_counter,
                    "score": float(score),
                }
            )
            rank_counter += 1

    return pd.DataFrame(rows)

def get_feature_extraction(df: pd.DataFrame, n_mfcc: int = 20, sr: int = 24000) -> pd.DataFrame:
    """
    Build MFCC baseline embeddings for each clip row.

    Output columns:
      - clip_id
      - track_id
      - label
      - embedding (np.ndarray, shape=(2 * n_mfcc,))
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

    def _load_audio_mono_resampled(path: str, target_sr: int) -> np.ndarray:
        orig_sr, y = wavfile.read(path)

        if y.dtype.kind in ("i", "u"):
            max_abs = np.iinfo(y.dtype).max
            y = y.astype(np.float32) / max_abs
        else:
            y = y.astype(np.float32)

        if y.ndim == 2:
            y = y.mean(axis=1)

        if orig_sr != target_sr:
            gcd = np.gcd(orig_sr, target_sr)
            up = target_sr // gcd
            down = orig_sr // gcd
            y = signal.resample_poly(y, up, down).astype(np.float32)

        return y

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
        power_spec = (np.abs(zxx) ** 2).astype(np.float32)  # (freq_bins, frames)
        mel_fb = _make_mel_filterbank(sample_rate, n_fft, n_mels=n_mels)
        mel_spec = mel_fb @ power_spec
        log_mel = np.log(np.maximum(mel_spec, eps))
        mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:num_mfcc, :]
        return mfcc.astype(np.float32)

    rows = []
    for row in df.itertuples(index=False):
        clip_id = row.clip_id
        track_id = row.track_id
        label = row.label
        path = row.path

        # Load to protocol sample rate and mono.
        y = _load_audio_mono_resampled(path=path, target_sr=sr)
        if y.size == 0:
            raise ValueError(f"empty audio clip encountered: {clip_id} ({path})")

        # MFCC shape: (n_mfcc, n_frames)
        mfcc = _mfcc_scipy(y=y, sample_rate=sr, num_mfcc=n_mfcc)
        if mfcc.ndim != 2 or mfcc.shape[0] != n_mfcc:
            raise ValueError(
                f"unexpected MFCC shape for {clip_id}: {mfcc.shape}, expected ({n_mfcc}, n_frames)"
            )

        # Aggregate over time: [mean(mfcc_i), std(mfcc_i)] for i in 1..n_mfcc
        mean_vec = mfcc.mean(axis=1)
        std_vec = mfcc.std(axis=1)
        emb = np.concatenate([mean_vec, std_vec]).astype(np.float32)

        # L2 normalize for cosine-distance retrieval.
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        rows.append(
            {
                "clip_id": clip_id,
                "track_id": track_id,
                "label": label,
                "embedding": emb,
            }
        )

    return pd.DataFrame(rows)



def main():
    test_pd, train_pd, val_pd = load_datasets()
    eval_context = build_protocol_eval_context(test_pd)
    print(
        "Protocol check passed:",
        f"queries={len(eval_context['queries_df'])},",
        f"candidates={len(eval_context['candidates_df'])},",
        f"k={eval_context['k_values']},",
        f"tasks={eval_context['tasks']}",
    )
    print("Extracting MFCC embeddings for test clips...")
    test_embeddings_df = get_feature_extraction(test_pd)

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
    metrics_df = evaluate_recall_at_k_from_rankings(
        rankings_by_query=rankings_by_query,
        eval_context=eval_context,
    )

    metrics_df["run_name"] = "mfcc_baseline"
    metrics_df["split"] = "test"
    metrics_df["candidate_pool"] = "test_all_clips"
    metrics_df["self_match_excluded"] = True
    metrics_df["tie_break_rule"] = "score_desc_then_clip_id_asc"
    metrics_df = metrics_df[
        [
            "run_name",
            "split",
            "task",
            "R@1",
            "R@5",
            "R@10",
            "num_queries",
            "candidate_pool",
            "self_match_excluded",
            "tie_break_rule",
        ]
    ]

    metrics_df.to_csv(BASELINE_RESULTS_FILE, index=False)
    rankings_scores_df.to_csv(BASELINE_RANKINGS_FILE, index=False)
    print(f"Saved baseline metrics to: {BASELINE_RESULTS_FILE}")
    print(f"Saved baseline rankings to: {BASELINE_RANKINGS_FILE}")
    print(metrics_df.to_string(index=False))

if __name__ == '__main__':
    main()
