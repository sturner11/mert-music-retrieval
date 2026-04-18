from typing import Dict, Tuple

import numpy as np
import pandas as pd


def build_protocol_eval_context(
    test_df: pd.DataFrame,
    k_values: Tuple[int, ...],
) -> Dict[str, object]:
    """
    Enforce locked retrieval protocol:
    - all test clips are queries
    - candidate pool is all test clips
    - exact same clip_id excluded at scoring time
    - evaluate same_track and same_genre at requested K
    """
    required_cols = {"clip_id", "track_id", "label"}
    missing = required_cols.difference(test_df.columns)
    if missing:
        raise ValueError(f"test manifest missing required columns: {sorted(missing)}")

    if test_df["clip_id"].duplicated().any():
        dups = test_df.loc[test_df["clip_id"].duplicated(), "clip_id"].head(5).tolist()
        raise ValueError(f"clip_id must be unique in test manifest; found duplicates like: {dups}")

    queries_df = test_df.copy().reset_index(drop=True)
    candidates_df = test_df.copy().reset_index(drop=True)
    return {
        "queries_df": queries_df,
        "candidates_df": candidates_df,
        "k_values": tuple(k_values),
        "tasks": ("same_track", "same_genre"),
        "exclude_self_clip": True,
    }


def build_rankings_cosine(
    query_embeddings_df: pd.DataFrame,
    candidate_embeddings_df: pd.DataFrame,
    exclude_self_clip: bool = True,
) -> Dict[str, list]:
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
        scores = candidate_matrix @ query_matrix[i]
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


def evaluate_recall_at_k_from_rankings(
    rankings_by_query: Dict[str, list],
    eval_context: Dict[str, object],
) -> pd.DataFrame:
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
