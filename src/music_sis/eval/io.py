from pathlib import Path

import pandas as pd


def finalize_metrics_table(
    metrics_df: pd.DataFrame,
    run_name: str,
    split: str = "test",
    candidate_pool: str = "test_all_clips",
    self_match_excluded: bool = True,
    tie_break_rule: str = "score_desc_then_clip_id_asc",
) -> pd.DataFrame:
    df = metrics_df.copy()
    df["run_name"] = run_name
    df["split"] = split
    df["candidate_pool"] = candidate_pool
    df["self_match_excluded"] = self_match_excluded
    df["tie_break_rule"] = tie_break_rule
    return df[
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


def write_retrieval_artifacts(
    metrics_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    metrics_path: Path,
    rankings_path: Path,
) -> None:
    metrics_df.to_csv(metrics_path, index=False)
    rankings_df.to_csv(rankings_path, index=False)
