import argparse
import random
import shutil
import subprocess
from typing import Dict, List, Optional

import pandas as pd

from baseline_retrieval_run import (
    load_datasets,
)


def _pick_player() -> Optional[List[str]]:
    """Choose a local audio player command available on this machine."""
    if shutil.which("afplay"):
        return ["afplay"]  # macOS
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    if shutil.which("mpv"):
        return ["mpv", "--no-video", "--really-quiet"]
    return None


def _find_group_examples(
    query_clip_id: str,
    ranked_clip_ids: List[str],
    meta_by_clip: Dict[str, Dict[str, str]],
) -> Dict[str, Optional[str]]:
    """Find one clip for each comparison group from ranked neighbors."""
    query_track = meta_by_clip[query_clip_id]["track_id"]
    query_label = meta_by_clip[query_clip_id]["label"]

    same_track = None
    same_genre_diff_track = None
    different_genre = None

    for cid in ranked_clip_ids:
        if cid == query_clip_id:
            continue
        c_track = meta_by_clip[cid]["track_id"]
        c_label = meta_by_clip[cid]["label"]

        if same_track is None and c_track == query_track:
            same_track = cid
            continue
        if same_genre_diff_track is None and c_label == query_label and c_track != query_track:
            same_genre_diff_track = cid
            continue
        if different_genre is None and c_label != query_label:
            different_genre = cid
            continue

        if same_track and same_genre_diff_track and different_genre:
            break

    return {
        "same_track": same_track,
        "same_genre_diff_track": same_genre_diff_track,
        "different_genre": different_genre,
    }


def _play_clip(player_cmd: List[str], clip_path: str) -> None:
    subprocess.run([*player_cmd, clip_path], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play grouped retrieval examples for human listening checks."
    )
    parser.add_argument("--n-queries", type=int, default=3, help="How many query sets to play.")
    parser.add_argument("--seed", type=int, default=478, help="Random seed for query sampling.")
    parser.add_argument(
        "--rankings-csv",
        type=str,
        default="/Users/samuelturner/Documents/mert-music-retrieval/artifacts/baseline_rankings.csv",
        help="Path to rankings CSV from baseline_retrieval_run.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the clip groups without playing audio.",
    )
    parser.add_argument(
        "--close-cross-genre",
        type=int,
        default=3,
        help="Number of high-scoring different-genre neighbors to display per query.",
    )
    args = parser.parse_args()

    test_df, _, _ = load_datasets()
    rankings_df = pd.read_csv(args.rankings_csv)
    expected_cols = {"query_clip_id", "candidate_clip_id", "rank", "score"}
    missing = expected_cols.difference(rankings_df.columns)
    if missing:
        raise ValueError(f"rankings CSV missing required columns: {sorted(missing)}")

    meta_df = test_df[["clip_id", "track_id", "label", "path"]].copy()
    meta_by_clip = meta_df.set_index("clip_id").to_dict("index")

    rng = random.Random(args.seed)
    query_ids = test_df["clip_id"].tolist()
    rng.shuffle(query_ids)
    query_ids = query_ids[: max(1, args.n_queries)]

    player_cmd = _pick_player()
    if not args.dry_run and player_cmd is None:
        raise RuntimeError("No supported audio player found (afplay, ffplay, mpv).")

    for i, qid in enumerate(query_ids, start=1):
        q_rank_df = rankings_df[rankings_df["query_clip_id"] == qid].sort_values("rank")
        ranked_clip_ids = q_rank_df["candidate_clip_id"].tolist()
        score_by_candidate = dict(
            zip(q_rank_df["candidate_clip_id"].tolist(), q_rank_df["score"].tolist())
        )
        groups = _find_group_examples(qid, ranked_clip_ids, meta_by_clip)
        qmeta = meta_by_clip[qid]

        print(f"\n=== Query Set {i} ===")
        print(f"query: {qid} | track={qmeta['track_id']} | label={qmeta['label']}")

        ordered = [
            ("query", qid),
            ("same_track", groups["same_track"]),
            ("same_genre_diff_track", groups["same_genre_diff_track"]),
            ("different_genre", groups["different_genre"]),
        ]

        for group_name, cid in ordered:
            if cid is None:
                print(f"{group_name}: NONE FOUND")
                continue

            cmeta = meta_by_clip[cid]
            score = score_by_candidate.get(cid)
            score_text = f"{score:.6f}" if score is not None else "N/A"
            print(
                f"{group_name}: {cid} | track={cmeta['track_id']} | "
                f"label={cmeta['label']} | score={score_text} | path={cmeta['path']}"
            )
            if not args.dry_run:
                _play_clip(player_cmd, cmeta["path"])

        if args.close_cross_genre > 0:
            q_label = qmeta["label"]
            cross_genre_rows = q_rank_df[
                q_rank_df["candidate_clip_id"].map(lambda cid: meta_by_clip[cid]["label"] != q_label)
            ].head(args.close_cross_genre)
            if not cross_genre_rows.empty:
                print("close_cross_genre_neighbors:")
                for row in cross_genre_rows.itertuples(index=False):
                    cid = row.candidate_clip_id
                    cmeta = meta_by_clip[cid]
                    print(
                        f"  rank={int(row.rank)} score={float(row.score):.6f} "
                        f"clip={cid} track={cmeta['track_id']} label={cmeta['label']}"
                    )


if __name__ == "__main__":
    main()
