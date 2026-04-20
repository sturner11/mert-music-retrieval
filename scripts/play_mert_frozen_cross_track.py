import argparse
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(
    os.environ.get("MMR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_sis.config import MERT_FROZEN_RANKINGS_FILE
from music_sis.data.split_manifests import load_split_dataframes


def _pick_player() -> Optional[List[str]]:
    if shutil.which("afplay"):
        return ["afplay"]  # macOS
    if shutil.which("ffplay"):
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
    if shutil.which("mpv"):
        return ["mpv", "--no-video", "--really-quiet"]
    return None


def _play_clip(player_cmd: List[str], clip_path: str) -> None:
    subprocess.run([*player_cmd, clip_path], check=False)


def _top_cross_track_neighbor(
    query_clip_id: str,
    rankings_df: pd.DataFrame,
    meta_by_clip: Dict[str, Dict[str, str]],
) -> Optional[Tuple[str, float]]:
    query_track = meta_by_clip[query_clip_id]["track_id"]
    q_rows = rankings_df[rankings_df["query_clip_id"] == query_clip_id].sort_values("rank")

    for row in q_rows.itertuples(index=False):
        cid = row.candidate_clip_id
        if cid == query_clip_id:
            continue
        if cid not in meta_by_clip:
            continue
        if meta_by_clip[cid]["track_id"] != query_track:
            return cid, float(row.score)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play top MERT-frozen neighbors that are not from the same track."
    )
    parser.add_argument("--n-queries", type=int, default=5, help="Number of query clips to sample.")
    parser.add_argument("--seed", type=int, default=478, help="Random seed for sampling query clips.")
    parser.add_argument(
        "--rankings-csv",
        type=str,
        default=str(MERT_FROZEN_RANKINGS_FILE),
        help="Path to MERT frozen rankings CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print chosen query/neighbor pairs without playing audio.",
    )
    args = parser.parse_args()

    split_frames = load_split_dataframes()
    test_df = split_frames["test"]
    rankings_df = pd.read_csv(args.rankings_csv)

    expected_cols = {"query_clip_id", "candidate_clip_id", "rank", "score"}
    missing = expected_cols.difference(rankings_df.columns)
    if missing:
        raise ValueError(f"rankings CSV missing required columns: {sorted(missing)}")

    meta_df = test_df[["clip_id", "track_id", "label", "path"]].copy()
    meta_by_clip = meta_df.set_index("clip_id").to_dict("index")

    player_cmd = _pick_player()
    if not args.dry_run and player_cmd is None:
        raise RuntimeError("No supported audio player found (afplay, ffplay, mpv).")

    query_ids = test_df["clip_id"].tolist()
    rng = random.Random(args.seed)
    rng.shuffle(query_ids)
    query_ids = query_ids[: max(1, args.n_queries)]

    printed = 0
    for qid in query_ids:
        neighbor = _top_cross_track_neighbor(
            query_clip_id=qid,
            rankings_df=rankings_df,
            meta_by_clip=meta_by_clip,
        )
        if neighbor is None:
            continue

        cid, score = neighbor
        qmeta = meta_by_clip[qid]
        cmeta = meta_by_clip[cid]
        same_genre = qmeta["label"] == cmeta["label"]

        printed += 1
        print(f"\n=== Pair {printed} ===")
        print(
            f"query:    {qid} | track={qmeta['track_id']} | label={qmeta['label']} | path={qmeta['path']}"
        )
        print(
            f"neighbor: {cid} | track={cmeta['track_id']} | label={cmeta['label']} "
            f"| score={score:.6f} | same_genre={same_genre} | path={cmeta['path']}"
        )

        if not args.dry_run:
            _play_clip(player_cmd, qmeta["path"])
            _play_clip(player_cmd, cmeta["path"])

    if printed == 0:
        raise RuntimeError("No valid cross-track neighbors found to play.")


if __name__ == "__main__":
    main()
