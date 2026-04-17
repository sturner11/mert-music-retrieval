#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(os.environ.get("MMR_PROJECT_ROOT", "/Users/samuelturner/Documents/mert-music-retrieval"))
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
OUT_CSV = PROJECT_ROOT / "artifacts" / "decode_preprocess_audit.csv"
OUT_MD = PROJECT_ROOT / "artifacts" / "decode_preprocess_report.md"
TARGET_SPLITS = ("train", "val", "test")
EXCLUDED_TRACK_IDS = {"jazz/jazz.00054"}


@dataclass
class Row:
    split: str
    track_id: str
    path: str
    label: str


def read_split(split_name: str) -> list[Row]:
    split_path = SPLITS_DIR / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    rows: list[Row] = []
    with split_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            track_id = r["track_id"]
            if track_id in EXCLUDED_TRACK_IDS:
                continue
            rows.append(Row(split_name, track_id, r["path"], r["label"]))
    return rows


def classify_decode(path: Path) -> tuple[str, dict[str, str]]:
    if not path.exists():
        return "missing_file", {}
    try:
        audio, sr = sf.read(path, always_2d=False)
    except Exception:
        return "decode_error", {}

    audio = np.asarray(audio)
    if audio.size == 0:
        return "empty_audio", {"sample_rate": str(sr)}
    if not np.isfinite(audio).all():
        return "invalid_numeric", {"sample_rate": str(sr)}
    if sr <= 0:
        return "invalid_sample_rate", {"sample_rate": str(sr)}

    if audio.ndim == 1:
        channels = 1
        num_samples = int(audio.shape[0])
    elif audio.ndim == 2:
        channels = int(audio.shape[1])
        num_samples = int(audio.shape[0])
    else:
        return "invalid_shape", {"sample_rate": str(sr), "shape": str(tuple(audio.shape))}

    duration_sec = num_samples / float(sr)
    return "ok", {
        "sample_rate": str(sr),
        "channels": str(channels),
        "num_samples": str(num_samples),
        "duration_sec": f"{duration_sec:.3f}",
    }


def main() -> None:
    rows: list[Row] = []
    for split in TARGET_SPLITS:
        rows.extend(read_split(split))

    out_rows: list[dict[str, str]] = []
    status_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {s: 0 for s in TARGET_SPLITS}
    split_ok: dict[str, int] = {s: 0 for s in TARGET_SPLITS}

    for r in rows:
        split_counts[r.split] += 1
        status, meta = classify_decode(Path(r.path))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "ok":
            split_ok[r.split] += 1

        out_rows.append(
            {
                "split": r.split,
                "track_id": r.track_id,
                "label": r.label,
                "path": r.path,
                "decode_status": status,
                "sample_rate": meta.get("sample_rate", ""),
                "channels": meta.get("channels", ""),
                "num_samples": meta.get("num_samples", ""),
                "duration_sec": meta.get("duration_sec", ""),
            }
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        fieldnames = [
            "split",
            "track_id",
            "label",
            "path",
            "decode_status",
            "sample_rate",
            "channels",
            "num_samples",
            "duration_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    md_lines = [
        "# Decode Preprocess Report",
        "",
        f"- Input tracks: {len(rows)}",
        f"- Output audit: `{OUT_CSV}`",
        "",
        "## Decode Status Counts",
    ]
    for k in sorted(status_counts):
        md_lines.append(f"- {k}: {status_counts[k]}")

    md_lines.extend(["", "## Split Decode Coverage"])
    for s in TARGET_SPLITS:
        total = split_counts[s]
        ok = split_ok[s]
        md_lines.append(f"- {s}: {ok}/{total} ok")

    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
