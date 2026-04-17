#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT_ROOT = Path(os.environ.get("MMR_PROJECT_ROOT", "/Users/samuelturner/Documents/mert-music-retrieval"))
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
OUT_AUDIO_ROOT = PROJECT_ROOT / "data" / "interim" / "gtzan_5s"
OUT_SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits_5s"
OUT_SUMMARY = PROJECT_ROOT / "artifacts" / "preprocessing_5s_summary.md"

TARGET_SR = 24_000
CLIP_SECONDS = 5.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)
MIN_PEAK = 1e-8
EXCLUDED_TRACK_IDS = {"jazz/jazz.00054"}


@dataclass
class Row:
    track_id: str
    path: str
    label: str


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            track_id = r["track_id"]
            if track_id in EXCLUDED_TRACK_IDS:
                continue
            rows.append(Row(track_id=track_id, path=r["path"], label=r["label"]))
    return rows


def write_clip_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["clip_id", "track_id", "path", "label", "split", "start_sec", "duration_sec"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def load_mono_resampled(path: Path, target_sr: int) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    audio, sr = sf.read(path, always_2d=False)
    original_sr = int(sr)
    original_channels = 1
    if audio.ndim == 2:
        original_channels = int(audio.shape[1])
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    was_resampled = False
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up=up, down=down).astype(np.float32)
        was_resampled = True

    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    was_normalized = False
    if peak_before > MIN_PEAK:
        audio = audio / peak_before
        was_normalized = True
    peak_after = float(np.max(np.abs(audio))) if audio.size else 0.0

    meta = {
        "original_sr": original_sr,
        "original_channels": original_channels,
        "was_resampled": was_resampled,
        "was_normalized": was_normalized,
        "peak_before": peak_before,
        "peak_after": peak_after,
    }
    return audio.astype(np.float32, copy=False), meta


def stable_offset(track_id: str, max_offset: int) -> int:
    if max_offset <= 0:
        return 0
    h = hashlib.sha1(track_id.encode("utf-8")).hexdigest()
    return int(h, 16) % (max_offset + 1)


def segment_track(audio: np.ndarray, track_id: str) -> list[np.ndarray]:
    if audio.size <= CLIP_SAMPLES:
        out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        out[: audio.size] = audio
        return [out]

    n_full = audio.size // CLIP_SAMPLES
    clips = [audio[i * CLIP_SAMPLES : (i + 1) * CLIP_SAMPLES] for i in range(n_full)]

    # If there is a remainder, include one deterministic tail crop so we do not bias intros only.
    rem = audio.size - n_full * CLIP_SAMPLES
    if rem > 0:
        max_off = audio.size - CLIP_SAMPLES
        off = stable_offset(track_id, max_off)
        clips.append(audio[off : off + CLIP_SAMPLES])
    return [c.astype(np.float32, copy=False) for c in clips]


def main() -> None:
    OUT_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "# 5s Preprocessing Summary",
        "",
        f"- Target sample rate: {TARGET_SR}",
        f"- Clip seconds: {CLIP_SECONDS}",
        f"- Output audio root: `{OUT_AUDIO_ROOT}`",
        f"- Output manifest root: `{OUT_SPLITS_DIR}`",
        "",
    ]

    for split_name in ("train", "val", "test"):
        in_manifest = SPLITS_DIR / f"{split_name}.csv"
        if not in_manifest.exists():
            raise FileNotFoundError(f"Missing split manifest: {in_manifest}")

        in_rows = read_rows(in_manifest)
        clip_rows: list[dict[str, str]] = []
        fail_count = 0
        resampled_count = 0
        normalized_count = 0
        mono_fold_count = 0
        sr_counts: dict[int, int] = {}

        for r in in_rows:
            src = Path(r.path)
            try:
                audio, meta = load_mono_resampled(src, TARGET_SR)
                clips = segment_track(audio, r.track_id)
            except Exception:
                fail_count += 1
                continue

            orig_sr = int(meta["original_sr"])
            sr_counts[orig_sr] = sr_counts.get(orig_sr, 0) + 1
            if bool(meta["was_resampled"]):
                resampled_count += 1
            if bool(meta["was_normalized"]):
                normalized_count += 1
            if int(meta["original_channels"]) > 1:
                mono_fold_count += 1

            for i, clip in enumerate(clips):
                clip_id = f"{r.track_id.replace('/', '__')}__seg{i:02d}"
                out_dir = OUT_AUDIO_ROOT / split_name / r.label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{clip_id}.wav"
                sf.write(out_path, clip, TARGET_SR, subtype="PCM_16")

                clip_rows.append(
                    {
                        "clip_id": clip_id,
                        "track_id": r.track_id,
                        "path": str(out_path.resolve()),
                        "label": r.label,
                        "split": split_name,
                        "start_sec": f"{i * CLIP_SECONDS:.2f}",
                        "duration_sec": f"{CLIP_SECONDS:.2f}",
                    }
                )

        out_manifest = OUT_SPLITS_DIR / f"{split_name}.csv"
        write_clip_manifest(out_manifest, clip_rows)

        summary_lines.extend(
            [
                f"## {split_name}",
                f"- Input tracks: {len(in_rows)}",
                f"- Output clips: {len(clip_rows)}",
                f"- Failed tracks: {fail_count}",
                f"- Tracks resampled to {TARGET_SR}: {resampled_count}",
                f"- Tracks amplitude-normalized: {normalized_count}",
                f"- Tracks mono-folded from multi-channel: {mono_fold_count}",
                f"- Original sample rates seen: {dict(sorted(sr_counts.items()))}",
                "",
            ]
        )

    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
