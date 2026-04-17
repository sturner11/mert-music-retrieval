# Phase 2 Spec (Setup + Data Pipeline)

## Objective
Build a reproducible, leakage-safe GTZAN data pipeline that is ready for MERT retrieval training in Phase 3.

## Locked Defaults
- Dataset root: `/Users/samuelturner/Documents/mert-music-retrieval/data/raw/gtzan`
- Similarity target: performance feel + sound/timbre
- Sample rate: `24000` Hz (single canonical resample)
- Channels: mono (mean fold across channels)
- Clip length: `5.0` seconds
- Clip policy: deterministic crop using hash-based offset per `track_id`
- Normalization policy: peak normalization to `[-1, 1]` after decode/resample
- Metrics: Recall@1, Recall@5, Recall@10 (primary: Recall@10)
- Split policy: track-level split before any clip generation
- Random seed: `478`

## Retrieval Objective Statement
Learn embeddings where nearest neighbors preserve performance feel (groove, articulation, energy) and sonic character (timbre/production profile).

## Success Criteria
Phase 2 is successful when train/val/test manifests are generated with zero overlap and the preprocessing/smoke-test checks pass on available files.
