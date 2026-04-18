### Phase 3 Protocol

#### 1) Dataset Version (Frozen)
- Dataset root: `/Users/samuelturner/Documents/mert-music-retrieval/data/raw/gtzan`
- Dataset source: Kaggle `andradaolteanu/gtzan-dataset-music-genre-classification` (frozen on 2026-04-16)
- Sample rate: `24000` Hz
- Channels: mono (mean fold across channels)
- Clip length: `5.0` seconds
- Clip policy: deterministic crop using hash-based offset per `track_id`
- Normalization policy: peak normalization to `[-1, 1]` after decode/resample
- Split policy: track-level split before clip generation
- Random seed: `478`
- Known exclusion: one train file dropped due to decoder error: `jazz.00054.wav`
- Freeze rule: baseline and all MERT runs must use this exact dataset/split snapshot with no further data edits.

#### 2) Split Manifests and Counts (`splits_5s`)
- Train manifest: `/Users/samuelturner/Documents/mert-music-retrieval/artifacts/splits_5s/train.csv`
- Val manifest: `/Users/samuelturner/Documents/mert-music-retrieval/artifacts/splits_5s/val.csv`
- Test manifest: `/Users/samuelturner/Documents/mert-music-retrieval/artifacts/splits_5s/test.csv`
- Clip counts:
  - Train: `5584`
  - Val: `700`
  - Test: `699`

#### 3) Leakage Check (Locked Result)
- Unique `track_id` counts:
  - Train: `799`
  - Val: `100`
  - Test: `100`
- `track_id` overlap across split pairs:
  - Train-Val: `0`
  - Train-Test: `0`
  - Val-Test: `0`
- `clip_id` overlap across split pairs:
  - Train-Val: `0`
  - Train-Test: `0`
  - Val-Test: `0`

#### 4) Metrics and Relevance
- Primary reported metrics:
  - `R@1`
  - `R@5`
  - `R@10`
- Relevance definitions:
  - Primary task: same `track_id` (query clip excluded from candidates)
  - Secondary task: same genre via `label` (query clip excluded from candidates)
- Reporting scope:
  - Report both primary and secondary retrieval metrics for every run (baseline, frozen MERT, partial-unfreeze, and ablations).

#### 5) Evaluation Procedure (Locked)
- Query set: all clips in test manifest (`699` queries).
- Candidate pool definition: all clips in test manifest.
- Self-match policy: remove the exact same `clip_id` as the query from the candidate pool before ranking.
- Rank cutoffs used for reporting: top `1`, top `5`, top `10`.
- Run policy: single run per experiment configuration.
- Tie handling: deterministic secondary sort (`score` descending, then `clip_id` ascending).

#### 6) Checkpoint Selection Rule
- Select checkpoint with best validation `R@10` on the primary task.
- Tie-breaker for checkpoint selection: higher validation `R@1`.
- If still tied, choose the earlier `global_step` (deterministic final tie-break).
- Test set is used only for final comparison after checkpoint selection.

#### 7) Protocol Lock Note
- Locked on: `2026-04-17` (America/Denver)
- Scope: baseline, MERT frozen, MERT partial-unfreeze, and ablations must all follow this protocol.

#### 8) Validation Cadence (Training Runs)
- Run validation retrieval at least once per epoch.
- Required step cadence: every `250` optimizer steps (plus end-of-epoch validation).
- The "best checkpoint" is selected from all recorded validation points using the rule in Section 6.

#### 9) Baseline Feature Lock (MFCC + NN)
- Input: 24 kHz mono, 5.0s clips from locked manifests.
- Feature: `20` MFCC coefficients per frame.
- Clip-level embedding: time-average + time-std of each MFCC coefficient (final vector size `40`).
- Retrieval distance: cosine distance on L2-normalized clip embeddings.
- Candidate/query/eval rules follow Sections 4 and 5 exactly.

#### 10) Qualitative Retrieval Template
- Required fields per example:
  - Query
  - Top-5 neighbors
  - Relevance labels
  - Short note
