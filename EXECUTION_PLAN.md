# 30-Hour Execution Guide

## Summary
This plan is a realistic 4-day, 30-hour project focused on learning and experimentation.

**Goal:** Evaluate whether MERT-based fine-tuned embeddings can improve performance+sound song retrieval over simple baselines.

## Day-by-Day Plan

### Day 1 (8h): Problem Framing + Data Pipeline
- [ ] Finalize project question:
  - "Can embedding-based audio retrieval find style-similar songs better than metadata-like baselines?"
- [ ] Define evaluation:
  - Primary metric: Recall@10
  - Secondary metrics: Recall@1, Recall@5, qualitative listening checks
- [ ] Prepare GTZAN:
  - Verify file integrity
  - Verify class balance
  - Choose clip length and sample rate
- [ ] Define split protocol and leakage safeguards
- [ ] Run a tiny dry training run to validate pipeline assumptions

### Day 2 (8h): Baseline + First MERT Fine-Tune
- [ ] Build a simple baseline (e.g., MFCC + nearest neighbors)
- [ ] Train first MERT retrieval run:
  - Linear retrieval head on frozen MERT embeddings
  - Contrastive objective for similarity search
- [ ] Track experiments with:
  - Augmentation on/off
  - Embedding dimension
  - Temperature/loss settings
  - Batch size
- [ ] End-of-day outputs:
  - One baseline score
  - One MERT-finetune score
  - Failure mode notes

### Day 3 (8h): Controlled Experiments + Analysis
- [ ] Run 3-5 targeted experiments (no broad sweeps)
- [ ] For each run, record:
  - Hypothesis
  - Change made
  - Result
  - Interpretation
- [ ] Perform retrieval error analysis:
  - Good examples
  - Bad examples
  - Potential artist/recording-quality bias cases
- [ ] Compare freezing strategies and evaluate on held-out split:
  - Fully frozen encoder
  - Top-layer unfreeze
  - Partial unfreeze (upper block)

### Day 4 (6h): Demo + Writeup Packaging
- [ ] Build minimal retrieval demo:
  - Query song -> top-k similar songs + scores
- [ ] Create final report artifacts:
  - Main metric table
  - Small ablation table
  - 2-3 qualitative retrieval examples
- [ ] Write final PDF sections:
  - Problem
  - Data + EDA
  - Method
  - Results + limitations
- [ ] Finalize time log entries (activities + roadblocks)

## High-Value Experiment Menu (Pick 5-7 Total)
- [ ] Clip duration (3s vs 5s vs 10s)
- [ ] Mel bins (64 vs 128)
- [ ] Augmentations (time mask, freq mask, noise, pitch shift)
- [ ] Embedding dimension/head size (64 vs 128 vs 256)
- [ ] Contrastive temperature/margin
- [ ] L2 normalization on/off before retrieval
- [ ] MERT layer choice (last vs weighted layer pooling)
- [ ] Freeze policy (frozen vs partial vs full fine-tune)
- [ ] Retrieval `k` (5 vs 10 vs 20)
- [ ] Distance metric (cosine vs euclidean)

## Places to Learn and Compare Ideas

### Core Conceptual Reference
- [1508.06576v2.pdf](/Users/samuelturner/Documents/music_sis/1508.06576v2.pdf)

### MIR and Evaluation
- ISMIR papers/tutorials on music similarity and retrieval embeddings
- MIREX task definitions for ranking and retrieval evaluation framing

### Tools / Libraries
- `librosa` docs (feature and spectrogram design)
- `torchaudio` docs (audio augmentations and loading)
- `transformers` docs (MERT loading and fine-tuning)
- FAISS docs (nearest-neighbor indexing)

### Dataset Caveats
- GTZAN known issues and leakage discussions (to include in limitations)

## Report + Grading Alignment Checklist
- [ ] Training/fine-tuning requirement clearly satisfied
- [ ] Limitations section includes bias/leakage/style subjectivity
- [ ] Iterations and learning process are documented
- [ ] Hour constraints are respected:
  - [ ] Research <= 5h
  - [ ] Prep <= 10h
  - [ ] Core DL >= 20h
- [ ] Final narrative reflects ambition, learning, and measured progress

## Hour Budget (Locked)
- Research/reading: **2h**
- Prep work: **6h**
- Core DL work: **22h**
- Total: **30h**

## Project Assumptions
- Single consumer GPU compute target
- Scope fixed to one dataset (GTZAN) for feasibility
- Start with `m-a-p/MERT-v1-95M` for feasibility on consumer hardware
- Learning quality prioritized over perfect final performance
