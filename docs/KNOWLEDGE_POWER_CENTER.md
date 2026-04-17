# Beginning Research: Music Similarity Embeddings Power Center

## Mission
Build a song/audio embedding system that retrieves musically useful "similar songs" for real listening workflows.

## Current Strategic Decision
We are focusing on:
1. Performance style signal
2. Sound (timbre/production) signal

We are not prioritizing full composition-style modeling in phase 1.

## Foundation Model Direction
We are adopting a **MERT-first** strategy for phase 1:
1. Start from pre-trained `m-a-p/MERT-v1-95M`.
2. Train retrieval objective for our specific similarity goal.
3. Fine-tune progressively (frozen encoder -> partial unfreeze -> deeper unfreeze as needed).

## Why This Focus Is Right Now
1. Performance + sound gives strong practical retrieval quality quickly.
2. It matches common user intent: "find songs with a similar vibe/feel."
3. It avoids immediate dependency on high-quality symbolic transcription.
4. It is a strong MVP path before deeper composition-level modeling.

## Core Facts To Remember
1. Music style is multi-level (score, control/performance, sound).
2. "Similar" depends on objective, so objective must be explicit.
3. Performance style means how music is played: timing, dynamics, articulation, groove.
4. Sound style means how music sounds: timbre, instrumentation, texture, production profile.
5. Composition style can be added later for cover-version and melodic/harmonic similarity.
6. SSL does not mean "no design"; we still define objective, data, and losses.
7. Foundation encoders learn general features; fine-tuning aligns them to our retrieval definition.

## MERT Notes (What It Is)
MERT is a self-supervised, PLM-style music encoder trained with masked prediction.

Key mechanism:
1. Mask parts of audio sequence.
2. Predict pseudo-targets from teacher signals.
3. Learn reusable representations across many MIR tasks.

MERT combines:
1. Acoustic teacher targets (RVQ-VAE/EnCodec token prediction): strong timbre/sound supervision.
2. Musical teacher targets (CQT reconstruction): pitch/harmonic inductive bias.

Why it fits our project:
1. It is audio-native and captures sound + musical structure.
2. It gives better starting features than small from-scratch models.
3. We can still do substantial project work by fine-tuning and running retrieval experiments.

## data2vec SSL Framework Notes (Core Mechanism)
Music2Vec follows data2vec-style SSL. This is the core training recipe:

1. Build two models with the same architecture:
   - Student model (trained directly)
   - Teacher model (updated as EMA of student weights)

2. Feed different views of the same audio:
   - Student gets masked input sequence
   - Teacher gets unmasked/full input sequence

3. Apply masking on student input:
   - Random spans of feature tokens are masked
   - Student must infer hidden regions from context

4. Produce teacher targets from contextual layers:
   - Teacher outputs contextual representations from Transformer layers
   - Target is usually formed from top-layer representations (often top-K averaged)

5. Train student to predict teacher latent targets:
   - Student outputs contextual vectors for masked positions
   - Loss regresses student vectors toward teacher latent targets

6. Update teacher by momentum/EMA:
   - Teacher is not directly optimized by gradient
   - Teacher parameters track student smoothly over time

7. Why this helps embeddings:
   - Predicting contextualized latent targets encourages global musical understanding
   - The model learns transferable structure beyond local frame reconstruction

## Project Scope (Phase 1)
### In Scope
1. Learn embeddings from audio that capture:
   - Sound profile (timbre/production)
   - Performance feel (rhythm, dynamics, phrasing)
2. Build retrieval pipeline for nearest-neighbor song search.
3. Evaluate with curated queries and expected matches.
4. Fine-tune MERT for retrieval-specific behavior.

### Out of Scope (for now)
1. Full score-level disentanglement.
2. Explicit melody/chord symbolic modeling.
3. Universal "all-style" transfer system.

## Retrieval Definition For This Project
A good match is a song that is close on:
1. Sonic character (production/instrument texture)
2. Performance feel (energy, groove, articulation)

A match does not have to share exact melody/chords in phase 1.

## Evaluation Checklist
Use this when testing results:
1. Does the result feel similar in groove/energy?
2. Does the result sound similar in texture/production?
3. Are top-5 results consistently useful for listening/discovery?
4. Are we overfitting to one cue (only timbre or only tempo)?
5. Does MERT fine-tuning beat simple baselines on Recall@K?

## Failure Modes To Watch
1. Sonically similar but musically dull/irrelevant matches.
2. Tempo-only matching without deeper feel similarity.
3. Production-heavy bias that ignores expression.
4. Retrieval collapse around popular/mastering artifacts.
5. Representation drift after aggressive full-model fine-tuning.

## Expansion Trigger (When To Add Composition Features)
Add composition-level features if we see repeated failures like:
1. Missed covers/reharmonizations that humans judge as similar.
2. Good vibe matches but poor melodic/harmonic relevance.
3. User requests centered on "same musical idea" rather than "same vibe."

## Paper Anchors
1. MuLan (joint audio-text embedding): https://arxiv.org/pdf/2208.12415
2. Music Style Transfer: A Position Paper (3 style types): https://arxiv.org/pdf/1803.06841
3. MERT (SSL music foundation encoder): https://arxiv.org/abs/2306.00107

## Practical MERT Resource Notes
1. Start with `MERT-v1-95M` for feasibility on consumer hardware.
2. `MERT-v1-95M` HF repo footprint is about 1.7 GB.
3. Begin with frozen-encoder + retrieval head, then unfreeze top layers.
4. If compute is tight, prefer shorter clips and gradient accumulation.

## MERT Fine-Tuning Workflow (Project Default)
1. Extract MERT embeddings on training clips.
2. Train contrastive retrieval head (frozen encoder).
3. Evaluate Recall@1/5/10 and qualitative neighbors.
4. Unfreeze upper encoder layers and repeat.
5. Select best checkpoint by retrieval metrics + listening quality.

## Working Principles
1. Start simple, evaluate honestly, iterate fast.
2. Let failure cases drive feature expansion.
3. Keep objective explicit: performance + sound first.
4. Track decisions in this file so reasoning compounds over time.

## Weekly Notes Template
Copy this block each week.

### Week Of: YYYY-MM-DD
- Goal:
- What we tried:
- What worked:
- What failed:
- Retrieval examples (good/bad):
- Next experiment:
- Decision update:
