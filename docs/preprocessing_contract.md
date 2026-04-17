# Preprocessing Contract for MERT Input

## Ordered Processing Steps
1. Resolve path from manifest row.
2. Decode audio file; on decode failure, mark as `decode_error`.
3. Convert to mono by channel mean (no left-channel shortcut).
4. Resample once to `24000` Hz.
5. Peak-normalize to `[-1, 1]`.
6. Apply deterministic 5.0s crop using hash(`track_id`) offset.
7. Return waveform tensor/array and metadata (`track_id`, `label`, `path`).

## Edge Cases
- File missing: `missing_file` (exclude from batch and log).
- Decode failure/corrupt: `decode_error` (quarantine and log).
- Silent or near-silent audio: `silent_audio` (log; include/exclude policy decided at training run start, default exclude).
- Short clip (< 5.0s): right-pad with zeros to exact clip length.
- NaN/Inf after processing: `invalid_numeric` and quarantine.

## Consistency Requirements
- Train/val/test use identical preprocessing graph.
- No split-specific normalization differences.
- No second resampling stage downstream.

## Logging Requirements
For each preprocessing run, log:
- total files attempted
- pass count
- fail count by reason
- first 5 failing paths per reason category
