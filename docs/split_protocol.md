# GTZAN Split Protocol (Leakage-Safe)

## Scope
This protocol applies to all GTZAN files under:
`/Users/samuelturner/Documents/mert-music-retrieval/data/raw/gtzan`

## Grouping + IDs
- Unit of split: full track file (never per clip first).
- Stable `track_id` format: `<genre>/<stem>`.
- Label source: parent directory name (`genre`) if directory structure exists.

## Fixed Split Rules
- Seed: `478`
- Ratios: `80/10/10` for train/val/test
- Method: deterministic shuffle at track level, then contiguous slicing by ratio

## Leakage Safeguards
- No `track_id` appears in more than one split.
- Split assignment occurs before segmentation/cropping.
- Any duplicate path or duplicate `track_id` is treated as a validation failure.
- Summary counts by split and by label are recorded after generation.

## Manifest Schema
Each split CSV must contain:
- `track_id`
- `path` (absolute path)
- `label`

## Validation Checklist
- [ ] 0 overlaps in `track_id` across train/val/test
- [ ] 0 duplicate rows within each split
- [ ] all manifest paths exist
- [ ] non-empty splits (when data is present)
