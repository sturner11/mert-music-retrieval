### Baseline Retrieval Notes (MFCC + Cosine NN)

Protocol context:
- Query set: all test clips
- Candidate pool: all test clips (self `clip_id` excluded)
- Similarity: cosine over L2-normalized MFCC mean+std embeddings

### Example 1
Query:
- `jazz__jazz.00078__seg06` (`track_id`: `jazz/jazz.00078`, `label`: `jazz`)

Top-5 neighbors:
1. `jazz__jazz.00078__seg04` (score `0.999986`)
2. `jazz__jazz.00078__seg03` (score `0.999444`)
3. `jazz__jazz.00078__seg01` (score `0.999336`)
4. `jazz__jazz.00074__seg02` (score `0.998943`)
5. `jazz__jazz.00074__seg04` (score `0.998880`)

Relevance labels:
- same-track: `[Y, Y, Y, N, N]`
- same-genre: `[Y, Y, Y, Y, Y]`

Short note:
- Very strong local consistency: same-track segments dominate top ranks, then same-genre neighbors appear with only a small score drop.

### Example 2
Query:
- `rock__rock.00052__seg00` (`track_id`: `rock/rock.00052`, `label`: `rock`)

Top-5 neighbors:
1. `rock__rock.00052__seg01` (score `0.999109`)
2. `country__country.00035__seg00` (score `0.997741`)
3. `country__country.00035__seg01` (score `0.997702`)
4. `country__country.00028__seg03` (score `0.997305`)
5. `country__country.00028__seg06` (score `0.997264`)

Relevance labels:
- same-track: `[Y, N, N, N, N]`
- same-genre: `[Y, N, N, N, N]`

Short note:
- After the first same-track hit, the neighborhood shifts to cross-genre (`country`), suggesting MFCC timbre similarity can override genre semantics.

### Example 3
Query:
- `classical__classical.00097__seg00` (`track_id`: `classical/classical.00097`, `label`: `classical`)

Top-5 neighbors:
1. `classical__classical.00097__seg06` (score `0.999813`)
2. `classical__classical.00097__seg01` (score `0.998796`)
3. `classical__classical.00097__seg04` (score `0.998260`)
4. `classical__classical.00097__seg05` (score `0.998217`)
5. `jazz__jazz.00028__seg01` (score `0.997461`)

Relevance labels:
- same-track: `[Y, Y, Y, Y, N]`
- same-genre: `[Y, Y, Y, Y, N]`

Short note:
- Same-track retrieval is very stable for this query; first mismatch appears at rank 5 and is still acoustically close by MFCC score.

### Baseline Interpretation
- Strength: Excellent near-neighbor recovery for same-track clips in this 5s setup.
- Weakness: Some high-scoring cross-genre confusions appear early for certain queries.
- Hypothesis for MERT: Learned embeddings may reduce semantically awkward cross-genre matches while preserving high same-track recall.
