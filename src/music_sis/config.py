import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get(
        "MMR_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
).resolve()

SPLITS_5S_DIR = PROJECT_ROOT / "artifacts" / "splits_5s"
TRAIN_SPLIT_FILE = SPLITS_5S_DIR / "train.csv"
VAL_SPLIT_FILE = SPLITS_5S_DIR / "val.csv"
TEST_SPLIT_FILE = SPLITS_5S_DIR / "test.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BASELINE_RESULTS_FILE = ARTIFACTS_DIR / "baseline_results.csv"
BASELINE_RANKINGS_FILE = ARTIFACTS_DIR / "baseline_rankings.csv"
MERT_FROZEN_RESULTS_FILE = ARTIFACTS_DIR / "mert_frozen_results.csv"
MERT_FROZEN_RANKINGS_FILE = ARTIFACTS_DIR / "mert_frozen_rankings.csv"
MERT_FROZEN_CHECKPOINT_FILE = ARTIFACTS_DIR / "mert_frozen_best_head.pt"

REQUIRED_K_VALUES = (1, 5, 10)
DEFAULT_MERT_NAME = "m-a-p/MERT-v1-95M"
