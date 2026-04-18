from typing import Dict

import pandas as pd

from music_sis.config import TEST_SPLIT_FILE, TRAIN_SPLIT_FILE, VAL_SPLIT_FILE


def load_split_dataframes() -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(TRAIN_SPLIT_FILE),
        "val": pd.read_csv(VAL_SPLIT_FILE),
        "test": pd.read_csv(TEST_SPLIT_FILE),
    }
