import pandas as pd

import config


class DataLoader:
    def __init__(self, data_file: str = None):
        self.data_file = data_file or config.DATA_FILE

    def load_data(self) -> pd.DataFrame:
        return pd.read_excel(self.data_file, dtype={"internalReviewId": str})

    def get_valid_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            df["reviewText"].notna() &
            (df["reviewText"].astype(str).str.strip() != "")
        ].copy()

    def get_total_valid_count(self, df: pd.DataFrame) -> int:
        return len(self.get_valid_entries(df))

    def filter_by_ids(self, df: pd.DataFrame, ids: list) -> pd.DataFrame:
        return df[df["internalReviewId"].isin(ids)].drop_duplicates(subset=["internalReviewId"])

    def sample_random(self, df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
        return df.sample(n=min(n, len(df)), random_state=random_state)
