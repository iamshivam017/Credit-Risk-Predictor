"""
src/data_cleaning.py
---------------------
Loads the internal + external bureau datasets, cleans the -99999
missing-value sentinel, and merges them on PROSPECTID.

Public API
----------
load_raw() -> (df_internal, df_external)
clean_internal(df_internal) -> df_internal
clean_external(df_external) -> (df_external, dropped_columns)
merge_datasets(df_internal, df_external) -> df_merged
load_and_clean_merged(use_cache=True) -> df_merged
"""

from typing import List, Tuple

import pandas as pd

from src.utils import (
    EXTERNAL_PATH,
    INTERNAL_PATH,
    MERGED_CACHE_PATH,
    SENTINEL_COLUMN_DROP_THRESHOLD,
    SENTINEL_VALUE,
)


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw Excel sources, unmodified."""
    df_internal = pd.read_excel(INTERNAL_PATH)
    df_external = pd.read_excel(EXTERNAL_PATH)
    return df_internal, df_external


def clean_internal(df_internal: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where the sentinel value appears in Age_Oldest_TL.

    Only ~40 of 51,336 rows are affected — dropping is preferred over
    imputing since the fraction lost is negligible.
    """
    return df_internal.loc[df_internal["Age_Oldest_TL"] != SENTINEL_VALUE].copy()


def clean_external(df_external: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean the external dataset in two steps:

    1. Drop any column (except PROSPECTID/Approved_Flag) where more than
       `SENTINEL_COLUMN_DROP_THRESHOLD` rows are the sentinel value —
       imputing a column that's mostly missing fabricates signal rather
       than recovering it.
    2. Drop any remaining rows containing the sentinel in a surviving
       numeric column.

    Returns
    -------
    (cleaned_df, list_of_dropped_column_names)
    """
    df = df_external.copy()

    sentinel_counts = (
        df.drop(columns=["PROSPECTID", "Approved_Flag"]) == SENTINEL_VALUE
    ).sum()
    dropped_columns = sentinel_counts[
        sentinel_counts > SENTINEL_COLUMN_DROP_THRESHOLD
    ].index.tolist()

    df = df.drop(columns=dropped_columns)

    for col in df.select_dtypes(include="number").columns:
        df = df.loc[df[col] != SENTINEL_VALUE]

    return df, dropped_columns


def merge_datasets(df_internal: pd.DataFrame, df_external: pd.DataFrame) -> pd.DataFrame:
    """Inner join on PROSPECTID — keeps only applicants clean in both sources."""
    return pd.merge(df_internal, df_external, how="inner", on="PROSPECTID")


def load_and_clean_merged(use_cache: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load raw -> clean internal -> clean external -> merge.

    If `use_cache` is True and a cached parquet file exists at
    MERGED_CACHE_PATH, it's loaded instead of re-reading the (large)
    source Excel files — this is what makes the Streamlit dashboard
    boot quickly instead of re-parsing a 13MB Excel file on every run.
    Call `refresh_cache()` (or delete data/processed/merged_clean.parquet)
    after the raw source files change.
    """
    if use_cache and MERGED_CACHE_PATH.exists():
        return pd.read_pickle(MERGED_CACHE_PATH)

    df_internal, df_external = load_raw()
    df_internal = clean_internal(df_internal)
    df_external, _dropped = clean_external(df_external)
    df_merged = merge_datasets(df_internal, df_external)

    df_merged.to_pickle(MERGED_CACHE_PATH)
    return df_merged


def refresh_cache() -> pd.DataFrame:
    """Force a re-read of the raw source files, bypassing the parquet cache."""
    return load_and_clean_merged(use_cache=False)


if __name__ == "__main__":
    # `python -m src.data_cleaning` — quick standalone sanity check.
    df = load_and_clean_merged(use_cache=False)
    print(f"Merged dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df["Approved_Flag"].value_counts())
