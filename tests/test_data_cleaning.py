"""tests/test_data_cleaning.py — Tests for the data loading/cleaning pipeline."""

import pandas as pd
import pytest

from src.data_cleaning import (
    clean_external,
    clean_internal,
    load_and_clean_merged,
    load_raw,
    merge_datasets,
)
from src.utils import SENTINEL_VALUE


@pytest.fixture(scope="module")
def raw_data():
    return load_raw()


class TestCleanInternal:
    def test_removes_all_sentinel_rows(self, raw_data):
        df_internal, _ = raw_data
        cleaned = clean_internal(df_internal)
        assert (cleaned["Age_Oldest_TL"] == SENTINEL_VALUE).sum() == 0

    def test_removes_small_fraction_of_rows(self, raw_data):
        df_internal, _ = raw_data
        cleaned = clean_internal(df_internal)
        dropped = len(df_internal) - len(cleaned)
        assert 0 < dropped < 100  # known to be ~40


class TestCleanExternal:
    def test_no_sentinel_remains_in_numeric_columns(self, raw_data):
        _, df_external = raw_data
        cleaned, _dropped = clean_external(df_external)
        numeric_cols = cleaned.select_dtypes(include="number").columns
        assert (cleaned[numeric_cols] == SENTINEL_VALUE).sum().sum() == 0

    def test_drops_at_least_one_sparse_column(self, raw_data):
        _, df_external = raw_data
        _cleaned, dropped = clean_external(df_external)
        assert len(dropped) > 0

    def test_keeps_prospectid_and_target(self, raw_data):
        _, df_external = raw_data
        cleaned, _dropped = clean_external(df_external)
        assert "PROSPECTID" in cleaned.columns
        assert "Approved_Flag" in cleaned.columns


class TestMerge:
    def test_merged_shape_matches_expected(self, raw_data):
        df_internal, df_external = raw_data
        df_internal = clean_internal(df_internal)
        df_external, _ = clean_external(df_external)
        merged = merge_datasets(df_internal, df_external)
        # Known-good values from the documented project report.
        assert merged.shape[0] == 42_064
        assert merged.shape[1] == 79

    def test_no_duplicate_prospectid(self, raw_data):
        df_internal, df_external = raw_data
        df_internal = clean_internal(df_internal)
        df_external, _ = clean_external(df_external)
        merged = merge_datasets(df_internal, df_external)
        assert merged["PROSPECTID"].is_unique

    def test_target_has_exactly_four_classes(self, raw_data):
        df_internal, df_external = raw_data
        df_internal = clean_internal(df_internal)
        df_external, _ = clean_external(df_external)
        merged = merge_datasets(df_internal, df_external)
        assert set(merged["Approved_Flag"].unique()) == {"P1", "P2", "P3", "P4"}


class TestFullPipeline:
    def test_load_and_clean_merged_matches_expected_shape(self):
        df = load_and_clean_merged(use_cache=False)
        assert df.shape[0] == 42_064
        assert df.shape[1] == 79

    def test_cache_roundtrip_matches(self):
        df_fresh = load_and_clean_merged(use_cache=False)
        df_cached = load_and_clean_merged(use_cache=True)
        assert df_fresh.shape == df_cached.shape
        pd.testing.assert_frame_equal(
            df_fresh.reset_index(drop=True), df_cached.reset_index(drop=True)
        )
