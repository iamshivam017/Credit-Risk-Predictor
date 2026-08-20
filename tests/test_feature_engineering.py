"""tests/test_feature_engineering.py — Tests for encoding and feature-set logic."""

import pandas as pd
import pytest

from src.data_cleaning import load_and_clean_merged
from src.feature_engineering import align_to_deployment_schema, engineer_features, get_feature_sets
from src.utils import EDUCATION_ORDER


@pytest.fixture(scope="module")
def merged_df():
    return load_and_clean_merged()


@pytest.fixture(scope="module")
def encoded(merged_df):
    return engineer_features(merged_df)


class TestEngineerFeatures:
    def test_education_is_numeric_and_in_range(self, encoded):
        df_encoded, _ = encoded
        assert pd.api.types.is_numeric_dtype(df_encoded["EDUCATION"])
        assert df_encoded["EDUCATION"].min() >= min(EDUCATION_ORDER.values())
        assert df_encoded["EDUCATION"].max() <= max(EDUCATION_ORDER.values())

    def test_target_column_created(self, encoded):
        df_encoded, _ = encoded
        assert "target" in df_encoded.columns
        assert set(df_encoded["target"].unique()) == {0, 1, 2, 3}

    def test_target_encoder_recovers_original_labels(self, encoded):
        _df_encoded, target_encoder = encoded
        assert list(target_encoder.classes_) == ["P1", "P2", "P3", "P4"]

    def test_no_raw_categorical_columns_remain(self, encoded):
        df_encoded, _ = encoded
        for col in ["MARITALSTATUS", "GENDER", "last_prod_enq2", "first_prod_enq2"]:
            assert col not in df_encoded.columns


class TestFeatureSets:
    def test_deployment_feature_count(self, encoded):
        df_encoded, _ = encoded
        deploy_features, _full_features = get_feature_sets(df_encoded)
        assert len(deploy_features) == 50

    def test_full_feature_count(self, encoded):
        df_encoded, _ = encoded
        _deploy_features, full_features = get_feature_sets(df_encoded)
        assert len(full_features) == 51

    def test_full_features_is_deployment_plus_credit_score(self, encoded):
        df_encoded, _ = encoded
        deploy_features, full_features = get_feature_sets(df_encoded)
        assert set(full_features) - set(deploy_features) == {"Credit_Score"}

    def test_credit_score_not_in_deployment_features(self, encoded):
        df_encoded, _ = encoded
        deploy_features, _full_features = get_feature_sets(df_encoded)
        assert "Credit_Score" not in deploy_features


class TestAlignToDeploymentSchema:
    def test_output_matches_expected_columns_exactly(self, encoded):
        df_encoded, _ = encoded
        deploy_features, _ = get_feature_sets(df_encoded)

        raw_row = pd.DataFrame([{
            "GENDER": "M", "MARITALSTATUS": "Married", "EDUCATION": "GRADUATE",
            "NETMONTHLYINCOME": 30000, "last_prod_enq2": "PL", "first_prod_enq2": "PL",
        }])
        aligned = align_to_deployment_schema(raw_row, deploy_features)
        assert list(aligned.columns) == deploy_features

    def test_missing_fields_filled_with_zero(self, encoded):
        df_encoded, _ = encoded
        deploy_features, _ = get_feature_sets(df_encoded)

        raw_row = pd.DataFrame([{
            "GENDER": "F", "MARITALSTATUS": "Single", "EDUCATION": "SSC",
            "last_prod_enq2": "CC", "first_prod_enq2": "CC",
        }])
        aligned = align_to_deployment_schema(raw_row, deploy_features)
        # Any numeric field not supplied should default to 0, not NaN.
        assert aligned.isnull().sum().sum() == 0

    def test_unseen_category_does_not_crash(self, encoded):
        df_encoded, _ = encoded
        deploy_features, _ = get_feature_sets(df_encoded)

        raw_row = pd.DataFrame([{
            "GENDER": "M", "MARITALSTATUS": "Married", "EDUCATION": "GRADUATE",
            "last_prod_enq2": "SomeBrandNewCategoryNeverSeenBefore", "first_prod_enq2": "PL",
        }])
        aligned = align_to_deployment_schema(raw_row, deploy_features)
        assert list(aligned.columns) == deploy_features
