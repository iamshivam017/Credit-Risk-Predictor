"""tests/test_model.py — Tests for the trained models and prediction pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.dashboard_utils import predict_batch, predict_single
from src.utils import MODEL_DEPLOYMENT_PATH, MODEL_FULL_PATH, METADATA_PATH, UNSEEN_PATH

pytestmark = pytest.mark.skipif(
    not (MODEL_DEPLOYMENT_PATH.exists() and METADATA_PATH.exists()),
    reason="Models not trained yet — run `python train.py` first.",
)


@pytest.fixture(scope="module")
def deployment_model():
    import joblib
    return joblib.load(MODEL_DEPLOYMENT_PATH)


@pytest.fixture(scope="module")
def full_model():
    import joblib
    return joblib.load(MODEL_FULL_PATH)


@pytest.fixture(scope="module")
def metadata():
    import json
    with open(METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def deploy_features(metadata):
    return metadata["feature_sets"]["deployment_features"]


class TestModelArtifacts:
    def test_deployment_model_has_no_credit_score_feature(self, deployment_model):
        feature_names = list(deployment_model.feature_names_in_)
        assert "Credit_Score" not in feature_names

    def test_full_model_has_credit_score_feature(self, full_model):
        feature_names = list(full_model.feature_names_in_)
        assert "Credit_Score" in feature_names

    def test_deployment_model_predicts_four_classes(self, deployment_model):
        assert len(deployment_model.classes_) == 4


class TestMetadata:
    def test_metadata_reports_expected_dataset_size(self, metadata):
        assert metadata["dataset"]["merged_rows"] == 42_064

    def test_deployment_accuracy_in_plausible_range(self, metadata):
        # Documented benchmark is ~77-78%; allow a small margin for
        # environment-to-environment XGBoost variance.
        acc = metadata["model_b_detail"]["accuracy"]
        assert 0.70 <= acc <= 0.85

    def test_full_accuracy_much_higher_than_deployment(self, metadata):
        acc_full = metadata["model_a_detail"]["accuracy"]
        acc_deploy = metadata["model_b_detail"]["accuracy"]
        # The leakage gap should be large and clearly visible — this is
        # the core empirical claim of the whole project.
        assert acc_full - acc_deploy > 0.15


class TestPredictSingle:
    def _sample_applicant(self):
        return {
            "GENDER": "M", "MARITALSTATUS": "Married", "EDUCATION": "12TH",
            "NETMONTHLYINCOME": 51000, "Time_With_Curr_Empr": 114,
            "Age_Oldest_TL": 72, "Age_Newest_TL": 18, "Secured_TL": 1, "Unsecured_TL": 4,
            "last_prod_enq2": "PL", "first_prod_enq2": "PL",
        }

    def test_returns_valid_class(self, deployment_model, deploy_features):
        result = predict_single(deployment_model, self._sample_applicant(), deploy_features)
        assert result["predicted_class"] in {"P1", "P2", "P3", "P4"}

    def test_confidence_is_a_probability(self, deployment_model, deploy_features):
        result = predict_single(deployment_model, self._sample_applicant(), deploy_features)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_class_probabilities_sum_to_one(self, deployment_model, deploy_features):
        result = predict_single(deployment_model, self._sample_applicant(), deploy_features)
        total = sum(result["class_probabilities"].values())
        assert abs(total - 1.0) < 1e-6

    def test_missing_fields_do_not_crash(self, deployment_model, deploy_features):
        minimal_applicant = {"GENDER": "F", "MARITALSTATUS": "Single", "EDUCATION": "GRADUATE",
                              "last_prod_enq2": "CC", "first_prod_enq2": "CC"}
        result = predict_single(deployment_model, minimal_applicant, deploy_features)
        assert result["predicted_class"] in {"P1", "P2", "P3", "P4"}


class TestPredictBatch:
    def test_scores_the_real_unseen_dataset(self, deployment_model, deploy_features):
        df_unseen = pd.read_excel(UNSEEN_PATH)
        result = predict_batch(deployment_model, df_unseen, deploy_features)
        assert len(result) == len(df_unseen)
        assert "predicted_class" in result.columns
        assert "confidence" in result.columns
        assert result["predicted_class"].isin(["P1", "P2", "P3", "P4"]).all()

    def test_confidence_values_are_valid_probabilities(self, deployment_model, deploy_features):
        df_unseen = pd.read_excel(UNSEEN_PATH)
        result = predict_batch(deployment_model, df_unseen, deploy_features)
        assert (result["confidence"] >= 0.0).all()
        assert (result["confidence"] <= 1.0).all()
