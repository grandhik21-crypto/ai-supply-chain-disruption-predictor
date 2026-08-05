import numpy as np
import pytest

from supply_chain_predictor.data import generate_supplier_dataset
from supply_chain_predictor.model import FEATURE_NAMES, DisruptionModel


def _trained_model():
    df = generate_supplier_dataset(n_samples=1500, seed=7)
    model = DisruptionModel(seed=7)
    model.fit(df[FEATURE_NAMES], df["disrupted"])
    return model


def test_dataset_shape_and_labels():
    df = generate_supplier_dataset(n_samples=500, seed=1)
    assert list(df.columns) == FEATURE_NAMES + ["disrupted"]
    assert len(df) == 500
    assert set(df["disrupted"].unique()) <= {0, 1}


def test_predict_before_fit_raises():
    model = DisruptionModel()
    with pytest.raises(RuntimeError):
        model.predict_proba([[10, 0.9, 30, 0.8, 0.2, 0.3, 0]])


def test_probabilities_are_valid():
    model = _trained_model()
    probs = model.predict_proba(
        [
            [7, 0.98, 45, 0.9, 0.1, 0.15, 0],
            [50, 0.6, 5, 0.3, 0.9, 0.85, 1],
        ]
    )
    assert probs.shape == (2,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_risky_supplier_scores_higher_than_stable_one():
    model = _trained_model()
    stable = model.predict_proba([[7, 0.98, 45, 0.9, 0.1, 0.15, 0]])[0]
    fragile = model.predict_proba([[50, 0.6, 5, 0.3, 0.9, 0.85, 1]])[0]
    assert fragile > stable


def test_feature_contributions_cover_all_features():
    model = _trained_model()
    contributions = model.feature_contributions([[50, 0.6, 5, 0.3, 0.9, 0.85, 1]])
    assert set(contributions) == set(FEATURE_NAMES)
