import pytest

from supply_chain_predictor import SupplyChainPredictor

STABLE = {
    "lead_time_days": 7,
    "on_time_delivery_rate": 0.98,
    "inventory_days_of_supply": 45,
    "supplier_financial_health": 0.9,
    "geopolitical_risk_index": 0.1,
    "demand_volatility": 0.15,
    "single_source": 0,
}

FRAGILE = {
    "lead_time_days": 50,
    "on_time_delivery_rate": 0.6,
    "inventory_days_of_supply": 5,
    "supplier_financial_health": 0.3,
    "geopolitical_risk_index": 0.9,
    "demand_volatility": 0.85,
    "single_source": 1,
}


@pytest.fixture(scope="module")
def predictor():
    return SupplyChainPredictor()


def test_missing_features_raise(predictor):
    with pytest.raises(KeyError):
        predictor.assess({"lead_time_days": 5}, [])


def test_assessment_fields(predictor):
    result = predictor.assess(STABLE, [])
    assert 0.0 <= result.overall_risk <= 1.0
    assert result.risk_level in {"Low", "Moderate", "High", "Critical"}
    assert result.recommendation
    assert len(result.top_factors) == 3


def test_fragile_supplier_is_riskier(predictor):
    stable = predictor.assess(STABLE, [])
    fragile = predictor.assess(FRAGILE, [])
    assert fragile.overall_risk > stable.overall_risk


def test_news_increases_risk(predictor):
    without_news = predictor.assess(STABLE, [])
    with_news = predictor.assess(
        STABLE, ["Major port strike and semiconductor shortage disrupt shipments"]
    )
    assert with_news.news_risk > 0.0
    assert with_news.overall_risk > without_news.overall_risk


def test_no_news_uses_full_model_weight(predictor):
    result = predictor.assess(STABLE, [])
    assert result.ml_weight == 1.0
    assert result.news_weight == 0.0


def test_is_deterministic(predictor):
    first = predictor.assess(FRAGILE, ["port congestion delay"]).to_dict()
    second = predictor.assess(FRAGILE, ["port congestion delay"]).to_dict()
    assert first == second
