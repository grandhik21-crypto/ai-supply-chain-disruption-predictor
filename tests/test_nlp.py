from supply_chain_predictor.nlp import NewsRiskAnalyzer


def test_empty_input_is_zero_risk():
    analyzer = NewsRiskAnalyzer()
    result = analyzer.analyze([])
    assert result.score == 0.0
    assert result.headline_count == 0


def test_score_is_bounded():
    analyzer = NewsRiskAnalyzer()
    heavy = ["strike shortage war earthquake bankruptcy sanction embargo shutdown"]
    result = analyzer.analyze(heavy)
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.5


def test_risk_terms_increase_score():
    analyzer = NewsRiskAnalyzer()
    calm = analyzer.analyze(["Shipments on schedule, operations normal"])
    risky = analyzer.analyze(["Factory fire causes major production shutdown"])
    assert risky.score > calm.score


def test_matched_terms_are_reported():
    analyzer = NewsRiskAnalyzer()
    result = analyzer.analyze(["Port congestion triggers shipment delay"])
    assert "congestion" in result.matched_risk_terms
    assert "delay" in result.matched_risk_terms


def test_string_input_is_accepted():
    analyzer = NewsRiskAnalyzer()
    result = analyzer.analyze("supplier bankruptcy announced")
    assert result.headline_count == 1
    assert result.score > 0.0
