"""Four-pillar IC recommendation contract tests.

Run: python tradingagents/agents/managers/test_ic_recommendation.py
"""

from tradingagents.agents.managers.ic_recommendation import (
    compute_ic_recommendation as IC,
)


ELIGIBLE = ["FUND-01", "VAL-01", "NEWS-01", "RISK-01", "BREAK-01"]


def _base(**overrides):
    payload = {
        "scenario_expected_return_pct": 0.0,
        "downside_pct": -12.0,
        "payoff_ratio": 1.5,
        "thesis_state": "adequate",
        "thesis_direction": "bullish",
        "valuation_state": "fair",
        "catalyst_state": "visible",
        "priced_in": "partial",
        "durability_state": "acceptable",
        "hard_veto": False,
        "evidence_quality": "complete",
        "style": "blue_chip",
        "theme_premium_pct": 0.0,
        "theme_stage": "none",
        "crowded_long": False,
        "eligible_evidence_ids": ELIGIBLE,
        "thesis_evidence_ids": ["FUND-01"],
        "valuation_evidence_ids": ["VAL-01"],
        "catalyst_evidence_ids": ["NEWS-01"],
        "durability_evidence_ids": ["RISK-01"],
        "thesis_breaker_evidence_ids": [],
    }
    payload.update(overrides)
    return payload


def test_buy_requires_high_return_sound_thesis_and_visible_catalyst():
    result = IC.invoke(_base(
        scenario_expected_return_pct=40.0,
        valuation_state="attractive",
        thesis_state="strong",
        catalyst_state="strong",
        durability_state="resilient",
    ))
    assert result["research_rating"] == "BUY", result
    assert "EXPECTED_RETURN_HIGH" in result["rating_reason_codes"]
    assert "THESIS_STRONG" in result["rating_reason_codes"]


def test_positive_configuration_return_maps_to_overweight():
    result = IC.invoke(_base(
        scenario_expected_return_pct=20.0,
        valuation_state="attractive",
    ))
    assert result["research_rating"] == "OVERWEIGHT", result


def test_cheap_stock_with_weak_thesis_is_not_positive():
    result = IC.invoke(_base(
        scenario_expected_return_pct=40.0,
        valuation_state="attractive",
        thesis_state="weak",
        thesis_direction="bearish",
    ))
    assert result["research_rating"] == "HOLD", result
    assert "WEAK_THESIS_CAP" in result["rating_reason_codes"]


def test_good_company_that_is_only_expensive_is_not_a_sell():
    result = IC.invoke(_base(
        scenario_expected_return_pct=-40.0,
        valuation_state="stretched",
        thesis_state="strong",
        thesis_direction="bullish",
        catalyst_state="visible",
        durability_state="resilient",
    ))
    assert result["research_rating"] == "HOLD", result
    assert "NO_NON_VALUATION_BEAR_CASE" in result["rating_reason_codes"]


def test_negative_return_plus_fragile_durability_maps_to_underweight():
    result = IC.invoke(_base(
        scenario_expected_return_pct=-20.0,
        valuation_state="stretched",
        thesis_state="mixed",
        thesis_direction="mixed",
        catalyst_state="adverse",
        durability_state="fragile",
    ))
    assert result["research_rating"] == "UNDERWEIGHT", result


def test_deep_negative_return_and_broken_durability_maps_to_sell():
    result = IC.invoke(_base(
        scenario_expected_return_pct=-40.0,
        valuation_state="stretched",
        thesis_state="weak",
        thesis_direction="bearish",
        catalyst_state="adverse",
        durability_state="broken",
    ))
    assert result["research_rating"] == "SELL", result


def test_verified_thesis_breaker_overrides_stale_positive_return():
    result = IC.invoke(_base(
        scenario_expected_return_pct=5.0,
        valuation_state="fair",
        thesis_state="weak",
        thesis_direction="bearish",
        catalyst_state="adverse",
        durability_state="broken",
        hard_veto=True,
        thesis_breaker_evidence_ids=["BREAK-01"],
    ))
    assert result["research_rating"] == "SELL", result
    assert "THESIS_BREAK_OVERRIDE" in result["rating_reason_codes"]


def test_unverified_thesis_breaker_cannot_override_return_direction():
    result = IC.invoke(_base(
        scenario_expected_return_pct=5.0,
        thesis_state="weak",
        thesis_direction="bearish",
        durability_state="broken",
        hard_veto=True,
        thesis_breaker_evidence_ids=["NOT-ELIGIBLE"],
    ))
    assert "error" in result
    assert "NOT-ELIGIBLE" in result["error"]


def test_insufficient_evidence_caps_directional_rating_at_hold():
    result = IC.invoke(_base(
        scenario_expected_return_pct=40.0,
        valuation_state="attractive",
        thesis_state="strong",
        catalyst_state="strong",
        evidence_quality="insufficient",
    ))
    assert result["research_rating"] == "HOLD", result
    assert "INSUFFICIENT_EVIDENCE_CAP" in result["rating_reason_codes"]


def test_partial_evidence_and_crowding_cap_buy_at_overweight():
    result = IC.invoke(_base(
        scenario_expected_return_pct=40.0,
        valuation_state="attractive",
        thesis_state="strong",
        catalyst_state="strong",
        evidence_quality="partial",
        crowded_long=True,
    ))
    assert result["research_rating"] == "OVERWEIGHT", result
    assert "PARTIAL_EVIDENCE_CAP" in result["rating_reason_codes"]
    assert "CROWDED_LONG_CAP" in result["rating_reason_codes"]


def test_high_beta_style_uses_wider_return_thresholds():
    result = IC.invoke(_base(
        scenario_expected_return_pct=20.0,
        valuation_state="attractive",
        style="high_beta_growth",
    ))
    assert result["thresholds"]["configuration_pct"] == 22.5
    assert result["research_rating"] == "HOLD", result


def test_invalid_enum_is_rejected_deterministically():
    result = IC.invoke(_base(thesis_state="excellent"))
    assert result == {"error": "thesis_state 非法：excellent"}


def test_tool_signature_has_no_short_term_execution_inputs():
    schema = IC.args_schema.model_json_schema()["properties"]
    assert "entry_timing" not in schema
    assert "market_mode" not in schema
    assert "position_cap_pct" not in schema
    assert "trade_action" not in schema


def test_same_input_is_reproducible():
    payload = _base(
        scenario_expected_return_pct=20.0,
        valuation_state="attractive",
    )
    assert IC.invoke(payload) == IC.invoke(payload)


def test_tltw_empty_evidence_object_matches_empty_list():
    expected = IC.invoke(_base(style="etf", thesis_breaker_evidence_ids=[]))
    actual = IC.invoke(_base(style="etf", thesis_breaker_evidence_ids={}))
    assert actual == expected


def test_all_optional_evidence_fields_accept_empty_objects_only():
    fields = [name for name in _base() if name.endswith("evidence_ids")]
    for field in fields:
        base = _base()
        if field == "eligible_evidence_ids":
            base.update({name: [] for name in fields})
        assert IC.invoke({**base, field: {}}) == IC.invoke({**base, field: []}), field


def test_nonempty_objects_and_scalar_evidence_stay_invalid():
    from pydantic import ValidationError
    for value in ({"BREAK-01": True}, "BREAK-01", 0, False, [{"id": "BREAK-01"}]):
        try:
            IC.invoke(_base(thesis_breaker_evidence_ids=value))
        except ValidationError:
            continue
        raise AssertionError(f"Malformed evidence must be rejected: {value!r}")


def test_empty_object_does_not_bypass_hard_veto_evidence_gate():
    result = IC.invoke(_base(hard_veto=True, thesis_breaker_evidence_ids={}))
    assert "error" in result
    assert "缺少" in result["error"]


def test_evidence_tool_schema_remains_array_or_null():
    properties = IC.args_schema.model_json_schema()["properties"]
    for name, schema in properties.items():
        if name.endswith("evidence_ids"):
            variants = {item["type"] for item in schema["anyOf"]}
            assert variants == {"array", "null"}, (name, schema)


if __name__ == "__main__":
    import sys

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {test.__name__}: [{type(exc).__name__}] {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
