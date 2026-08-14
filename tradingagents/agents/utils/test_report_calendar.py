from tradingagents.agents.utils.report_calendar import normalize_reporting_event


def test_half_year_report_does_not_drift_into_q3():
    result = normalize_reporting_event(
        "2026年半年度报告", "2026Q3", "2026-07-15"
    )
    assert result["reporting_period"] == "2026H1"
    assert result["event_date"] == "2026-08-31"
    assert result["date_basis"] == "legal_deadline"


def test_exact_reserved_q3_date_is_preserved():
    result = normalize_reporting_event(
        "2026年第三季度报告", "2026-10-26", "2026-08-13",
        event_date_basis="official_reservation",
        source_tier="official",
        verification_status="verified",
        source_url="https://static.cninfo.com.cn/finalpage/2026-08-13/DOC-Q3.PDF",
        document_id="DOC-Q3",
    )
    assert result["reporting_period"] == "2026Q3"
    assert result["event_date"] == "2026-10-26"
    assert result["date_basis"] == "exact_reservation"


def test_model_supplied_exact_date_without_traceable_official_basis_uses_deadline():
    result = normalize_reporting_event(
        "2026年第三季度报告", "2026-10-26", "2026-08-13",
        event_date_basis="official_reservation",
        source_tier="official",
        verification_status="verified",
        source_url="https://example.com/fake/DOC-Q3.pdf",
        document_id="DOC-Q3",
    )
    assert result["event_date"] == "2026-10-31"
    assert result["date_basis"] == "legal_deadline"


def test_legal_deadline_is_not_mislabeled_as_exact_reservation():
    h1 = normalize_reporting_event(
        "2026年半年度报告", "2026-08-31", "2026-08-13"
    )
    q3 = normalize_reporting_event(
        "2026年第三季度报告", "2026-10-31", "2026-08-13"
    )
    assert h1["date_basis"] == "legal_deadline"
    assert q3["date_basis"] == "legal_deadline"


def test_forecast_announcement_uses_its_publication_date():
    result = normalize_reporting_event(
        "2026年半年度业绩预增公告", "2026Q3", "2026-07-14"
    )
    assert result["event_date"] == "2026-07-14"
    assert result["date_basis"] == "publication_date"


def test_h1_performance_increase_shorthand_uses_publication_date():
    result = normalize_reporting_event(
        "2026H1业绩预增1172%-1368%", "2026-08-31", "2026-07-15"
    )
    assert result["reporting_period"] == "2026H1"
    assert result["event_date"] == "2026-07-15"
    assert result["date_basis"] == "publication_date"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
