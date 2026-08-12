from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RM_PATH = ROOT / "tradingagents/agents/managers/research_manager.py"
PM_PATH = ROOT / "tradingagents/agents/managers/portfolio_manager.py"
RESEARCHER_PATHS = (
    ROOT / "tradingagents/agents/researchers/bull_researcher.py",
    ROOT / "tradingagents/agents/researchers/bear_researcher.py",
)


def test_rm_and_researchers_receive_ic_packet_as_primary_evidence_index():
    for path in (*RESEARCHER_PATHS, RM_PATH):
        source = path.read_text(encoding="utf-8")
        assert 'state.get("ic_packet", "")' in source, path
        assert "证据 ID" in source, path
        assert "IC 决策包" in source, path


def test_researchers_require_material_arguments_to_cite_existing_ids():
    for path in RESEARCHER_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "每条有效论据必须引用至少一个本包存在的证据 ID" in source, path
        assert "禁止编造证据 ID" in source, path


def test_rm_summary_has_four_auditable_reference_fields():
    source = RM_PATH.read_text(encoding="utf-8")
    for field in (
        "rating_evidence_ids",
        "target_price_evidence_ids",
        "earnings_evidence_ids",
        "key_conflict_ids",
    ):
        assert field in source
    assert "RM 只能决定长期 thesis" in source


def test_pm_uses_ic_packet_instead_of_four_full_analyst_reports():
    source = PM_PATH.read_text(encoding="utf-8")
    assert 'state.get("ic_packet", "")' in source
    assert "### 4 个 analyst 原始报告" not in source
    assert "PM 是最终交易决策人" in source
    assert "不得让舆情单独决定目标价" in source


def test_pm_summary_has_four_auditable_reference_fields():
    source = PM_PATH.read_text(encoding="utf-8")
    for field in (
        "short_term_evidence_ids",
        "long_term_evidence_ids",
        "position_evidence_ids",
        "target_price_evidence_ids",
    ):
        assert field in source
    assert "字段值只能引用 IC 决策包中存在的证据 ID" in source


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: [{type(exc).__name__}] {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
