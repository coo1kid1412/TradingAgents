"""归档应使用报告内实际分析日，并保留市场风险双周期字段。"""

from tradingagents.harness.archive import _merge_pred_fields, resolve_trade_date
from tradingagents.harness.extractor import ExtractResult, _find_yaml_block


def test_archive_prefers_pm_trade_date_over_report_generation_date():
    extract = ExtractResult(pm_summary={"trade_date": "2026-06-20"})

    assert resolve_trade_date({"trade_date": "2026-06-25"}, extract) == "2026-06-20"


def test_archive_extracts_market_risk_and_dual_horizon_fields():
    extract = ExtractResult(pm_summary={
        "market_risk_level": "高", "market_entry_gate": "WAIT",
        "market_position_cap_pct": 3, "short_term_trend": "下跌",
        "short_term_confidence": "中", "theme_outlook_12m": "扩张",
    })
    fields = _merge_pred_fields(extract)

    assert fields["market_risk_level"] == "高"
    assert fields["short_term_trend"] == "下跌"
    assert fields["theme_outlook_12m"] == "扩张"


def test_yaml_extractor_accepts_document_markers_inside_fence():
    text = """```yaml
---
PM_SUMMARY:
  pm_rating: OVERWEIGHT
  entry_timing: 暂不介入
---
```"""
    parsed = _find_yaml_block(text, "PM_SUMMARY")
    assert parsed == {"pm_rating": "OVERWEIGHT", "entry_timing": "暂不介入"}


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {fn.__name__}: [{type(e).__name__}] {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
