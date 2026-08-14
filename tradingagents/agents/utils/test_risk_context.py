from tradingagents.agents.utils.risk_context import build_risk_data_packet


def test_risk_packet_contains_only_compact_traceable_inputs():
    state = {
        "market_report": """# 技术报告

```yaml
SUMMARY:
  price_data_date: 2026-08-13
  price_data_status: official_daily
  key_support: 118.2
  key_resistance: 136.4
  atr_pct: 4.8
  trend_daily: 下行
  ignored_long_field: 不应进入风险数据包
```

这是一段很长的技术分析正文，不应进入风险数据包。""",
        "stock_profile": """画像长正文
| liquidity | **deep**（深） | 60 日日均成交额 ≈ 12.30 亿（OHLCV 直接算）|
SYS_SHORT_TERM_STRUCTURE: class=weak_rebound_in_downtrend | ma10_slope_5d_pct=1.00
SYS_ENTRY_CAPITAL_FLOW_REGIME: 恶化
普通画像正文不应进入数据包。""",
        "market_risk_snapshot": {
            "as_of_date": "2026-08-13",
            "entry_gate": "WAIT",
            "position_cap_pct": 0,
        },
        "research_evidence_ledger": {
            "analysis_status": "complete",
            "warnings": [],
            "cards": [
                {
                    "claim_id": "NEWS-CAT-01",
                    "owner": "news",
                    "claim": "官方业绩预告",
                    "as_of": "2026-08-12",
                    "quality_status": "valid",
                    "decision_eligible": True,
                    "source_name": "交易所公告",
                    "source_url": "https://static.cninfo.com.cn/finalpage/DOC-1.pdf",
                    "document_id": "DOC-1",
                    "source_tier": "official",
                    "verification_status": "verified",
                },
                {
                    "claim_id": "NEWS-CAT-02",
                    "owner": "news",
                    "claim": "社交媒体传闻",
                    "quality_status": "partial",
                    "decision_eligible": False,
                },
            ],
        },
    }

    packet = build_risk_data_packet(state)

    assert packet["technical"]["atr_pct"] == 4.8
    assert packet["technical"]["key_support"] == 118.2
    assert "ignored_long_field" not in packet["technical"]
    assert packet["profile_truth"][0].startswith("| liquidity |")
    assert any("weak_rebound_in_downtrend" in line for line in packet["profile_truth"])
    assert [item["claim_id"] for item in packet["eligible_evidence"]] == ["NEWS-CAT-01"]
    assert packet["reference_ids"] == [
        "TECHNICAL-SUMMARY",
        "STOCK-PROFILE-TRUTH",
        "MARKET-RISK-SNAPSHOT",
        "NEWS-CAT-01",
    ]
    assert packet["official_event_evidence_ids"] == ["NEWS-CAT-01"]
    rendered = str(packet)
    assert "很长的技术分析正文" not in rendered
    assert "社交媒体传闻" not in rendered


if __name__ == "__main__":
    test_risk_packet_contains_only_compact_traceable_inputs()
    print("  PASS test_risk_packet_contains_only_compact_traceable_inputs")
