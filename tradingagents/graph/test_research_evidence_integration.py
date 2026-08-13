from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph import propagation as propagation_module
from tradingagents.graph.propagation import Propagator


ROOT = Path(__file__).resolve().parents[2]


def test_initial_state_contains_empty_research_evidence_fields():
    original_resolver = propagation_module.resolve_ticker
    original_risk_loader = propagation_module.load_market_risk_for_ticker
    propagation_module.resolve_ticker = lambda _: SimpleNamespace(
        market="a_share",
        code="688114",
        exchange="SH",
        name="华大智造",
        original_input="688114",
    )
    propagation_module.load_market_risk_for_ticker = lambda *_: {}
    try:
        state = Propagator().create_initial_state("688114", "2026-08-12")
    finally:
        propagation_module.resolve_ticker = original_resolver
        propagation_module.load_market_risk_for_ticker = original_risk_loader

    assert state["research_evidence_ledger"] == {}
    assert state["ic_packet"] == ""


def test_agent_state_declares_research_evidence_contract():
    annotations = AgentState.__annotations__
    assert "research_evidence_ledger" in annotations
    assert "ic_packet" in annotations


def test_graph_routes_consensus_through_research_evidence_before_bull():
    source = (ROOT / "tradingagents/graph/setup.py").read_text(encoding="utf-8")
    assert "create_research_evidence_node" in source
    assert 'workflow.add_node("Research Evidence Officer"' in source
    assert 'workflow.add_edge("Consensus Officer", "Research Evidence Officer")' in source
    assert 'workflow.add_edge("Research Evidence Officer", "Bull Researcher")' in source
    assert 'workflow.add_edge("Consensus Officer", "Bull Researcher")' not in source


def test_progress_logging_and_report_persistence_include_ic_packet():
    graph_source = (ROOT / "tradingagents/graph/trading_graph.py").read_text(encoding="utf-8")
    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    cli_source = (ROOT / "cli/main.py").read_text(encoding="utf-8")

    assert '"ic_packet": "IC 决策包"' in graph_source
    assert '"research_evidence_ledger": final_state.get(' in graph_source
    assert '"ic_packet": final_state.get(' in graph_source
    assert '("ic_packet", "ic_packet.md", "Research Evidence Officer")' in main_source
    assert '"Research Evidence Officer": "研究证据官"' in main_source
    assert '(analysts_dir / "ic_packet.md").write_text' in cli_source
    assert 'final_state["ic_packet"]' in cli_source


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
