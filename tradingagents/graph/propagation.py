# TradingAgents/graph/propagation.py

import logging
from typing import Dict, Any, List, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.ticker_resolver import resolve_ticker

logger = logging.getLogger(__name__)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        Resolves the ticker first via AKShare/Tushare/yfinance to validate
        and retrieve the company/fund name. Raises TickerNotFoundError if
        the ticker cannot be resolved by any data source.
        """
        resolved = resolve_ticker(company_name)

        # 构建规范化 ticker —— 供所有 agent 工具调用使用
        # A股用纯6位代码（is_a_share 可识别、各 vendor 可转换）
        # 港股用 code.HK，美股用纯 ticker
        if resolved.market == "a_share":
            normalized_ticker = resolved.code
        elif resolved.exchange:
            normalized_ticker = f"{resolved.code}.{resolved.exchange}"
        else:
            normalized_ticker = resolved.original_input

        logger.info(
            "Ticker resolved: %s → %s (%s, market=%s, ticker=%s)",
            company_name, resolved.name, resolved.code,
            resolved.market, normalized_ticker,
        )

        return {
            "messages": [("human", company_name)],
            "company_of_interest": normalized_ticker,
            "company_name": resolved.name,
            "trade_date": str(trade_date),
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
