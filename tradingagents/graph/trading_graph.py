# TradingAgents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_news,
    get_insider_transactions,
    get_global_news,
    get_announcements,
    get_cls_telegraph,
    get_research_reports,
    get_news_from_search,
    get_xueqiu_posts,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        # Create LLM instances with role-specific temperatures
        # 四个基础分析师 + 交易员：使用 CLI 选择的模型（受 use_deep_think_for_analysts 控制）
        use_deep = self.config.get("use_deep_think_for_analysts", True)
        self.market_llm = self._create_templllm(self.config.get("temperature_market", 0.5), use_deep_think=use_deep)
        self.sentiment_llm = self._create_templllm(self.config.get("temperature_sentiment", 0.5), use_deep_think=use_deep)
        self.news_llm = self._create_templllm(self.config.get("temperature_news", 0.5), use_deep_think=use_deep)
        self.fundamentals_llm = self._create_templllm(self.config.get("temperature_fundamentals", 0.2), use_deep_think=use_deep)
        self.trader_llm = self._create_templllm(self.config.get("temperature_trader", 0.3), use_deep_think=use_deep)
        # 其他角色：固定配置，不受 CLI 选择影响
        self.research_manager_llm = self._create_templllm(self.config.get("temperature_research_manager", 0.4), use_deep_think=True)  # 固定 deep think
        self.portfolio_manager_llm = self._create_templllm(self.config.get("temperature_portfolio_manager", 0.3), use_deep_think=True)  # 固定 deep think
        # 风控分析师：固定使用 quick_think_llm
        self.aggressive_risk_llm = self._create_templllm(self.config.get("temperature_aggressive_risk", 0.6), use_deep_think=False)
        self.conservative_risk_llm = self._create_templllm(self.config.get("temperature_conservative_risk", 0.6), use_deep_think=False)
        self.neutral_risk_llm = self._create_templllm(self.config.get("temperature_neutral_risk", 0.6), use_deep_think=False)
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
            # Role-specific LLMs with different temperatures
            market_llm=self.market_llm,
            sentiment_llm=self.sentiment_llm,
            news_llm=self.news_llm,
            fundamentals_llm=self.fundamentals_llm,
            trader_llm=self.trader_llm,
            research_manager_llm=self.research_manager_llm,
            portfolio_manager_llm=self.portfolio_manager_llm,
            aggressive_risk_llm=self.aggressive_risk_llm,
            conservative_risk_llm=self.conservative_risk_llm,
            neutral_risk_llm=self.neutral_risk_llm,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        elif provider == "minimax":
            max_tokens = self.config.get("minimax_max_tokens")
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

        return kwargs

    def _create_templllm(self, temperature: float, use_deep_think: bool = True) -> Any:
        """Create an LLM instance with a specific temperature.
        
        Args:
            temperature: Temperature value for controlling randomness (0.0-1.0)
            use_deep_think: If True, use deep_think_llm; otherwise use quick_think_llm
            
        Returns:
            Configured LLM instance with the specified temperature
        """
        llm_kwargs = self._get_provider_kwargs()
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks
        
        # Choose model based on use_deep_think flag
        model_name = self.config["deep_think_llm"] if use_deep_think else self.config["quick_think_llm"]
        
        client = create_llm_client(
            provider=self.config["llm_provider"],
            model=model_name,
            base_url=self.config.get("backend_url"),
            temperature=temperature,
            **llm_kwargs,
        )
        return client.get_llm()

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                    # Xueqiu (雪球) social media sentiment
                    get_xueqiu_posts,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    # Company announcements (公告)
                    get_announcements,
                    # CLS telegraph (财联社电报)
                    get_cls_telegraph,
                    # Research reports (个股研报)
                    get_research_reports,
                    # Brave Search real-time web news
                    get_news_from_search,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date.

        When ``checkpoint_enabled`` is set in config, the graph is recompiled
        with a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            from .checkpointer import checkpoint_step, get_checkpointer

            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.graph_setup._workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            _last_printed_msg_id = None
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    last_msg = chunk["messages"][-1]
                    msg_id = getattr(last_msg, "id", id(last_msg))
                    if msg_id != _last_printed_msg_id:
                        last_msg.pretty_print()
                        _last_printed_msg_id = msg_id
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        eval_base = self.config.get("eval_results_dir", self.config.get("results_dir", "eval_results"))
        directory = Path(eval_base) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_file = directory / f"full_states_log_{trade_date}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
