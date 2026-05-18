# TradingAgents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional, Set

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.seed_lessons import get_seed_lessons
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

        # 使用 get_llm_wrapped 注入 wall-clock 超时保护，防止 LLM API 假死导致进程死锁
        self.deep_thinking_llm = deep_client.get_llm_wrapped()
        self.quick_thinking_llm = quick_client.get_llm_wrapped()
        
        # Create LLM instances with role-specific temperatures
        # 四个基础分析师：使用 CLI 选择的模型（受 use_deep_think_for_analysts 控制）
        use_deep = self.config.get("use_deep_think_for_analysts", True)
        self.market_llm = self._create_templllm(self.config.get("temperature_market", 0.2), use_deep_think=use_deep)
        self.sentiment_llm = self._create_templllm(self.config.get("temperature_sentiment", 0.5), use_deep_think=use_deep)
        self.news_llm = self._create_templllm(self.config.get("temperature_news", 0.5), use_deep_think=use_deep)
        self.fundamentals_llm = self._create_templllm(self.config.get("temperature_fundamentals", 0.2), use_deep_think=use_deep)
        # 交易员：默认 quick_think（优化01后只做执行评估），可通过 use_deep_for_trader 回退
        self.trader_llm = self._create_templllm(self.config.get("temperature_trader", 0.3), use_deep_think=self.config.get("use_deep_for_trader", False))
        # RM/PM：固定 deep_think，最终决策推理需要深度
        self.research_manager_llm = self._create_templllm(self.config.get("temperature_research_manager", 0.3), use_deep_think=True)
        self.portfolio_manager_llm = self._create_templllm(self.config.get("temperature_portfolio_manager", 0.3), use_deep_think=True)
        # 多/空研究员：默认 quick_think（修辞密度高但推理深度低），可通过 config flag 回退
        self.bull_researcher_llm = self._create_templllm(self.config.get("temperature_bull_researcher", 0.5), use_deep_think=self.config.get("use_deep_for_bull_researcher", False))
        self.bear_researcher_llm = self._create_templllm(self.config.get("temperature_bear_researcher", 0.5), use_deep_think=self.config.get("use_deep_for_bear_researcher", False))
        # 风控分析师：固定使用 quick_think_llm
        self.aggressive_risk_llm = self._create_templllm(self.config.get("temperature_aggressive_risk", 0.4), use_deep_think=False)
        self.conservative_risk_llm = self._create_templllm(self.config.get("temperature_conservative_risk", 0.4), use_deep_think=False)
        self.neutral_risk_llm = self._create_templllm(self.config.get("temperature_neutral_risk", 0.4), use_deep_think=False)
        
        # Initialize memories
        # Memory 系统简化：旧版给 bull/bear/trader/invest_judge 各分配一个 memory 实例，
        # 但当前工作流（单股票实时分析）下 reflect_* 没有 backtest 闭环触发，且 seed_lessons
        # 只注入 PM——其他 memory 永远是空字符串，纯粹是 prompt 装饰。改造 A 清理：只保留
        # portfolio_manager_memory 一份真正在用的。
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
            bull_researcher_llm=self.bull_researcher_llm,
            bear_researcher_llm=self.bear_researcher_llm,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict
        self._seed_lessons_loaded: Set[str] = set()  # track which markets' seed lessons have been injected

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}

        # Pass LLM timeout from config (prevents indefinite hangs)
        llm_timeout = self.config.get("llm_timeout")
        if llm_timeout:
            kwargs["timeout"] = llm_timeout

        # Pass LLM retry count from config (handles transient 429/5xx errors)
        max_retries = self.config.get("llm_max_retries")
        if max_retries is not None:
            kwargs["max_retries"] = max_retries

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
        # 使用 get_llm_wrapped 注入 wall-clock 超时保护
        return client.get_llm_wrapped()

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

    def _inject_seed_lessons(self, market: str):
        """Inject seed lessons into all 5 memory instances for the given market.

        Common lessons are injected once (on first call regardless of market).
        Market-specific lessons are injected once per unique market.

        Args:
            market: Market identifier from ResolvedTicker ("a_share"/"hk"/"us"/"other")
        """
        # Inject common lessons only once across all markets (idempotent)
        if "common" not in self._seed_lessons_loaded:
            common = get_seed_lessons("other")  # returns only COMMON_LESSONS
            self._add_to_all_memories(common, "common")
            self._seed_lessons_loaded.add("common")

        # Inject market-specific lessons once per unique market
        if market not in self._seed_lessons_loaded and market != "other":
            # get_seed_lessons(market) returns common + market-specific.
            # Extract only the market-specific portion.
            common_count = len(get_seed_lessons("other"))
            all_for_market = get_seed_lessons(market)
            market_specific = all_for_market[common_count:]
            if market_specific:
                self._add_to_all_memories(market_specific, f"{market}-specific")
            self._seed_lessons_loaded.add(market)

    def _add_to_all_memories(self, lessons, label: str):
        """Inject lessons into portfolio_manager_memory only.

        Seed lessons serve as guardrails for the final decision maker.
        Injecting into only portfolio_manager (vs. all 5 memories) prevents
        prior knowledge from biasing the bull/bear debate and intermediate
        decisions — the debate should be driven by analyst data, not priors.
        """
        logger.info(
            "Injecting %d %s seed lessons into portfolio_manager_memory",
            len(lessons), label,
        )
        self.portfolio_manager_memory.add_situations(lessons)

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

        # Inject seed lessons based on the resolved market (idempotent per market)
        self._inject_seed_lessons(init_agent_state.get("market", "other"))

        args = self.propagator.get_graph_args()

        # 阶段进度映射：state 字段 → 阶段标签
        _PHASE_LABELS = {
            "market_report": "市场分析师",
            "sentiment_report": "舆情分析师",
            "news_report": "新闻分析师",
            "fundamentals_report": "基本面分析师",
            "stock_profile": "股票画像识别官",
            "consensus_snapshot": "共识识别官",
            "investment_plan": "研究主管 (RM)",
            # "trader_investment_plan": "交易员",  # DEPRECATED in optimization 05
            "final_trade_decision": "投资组合经理 (PM)",
        }
        # 阶段排序（用于显示序号）
        _PHASE_ORDER = [
            "market_report", "sentiment_report", "news_report", "fundamentals_report",
            "stock_profile",
            "consensus_snapshot",
            "investment_plan", "final_trade_decision",  # trader_investment_plan removed in 05
        ]

        if self.debug:
            import sys
            # Debug mode with tracing
            trace = []
            _last_printed_msg_id = None
            _completed_phases = set()
            _prev_state = {}

            print(f"[{company_name}] 正在启动分析流程...", flush=True)
            for chunk in self.graph.stream(init_agent_state, **args):
                for field in _PHASE_ORDER:
                    if field in _completed_phases:
                        continue
                    new_val = chunk.get(field, "")
                    old_val = _prev_state.get(field, "")
                    if new_val and not old_val:
                        phase_idx = _PHASE_ORDER.index(field) + 1
                        total = len(_PHASE_ORDER)
                        label = _PHASE_LABELS[field]
                        print(f"\n[{company_name}] ✓ Phase {phase_idx}/{total}: {label} 完成", flush=True)
                        _completed_phases.add(field)
                _prev_state = {k: v for k, v in chunk.items() if k in _PHASE_ORDER}

                if len(chunk["messages"]) == 0:
                    pass
                else:
                    last_msg = chunk["messages"][-1]
                    msg_id = getattr(last_msg, "id", id(last_msg))
                    if msg_id != _last_printed_msg_id:
                        last_msg.pretty_print()
                        sys.stdout.flush()
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
            "stock_profile": final_state.get("stock_profile", ""),
            "consensus_snapshot": final_state.get("consensus_snapshot", ""),
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
            # "trader_investment_decision": final_state["trader_investment_plan"],  # DEPRECATED in optimization 05
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
        """Reflect on decisions and update memory based on returns.

        改造 A 简化：只对 PM 反思（其他 memory 实例已删除）。
        reflect_bull/bear/trader/invest_judge 在 backtest 闭环存在时可重新启用，
        但需先恢复对应 memory 实例（trading_graph.py __init__）。
        """
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
