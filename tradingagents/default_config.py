import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv(
        "TRADINGAGENTS_RESULTS_DIR",
        os.path.join(_PROJECT_ROOT, "results"),
    ),
    "eval_results_dir": os.getenv(
        "TRADINGAGENTS_EVAL_DIR",
        os.path.join(_PROJECT_ROOT, "eval_results"),
    ),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": "https://api.openai.com/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    "minimax_max_tokens": 8192,         # MiniMax default max_tokens (upstream default 256 too small)
    # Temperature settings for different agent roles
    "temperature_market": 0.5,          # 市场分析师：适度创意发现不同技术视角
    "temperature_sentiment": 0.5,       # 舆情分析师：适度创意理解复杂情绪
    "temperature_news": 0.5,            # 新闻分析师：适度创意发现隐藏关联
    "temperature_fundamentals": 0.2,    # 基本面分析师：低随机性确保客观分析
    "temperature_trader": 0.3,          # 交易员：平衡稳定性和灵活性
    "temperature_research_manager": 0.4,    # 研究主管：偏向稳定综合判断
    "temperature_portfolio_manager": 0.3,   # 投资组合经理：最终决策需要高稳定性
    # 多头/空头研究员使用默认 temperature (0.7)，保持辩论的多样性
    "temperature_aggressive_risk": 0.6,     # 激进风控分析师：适度多样性
    "temperature_conservative_risk": 0.6,   # 保守风控分析师：适度多样性
    "temperature_neutral_risk": 0.6,        # 中立风控分析师：适度多样性
    # 使用 deep think 模型作为分析师和交易员（True=deep_think_llm, False=quick_think_llm）
    "use_deep_think_for_analysts": True,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "Chinese",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    # A股代码（如 600519）会自动路由到 akshare → tushare → yfinance
    # 以下配置仅影响非A股代码的供应商选择
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: akshare, tushare, alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: akshare, tushare, alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: akshare, tushare, alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: akshare, tushare, alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
