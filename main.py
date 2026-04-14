import os
import re
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (use project root, not CWD)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# 确保代理可用（Google Gemini 等国际 API 需要代理）
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")

# 国内数据源绕过代理（AKShare/Tushare/交易所/雪球 等）
# 注意：Python urllib 不支持 *.domain 格式，必须用 .domain（POSIX 标准）
_DOMESTIC_NO_PROXY = ",".join([
    ".eastmoney.com",       # 东方财富 (push2/quote/emweb 等子域)
    ".sina.com.cn",         # 新浪财经
    ".sse.com.cn",          # 上交所
    ".szse.cn",             # 深交所
    ".bse.cn",              # 北交所
    ".tushare.pro",         # Tushare
    ".xueqiu.com",          # 雪球
    ".baidu.com",           # 百度
    ".akshare.xyz",         # AKShare
    ".minimaxi.com",        # MiniMax (国内 LLM)
    "api.tauric.ai",        # Tauric
    ".pypi.org",            # pypi
])
_existing = os.environ.get("NO_PROXY", "")
os.environ["NO_PROXY"] = f"{_existing},{_DOMESTIC_NO_PROXY}" if _existing else _DOMESTIC_NO_PROXY

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "minimax"
config["backend_url"] = "https://api.minimaxi.com/v1"
config["deep_think_llm"] = "MiniMax-M2.7"
config["quick_think_llm"] = "MiniMax-M2.7"
# 默认使用 deep_think_llm 作为分析师和交易员的模型（可通过此开关切换）
# use_deep_think_for_analysts: True=使用 deep_think_llm（默认）, False=使用 quick_think_llm
config["use_deep_think_for_analysts"] = True
config["max_debate_rounds"] = 3
config["max_risk_discuss_rounds"] = 3

# A股代码会自动路由到 akshare → tushare → yfinance
# 以下配置仅影响非A股（美股等）的供应商选择
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# Initialize with custom config (全部 4 个分析师)
ta = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    debug=True,
    config=config,
)

# A股测试：淳中科技 603516
_ticker = "688008"
final_state, decision = ta.propagate(_ticker, "2026-04-14")
print(decision)


# ---------------------------------------------------------------------------
#  保存报告到 reports/ 目录（与 CLI 一致的路径格式和目录结构）
# ---------------------------------------------------------------------------
_AGENT_CN = {
    "Market Analyst": "市场分析师",
    "Social Analyst": "舆情分析师",
    "News Analyst": "新闻分析师",
    "Fundamentals Analyst": "基本面分析师",
    "Bull Researcher": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Research Manager": "研究主管",
    "Trader": "交易员",
    "Aggressive Analyst": "激进风控分析师",
    "Conservative Analyst": "保守风控分析师",
    "Neutral Analyst": "中立风控分析师",
    "Portfolio Manager": "投资组合经理",
}


def _save_report(state, ticker: str, save_path: Path):
    """Save complete analysis report to disk (mirrors CLI save_report_to_disk)."""
    save_path.mkdir(parents=True, exist_ok=True)
    cn = _AGENT_CN.get
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    for key, fname, label in [
        ("market_report", "market.md", "Market Analyst"),
        ("sentiment_report", "sentiment.md", "Social Analyst"),
        ("news_report", "news.md", "News Analyst"),
        ("fundamentals_report", "fundamentals.md", "Fundamentals Analyst"),
    ]:
        if state.get(key):
            analysts_dir.mkdir(exist_ok=True)
            (analysts_dir / fname).write_text(state[key], encoding="utf-8")
            analyst_parts.append((cn(label, label), state[key]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## 一、分析师团队报告\n\n{content}")

    # 2. Research (investment debate)
    if state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = state["investment_debate_state"]
        research_parts = []
        for dkey, fname, label in [
            ("bull_history", "bull.md", "Bull Researcher"),
            ("bear_history", "bear.md", "Bear Researcher"),
            ("judge_decision", "manager.md", "Research Manager"),
        ]:
            if debate.get(dkey):
                research_dir.mkdir(exist_ok=True)
                (research_dir / fname).write_text(debate[dkey], encoding="utf-8")
                research_parts.append((cn(label, label), debate[dkey]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## 二、研究团队决策\n\n{content}")

    # 3. Trading
    if state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(
            state["trader_investment_plan"], encoding="utf-8"
        )
        sections.append(
            f"## 三、交易团队方案\n\n### {cn('Trader', '交易员')}\n"
            f"{state['trader_investment_plan']}"
        )

    # 4. Risk Management
    if state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = state["risk_debate_state"]
        risk_parts = []
        for rkey, fname, label in [
            ("aggressive_history", "aggressive.md", "Aggressive Analyst"),
            ("conservative_history", "conservative.md", "Conservative Analyst"),
            ("neutral_history", "neutral.md", "Neutral Analyst"),
        ]:
            if risk.get(rkey):
                risk_dir.mkdir(exist_ok=True)
                (risk_dir / fname).write_text(risk[rkey], encoding="utf-8")
                risk_parts.append((cn(label, label), risk[rkey]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## 四、风险管理团队决策\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(
                risk["judge_decision"], encoding="utf-8"
            )
            sections.append(
                f"## 五、投资组合管理决策\n\n### {cn('Portfolio Manager', '投资组合经理')}\n"
                f"{risk['judge_decision']}"
            )

    # Write consolidated report
    header = (
        f"# 交易分析报告：{ticker}\n\n"
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    report_file = save_path / "complete_report.md"
    report_file.write_text(header + "\n\n".join(sections), encoding="utf-8")
    return report_file


# Build report folder path: {ticker}_{name}_{YYYYMMDD}_{HHMMSS}
_company_name_safe = re.sub(
    r'[\\/:*?"<>|]', "", final_state.get("company_name", "")
).strip()
# fallback 时 name == code，不重复拼接
if _company_name_safe and _company_name_safe == _ticker:
    _company_name_safe = ""
_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if _company_name_safe:
    _folder = f"{_ticker}_{_company_name_safe}_{_timestamp}"
else:
    _folder = f"{_ticker}_{_timestamp}"
_report_path = Path(_PROJECT_ROOT) / "reports" / _folder

_report_file = _save_report(final_state, _ticker, _report_path)
print(f"\n报告已保存至: {_report_path.resolve()}")
print(f"  完整报告: {_report_file.name}")
