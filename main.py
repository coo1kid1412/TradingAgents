import logging
import os
import re
import sys
import time
import datetime
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables from .env file (use project root, not CWD)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# Logger setup: 让 dataflows / agents 中的 logger.warning/error 可见到终端
# 用于排查 tushare/akshare fallback 链上的具体失败原因
# 同时写到 logs/run_<timestamp>.log 文件，避免开头日志被滚动出屏幕后丢失
_LOG_DIR = Path(_PROJECT_ROOT) / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.WARNING,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
# 抑制几个第三方库的噪音 WARNING
for noisy in ("urllib3", "httpx", "httpcore", "matplotlib", "PIL"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
# 启动横幅写入日志文件（方便日后辨认是哪一次运行）
logging.getLogger(__name__).warning(
    "=== main.py 启动 ===  日志文件: %s", _LOG_FILE
)

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


# ---------------------------------------------------------------------------
#  分析配置（单股票模式）
# ---------------------------------------------------------------------------
# 要分析的股票代码（单只）
# 多股票并发已彻底移除——LLM API 偶发假死 + multiprocessing.join 会形成死锁链
# 如需分析多只，请顺序多次运行本脚本
_TICKER = "603986"

# 分析日期（默认今天）
_ANALYSIS_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# 辩论轮数配置（研究团队多空辩论 与 风控团队讨论 可独立配置）
# 范围：1-3
# - _BULL_BEAR_ROUNDS（多头 vs 空头）：2 = 立论 + 反驳（最小有效辩论单位）
# - _RISK_ROUNDS（激进/保守/中立风控）：1 = 三个维度并行审查，PM 综合（多轮重复度高）
_BULL_BEAR_ROUNDS = 3
_RISK_ROUNDS = 1


def _clamp_debate_rounds(value: int, name: str, default: int, min_val: int = 1, max_val: int = 3) -> int:
    """限制辩论轮数在合理范围内（默认 1-3）"""
    if not isinstance(value, int):
        print(f"警告: {name} 应为整数，使用默认值 {default}")
        return default
    if value < min_val:
        print(f"警告: {name}={value} 小于最小值 {min_val}，已调整为 {min_val}")
        return min_val
    if value > max_val:
        print(f"警告: {name}={value} 大于最大值 {max_val}，已调整为 {max_val}")
        return max_val
    return value


def _build_config() -> dict:
    """构建分析配置"""
    from tradingagents.default_config import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "minimax"
    config["backend_url"] = "https://api.minimaxi.com/v1"
    config["deep_think_llm"] = "MiniMax-M2.7"
    config["quick_think_llm"] = "MiniMax-M2.7"
    config["use_deep_think_for_analysts"] = True
    # P1 LLM 选择配置（perf_02 保留）
    config["use_deep_for_trader"] = False       # trader 默认 quick_think
    config["use_deep_for_bull_researcher"] = False  # bull 默认 quick_think
    config["use_deep_for_bear_researcher"] = False  # bear 默认 quick_think
    
    # 分别设置多空辩论与风控辩论轮数（带保护机制）
    config["max_debate_rounds"] = _clamp_debate_rounds(_BULL_BEAR_ROUNDS, "_BULL_BEAR_ROUNDS", default=2)
    config["max_risk_discuss_rounds"] = _clamp_debate_rounds(_RISK_ROUNDS, "_RISK_ROUNDS", default=1)
    
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    return config


# ---------------------------------------------------------------------------
#  报告保存工具函数
# ---------------------------------------------------------------------------
_AGENT_CN = {
    "Market Analyst": "市场分析师",
    "Social Analyst": "舆情分析师",
    "News Analyst": "新闻分析师",
    "Fundamentals Analyst": "基本面分析师",
    "Stock Profile Officer": "股票画像识别官",
    "Consensus Officer": "共识识别官",
    "Bull Researcher": "多头研究员",
    "Bear Researcher": "空头研究员",
    "Research Manager": "研究主管",
    "Trader": "交易员",
    "Aggressive Analyst": "流动性风控分析师",
    "Conservative Analyst": "事件风控分析师",
    "Neutral Analyst": "尾部风控分析师",
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
        ("stock_profile", "stock_profile.md", "Stock Profile Officer"),
        ("consensus_snapshot", "consensus.md", "Consensus Officer"),
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

    # 3. Trading (DEPRECATED in optimization 05: Trader node removed)
    # 保留此段以兼容旧报告，新运行不再产生 trader 章节
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


# ---------------------------------------------------------------------------
#  单支股票分析函数（在独立进程中执行）
# ---------------------------------------------------------------------------
def analyze_single_stock(ticker: str, analysis_date: str, config: dict) -> Tuple[str, bool, str]:
    """
    分析单支股票（主进程直接执行）

    Returns:
        (ticker, success, report_path_or_error)
    """
    # MiniMax 529 整点高峰重试配置
    _MAX_529_RETRIES = 2          # 最多额外重试次数（不含首次执行）
    _529_RETRY_WAIT_BASE = 120    # 基础等待秒数（指数递增：120→240）

    for attempt in range(1 + _MAX_529_RETRIES):
        try:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents import profiling

            if attempt == 0:
                print(f"\n{'='*60}", flush=True)
                print(f"[{ticker}] 开始分析 (日期: {analysis_date})", flush=True)
                print(f"{'='*60}\n", flush=True)
                profiling.reset()  # 开始新一支股票分析，重置计时器
            else:
                wait_sec = _529_RETRY_WAIT_BASE * attempt
                print(f"\n[{ticker}] MiniMax 529 重试 (第 {attempt} 次)，等待 {wait_sec}s...", flush=True)
                time.sleep(wait_sec)

            # 创建独立的 TradingAgentsGraph 实例
            ta = TradingAgentsGraph(
                selected_analysts=["market", "social", "news", "fundamentals"],
                debug=True,
                config=config,
            )

            # 执行分析
            final_state, decision = ta.propagate(ticker, analysis_date)

            # 打印决策
            print(f"\n[{ticker}] 分析完成，决策: {decision}\n", flush=True)

            # 保存报告
            company_name_safe = re.sub(
                r'[\\/:*?"<>|]', "", final_state.get("company_name", "")
            ).strip()

            if company_name_safe and company_name_safe == ticker:
                company_name_safe = ""

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if company_name_safe:
                folder = f"{ticker}_{company_name_safe}_{timestamp}"
            else:
                folder = f"{ticker}_{timestamp}"

            report_path = Path(_PROJECT_ROOT) / "reports" / folder
            report_file = _save_report(final_state, ticker, report_path)

            print(f"[{ticker}] 报告已保存至: {report_path.resolve()}")
            print(f"[{ticker}] 完整报告: {report_file.name}\n")

            # 打印性能分析摘要
            try:
                profiling.print_summary(label=ticker)
            except Exception as _e:
                print(f"[{ticker}] 性能摘要生成失败: {_e}", flush=True)

            return (ticker, True, str(report_path))

        except Exception as e:
            error_str = str(e)
            is_529 = "529" in error_str and ("overloaded" in error_str.lower() or "繁忙" in error_str)

            if is_529 and attempt < _MAX_529_RETRIES:
                print(f"\n[{ticker}] MiniMax 529 整点高峰错误，将自动重试...", flush=True)
                continue

            # 非重试性错误或重试耗尽
            error_msg = f"[{ticker}] 分析失败: {error_str}"
            print(f"\n{'!'*60}", flush=True)
            print(error_msg, flush=True)
            print(f"{'!'*60}\n", flush=True)
            import traceback
            traceback.print_exc()
            return (ticker, False, error_msg)


# ---------------------------------------------------------------------------
#  主函数：单股票顺序执行
# ---------------------------------------------------------------------------
def main():
    """主函数：单股票主进程直接执行（不使用 multiprocessing）"""

    ticker = _TICKER.strip()
    if not ticker:
        print("错误：未指定要分析的股票代码，请修改 _TICKER 配置")
        sys.exit(1)

    config = _build_config()

    print("=" * 60)
    print("单股票分析模式（主进程直接执行）")
    print("=" * 60)
    print(f"分析日期: {_ANALYSIS_DATE}")
    print(f"股票: {ticker}")
    print("=" * 60)

    try:
        ticker_result, success, result_path = analyze_single_stock(ticker, _ANALYSIS_DATE, config)

        print("\n" + "=" * 60)
        if success:
            print(f"分析完成: {ticker}")
            print(f"报告路径: {result_path}")
        else:
            print(f"分析失败: {ticker}")
            print(f"错误信息: {result_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n用户中断分析")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n分析异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
