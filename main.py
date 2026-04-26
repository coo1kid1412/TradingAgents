import os
import re
import sys
import time
import datetime
import multiprocessing
from pathlib import Path
from typing import List, Tuple
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


# ---------------------------------------------------------------------------
#  并发配置
# ---------------------------------------------------------------------------
# 要分析的股票列表（逗号分隔）
_TICKERS = "603516"

# 最多同时分析的股票数（不超过 3）
_MAX_CONCURRENT = 3

# 每支股票启动间隔（秒），默认 5 分钟 = 300 秒
# 用于错开 LLM API 和数据源的并发请求，避免触发限流
_START_INTERVAL_SECONDS = 300

# 分析日期（默认今天）
_ANALYSIS_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# 辩论轮数配置（同时控制研究团队辩论和风控团队讨论）
# 范围：1-3，默认 3
# - max_debate_rounds: 多头 vs 空头的辩论轮数
# - max_risk_discuss_rounds: 激进/保守/中立风控分析师的讨论轮数
_DEBATE_ROUNDS = 3


def _clamp_debate_rounds(value: int, min_val: int = 1, max_val: int = 3) -> int:
    """限制辩论轮数在合理范围内（默认 1-3）"""
    if not isinstance(value, int):
        print(f"警告: 辩论轮数应为整数，使用默认值 3")
        return 3
    if value < min_val:
        print(f"警告: 辩论轮数 {value} 小于最小值 {min_val}，已调整为 {min_val}")
        return min_val
    if value > max_val:
        print(f"警告: 辩论轮数 {value} 大于最大值 {max_val}，已调整为 {max_val}")
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
    
    # 统一设置辩论轮数（带保护机制）
    debate_rounds = _clamp_debate_rounds(_DEBATE_ROUNDS)
    config["max_debate_rounds"] = debate_rounds
    config["max_risk_discuss_rounds"] = debate_rounds
    
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


# ---------------------------------------------------------------------------
#  单支股票分析函数（在独立进程中执行）
# ---------------------------------------------------------------------------
def analyze_single_stock(ticker: str, analysis_date: str, config: dict) -> Tuple[str, bool, str]:
    """
    在独立进程中分析单支股票
    
    Returns:
        (ticker, success, report_path_or_error)
    """
    try:
        # 每个进程独立导入，避免 multiprocessing 序列化问题
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        
        print(f"\n{'='*60}")
        print(f"[{ticker}] 开始分析 (日期: {analysis_date})")
        print(f"{'='*60}\n")
        
        # 创建独立的 TradingAgentsGraph 实例
        ta = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=True,
            config=config,
        )
        
        # 执行分析
        final_state, decision = ta.propagate(ticker, analysis_date)
        
        # 打印决策
        print(f"\n[{ticker}] 分析完成，决策: {decision}\n")
        
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
        
        return (ticker, True, str(report_path))
        
    except Exception as e:
        error_msg = f"[{ticker}] 分析失败: {str(e)}"
        print(f"\n{'!'*60}")
        print(error_msg)
        print(f"{'!'*60}\n")
        import traceback
        traceback.print_exc()
        return (ticker, False, error_msg)


# ---------------------------------------------------------------------------
#  主函数：顺序启动多进程
# ---------------------------------------------------------------------------
def main():
    """主函数：智能选择单进程或多进程模式"""
    
    # 解析股票代码
    tickers = [t.strip() for t in _TICKERS.split(",") if t.strip()]
    
    if not tickers:
        print("错误：未指定要分析的股票代码，请修改 _TICKERS 配置")
        sys.exit(1)
    
    # 限制最多分析数量
    if len(tickers) > _MAX_CONCURRENT:
        print(f"警告：指定了 {len(tickers)} 支股票，但最多只分析 {_MAX_CONCURRENT} 支")
        print(f"将分析前 {_MAX_CONCURRENT} 支: {', '.join(tickers[:_MAX_CONCURRENT])}\n")
        tickers = tickers[:_MAX_CONCURRENT]
    
    # 构建配置
    config = _build_config()
    
    # 智能模式选择：单支股票用单进程，多支股票用多进程
    if len(tickers) == 1:
        # 单支股票：直接在主进程运行（避免多进程开销）
        ticker = tickers[0]
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
    else:
        # 多支股票：使用多进程并发
        print("=" * 60)
        print("多股票并发分析模式")
        print("=" * 60)
        print(f"分析日期: {_ANALYSIS_DATE}")
        print(f"股票列表: {', '.join(tickers)}")
        print(f"启动间隔: {_START_INTERVAL_SECONDS} 秒 ({_START_INTERVAL_SECONDS/60:.1f} 分钟)")
        print(f"预计总启动时间: {(len(tickers)-1) * _START_INTERVAL_SECONDS / 60:.1f} 分钟")
        print("=" * 60)
        
        # 存储进程和结果
        results = []
        
        # 顺序启动进程
        for idx, ticker in enumerate(tickers):
            if idx > 0:
                # 延迟启动，错开 LLM API 和数据源请求
                print(f"\n等待 {_START_INTERVAL_SECONDS} 秒后启动下一支股票...")
                time.sleep(_START_INTERVAL_SECONDS)
            
            print(f"\n>>> 启动 [{ticker}] 的分析进程 (第 {idx+1}/{len(tickers)} 支)")
            
            # 创建独立进程
            process = multiprocessing.Process(
                target=analyze_single_stock,
                args=(ticker, _ANALYSIS_DATE, config),
                name=f"StockAnalysis-{ticker}"
            )
            
            # 启动进程
            process.start()
            results.append((ticker, process))
            
            print(f"[{ticker}] 进程已启动 (PID: {process.pid})")
        
        # 等待所有进程完成
        print("\n" + "=" * 60)
        print("等待所有分析任务完成...")
        print("=" * 60 + "\n")
        
        for ticker, process in results:
            process.join()
            exit_code = process.exitcode
            status = "成功" if exit_code == 0 else f"失败 (退出码: {exit_code})"
            print(f"[{ticker}] 进程结束: {status}")
        
        # 输出总结
        print("\n" + "=" * 60)
        print("分析任务全部完成")
        print("=" * 60)
        for ticker, process in results:
            status = "成功" if process.exitcode == 0 else "失败"
            print(f"  {ticker}: {status}")
        print("=" * 60)


if __name__ == "__main__":
    # Windows/macOS 多进程保护
    multiprocessing.set_start_method("spawn", force=True)
    main()
