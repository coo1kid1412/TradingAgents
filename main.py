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
_TICKERS = "300857"

# 最多同时分析的股票数（不超过 3）
_MAX_CONCURRENT = 3

# 每支股票启动间隔（秒），默认 5 分钟 = 300 秒
# 用于错开 LLM API 和数据源的并发请求，避免触发限流
_START_INTERVAL_SECONDS = 300

# 分析日期（默认今天）
_ANALYSIS_DATE = datetime.datetime.now().strftime("%Y-%m-%d")


def _build_config() -> dict:
    """构建分析配置"""
    from tradingagents.default_config import DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "minimax"
    config["backend_url"] = "https://api.minimaxi.com/v1"
    config["deep_think_llm"] = "MiniMax-M2.7"
    config["quick_think_llm"] = "MiniMax-M2.7"
    config["use_deep_think_for_analysts"] = True
    config["max_debate_rounds"] = 3
    config["max_risk_discuss_rounds"] = 3
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
    import signal
    import traceback
    
    # 内存监控函数
    def log_memory_usage(stage: str):
        """记录当前内存使用"""
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            print(f"[{ticker}] 内存使用 [{stage}]: {mem_mb:.1f} MB")
        except ImportError:
            pass  # psutil 未安装则跳过
    
    # 保存中间状态（防止崩溃后丢失）
    def save_intermediate_report(final_state, stage_name: str):
        """保存中间状态报告"""
        try:
            temp_path = Path(_PROJECT_ROOT) / "reports" / f"{ticker}_temp_{stage_name}"
            temp_path.mkdir(parents=True, exist_ok=True)
            
            # 保存完整 state 到 JSON
            import json
            state_file = temp_path / "state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                # 简化 state 只保存关键字段
                simple_state = {
                    'ticker': ticker,
                    'date': analysis_date,
                    'stage': stage_name,
                    'decision': final_state.get('final_trade_decision', 'N/A'),
                    'company_name': final_state.get('company_name', ''),
                }
                json.dump(simple_state, f, indent=2, ensure_ascii=False)
            
            # 如果有 trader 决策，单独保存
            if final_state.get('trader_investment_plan'):
                trader_file = temp_path / "trader_decision.md"
                trader_file.write_text(final_state['trader_investment_plan'], encoding='utf-8')
            
            print(f"[{ticker}] 中间状态已保存: {temp_path}")
        except Exception as e:
            print(f"[{ticker}] 保存中间状态失败: {e}")
    
    try:
        # 每个进程独立导入，避免 multiprocessing 序列化问题
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        
        print(f"\n{'='*60}")
        print(f"[{ticker}] 开始分析 (日期: {analysis_date})")
        print(f"{'='*60}\n")
        
        log_memory_usage("启动时")
        
        # 创建独立的 TradingAgentsGraph 实例
        ta = TradingAgentsGraph(
            selected_analysts=["market", "social", "news", "fundamentals"],
            debug=True,
            config=config,
        )
        
        log_memory_usage("Graph 初始化后")
        
        # 执行分析
        print(f"[{ticker}] 开始执行分析流程...")
        final_state, decision = ta.propagate(ticker, analysis_date)
        
        log_memory_usage("分析完成后")
        
        # 打印决策
        print(f"\n[{ticker}] 分析完成，决策: {decision}\n")
        
        # 保存中间状态（在保存正式报告前）
        save_intermediate_report(final_state, "before_save")
        
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
        
    except MemoryError as e:
        error_msg = f"[{ticker}] 内存不足: {str(e)}"
        print(f"\n{'!'*60}")
        print(error_msg)
        print(f"{'!'*60}\n")
        print("建议：")
        print("  1. 关闭其他占用内存的程序")
        print("  2. 减少并发分析的股票数量")
        print("  3. 增加系统内存")
        return (ticker, False, error_msg)
        
    except Exception as e:
        error_msg = f"[{ticker}] 分析失败: {str(e)}"
        print(f"\n{'!'*60}")
        print(error_msg)
        print(f"{'!'*60}\n")
        traceback.print_exc()
        return (ticker, False, error_msg)


# ---------------------------------------------------------------------------
#  主函数：顺序启动多进程
# ---------------------------------------------------------------------------
def main():
    """主函数：顺序启动多个进程分析股票"""
    
    # 解析股票代码
    tickers = [t.strip() for t in _TICKERS.split(",") if t.strip()]
    
    if not tickers:
        print("错误：未指定要分析的股票代码，请修改 _TICKERS 配置")
        sys.exit(1)
    
    # 系统资源检查
    print("=" * 60)
    print("系统资源检查")
    print("=" * 60)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_available_gb = mem.available / 1024 / 1024 / 1024
        mem_total_gb = mem.total / 1024 / 1024 / 1024
        mem_percent = mem.percent
        
        print(f"内存: {mem_available_gb:.1f} GB 可用 / {mem_total_gb:.1f} GB 总计 ({mem_percent}% 已用)")
        
        if mem_available_gb < 4:
            print("⚠️  警告：可用内存不足 4GB，可能导致进程被 OOM Killer 终止")
            print("建议：")
            print("  1. 关闭浏览器等占用内存的程序")
            print("  2. 减少同时分析的股票数量")
            print("  3. 增加系统内存或使用 swap")
            response = input("\n是否继续？(y/N): ").strip().lower()
            if response != 'y':
                print("已取消")
                sys.exit(0)
    except ImportError:
        print("psutil 未安装，跳过内存检查 (pip install psutil)")
    except Exception as e:
        print(f"内存检查失败: {e}")
    
    # 限制最多分析数量
    if len(tickers) > _MAX_CONCURRENT:
        print(f"警告：指定了 {len(tickers)} 支股票，但最多只分析 {_MAX_CONCURRENT} 支")
        print(f"将分析前 {_MAX_CONCURRENT} 支: {', '.join(tickers[:_MAX_CONCURRENT])}\n")
        tickers = tickers[:_MAX_CONCURRENT]
    
    print("\n" + "=" * 60)
    print("多股票并发分析启动")
    print("=" * 60)
    print(f"分析日期: {_ANALYSIS_DATE}")
    print(f"股票列表: {', '.join(tickers)}")
    print(f"启动间隔: {_START_INTERVAL_SECONDS} 秒 ({_START_INTERVAL_SECONDS/60:.1f} 分钟)")
    print(f"预计总启动时间: {(len(tickers)-1) * _START_INTERVAL_SECONDS / 60:.1f} 分钟")
    print("=" * 60)
    
    # 构建配置
    config = _build_config()
    
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
