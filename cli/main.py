from typing import Optional
import os
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console, Group
from rich.progress_bar import ProgressBar
from dotenv import load_dotenv

# Load environment variables from .env file (use project root, not CWD)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# 确保代理可用（Google Gemini 等国际 API 需要代理）
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")

# 国内数据源绕过代理（AKShare/Tushare/公告 API 等）
os.environ.setdefault(
    "NO_PROXY",
    "*.eastmoney.com,*.sina.com.cn,*.tushare.pro,*.baidu.com,api.tauric.ai,*.akshare.xyz,*.minimaxi.com"
)
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
import threading
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler
from cli.i18n import AGENT_NAMES, TEAM_NAMES, REPORT_TITLES, STATUS_TEXT, STATUS_COLORS, MESSAGE_TYPES, t

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: 多智能体大模型金融交易分析框架",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Social Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Social Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._last_message_id = None
        # Progress tracking
        self._agent_start_times = {}
        self._investment_debate_count = 0
        self._risk_debate_count = 0
        self._max_debate_rounds = 1
        self._max_risk_rounds = 1
        self._streaming_complete = False
        self._streaming_error = None
        self._trace = []

    def init_for_analysis(self, selected_analysts, max_debate_rounds=1, max_risk_rounds=1):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
            max_debate_rounds: Maximum investment debate rounds
            max_risk_rounds: Maximum risk discussion rounds
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._last_message_id = None
        # Reset progress tracking
        self._agent_start_times = {}
        self._investment_debate_count = 0
        self._risk_debate_count = 0
        self._max_debate_rounds = max_debate_rounds
        self._max_risk_rounds = max_risk_rounds
        self._streaming_complete = False
        self._streaming_error = None
        self._trace = []

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed)."""
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            old_status = self.agent_status.get(agent)
            self.agent_status[agent] = status
            self.current_agent = agent
            # Track timing for progress display
            if status == "in_progress" and old_status != "in_progress":
                self._agent_start_times[agent] = time.time()
            elif status == "completed":
                self._agent_start_times.pop(agent, None)

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content
               
        if latest_section and latest_content:
            # Format the current section for display using Chinese titles
            title = t(latest_section, REPORT_TITLES)
            self.current_report = (
                f"### {title}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## 分析师团队报告")
            for section_key in analyst_sections:
                if self.report_sections.get(section_key):
                    title = t(section_key, REPORT_TITLES)
                    report_parts.append(
                        f"### {title}\n{self.report_sections[section_key]}"
                    )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## 研究团队决策")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## 交易团队方案")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## 投资组合管理决策")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None

    def update_debate_count(self, count):
        """Update the investment debate round counter."""
        self._investment_debate_count = count

    def update_risk_count(self, count):
        """Update the risk discussion round counter."""
        self._risk_debate_count = count

    def get_progress_info(self):
        """Return (percentage, phase_text, thinking_text) for display."""
        agents_completed = sum(1 for s in self.agent_status.values() if s == "completed")
        agents_total = len(self.agent_status)
        percentage = int(agents_completed / agents_total * 100) if agents_total > 0 else 0
        phase = self._get_current_phase()
        thinking = self._get_thinking_info()
        return percentage, phase, thinking

    def _get_current_phase(self):
        """Determine the current analysis phase with round info."""
        # Phase 1: Analysts
        analyst_statuses = [
            self.agent_status.get(self.ANALYST_MAPPING[k])
            for k in self.selected_analysts
            if k in self.ANALYST_MAPPING
        ]
        if analyst_statuses and any(s != "completed" for s in analyst_statuses):
            active = [
                k for k in self.selected_analysts
                if self.agent_status.get(self.ANALYST_MAPPING[k]) == "in_progress"
            ]
            if active:
                agent_cn = AGENT_NAMES.get(self.ANALYST_MAPPING[active[0]], active[0])
                return f"\u4e00\u3001\u5206\u6790\u5e08\u56e2\u961f - {agent_cn}"
            return "\u4e00\u3001\u5206\u6790\u5e08\u56e2\u961f"

        # Phase 2: Research debate
        if self.agent_status.get("Research Manager") != "completed":
            cnt = self._investment_debate_count
            max_r = self._max_debate_rounds
            if cnt >= 2 * max_r:
                return "\u4e8c\u3001\u7814\u7a76\u56e2\u961f - \u7814\u7a76\u4e3b\u7ba1\u51b3\u7b56"
            if cnt > 0 or self.agent_status.get("Bull Researcher") == "in_progress":
                current_round = min(cnt // 2 + 1, max_r)
                return f"\u4e8c\u3001\u7814\u7a76\u8fa9\u8bba \u7b2c{current_round}\u8f6e/\u5171{max_r}\u8f6e"
            return "\u4e8c\u3001\u7814\u7a76\u56e2\u961f"

        # Phase 3: Trader
        if self.agent_status.get("Trader") != "completed":
            return "\u4e09\u3001\u4ea4\u6613\u56e2\u961f"

        # Phase 4: Risk discussion
        if self.agent_status.get("Portfolio Manager") != "completed":
            cnt = self._risk_debate_count
            max_r = self._max_risk_rounds
            if cnt >= 3 * max_r:
                return "\u56db\u3001\u98ce\u9669\u7ba1\u7406 - \u6295\u8d44\u7ec4\u5408\u7ecf\u7406\u51b3\u7b56"
            if cnt > 0 or self.agent_status.get("Aggressive Analyst") == "in_progress":
                current_round = min(cnt // 3 + 1, max_r)
                return f"\u56db\u3001\u98ce\u63a7\u8ba8\u8bba \u7b2c{current_round}\u8f6e/\u5171{max_r}\u8f6e"
            return "\u56db\u3001\u98ce\u9669\u7ba1\u7406\u56e2\u961f"

        return "\u2713 \u5206\u6790\u5b8c\u6210"

    def _get_thinking_info(self):
        """Get info about currently thinking agent with elapsed time."""
        for agent, status in self.agent_status.items():
            if status == "in_progress" and agent in self._agent_start_times:
                elapsed = int(time.time() - self._agent_start_times[agent])
                agent_cn = AGENT_NAMES.get(agent, agent)
                return f"{agent_cn} \u601d\u8003\u4e2d ({elapsed}s)"
        return None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=5),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def _render_status_cell(status, agent_name=None):
    """Render a status cell with Chinese text and color."""
    status_cn = t(status, STATUS_TEXT)
    if status == "in_progress":
        elapsed_str = ""
        if agent_name and agent_name in message_buffer._agent_start_times:
            elapsed = int(time.time() - message_buffer._agent_start_times[agent_name])
            elapsed_str = f" ({elapsed}s)"
        return Spinner("dots", text=f"[blue]{status_cn}{elapsed_str}[/blue]", style="bold cyan")
    color = STATUS_COLORS.get(status, "white")
    return f"[{color}]{status_cn}[/{color}]"


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]欢迎使用 TradingAgents 智能交易分析系统[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="TradingAgents 智能交易分析",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,
        title=None,
        padding=(0, 2),
        expand=True,
    )
    progress_table.add_column("团队", style="cyan", justify="center", width=20)
    progress_table.add_column("智能体", style="green", justify="center", width=20)
    progress_table.add_column("状态", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        team_cn = t(team, TEAM_NAMES)
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        status_cell = _render_status_cell(status, first_agent)
        progress_table.add_row(team_cn, t(first_agent, AGENT_NAMES), status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            status_cell = _render_status_cell(status, agent)
            progress_table.add_row("", t(agent, AGENT_NAMES), status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    # Get progress info for panel title and progress bar
    percentage, phase_text, thinking_text = message_buffer.get_progress_info()
    progress_title = f"分析进度 -- {phase_text}" if phase_text else "分析进度"

    # Build progress panel with bar + table
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)
    progress_content = Group(
        Text(f"总进度: {percentage}% ({agents_completed}/{agents_total} 步)", justify="center"),
        ProgressBar(completed=agents_completed, total=max(agents_total, 1), width=None),
        Text(""),  # spacer
        progress_table,
    )

    layout["progress"].update(
        Panel(progress_content, title=progress_title, border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,
        box=box.MINIMAL,
        show_lines=True,
        padding=(0, 1),
    )
    messages_table.add_column("时间", style="cyan", width=8, justify="center")
    messages_table.add_column("类型", style="green", width=10, justify="center")
    messages_table.add_column(
        "内容", style="white", no_wrap=False, ratio=1
    )

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "工具", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        msg_type_cn = t(msg_type, MESSAGE_TYPES)
        all_messages.append((timestamp, msg_type_cn, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="消息与工具调用",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="当前报告",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]等待分析报告中...[/italic]",
                title="当前报告",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with progress and statistics
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Progress bar characters
    bar_width = 20
    filled = int(bar_width * percentage / 100)
    progress_bar = "\u2501" * filled + "\u2500" * (bar_width - filled)

    # Row 1: Progress + Phase + Thinking agent
    row1_parts = [f"\u8fdb\u5ea6: {percentage}% {progress_bar}"]
    if phase_text:
        row1_parts.append(f"\u9636\u6bb5: {phase_text}")
    if thinking_text:
        row1_parts.append(thinking_text)

    # Row 2: Detailed stats
    row2_parts = [f"\u667a\u80fd\u4f53: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        row2_parts.append(f"\u5927\u6a21\u578b: {stats['llm_calls']}")
        row2_parts.append(f"\u5de5\u5177: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"\u4ee4\u724c: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "\u4ee4\u724c: --"
        row2_parts.append(tokens_str)

    row2_parts.append(f"\u62a5\u544a: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        row2_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(row1_parts))
    stats_table.add_row(" | ".join(row2_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: 多智能体大模型金融交易分析框架 - CLI[/bold green]\n\n"
    welcome_content += "[bold]分析流程:[/bold]\n"
    welcome_content += "一、分析师团队 → 二、研究团队 → 三、交易员 → 四、风险管理 → 五、投资组合管理\n\n"
    welcome_content += (
        "[dim]由 [Tauric Research](https://github.com/TauricResearch) 构建[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="欢迎使用 TradingAgents",
        subtitle="多智能体大模型金融交易分析框架",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure)
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]默认值: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "第一步：股票代码",
            "请输入要分析的股票代码，需要时包含交易所后缀（例: SPY, CNC.TO, 7203.T, 0700.HK）",
            "SPY",
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "第二步：分析日期",
            "输入分析日期（YYYY-MM-DD 格式）",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Select analysts
    console.print(
        create_question_box(
            "第三步：分析师团队", "选择参与分析的 LLM 分析师智能体"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]已选分析师:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth
    console.print(
        create_question_box(
            "第四步：研究深度", "选择研究深度级别"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 5: LLM provider
    console.print(
        create_question_box(
            "第五步：大模型供应商", "选择 LLM 服务提供商"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 6: Thinking agents
    console.print(
        create_question_box(
            "第六步：思考模型", "选择快速/深度思考模型"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 7: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box(
                "第七步：思考配置",
                "配置 Gemini 思考模式"
            )
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "第七步：推理配置",
                "配置 OpenAI 推理强度"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()
    elif provider_lower == "anthropic":
        console.print(
            create_question_box(
                "第七步：推理配置",
                "配置 Claude 推理等级"
            )
        )
        anthropic_effort = ask_anthropic_effort()

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]错误：分析日期不能是未来日期[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]错误：日期格式无效，请使用 YYYY-MM-DD 格式[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"])
        analyst_parts.append((t("Market Analyst", AGENT_NAMES), final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"])
        analyst_parts.append((t("Social Analyst", AGENT_NAMES), final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"])
        analyst_parts.append((t("News Analyst", AGENT_NAMES), final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"])
        analyst_parts.append((t("Fundamentals Analyst", AGENT_NAMES), final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## 一、分析师团队报告\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"])
            research_parts.append((t("Bull Researcher", AGENT_NAMES), debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"])
            research_parts.append((t("Bear Researcher", AGENT_NAMES), debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"])
            research_parts.append((t("Research Manager", AGENT_NAMES), debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## 二、研究团队决策\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"])
        sections.append(f"## 三、交易团队方案\n\n### {t('Trader', AGENT_NAMES)}\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"])
            risk_parts.append((t("Aggressive Analyst", AGENT_NAMES), risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"])
            risk_parts.append((t("Conservative Analyst", AGENT_NAMES), risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"])
            risk_parts.append((t("Neutral Analyst", AGENT_NAMES), risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## 四、风险管理团队决策\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"])
            sections.append(f"## 五、投资组合管理决策\n\n### {t('Portfolio Manager', AGENT_NAMES)}\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# 交易分析报告：{ticker}\n\n生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections))
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("完整分析报告", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append((t("Market Analyst", AGENT_NAMES), final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append((t("Social Analyst", AGENT_NAMES), final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append((t("News Analyst", AGENT_NAMES), final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append((t("Fundamentals Analyst", AGENT_NAMES), final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]一、分析师团队报告[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append((t("Bull Researcher", AGENT_NAMES), debate["bull_history"]))
        if debate.get("bear_history"):
            research.append((t("Bear Researcher", AGENT_NAMES), debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append((t("Research Manager", AGENT_NAMES), debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]二、研究团队决策[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]三、交易团队方案[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title=t("Trader", AGENT_NAMES), border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append((t("Aggressive Analyst", AGENT_NAMES), risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append((t("Conservative Analyst", AGENT_NAMES), risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append((t("Neutral Analyst", AGENT_NAMES), risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]四、风险管理团队决策[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]五、投资组合管理决策[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title=t("Portfolio Manager", AGENT_NAMES), border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk):
    """Update analyst statuses based on accumulated report state."""
    selected = message_buffer.selected_analysts
    found_active = False

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if not found_active and selected:
        if message_buffer.agent_status.get("Bull Researcher") == "pending":
            message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(tp for tp in text_parts if tp and not is_empty(tp))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result


def _streaming_worker(graph, init_agent_state, args, message_buffer, selections):
    """Background thread: process LangGraph streaming chunks.

    Updates message_buffer state but never touches Rich layout objects.
    """
    try:
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process messages if present (skip duplicates via message ID)
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]
                msg_id = getattr(last_message, "id", None)

                if msg_id != message_buffer._last_message_id:
                    message_buffer._last_message_id = msg_id

                    # Add message to buffer
                    msg_type, content = classify_message_type(last_message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    # Handle tool calls
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(
                                    tool_call["name"], tool_call["args"]
                                )
                            else:
                                message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(message_buffer, chunk)

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                message_buffer.update_debate_count(debate_state.get("count", 0))
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### {t('Bull Researcher', AGENT_NAMES)}\u5206\u6790\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### {t('Bear Researcher', AGENT_NAMES)}\u5206\u6790\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### {t('Research Manager', AGENT_NAMES)}\u51b3\u7b56\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                message_buffer.update_risk_count(risk_state.get("count", 0))
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### {t('Aggressive Analyst', AGENT_NAMES)}\u5206\u6790\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### {t('Conservative Analyst', AGENT_NAMES)}\u5206\u6790\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### {t('Neutral Analyst', AGENT_NAMES)}\u5206\u6790\n{neu_hist}"
                    )
                if judge:
                    if message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### {t('Portfolio Manager', AGENT_NAMES)}\u51b3\u7b56\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

            message_buffer._trace.append(chunk)

    except Exception as e:
        message_buffer._streaming_error = e
    finally:
        message_buffer._streaming_complete = True


def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts and debate config
    message_buffer.init_for_analysis(
        selected_analyst_keys,
        max_debate_rounds=config["max_debate_rounds"],
        max_risk_rounds=config["max_risk_discuss_rounds"],
    )

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"已选股票: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"分析日期: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"已选分析师: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"正在分析 {selections['ticker']}（{selections['analysis_date']}）..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis in a background thread so the main thread
        # can keep refreshing the display (elapsed time, progress, etc.)
        worker = threading.Thread(
            target=_streaming_worker,
            args=(graph, init_agent_state, args, message_buffer, selections),
            daemon=True,
        )
        worker.start()

        # Main thread: refresh display until streaming completes
        while not message_buffer._streaming_complete:
            update_display(layout, stats_handler=stats_handler, start_time=start_time)
            time.sleep(0.25)

        # Final display update after stream completes
        update_display(layout, stats_handler=stats_handler, start_time=start_time)
        worker.join(timeout=5.0)

        # Propagate any streaming errors
        if message_buffer._streaming_error:
            raise message_buffer._streaming_error

        trace = message_buffer._trace
        if not trace:
            console.print("[red]\u5206\u6790\u672a\u4ea7\u751f\u4efb\u4f55\u7ed3\u679c[/red]")
            return

        # Get final state and decision
        final_state = trace[-1]
        decision = graph.process_signal(final_state["final_trade_decision"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"分析已完成：{selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]分析完成！[/bold cyan]\n")

    # Prompt to save report
    save_choice = typer.prompt("是否保存报告？", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "保存路径（回车使用默认路径）",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]报告已保存至:[/green] {save_path.resolve()}")
            console.print(f"  [dim]完整报告:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]保存报告出错: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\n是否在屏幕上显示完整报告？", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


@app.command()
def analyze():
    run_analysis()


if __name__ == "__main__":
    app()
