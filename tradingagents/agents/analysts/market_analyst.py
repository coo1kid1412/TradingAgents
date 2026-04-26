import uuid
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config

# ── 技术指标目录（供 system prompt 使用） ──────────────────────────────
_INDICATOR_CATALOG = """
Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.
"""


def _prefetch_stock_data(ticker: str, current_date: str, messages: list):
    """Pre-fetch stock data and return (prefetch_msgs, success).

    Checks whether get_stock_data result already exists in messages.
    If not, programmatically calls the tool and constructs a proper
    AIMessage + ToolMessage pair that can be injected into the conversation.

    Returns:
        tuple: (list of messages to inject, bool indicating data available)
    """
    # 检查是否已有 get_stock_data 结果（ToolMessage.name == 'get_stock_data'）
    for m in messages:
        if getattr(m, "name", None) == "get_stock_data":
            return [], True

    print(f"[market_analyst] 预取注入: 正在为 {ticker} 强制获取行情数据...")
    try:
        end_dt = datetime.strptime(current_date, "%Y-%m-%d")
    except ValueError:
        end_dt = datetime.now()
    start_date = (end_dt - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        stock_data_result = get_stock_data.invoke({
            "symbol": ticker,
            "start_date": start_date,
            "end_date": current_date,
        })
        tool_call_id = f"prefetch_{uuid.uuid4().hex[:8]}"
        ai_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "get_stock_data",
                "args": {
                    "symbol": ticker,
                    "start_date": start_date,
                    "end_date": current_date,
                },
                "id": tool_call_id,
            }],
        )
        tool_msg = ToolMessage(
            content=str(stock_data_result),
            tool_call_id=tool_call_id,
            name="get_stock_data",
        )
        print(
            f"[market_analyst] 预取注入: {ticker} 行情数据已注入 "
            f"({len(str(stock_data_result))} chars)"
        )
        return [ai_msg, tool_msg], True
    except Exception as e:
        print(f"[market_analyst] 预取注入失败: {e}，LLM 将自行调用工具")
        return [], False


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(
            ticker, state.get("company_name", "")
        )

        tools = [
            get_stock_data,
            get_indicators,
        ]

        # ── 预取注入：强制获取行情数据 ────────────────────────────
        messages = list(state["messages"])
        prefetch_msgs, stock_data_available = _prefetch_stock_data(
            ticker, current_date, messages
        )
        if prefetch_msgs:
            messages = messages + prefetch_msgs

        # ── 根据数据是否已就绪，切换 prompt 指令 ─────────────────
        if stock_data_available:
            data_instruction = (
                "Stock price data (OHLCV) has already been retrieved and is "
                "available in the conversation above. You do NOT need to call "
                "get_stock_data again. Proceed directly to select the most "
                "relevant technical indicators using get_indicators, then write "
                "your analysis report."
            )
        else:
            data_instruction = (
                "Please call get_stock_data first to retrieve the CSV that is "
                "needed to generate indicators. "
                "**重要**：调用 get_stock_data 时，start_date 必须至少比 "
                "end_date 早 365 天（即回看至少一整年的行情数据）。"
                "系统会自动确保最低 365 天的数据覆盖，以充分捕捉中长期趋势、"
                "季节性规律和关键支撑/阻力位。"
            )

        system_message = (
            "You are a trading assistant tasked with analyzing financial "
            "markets. Your role is to select the **most relevant indicators** "
            "for a given market condition or trading strategy from the "
            "following list. The goal is to choose up to **8 indicators** that "
            "provide complementary insights without redundancy. Categories and "
            "each category's indicators are:"
            + _INDICATOR_CATALOG
            + "Select indicators that provide diverse and complementary "
            "information. Avoid redundancy (e.g., do not select both rsi and "
            "stochrsi). Also briefly explain why they are suitable for the "
            "given market context. When you tool call, please use the exact "
            "name of the indicators provided above as they are defined "
            "parameters, otherwise your call will fail. "
            + data_instruction
            + " Then use get_indicators with the specific indicator names. "
            "Write a very detailed and nuanced report of the trends you "
            "observe. Provide specific, actionable insights with supporting "
            "evidence to help traders make informed decisions."
            " Make sure to append a Markdown table at the end of the report "
            "to organize key points in the report, organized and easy to read."
            "\n\n**重要：请用中文撰写你的分析报告。** 股票代码（如 AAPL）、"
            "技术指标名称（如 RSI、MACD、SMA、EMA、ATR、VWMA 等）、"
            "以及评级关键词（BUY/SELL/HOLD）请保留英文原文。"
            "Markdown 表格的表头也请使用中文。"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "【语言要求】你必须使用中文撰写所有分析报告和回复内容。"
                    "股票代码、技术指标名称和评级关键词可保留英文。\n\n"
                    "You are a helpful AI assistant, collaborating with other "
                    "assistants. Use the provided tools to progress towards "
                    "answering the question. If you are unable to fully answer, "
                    "that's OK; another assistant with different tools will help "
                    "where you left off. Execute what you can to make progress. "
                    "If you or any other assistant has the FINAL TRANSACTION "
                    "PROPOSAL: **BUY/HOLD/SELL** or deliverable, prefix your "
                    "response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** "
                    "so the team knows to stop. "
                    "You have access to the following tools: {tool_names}.\n"
                    "{system_message}"
                    "For your reference, the current date is {current_date}. "
                    "{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(
            tool_names=", ".join([tool.name for tool in tools])
        )
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(messages)

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": prefetch_msgs + [result],
            "market_report": report,
        }

    return market_analyst_node
