import logging
import uuid
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    RISK_DEBATE_PHRASING_RULES,
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

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

    logger.info("预取注入: 正在为 %s 强制获取行情数据...", ticker)
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
        logger.info(
            "预取注入: %s 行情数据已注入 (%d chars)",
            ticker, len(str(stock_data_result)),
        )
        return [ai_msg, tool_msg], True
    except Exception as e:
        logger.warning("预取注入失败: %s，LLM 将自行调用工具", e)
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
                "行情数据（OHLCV）已在上方对话中获取。你无需再次调用 "
                "get_stock_data。请直接选择最相关的技术指标（通过 get_indicators），"
                "然后撰写分析报告。"
            )
        else:
            data_instruction = (
                "请先调用 get_stock_data 获取生成指标所需的 CSV 数据。"
                "**重要**：调用 get_stock_data 时，start_date 必须至少比 "
                "end_date 早 365 天（即回看至少一整年的行情数据），"
                "以充分捕捉中长期趋势、季节性规律和关键支撑/阻力位。"
            )

        system_message = (
            "你是一名专业技术分析师，负责分析金融市场技术面并撰写技术分析报告。\n\n"
            "## 分析流程\n"
            "1. 从下方指标目录中选择最多 **8 个互补性指标**（避免冗余，如不要同时选 rsi 和 stochrsi）\n"
            "2. 调用 get_indicators 获取数据\n"
            "3. 撰写结构化分析报告\n\n"
            "## 指标目录\n"
            + _INDICATOR_CATALOG
            + "\n"
            "选择指标时注意多样性和互补性，避免冗余。调用工具时请使用指标目录中的**精确名称**作为参数，否则调用会失败。"
            + data_instruction
            + "\n\n"
            "## 输出结构（必须按以下章节撰写）\n\n"
            "### 一、多周期趋势分析（核心，决定方向）\n"
            "**经典原则：大周期定方向，小周期找入场**。强制双周期分析，避免日线看似超卖但周线趋势恶化的'接飞刀'陷阱：\n\n"
            "**1.1 周线趋势（中长期方向，把日线数据按周聚合分析）**：\n"
            "- 周线 K 线形态（多头排列/空头排列/盘整）\n"
            "- 周线均线系统（如 20 周线、52 周线的位置关系）\n"
            "- 周线趋势阶段（启动/加速/顶部/回落/底部）\n"
            "- **周线方向结论**：上行 / 下行 / 震荡\n\n"
            "**1.2 日线择时（短期入场/退出信号，已有数据直接用）**：\n"
            "- 日线趋势是否与周线一致（一致则信号强，背离则警惕）\n"
            "- 日线 K 线近期形态（如反包、十字星、大阴线）\n"
            "- 日线在周线趋势中的相对位置（强趋势中的回调位 vs 弱趋势中的反弹位）\n\n"
            "**1.3 多周期一致性判定**（必输出）：\n"
            "- ✅ 周线上行 + 日线回调 = 强势回调，逢低入场机会\n"
            "- ⚠️ 周线下行 + 日线超卖反弹 = 弱反弹，**典型接飞刀陷阱**\n"
            "- ⚠️ 周线上行 + 日线突破乏力 = 趋势衰竭警告\n"
            "- ✅ 周线下行 + 日线持续走弱 = 趋势性下跌，回避\n\n"
            "### 二、核心技术指标分析\n"
            "逐一分析所选指标（RSI/MACD/均线/布林带/ATR 等），每个指标给出：\n"
            "- 当前数值与状态\n"
            "- 信号解读（看多/看空/中性）\n"
            "- 信号强度（★评级）\n\n"
            "**每个核心指标必须输出\\\"历史分位\\\"**：\n"
            "- 基于已获取的至少 365 天行情数据，估算当前指标值在过去 1 年的百分位\n"
            '- 例如："RSI=72，处于过去 1 年 87 分位（高位区间）"\n'
            '- 若数据不足 1 年（如新股），标注"数据不足，分位仅供参考"\n\n'
            "### 三、量价配合判断（必输出明确分类）\n"
            "不只看成交量绝对值，必须做**量价配合诊断**，从以下 6 种里选一个明确分类：\n\n"
            "| 量价模式 | 判定标准 | 含义 |\n"
            "|---------|---------|------|\n"
            "| **放量上涨** | 近 5 日均量 > 20 日均量 1.5 倍且收阳 | 主力增仓，趋势确认 |\n"
            "| **放量下跌** | 近 5 日均量 > 20 日均量 1.5 倍且收阴 | 主力出货/恐慌抛售 |\n"
            "| **缩量整理** | 近 5 日均量 < 20 日均量 0.7 倍 | 观望期，等方向选择 |\n"
            "| **无量背离** | 价格新高/新低但成交萎缩 | 趋势衰竭警告 |\n"
            "| **量价齐升** | 量价同向放大，温和上涨 | 健康上行 |\n"
            "| **正常** | 不符合上述任一模式 | 中性 |\n\n"
            "输出该标的**当前量价模式**+ 一句话解读。\n\n"
            "### 四、A 股资金面分析（仅 A 股，对资金面驱动的标的极重要）\n"
            "如果当前标的是 A 股（代码以 6/0/3/688/8 等开头），必须分析以下资金面维度。**这些数据可能散落在 news_report 或市场报告里，请提取整合**：\n\n"
            "- **北向资金（沪深港通）**：近 5/10/20 个交易日净流入/流出，持股比例变化方向\n"
            "- **融资融券**：融资余额绝对值 + 占流通市值比例 + 近期变化方向（增加/减少）\n"
            "- **龙虎榜**：近期是否登榜，机构席位/游资席位的多空方向\n"
            "- **大宗交易**：近 30 日大宗交易笔数、折溢价情况（折价 > 5% 通常是抛压）\n"
            "- **主力资金流向**：近 5 日特大单/大单的净流入/流出累计值\n\n"
            "**资金面综合判断**（必输出）：\n"
            "- ✅ 强势资金面：多维度协同净流入（如北向+融资双增 + 主力净流入）\n"
            "- ⚠️ 资金面分化：部分流入部分流出（如机构减仓+融资增加 = 散户接盘）\n"
            "- ❌ 资金面恶化：多维度协同净流出\n\n"
            "*若是非 A 股（港股/美股/ETF），本节标注'不适用，跳过'。*\n\n"
            "### 五、综合研判与交易建议\n"
            "多空力量对比表、关键价位表（支撑位/阻力位）、操作建议与评级（BUY/HOLD/SELL）\n\n"
            "### 六、技术指标汇总表（必须包含）\n"
            "| 指标名称 | 当前数值 | 参考值/阈值 | 状态 | 信号强度 |\n\n"
            "### 七、风险提示\n\n"
            "## 强制输出：SUMMARY 块（位于报告末尾）\n"
            "在报告所有正文章节和汇总表格之后，**必须**附加一个 YAML 代码块，"
            "格式严格如下（字段名、单位、取值集合不可变）：\n\n"
            "```yaml\n"
            "SUMMARY:\n"
            "  trend_weekly: 上行 / 下行 / 震荡        # 周线趋势\n"
            "  trend_daily: 上行 / 下行 / 震荡         # 日线趋势\n"
            "  multi_timeframe_alignment: 一致看多 / 一致看空 / 周强日弱 / 周弱日强 / 双向震荡   # 双周期一致性\n"
            "  trend: 上行 / 下行 / 震荡               # 综合方向（保留兼容）\n"
            "  momentum: 强 / 中 / 弱\n"
            "  rsi_value: <数值>\n"
            "  rsi_pct_1y: <0-100>\n"
            "  macd_signal: bullish / bearish / neutral\n"
            "  key_support: <数值>\n"
            "  key_resistance: <数值>\n"
            "  atr_pct: <0-100>\n"
            "  volume_state: 放量 / 缩量 / 正常\n"
            "  volume_price_pattern: 放量上涨 / 放量下跌 / 缩量整理 / 无量背离 / 量价齐升 / 正常   # 量价配合诊断\n"
            "  capital_flow_state: 强势 / 分化 / 恶化 / 不适用    # A 股资金面综合（非 A 股填'不适用'）\n"
            "  northbound_net_flow: 净流入 / 净流出 / 平衡 / 不适用\n"
            "  margin_change: 增加 / 减少 / 平稳 / 不适用\n"
            "  rating: BUY / HOLD / SELL                # 措辞评级（保守表达）\n"
            "  data_implied_direction: 偏多 / 偏空 / 中性  # 数据真实隐含方向（穿透措辞）\n"
            "  data_implied_reasoning: <≤30 字说明数据为何隐含此方向>\n"
            "  confidence: <1-5>\n"
            "```\n\n"
            "## SUMMARY 规则\n"
            '- 字段缺失时填 null 或 "不适用"，不允许省略字段名\n'
            "- 取值必须落在 schema 允许的集合内（如 trend ∈ {上行, 下行, 震荡}）\n"
            "- 数值字段保留 2 位小数；百分比字段直接填数字（不带 % 符号）\n"
            "- 该 SUMMARY 块是下游 RM / 风控团队的核心信息源，宁缺勿错\n\n"
            "**重要**：股票代码（如 AAPL）、技术指标名称（如 RSI、MACD、SMA、EMA、ATR、VWMA 等）、"
            "以及评级关键词（BUY/SELL/HOLD）请保留英文原文。Markdown 表格的表头请使用中文。\n\n"
            + RISK_DEBATE_PHRASING_RULES
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "【语言要求】你必须使用中文撰写所有分析报告和回复内容。"
                    "股票代码、技术指标名称和评级关键词可保留英文。\n\n"
                    "你是一个协作式 AI 助手。使用提供的工具推进分析。"
                    "如果你无法完全回答，其他助手会协助。"
                    "如果你或任何助手有最终交易建议 **BUY/HOLD/SELL**，"
                    "请在回复前加上 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**。"
                    "你可使用以下工具：{tool_names}。\n"
                    "{system_message}"
                    "当前日期：{current_date}。{instrument_context}",
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

        result_dict = {
            "messages": prefetch_msgs + [result],
            "market_report": report,
        }

        return result_dict

    return market_analyst_node
