"""股票画像识别节点（Stock Profile Node）

在 4 个 analyst 完成后、Consensus Officer 之前运行。
综合各分析师报告，提炼标的的"股性 + 推荐报告权重 + 决策风格 + 时间窗口"，
写入 state["stock_profile"]。

下游 consensus/bull/bear/RM/PM 据此知道：
- 该标的应该重看哪份报告（基本面 vs 技术 vs 新闻 vs 舆情）
- 该采用哪种决策风格（价值锚定 vs 催化驱动 vs 动量 vs 事件驱动）
- 当前是否处于关键时间窗口（财报/解禁/政策）→ 临时调整权重
"""

from tradingagents.agents.utils.agent_utils import build_instrument_context


def create_stock_profile_node(llm):
    def stock_profile_node(state) -> dict:
        instrument_context = build_instrument_context(
            state["company_of_interest"], state.get("company_name", "")
        )

        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        macro_context = state.get("macro_context", "")
        trade_date = state.get("trade_date", "")

        prompt = f"""【语言要求】你必须使用中文撰写以下分析。股票代码和技术指标名称可保留英文。

你是投研团队的**股票画像识别官（Stock Profile Officer）**。你的任务：在 Bull/Bear 辩论之前，识别这只标的的"股性"，并推荐 4 份分析师报告的使用权重和决策风格。

{instrument_context}
交易日期：{trade_date}

---

## 你的核心职责

不同股性的标的，4 份报告的价值差异巨大。你要做的：
1. **画像**：识别标的的市值层级、行业、风格、流动性、品种类型
2. **推荐权重**：给 4 份报告打分（权重加总 = 100%），权重越高的报告下游越要重点引用
3. **决策风格**：标的适合哪种决策框架（价值锚定 / 催化驱动 / 动量 / 事件驱动）
4. **时间窗口**：识别近 30 天内是否有关键事件窗口（财报/解禁/政策细则/重大公告），临时调权

**关键原则**：
- 不做方向判断，不给评级
- 推荐权重必须有依据（基于股性，不是凭感觉）
- 时间窗口调权要可追溯到具体事件

---

## 输出结构（必须严格按以下格式）

### 一、股票画像

| 维度 | 取值 | 判断依据 |
|------|------|---------|
| **市值层级** | 大盘(>1000亿) / 中盘(100-1000亿) / 小盘(<100亿) / 微盘(<30亿) | 引用 fundamentals SUMMARY 中的总市值 |
| **行业** | 成长(科技/AI/医药/新能源) / 价值(金融/能源/消费) / 周期(材料/地产/化工) / 题材(概念/热点) | 引用 fundamentals 行业归类 |
| **股性风格** | 稳健蓝筹 / 高弹性成长 / 题材炒作 / 周期波动 / 流动性差 / ETF/指数 | 综合判断 |
| **流动性档** | 深(日均>5亿) / 中(5000万-5亿) / 浅(<5000万) | 引用 market 报告日均成交额 |
| **品种类型** | A股个股 / 港股 / 美股 / ETF / LOF / 可转债 | 根据股票代码与报告内容判断 |

### 二、4 份报告推荐权重

| 报告 | 权重 | 推荐理由 |
|------|------|---------|
| 📊 Fundamentals 基本面 | __% | <为什么对该标的重要/不重要> |
| 📈 Market 技术面 | __% | __ |
| 📰 News 新闻 | __% | __ |
| 💬 Sentiment 舆情 | __% | __ |
| **合计** | **100%** | — |

**权重参考基准**（基于股性画像选最接近的一档作为起点，再按当前事件窗口调整）：

| 股性 | 基本面 | 技术面 | 新闻 | 舆情 |
|------|-------|-------|------|------|
| 大盘蓝筹 | 45% | 25% | 20% | 10% |
| 高弹性成长 | 35% | 30% | 20% | 15% |
| 题材炒作小盘 | 15% | 30% | 25% | 30% |
| 周期股 | 30% | 25% | 30% | 15% |
| 流动性差/微盘 | 25% | 35% | 15% | 25% |
| ETF/指数 | 15% | 45% | 30% | 10% |

### 三、决策风格

从以下 4 种里选 1 种最适合的：

| 风格 | 适用标的 | 决策特征 |
|------|---------|---------|
| **value_anchor 价值锚定** | 大盘蓝筹 + 稳定行业 | DCF/PE 锚定，等估值回归才动手，长 Time Stop |
| **catalyst_driven 催化驱动** | 成长股 + 明确事件 | 等催化兑现/证伪，中 Time Stop（3-6 月）|
| **momentum 动量** | 题材/高弹性成长 + 趋势明确 | 跟随技术指标，短 Time Stop（1-3 月），重视舆情拥挤度 |
| **event_driven 事件驱动** | 周期股 / 重组并购 / 政策受益 | 围绕单一事件窗口建仓退出，事件结束就清仓 |

**输出格式**：
> 推荐风格：[XXX]。理由：__（1-2 句话）

### 四、时间窗口事件（关键事件 + 权重临时调整）

识别 news/fundamentals 报告里近 30 天内的事件，按重要性列出：

| 事件 | 日期 | 类型 | 对权重的临时影响 |
|------|------|------|---------------|
| 例：Q2 财报披露 | 2026-08-15 | 财报 | 基本面权重 +15%（财报前 2 周内）|
| 例：限售股解禁 | 2026-09-01 | 解禁 | 新闻权重 +10% |
| 例：业绩说明会 | 2026-05-21 | 公司活动 | 新闻+舆情各 +5% |

**若无重大事件**：填"无（保持基础权重）"。

### 五、最终权重（基础权重 + 事件调整后的最终值）

| 报告 | 基础权重 | 事件调整 | 最终权重 |
|------|---------|---------|---------|
| Fundamentals | __% | +__% / —/ -__% | __% |
| Market | __% | __ | __% |
| News | __% | __ | __% |
| Sentiment | __% | __ | __% |
| **合计** | 100% | — | **100%** |

事件调整后**总和仍必须 = 100%**——如果一项权重 +X%，其他项需对应分摊 -X%。

### 六、给下游 agent 的使用建议

用 3-5 句话告诉下游 agent（Consensus/Bull/Bear/RM/PM）：
- 该标的应该**重点关注**哪份报告里的哪类信号？
- 应该**警惕**哪份报告里的什么误导？
- 该标的的"alpha 来源"通常在哪份报告里？

### 七、估值方法推荐（供 RM 多元交叉验证使用）

根据股性画像，推荐 RM 使用以下估值方法做交叉验证（**至少 3 种**）。请按 style 选择最合适的组合：

| style | 主估值法 | 次估值法 | 辅助法 | 不适用 |
|-------|---------|---------|--------|-------|
| blue_chip 蓝筹 | DCF | PE × EPS | 历史分位 + 同业可比 | — |
| high_beta_growth 高弹性成长 | PEG | PE × 预期 EPS | 同业可比（看龙头估值上限） | DCF（折现率过敏感）|
| theme_speculation 题材炒作 | 历史分位（自身上下沿）| 市值天花板 | 卖方目标价区间 | DCF / PB（基本面不主导）|
| cyclical 周期 | PB × BPS | 周期顶/底 PE | 同业可比 | DCF（现金流不稳）|
| illiquid 流动性差 | PB × BPS × 0.8（流动性折价）| PE × EPS（保守目标 PE）| 历史分位 | DCF |
| etf | 跟踪指数估值 | 折溢价 | — | 个股法全部不适用 |

#### 📌 行业框架卡库（按 fundamentals.industry 字段匹配，强制采用对应卡片）

**真实头部投研团队**每个行业都有专人深耕，知道该行业用什么估值、关键驱动、典型风险。下面是 12 个常见 A 股行业的"框架卡"——你识别到 industry 后**必须**采用对应卡片，禁止凭印象判断：

| 行业 | 估值方法（主→辅）| 关键驱动 | 景气信号 | 典型风险 |
|------|----------------|---------|---------|---------|
| **半导体设计**（含 IC 设计/存储/接口）| PEG / PE×EPS / 历史分位 | 技术节点 / 客户认证 / 周期位置 | HBM 价格、DDR5 渗透率、北美 capex | 出口管制 / 客户集中 / 周期下行 |
| **半导体设备**（光刻/刻蚀/封测设备）| PE×EPS / PB / 同业可比 | 资本开支周期 / 国产化率 | 晶圆厂资本开支、国产替代节奏 | 周期回落 / 技术追赶失败 |
| **CPO/光通信**（光模块/光器件）| PEG / 同业可比 / 卖方目标价 | AI 算力订单 / 800G/1.6T 时间表 | 北美 hyperscaler 订单、客户结构 | 单一大客户依赖 / 技术迭代风险 |
| **新能源车**（整车/三电）| PE×EPS / PEG / 同业可比 | 销量 / 渗透率 / 单车价值量 | 月度销量、单价、毛利率 | 价格战 / 补贴退坡 / 库存 |
| **消费白马**（食品饮料/必选）| PE / DCF / EV/EBITDA | 渠道 / 品牌 / 客单价 | 同店销售、经销商库存 | 库存堆积 / 大宗成本 / 渠道变化 |
| **互联网平台** | PE / PS / DCF | 用户活跃 / 货币化率 | MAU/DAU、ARPU、留存 | 监管 / 流量见顶 / 竞争加剧 |
| **生物医药**（创新药）| DCF（管线 NPV）/ EV/Sales / 同业可比 | 临床进度 / FDA/NMPA 批准 / 销售放量 | 临床数据公告、商业化进度 | 临床失败 / 集采 / 专利悬崖 |
| **CRO/CDMO** | PE / PEG / DCF | 订单 / 产能利用率 | 新签订单、产能 | 海外订单转移 / 价格战 |
| **银行** | PB / DDM（股息折现）/ 历史分位 | 利差 / 资产质量 / ROE | NIM / 不良率 / 拨备覆盖 / 信贷增速 | 利率倒挂 / 房地产敞口 / 资产质量恶化 |
| **券商** | PB / PE / 同业可比 | 市场成交量 / 投行业务 / 自营 | 日均成交、IPO/再融资节奏 | 市场低迷 / 监管收紧 |
| **房地产** | PB / NAV（资产重估）| 销售 / 土地储备 / 杠杆 | 销售面积、回款、融资成本 | 销售失速 / 债务违约 / 政策 |
| **公用事业**（电力/水务）| DCF / PB / 股息率 | 价格调整 / 容量增长 / 现金流 | 上网电价、煤价、装机容量 | 政策风险 / 燃料成本 |
| **算力租赁/IDC** | PE×EPS / EV/EBITDA / 同业可比 | 上架率 / 单机柜价格 / 电力成本 | 上架率、新签客户、电力成本 | 上架率不及预期 / 电力价格上行 |
| **AI 应用**（SaaS / 工具）| PS / 用户增长率 / DCF | ARR / 用户增长 / NRR | MAU、ARR 增速、NRR | 流量见顶 / 商业化不及预期 |

**行业框架卡使用规则**：

1. **必须**根据 fundamentals/news 报告中的 industry 字段匹配最接近的行业卡
2. 估值方法**直接采用**卡片推荐的"主→辅"组合，不允许自行选其他方法
3. 报告中要显式引用卡片内容：如"按行业框架卡【半导体设计】，主估值方法 PEG / 关键驱动：技术节点 + 客户认证..."
4. 若 industry 不在上述清单中（如冷门行业），明确标注"无标准行业卡，按 style 默认方法处理"，并自行说明依据
5. 卡片中的"典型风险"必须出现在最终输出的"风险清单"中（让风控有针对性参考）

**输出要求**：
- 显式给出 **主方法目标 PE/PB 区间**（如成长股目标 PE 30-50 倍、蓝筹股 PE 15-25 倍、周期股 PB 1-3 倍）
- 推荐区间必须有依据（行业平均 / 历史分位 / 同业可比）
- 明确数据完整度等级 L0-L3
- 说明该方法的局限性（防止 RM 盲信单一方法）

| 字段 | 说明 |
|------|------|
| `primary_method` | 主估值方法 |
| `secondary_methods` | 次要交叉验证方法（至少 2 个）|
| `target_pe_range` | 主方法的目标 PE 区间（如适用）|
| `target_pb_range` | 主方法的目标 PB 区间（如适用）|
| `data_completeness` | L0（完整）/ L1（缺行业 PE）/ L2（缺 EPS）/ L3（无估值数据）|
| `rationale` | 1-2 句话说明为什么这套方法适合该标的 |

---

### 八、主题热度识别（新增，影响下游估值容忍度）

A 股 / 美股都存在"主题轮动"现象——热门主题股在加速期可承受 PE 偏离合理估值 50-200% 仍持续上涨（如美股 NVDA / 国内 AI 算力），用一刀切的"偏离 +10% 就 UNDERWEIGHT"会错杀整轮上升周期。

**你必须识别该标的是否处于活跃主题中，并标注主题阶段，让下游 RM 据此放宽估值阈值。**

#### 主题识别信号（综合 news + sentiment + market）

| 信号 | 表现 |
|---|---|
| **news 主题密度** | 近 14 天 ≥5 条同主题正面新闻（如 AI 算力/CPO/算电/可控核聚变/算力租赁/量子计算/低空经济）|
| **sentiment KOL 抱团** | KOL 集中讨论同一主题 / 同板块抱团 |
| **行业景气信号** | 上下游订单激增、研报集中覆盖、政策窗口期 |
| **主题持续时间** | 第一次集中报道 → 启动期；持续 3-6 月 → 加速期；6+ 月 + 调整 → 顶部期；利空增多 → 退潮期 |

#### 当前 A 股主流热门主题（参考清单，非穷尽）

- AI 算力（GPU/HBM/液冷）
- CPO 光通信（光模块/光器件）
- 算力租赁（IDC/智算中心）
- 算电（HBM 配套/液冷散热/电源）
- 可控核聚变（聚变三代）
- 量子计算（量子通信/量子芯片）
- 低空经济（eVTOL/无人机/低空基建）
- 智能驾驶（L3/L4/激光雷达）
- 人形机器人（特斯拉链）
- 国产替代（半导体/工业软件）

#### 主题阶段判定（4 选 1）

| 阶段 | 触发信号 | 容忍系数 |
|------|---------|---------|
| 启动期（initiation）| 第一次集中报道 + 少数龙头股启动 + 卖方开始覆盖 | +30% 容忍 |
| 加速期（acceleration）| 主题已持续 3-6 月 + 多股共振 + 主流财经媒体反复报道 | **+50% 容忍**（最宽容）|
| 顶部期（peak）| 持续 6+ 月 + 调整出现 + 部分龙头 RSI 极端 + 卖方目标价频繁上调 | +20% 容忍（开始警惕）|
| 退潮期（fading）| 主题热度消退 + 利空增多 + 板块整体下跌 | **-20% 反向收紧**（主题反噬保护）|
| 不在主题（none）| 标的不属于任何当前主流主题 | 0% 容忍 |

⚠️ 主题判定必须 conservative——**只有有明确信号才标"加速期"，宁可标"顶部期"也别误判加速期**（否则下游会过度宽容）。

#### 📌 宏观环境对主题溢价的修正（强制读取 Macro Context）

宏观策略师已经在前面识别了利率周期、流动性松紧、行业宏观方向。**你必须在最终 premium_tolerance_pct 上叠加宏观修正**：

```
最终 premium_tolerance_pct = 基础溢价（按 theme_stage）+ macro_context.premium_adjustment_pct
```

其中 macro_context.premium_adjustment_pct 来自宏观策略师的 YAML：
- 流动性紧缩 + 利率上行 → -10% ~ -20%（如紧缩期 acceleration 从 +50% 降到 +30%）
- 流动性宽松 + 利率下行 → +0% ~ +10%（不另加成，但确保 acceleration 给到 +50% 上限）
- 中性 → 0%
- 行业强逆风（紧缩 + 行业政策不利） → -20%
- 行业强顺风 → +5%

**示例**：AI 算力 acceleration 基础 +50%，但当前流动性紧缩 + 利率上行 → 宏观修正 -20% → **最终 +30%**

⛔ 如果 macro_context 显示"紧缩 + 强逆风" 但你仍给主题 +50% 容忍 → 错误，必须叠加修正。

---

## 输入资料

**【最重要】宏观上下文（来自宏观策略师）**：

{macro_context if macro_context else "（宏观上下文缺失，按中性环境处理）"}

[置信度:高] Company fundamentals report:
{fundamentals_report}

[置信度:中高] Market research report:
{market_report}

[置信度:中] Latest world affairs news:
{news_report}

[置信度:中低] Social media sentiment report:
{sentiment_report}

---

**最终输出要求**：
- 用中文撰写，结构严格按上述六部分
- 末尾输出一段 YAML 摘要供下游程序化解析：

```yaml
PROFILE:
  market_cap_tier: large_cap / mid_cap / small_cap / micro_cap
  industry: <行业>
  style: blue_chip / high_beta_growth / theme_speculation / cyclical / illiquid / etf
  liquidity: deep / medium / shallow
  instrument_type: a_share_stock / hk_stock / us_stock / etf / lof / convertible_bond

REPORT_WEIGHTS:
  fundamentals: __  # 0-100 整数
  market: __
  news: __
  sentiment: __

DECISION_STYLE: value_anchor / catalyst_driven / momentum / event_driven

VALUATION_METHOD:
  primary_method: dcf / pe_eps / peg / pb_bps / ev_ebitda / historical_quantile / relative_position
  secondary_methods:
    - <方法 1>
    - <方法 2>
  target_pe_range: [__, __]    # 如适用，否则 null
  target_pb_range: [__, __]    # 如适用，否则 null
  data_completeness: L0 / L1 / L2 / L3
  rationale: <1-2 句话>

EVENT_WINDOWS:
  - event: <事件描述>
    date: YYYY-MM-DD
    impact: <对权重的临时调整>

THEMATIC_PREMIUM:
  is_active_theme: yes / no                              # 是否处于当前活跃主题
  theme_name: <如 AI算力 / CPO / 算电 / 可控核聚变 / 算力租赁 / 量子计算 / 低空经济 / 智能驾驶 / 人形机器人 / 国产替代 / 不在主题 等>
  theme_stage: initiation / acceleration / peak / fading / none  # 主题阶段
  premium_tolerance_pct: <整数>                          # 该主题对估值偏离的额外容忍：启动+30 / 加速+50 / 顶部+20 / 退潮-20 / 无0
  rationale: <1-2 句话，引用 news/sentiment 中的具体信号>
```
"""

        response = llm.invoke(prompt)
        return {"stock_profile": response.content}

    return stock_profile_node
