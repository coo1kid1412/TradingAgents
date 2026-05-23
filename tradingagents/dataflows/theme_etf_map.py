"""主题热点 → ETF + 代表股映射表。

设计原则：
- 维护性优先：单文件 dict 易读易改，每年人工 review 一次
- fuzzy 匹配：theme_name 是 LLM 写的字符串，不强求精确匹配 key
- 缺位容忍：某些主题没专门 ETF（如 PCB / 液冷），靠代表股做横向对比

数据来源（V1）：
- ETF 代码：实际市场流通的、有代表性的主题 ETF
- 代表股：每个主题 3-5 只龙头/活跃股（2026 Q2 market 状态）

未来扩展：
- 拉 ETF 持仓数据自动维护 peers（消除人工维护）
- 加权 RS（用市值/成交额加权 peers 平均收益）
"""

from __future__ import annotations

# 主题 → (ETF 代码 / None, 代表股清单)
THEME_MAP: dict[str, dict] = {
    "AI算力": {
        "etf": "512760",
        "peers": ["688041", "300474", "300308", "688008"],  # 海光信息 / 景嘉微 / 中际旭创 / 澜起科技
    },
    "CPO光通信": {
        "etf": None,
        "peers": ["300308", "300394", "002281", "300502"],  # 中际旭创 / 天孚通信 / 光迅科技 / 新易盛
    },
    "存储芯片": {
        "etf": "512760",
        "peers": ["603986", "688008", "688525", "002156"],  # 兆易创新 / 澜起 / 佰维存储 / 通富微电
    },
    "液冷": {
        "etf": None,
        "peers": ["301289", "002335", "300769", "603688"],  # 国缆检测 / 科华数据 / 德方纳米 / 石英股份
    },
    "PCB": {
        "etf": None,
        "peers": ["002463", "002618", "002709", "002463"],  # 沪电股份 / 深南电路 / 兴森科技
    },
    "电力": {
        "etf": "159611",
        "peers": ["600886", "600025", "001289", "601985"],  # 国投电力 / 华能水电 / 龙源电力 / 中国核电
    },
    "绿色电力": {
        "etf": "516160",
        "peers": ["600886", "601985", "002129", "300751"],  # 国投电力 / 中国核电 / TCL中环 / 迈为股份
    },
    "光伏": {
        "etf": "515790",
        "peers": ["601012", "002129", "300751", "688472"],  # 隆基绿能 / TCL中环 / 迈为股份 / 阿特斯
    },
    "新能源车": {
        "etf": "515030",
        "peers": ["300750", "002594", "002460", "300014"],  # 宁德时代 / 比亚迪 / 赣锋锂业 / 亿纬锂能
    },
    "锂电池": {
        "etf": "159755",
        "peers": ["300750", "002460", "300014", "002709"],
    },
    "半导体": {
        "etf": "512760",
        "peers": ["688008", "688041", "603986", "300782"],  # 澜起 / 海光 / 兆易 / 卓胜微
    },
    "商业航天": {
        "etf": None,
        "peers": ["600677", "688041", "002179", "002025"],  # 航天动力 / 海光 / 中航光电 / 航天电器
    },
    "创新药": {
        "etf": "159992",
        "peers": ["600196", "300760", "688235", "688180"],  # 复星医药 / 迈瑞 / 百济神州 / 君实生物
    },
    "国产替代": {
        "etf": None,
        "peers": ["688008", "688041", "300782", "688981", "688256"],  # 澜起 / 海光 / 卓胜微 / 中芯国际 / 寒武纪
    },
    "数字经济": {
        "etf": None,
        "peers": ["688111", "002230", "300624", "688023"],  # 金山办公 / 科大讯飞 / 万兴科技 / 思特奇
    },
    "信创": {
        "etf": None,
        "peers": ["688111", "002230", "002405", "688023"],
    },
}


# 主题别名（LLM 可能用不同写法，统一映射回标准 key）
_THEME_ALIASES: dict[str, str] = {
    "AI": "AI算力",
    "AIGC": "AI算力",
    "算力": "AI算力",
    "AI芯片": "AI算力",
    "算力基础设施": "AI算力",
    "光通信": "CPO光通信",
    "光模块": "CPO光通信",
    "CPO": "CPO光通信",
    "DRAM": "存储芯片",
    "存储": "存储芯片",
    "内存": "存储芯片",
    "存储超级周期": "存储芯片",
    "印制电路板": "PCB",
    "数据中心冷却": "液冷",
    "智算中心": "液冷",
    "电力新基建": "电力",
    "新能源": "绿色电力",
    "绿电": "绿色电力",
    "光伏新能源": "光伏",
    "新能源汽车": "新能源车",
    "电动车": "新能源车",
    "锂电": "锂电池",
    "半导体设计": "半导体",
    "集成电路": "半导体",
    "芯片": "半导体",
    "商业航空": "商业航天",
    "卫星互联网": "商业航天",
    "航天": "商业航天",
    "创新药械": "创新药",
    "生物医药": "创新药",
    "国产替代芯片": "国产替代",
    "自主可控": "国产替代",
    "数字经济产业": "数字经济",
    "信创产业": "信创",
}


def resolve_theme(theme_name: str | None) -> dict:
    """根据 theme_name（LLM 写的自由文本）fuzzy 匹配映射表。

    匹配优先级：
    1. theme_name 精确等于 THEME_MAP 的某个 key
    2. theme_name 包含 THEME_MAP 的某个 key（如 "AI算力 / 存储超级周期" 同时匹配 AI算力 和 存储芯片，取第一个）
    3. theme_name 精确等于别名表 _THEME_ALIASES 的某个 key
    4. theme_name 包含别名表的某个 key
    5. 全部 fail → 返回空对照集

    Returns:
        {"etf": str | None, "peers": list[str], "matched_theme": str | None}
    """
    if not theme_name or not theme_name.strip():
        return {"etf": None, "peers": [], "matched_theme": None}

    text = theme_name.strip()

    # 1. 精确匹配
    if text in THEME_MAP:
        data = THEME_MAP[text]
        return {"etf": data["etf"], "peers": data["peers"], "matched_theme": text}

    # 2. THEME_MAP key 子串匹配（优先级：长 key 先匹配，避免"半导体设计"先匹"半导体"）
    sorted_keys = sorted(THEME_MAP.keys(), key=lambda k: -len(k))
    for key in sorted_keys:
        if key in text:
            data = THEME_MAP[key]
            return {"etf": data["etf"], "peers": data["peers"], "matched_theme": key}

    # 3. 别名精确匹配
    if text in _THEME_ALIASES:
        canonical = _THEME_ALIASES[text]
        data = THEME_MAP[canonical]
        return {"etf": data["etf"], "peers": data["peers"], "matched_theme": canonical}

    # 4. 别名子串匹配（同样长 key 优先）
    sorted_aliases = sorted(_THEME_ALIASES.keys(), key=lambda k: -len(k))
    for alias in sorted_aliases:
        if alias in text:
            canonical = _THEME_ALIASES[alias]
            data = THEME_MAP[canonical]
            return {"etf": data["etf"], "peers": data["peers"], "matched_theme": canonical}

    return {"etf": None, "peers": [], "matched_theme": None}


# ============================================================================
# 行业 → ETF 映射（fallback 第二级：theme 没命中时用 industry 兜底）
# ============================================================================
INDUSTRY_ETF_MAP: dict[str, str] = {
    # 半导体大类
    "半导体": "512760",
    "半导体设计": "512760",
    "半导体设备": "512760",
    "集成电路": "512760",
    "芯片": "512760",
    "IC设计": "512760",
    "存储": "512760",
    # 新能源系
    "光伏": "515790",
    "太阳能": "515790",
    "锂电池": "159755",
    "锂电": "159755",
    "新能源车": "515030",
    "新能源汽车": "515030",
    "电池": "159755",
    "电动车": "515030",
    # 医药系
    "创新药": "159992",
    "生物医药": "159992",
    "医药": "159992",
    "医疗器械": "512170",
    "医疗服务": "512170",
    "CXO": "159992",
    # 金融
    "券商": "512000",
    "证券": "512000",
    "银行": "512800",
    "保险": "512070",
    # 消费
    "白酒": "159607",
    "食品饮料": "159928",
    "消费": "159928",
    "家电": "159996",
    # 周期
    "房地产": "512200",
    "地产": "512200",
    "煤炭": "515220",
    "钢铁": "515210",
    "有色金属": "512400",
    "化工": "512190",
    # 公用 / 资源
    "电力": "159611",
    "公用事业": "159611",
    "石油": "515220",
    # TMT
    "通信": "515050",
    "5G": "515050",
    "光通信": "512760",  # 借用半导体 ETF（光模块跟半导体高度相关）
    "计算机": "159998",
    "软件": "159998",
    "信创": "159998",
    # 国防 / 航天
    "军工": "512660",
    "国防": "512660",
    "航天": "512660",
    "航空航天": "512660",
}


def resolve_industry_etf(industry_name: str | None) -> str | None:
    """根据 stock_profile.industry 模糊匹配行业 ETF。

    匹配顺序：精确 → 长 key 子串。失败返回 None。
    """
    if not industry_name or not str(industry_name).strip():
        return None
    text = str(industry_name).strip()
    if text in INDUSTRY_ETF_MAP:
        return INDUSTRY_ETF_MAP[text]
    # 长 key 优先匹配
    for key in sorted(INDUSTRY_ETF_MAP.keys(), key=lambda k: -len(k)):
        if key in text:
            return INDUSTRY_ETF_MAP[key]
    return None


# ============================================================================
# 市场指数 ETF（fallback 第三级：theme + industry 都没命中时按 ticker 段兜底）
# ============================================================================
def resolve_market_etf_by_ticker(ticker: str) -> tuple[str, str] | None:
    """按 ticker 代码段决定"本股所在市场"的指数 ETF。

    Returns:
        (ETF 代码, 显示标签) 或 None（北交所/其他暂无对应指数）
    """
    if not ticker or len(ticker) < 6:
        return None
    prefix = ticker[:3]
    if prefix.startswith("688"):
        return ("588000", "科创50")
    if prefix.startswith("30"):
        return ("159915", "创业板")
    # 沪市主板 + 深市主板 + 深市中小板 → 都用沪深300
    if prefix[0] in ("6", "0", "2"):
        return ("510300", "沪深300")
    # 北交所等其他
    return None


# ============================================================================
# 最终兜底：永远加沪深300（V1 简单）
# ============================================================================
DEFAULT_FALLBACK_ETF = ("510300", "沪深300")
