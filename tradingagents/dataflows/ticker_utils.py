"""中国 A 股（含 ETF、可转债、北交所）代码检测与跨供应商格式转换工具。

格式对照：
  yfinance:  600519.SS (上交所), 000858.SZ (深交所)
  AKShare:   600519, 000858 (纯6位数字)
  Tushare:   600519.SH, 000858.SZ
"""

import re

# A股代码首位 → 交易所映射
# 上交所：6（主板）, 9（B股）, 5（ETF/基金/权证）
_SHANGHAI_PREFIXES = ("6", "5")
# 深交所：0（主板/中小板）, 2（B股）, 3（创业板）, 1（ETF/可转债/国债逆回购）
_SHENZHEN_PREFIXES = ("0", "2", "3", "1")
# 北交所/新三板：4（老三板/新三板）, 8（北交所/新三板）, 9（北交所新代码段 92xxxx）
_BEIJING_PREFIXES = ("4", "8", "9")

# 所有合法 A 股交易所后缀
_A_SHARE_SUFFIXES = {".SS", ".SH", ".SZ", ".BJ"}

_PURE_6DIGIT_RE = re.compile(r"^\d{6}$")

# 交易所前缀格式: SH600519, SZ000858, BJ831XXX
_EXCHANGE_PREFIX_RE = re.compile(r"^(SH|SZ|BJ)(\d{6})$", re.IGNORECASE)

# 所有中国市场代码前缀（上交所 + 深交所 + 北交所）
_ALL_CN_PREFIXES = _SHANGHAI_PREFIXES + _SHENZHEN_PREFIXES + _BEIJING_PREFIXES


def is_a_share(ticker: str) -> bool:
    """检测 ticker 是否为中国 A 股市场代码（含 ETF、可转债、北交所）。

    支持以下格式：
      - 纯6位数字: "600519", "518880", "159915"
      - yfinance 格式: "600519.SS", "000858.SZ"
      - Tushare 格式: "600519.SH", "000858.SZ"
      - 北交所格式: "831XXX.BJ"
      - 交易所前缀格式: "SH600519", "SZ000858", "BJ831XXX"

    >>> is_a_share("600519")
    True
    >>> is_a_share("518880")
    True
    >>> is_a_share("159915")
    True
    >>> is_a_share("600519.SS")
    True
    >>> is_a_share("SH518880")
    True
    >>> is_a_share("SZ000858")
    True
    >>> is_a_share("AAPL")
    False
    >>> is_a_share("0700.HK")
    False
    """
    ticker = ticker.strip().upper()

    # 纯6位数字
    if _PURE_6DIGIT_RE.match(ticker):
        return ticker[0] in _ALL_CN_PREFIXES

    # 带后缀: 600519.SS, 000858.SZ, 831XXX.BJ
    if "." in ticker:
        code, suffix = ticker.rsplit(".", 1)
        suffix = "." + suffix
        if suffix in _A_SHARE_SUFFIXES and _PURE_6DIGIT_RE.match(code):
            return True

    # 交易所前缀格式: SH600519, SZ000858, BJ831XXX
    if _EXCHANGE_PREFIX_RE.match(ticker):
        return True

    return False


def _extract_code(ticker: str) -> str:
    """提取纯6位数字部分。

    支持所有 A 股格式:
      "600519" → "600519"
      "600519.SS" → "600519"
      "600519.SH" → "600519"
      "SH600519" → "600519"
    """
    ticker = ticker.strip().upper()

    # 交易所前缀格式: SH600519 → 600519
    m = _EXCHANGE_PREFIX_RE.match(ticker)
    if m:
        return m.group(2)

    # 带后缀: 600519.SS → 600519
    if "." in ticker:
        return ticker.rsplit(".", 1)[0]

    return ticker


def _get_exchange(code: str) -> str:
    """根据代码首位判断交易所。返回 'SH'、'SZ' 或 'BJ'。"""
    first = code[0]
    if first in _SHANGHAI_PREFIXES:
        return "SH"
    if first in _SHENZHEN_PREFIXES:
        return "SZ"
    if first in _BEIJING_PREFIXES:
        return "BJ"
    raise ValueError(f"无法判断股票代码 {code} 的交易所归属")


def is_etf_or_lof(ticker: str) -> bool:
    """检测 ticker 是否为 ETF 或 LOF 基金代码。

    ETF/LOF 没有传统财务报表（资产负债表、利润表、现金流量表），
    也没有董监高交易、个股研报等信息。

    上交所 ETF: 5xxxxx
    深交所 ETF/LOF: 15xxxx, 16xxxx

    >>> is_etf_or_lof("518880")
    True
    >>> is_etf_or_lof("159915")
    True
    >>> is_etf_or_lof("160723")
    True
    >>> is_etf_or_lof("600519")
    False
    >>> is_etf_or_lof("SH518880")
    True
    """
    code = _extract_code(ticker)
    if not _PURE_6DIGIT_RE.match(code):
        return False
    return code[:1] == "5" or code[:2] in ("15", "16")


def to_akshare_format(ticker: str) -> str:
    """转为 AKShare 纯6位格式（行情/新闻接口）。

    >>> to_akshare_format("600519.SS")
    '600519'
    >>> to_akshare_format("600519.SH")
    '600519'
    >>> to_akshare_format("600519")
    '600519'
    >>> to_akshare_format("SH518880")
    '518880'
    """
    return _extract_code(ticker)


def to_akshare_report_format(ticker: str) -> str:
    """转为 AKShare 财报接口所需的 SH/SZ 前缀格式。

    stock_balance_sheet_by_report_em / stock_profit_sheet_by_report_em /
    stock_cash_flow_sheet_by_report_em 等接口使用此格式。

    >>> to_akshare_report_format("600519")
    'SH600519'
    >>> to_akshare_report_format("000858.SZ")
    'SZ000858'
    >>> to_akshare_report_format("600519.SS")
    'SH600519'
    """
    code = _extract_code(ticker)
    exchange = _get_exchange(code)
    return f"{exchange}{code}"


def to_tushare_format(ticker: str) -> str:
    """转为 Tushare 格式 "600519.SH" / "000858.SZ"。

    >>> to_tushare_format("600519.SS")
    '600519.SH'
    >>> to_tushare_format("600519")
    '600519.SH'
    >>> to_tushare_format("000858.SZ")
    '000858.SZ'
    """
    code = _extract_code(ticker)
    exchange = _get_exchange(code)
    return f"{code}.{exchange}"


def to_yfinance_format(ticker: str) -> str:
    """转为 yfinance 格式 "600519.SS" / "000858.SZ"。

    注意：yfinance 对上交所使用 .SS（而非 .SH）。

    >>> to_yfinance_format("600519.SH")
    '600519.SS'
    >>> to_yfinance_format("600519")
    '600519.SS'
    >>> to_yfinance_format("000858.SZ")
    '000858.SZ'
    """
    code = _extract_code(ticker)
    exchange = _get_exchange(code)
    yf_suffix = "SS" if exchange == "SH" else "SZ"
    return f"{code}.{yf_suffix}"


def to_akshare_date(date_str: str) -> str:
    """将 YYYY-MM-DD 日期转为 AKShare 所需的 YYYYMMDD 格式。

    >>> to_akshare_date("2024-01-15")
    '20240115'
    >>> to_akshare_date("20240115")
    '20240115'
    """
    return date_str.replace("-", "")


def to_standard_date(date_str: str) -> str:
    """将 YYYYMMDD 日期转为标准 YYYY-MM-DD 格式。

    >>> to_standard_date("20240115")
    '2024-01-15'
    >>> to_standard_date("2024-01-15")
    '2024-01-15'
    """
    d = date_str.replace("-", "")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
