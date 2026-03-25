"""A股代码检测与跨供应商格式转换工具。

格式对照：
  yfinance:  600519.SS (上交所), 000858.SZ (深交所)
  AKShare:   600519, 000858 (纯6位数字)
  Tushare:   600519.SH, 000858.SZ
"""

import re

# A股代码首位 → 交易所映射
_SHANGHAI_PREFIXES = ("6", "9")   # 主板、B股
_SHENZHEN_PREFIXES = ("0", "2", "3")  # 主板/中小板/创业板/B股

# 所有合法 A 股交易所后缀
_A_SHARE_SUFFIXES = {".SS", ".SH", ".SZ"}

_PURE_6DIGIT_RE = re.compile(r"^\d{6}$")


def is_a_share(ticker: str) -> bool:
    """检测 ticker 是否为中国 A 股代码。

    支持以下格式：
      - 纯6位数字: "600519", "000858"
      - yfinance 格式: "600519.SS", "000858.SZ"
      - Tushare 格式: "600519.SH", "000858.SZ"

    >>> is_a_share("600519")
    True
    >>> is_a_share("600519.SS")
    True
    >>> is_a_share("AAPL")
    False
    >>> is_a_share("0700.HK")
    False
    """
    ticker = ticker.strip().upper()

    # 纯6位数字
    if _PURE_6DIGIT_RE.match(ticker):
        first = ticker[0]
        return first in _SHANGHAI_PREFIXES or first in _SHENZHEN_PREFIXES

    # 带后缀
    if "." in ticker:
        code, suffix = ticker.rsplit(".", 1)
        suffix = "." + suffix
        if suffix in _A_SHARE_SUFFIXES and _PURE_6DIGIT_RE.match(code):
            return True

    return False


def _extract_code(ticker: str) -> str:
    """提取纯6位数字部分。"""
    ticker = ticker.strip().upper()
    if "." in ticker:
        return ticker.rsplit(".", 1)[0]
    return ticker


def _get_exchange(code: str) -> str:
    """根据代码首位判断交易所。返回 'SH' 或 'SZ'。"""
    first = code[0]
    if first in _SHANGHAI_PREFIXES:
        return "SH"
    if first in _SHENZHEN_PREFIXES:
        return "SZ"
    raise ValueError(f"无法判断股票代码 {code} 的交易所归属")


def to_akshare_format(ticker: str) -> str:
    """转为 AKShare 纯6位格式（行情/新闻接口）。

    >>> to_akshare_format("600519.SS")
    '600519'
    >>> to_akshare_format("600519.SH")
    '600519'
    >>> to_akshare_format("600519")
    '600519'
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
