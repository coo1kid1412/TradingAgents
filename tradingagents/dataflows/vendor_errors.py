"""统一的数据供应商异常层次，供 route_to_vendor() 捕获并触发 fallback。"""


class VendorRateLimitError(Exception):
    """数据供应商请求频率超限。触发 fallback 到下一个供应商。"""
    pass


class VendorUnavailableError(Exception):
    """数据供应商不可用（未配置 token、积分不足等）。触发 fallback。"""
    pass


class AKShareError(VendorRateLimitError):
    """AKShare 网络错误或上游限流。"""
    pass


class TushareRateLimitError(VendorRateLimitError):
    """Tushare Pro 请求频率超限。"""
    pass


class TushareUnavailableError(VendorUnavailableError):
    """Tushare Pro 不可用（Token 未配置或积分不足）。"""
    pass
