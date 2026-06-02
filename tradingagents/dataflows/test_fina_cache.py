"""fina_indicator 数据缓存测试。

财务指标季度更新、极稳定 → 缓存后命中即用（省 tushare 1次/小时限流额度），
限流/不可用时回退旧缓存。这是 SYS_GROWTH_YOY 稳定产出的总开关——earnings 腿确定性
+ PEG 确定性输入都依赖它。

运行：python tradingagents/dataflows/test_fina_cache.py
"""
import os
import tempfile
import time

os.environ["FINA_INDICATOR_CACHE_DIR"] = tempfile.mkdtemp()

import pandas as pd
from tradingagents.dataflows import tushare_vendor as tv

_DF = pd.DataFrame([
    {"end_date": "20260331", "eps": 1.2, "q_sales_yoy": 192.9, "q_netprofit_yoy": 343.45,
     "or_yoy": 65.1, "netprofit_yoy": 68.51, "dt_netprofit_yoy": 353.0,
     "profit_dedt": 7.4e8, "tob_operate_income": 6e9},
    {"end_date": "20251231", "eps": 4.86, "q_sales_yoy": 18.1, "q_netprofit_yoy": 40.0,
     "or_yoy": 50.0, "netprofit_yoy": 68.51, "dt_netprofit_yoy": 70.0,
     "profit_dedt": 11e8, "tob_operate_income": 2e10},
])


def test_roundtrip_preserves_growth_line():
    """写缓存 → 读回 → SYS_GROWTH_YOY 抽取逐字一致（JSON 往返不丢数据/不变格式）。"""
    g_orig = tv._format_growth_indicators(_DF)
    tv._write_fina_cache("TEST001.SZ", _DF)
    df2 = tv._read_fina_cache("TEST001.SZ", require_fresh=True)
    assert df2 is not None and not df2.empty
    assert tv._format_growth_indicators(df2) == g_orig


def test_fresh_vs_stale_ttl():
    """新鲜缓存(require_fresh)按 TTL 判定；过期则 require_fresh=True 返回 None、False 仍返回。"""
    tv._write_fina_cache("TEST002.SZ", _DF)
    assert tv._read_fina_cache("TEST002.SZ", require_fresh=True) is not None   # 刚写，新鲜
    # 人为把文件 mtime 改老到超 TTL
    path = tv._fina_cache_path("TEST002.SZ")
    old = time.time() - tv._FINA_CACHE_TTL_SEC - 100
    os.utime(path, (old, old))
    assert tv._read_fina_cache("TEST002.SZ", require_fresh=True) is None       # 过期 → 新鲜读取为 None
    assert tv._read_fina_cache("TEST002.SZ", require_fresh=False) is not None  # 回退读取仍拿得到


def test_missing_cache_returns_none():
    assert tv._read_fina_cache("NOPE999.SZ", require_fresh=True) is None
    assert tv._read_fina_cache("NOPE999.SZ", require_fresh=False) is None


def test_rate_limit_falls_back_to_stale(monkeypatch=None):
    """限流时 _fetch_fina_indicator_cached 回退旧缓存（旧增速 > 没增速）。"""
    from tradingagents.dataflows.vendor_errors import TushareRateLimitError
    tv._write_fina_cache("TEST003.SZ", _DF)
    # 让缓存过期（强制走 API 路径），再模拟 API 限流 → 应回退旧缓存
    path = tv._fina_cache_path("TEST003.SZ")
    old = time.time() - tv._FINA_CACHE_TTL_SEC - 100
    os.utime(path, (old, old))

    class _FakePro:
        def fina_indicator(self, **kw):
            raise TushareRateLimitError("小时级限流")

    orig = tv._safe_call
    tv._safe_call = lambda fn, **kw: fn(**kw)  # 让 _safe_call 直接调用 → 抛出限流
    try:
        out = tv._fetch_fina_indicator_cached(_FakePro(), "TEST003.SZ")
        assert out is not None and not out.empty   # 回退到旧缓存
    finally:
        tv._safe_call = orig


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
