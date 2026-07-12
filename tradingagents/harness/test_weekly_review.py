"""weekly_review cron 健康度判定的回归测试。

运行：python tradingagents/harness/test_weekly_review.py
"""

import datetime as dt

from tradingagents.harness.weekly_review import parse_cron_health

_TODAY = dt.date(2026, 6, 27)


def _run_block(day: int, failed: int = 0, noise_lines: int = 1300) -> str:
    """造一段 daily_update 单次运行日志（含大量噪音行，模拟真实 ~1300 行/次）。"""
    lines = [f"=== Harness Daily Update @ 2026-06-{day:02d} 21:00:01 ==="]
    lines += ["[INFO db] Harness DB initialized at ..."] * noise_lines
    lines.append(f"[21:07 INFO __main__] 真值采集统计：{{'promoted_from_not_due': 300, "
                 f"'fetched': 20, 'not_due': 280, 'failed': {failed}}}")
    return "\n".join(lines)


def test_daily_runs_counted_despite_huge_per_run_logs():
    """核心回归：每次运行 ~1300 行，旧实现取末尾 500 行把'天天跑'误报成'1 次'。
    全文按日期去重应数出窗口内每一天。"""
    text = "\n".join(_run_block(d) for d in range(18, 28))   # 06-18..06-27
    ok, desc = parse_cron_health(text, _TODAY)
    # cutoff=06-20，窗口内 06-20..06-27 = 8 天
    assert ok and "8 次运行正常" in desc, (ok, desc)


def test_single_run_in_window_is_unhealthy():
    text = _run_block(26)   # 仅 06-26 一次
    ok, desc = parse_cron_health(text, _TODAY)
    assert not ok and "仅 1 次" in desc, (ok, desc)


def test_price_cache_stale_warnings_are_not_vendor_failures():
    """关键回归：price_cache 逐票 stale 警告(退市/不活跃票常态噪音、9999=无缓存哨兵)
    不应被当成 vendor 失败——daily_update 自报 failed:0 即健康。"""
    text = "\n".join(_run_block(d) for d in range(20, 28))
    # 注入大量 stale 警告噪音
    text += "\n" + "\n".join(["vendor 链全部失败（cache 已落后 9999 天）"] * 96)
    ok, desc = parse_cron_health(text, _TODAY)
    assert ok and "无异常" in desc, (ok, desc)


def test_authoritative_failed_count_flags():
    """daily_update 自报 failed>0 才算真值采集失败。"""
    text = "\n".join(_run_block(d, failed=(3 if d == 25 else 0)) for d in range(20, 28))
    ok, desc = parse_cron_health(text, _TODAY)
    assert not ok and "真值采集失败 3 次" in desc, (ok, desc)


def test_traceback_flags():
    text = _run_block(26) + "\n" + "\n".join(_run_block(d) for d in range(20, 26))
    text += "\nTraceback (most recent call last):\n  File ...\nValueError: x"
    ok, desc = parse_cron_health(text, _TODAY)
    assert not ok and "Traceback" in desc, (ok, desc)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {fn.__name__}: [{type(e).__name__}] {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
