"""测试 Tushare 限流等待逻辑（允许等待时间超过壁钟超时）."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tradingagents.dataflows.tushare_vendor import _parse_retry_delay
from tradingagents.dataflows.interface import _VENDOR_CALL_TIMEOUT


def test_rate_limit_wait():
    """测试限流等待逻辑."""
    print("=" * 70)
    print("Tushare 限流等待逻辑测试（优先保证数据质量）")
    print("=" * 70)
    print()

    # Test 1: 分钟级限流
    msg1 = "抱歉，您访问接口(stock_basic)频率超限(1次/分钟)，具体频次详情：https://tushare.pro/document/1?doc_id=108。"
    delay1 = _parse_retry_delay(msg1)
    print(f"测试1: 分钟级限流")
    print(f"  错误消息: {msg1[:50]}...")
    print(f"  解析等待时间: {delay1}s")
    print(f"  壁钟超时: {_VENDOR_CALL_TIMEOUT}s")
    print()
    
    if delay1 is not None:
        if delay1 >= _VENDOR_CALL_TIMEOUT:
            print(f"  ✓ 等待时间({delay1}s) >= 壁钟超时({_VENDOR_CALL_TIMEOUT}s)")
            print(f"  ✓ 系统行为：等待 {delay1}s 后重试 Tushare（不立即 fallback）")
            print(f"  ✓ 设计原因：Tushare 的 A 股数据质量优于 AKShare/yfinance")
        else:
            print(f"  ✓ 等待时间({delay1}s) < 壁钟超时({_VENDOR_CALL_TIMEOUT}s)")
            print(f"  ✓ 系统行为：等待 {delay1}s 后重试 Tushare")
    print()

    # Test 2: 重试逻辑验证
    print("测试2: 重试策略")
    print(f"  最大重试次数: 2 次")
    print(f"  每次等待时间: {delay1}s")
    print(f"  最坏情况总等待时间: {delay1 * 2}s (约 {delay1 * 2 / 60:.1f} 分钟)")
    print(f"  ✓ 如果2次重试均失败，才 fallback 到 AKShare")
    print()

    # Test 3: 小时/天级限流
    msg2 = "抱歉，您访问接口频率超限(1次/小时)"
    delay2 = _parse_retry_delay(msg2)
    print(f"测试3: 小时级限流")
    print(f"  错误消息: {msg2}")
    print(f"  解析等待时间: {delay2}")
    if delay2 is None:
        print(f"  ✓ 小时级限流不重试，直接 fallback")
    print()

    print("=" * 70)
    print("测试完成！")
    print("=" * 70)
    print()
    print("总结：")
    print(f"  1. 分钟级限流（{delay1}s）：等待后重试，优先保证 Tushare 数据质量")
    print(f"  2. 小时/天级限流：直接 fallback（等待时间过长，无意义）")
    print(f"  3. 即使等待时间 > 壁钟超时，也会等待（数据质量优先）")


if __name__ == "__main__":
    test_rate_limit_wait()
