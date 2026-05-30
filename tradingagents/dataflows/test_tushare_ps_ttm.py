"""测试 tushare_vendor.py 中系统计算 PS(TTM) 的功能。"""

import pandas as pd
from tradingagents.dataflows.tushare_vendor import _compute_ttm_revenue_per_share_fina


def test_standard_quarterly_data():
    """测试标准季度数据计算 TTM 每股营业收入。"""
    # 模拟 Tushare fina_indicator 数据格式
    data = {
        "end_date": ["20231231", "20240331", "20240630", "20240930", "20241231", "20250331"],
        "tob_operate_income": [1000, 300, 700, 1100, 1200, 350],  # 累计值（亿元）
    }
    fina_df = pd.DataFrame(data)
    total_shares = 100  # 100亿股

    result = _compute_ttm_revenue_per_share_fina(fina_df, total_shares)

    # TTM = 2024年报 - 2024Q1 + 2025Q1 = 1200 - 300 + 350 = 1250
    # 每股 = 1250 / 100 = 12.5
    expected = 12.5
    assert result is not None, "计算结果不应为 None"
    assert abs(result - expected) < 0.01, f"期望 {expected}, 实际 {result}"
    print(f"✓ 标准季度数据测试通过: TTM每股营业收入 = {result} 元")


def test_missing_revenue_column():
    """测试缺少营收字段时返回 None。"""
    data = {
        "end_date": ["20231231", "20240331"],
        "eps": [1.5, 0.4],
    }
    fina_df = pd.DataFrame(data)
    total_shares = 100

    result = _compute_ttm_revenue_per_share_fina(fina_df, total_shares)
    assert result is None, "缺少营收字段时应返回 None"
    print("✓ 缺少营收字段测试通过")


def test_empty_dataframe():
    """测试空 DataFrame 返回 None。"""
    fina_df = pd.DataFrame()
    total_shares = 100

    result = _compute_ttm_revenue_per_share_fina(fina_df, total_shares)
    assert result is None, "空 DataFrame 应返回 None"
    print("✓ 空 DataFrame 测试通过")


def test_invalid_shares():
    """测试无效总股本返回 None。"""
    data = {
        "end_date": ["20231231", "20240331"],
        "tob_operate_income": [1000, 300],
    }
    fina_df = pd.DataFrame(data)

    # 测试 None
    result = _compute_ttm_revenue_per_share_fina(fina_df, None)
    assert result is None, "总股本为 None 应返回 None"

    # 测试 0
    result = _compute_ttm_revenue_per_share_fina(fina_df, 0)
    assert result is None, "总股本为 0 应返回 None"

    # 测试负数
    result = _compute_ttm_revenue_per_share_fina(fina_df, -100)
    assert result is None, "总股本为负数应返回 None"

    print("✓ 无效总股本测试通过")


def test_negative_revenue():
    """测试负数营收时返回 None。"""
    data = {
        "end_date": ["20231231", "20240331"],
        "tob_operate_income": [-1000, -300],
    }
    fina_df = pd.DataFrame(data)
    total_shares = 100

    result = _compute_ttm_revenue_per_share_fina(fina_df, total_shares)
    assert result is None, "负数营收应返回 None"
    print("✓ 负数营收测试通过")


def test_601138_scenario():
    """测试工业富联真实场景（2025年数据）。"""
    # 工业富联 2024年报营收约 6200亿元, 2025Q1约 1500亿元
    data = {
        "end_date": ["20231231", "20240331", "20240630", "20240930", "20241231", "20250331"],
        "tob_operate_income": [5000, 1400, 3000, 4700, 6200, 1500],  # 单位：亿元
    }
    fina_df = pd.DataFrame(data)

    # 工业富联总股本约 199.46亿股
    total_shares = 199.46

    result = _compute_ttm_revenue_per_share_fina(fina_df, total_shares)

    # TTM = 6200 - 1400 + 1500 = 6300 亿元
    # 每股 = 6300 / 199.46 ≈ 31.58 元
    expected = 6300 / 199.46

    assert result is not None, "计算结果不应为 None"
    assert abs(result - expected) < 0.1, f"期望约 {expected}, 实际 {result}"
    print(f"✓ 工业富联场景测试通过: TTM每股营业收入 ≈ {result:.2f} 元")


def run_all_tests():
    """运行所有测试。"""
    print("=" * 60)
    print("测试系统计算 PS(TTM) 功能")
    print("=" * 60)

    test_standard_quarterly_data()
    test_missing_revenue_column()
    test_empty_dataframe()
    test_invalid_shares()
    test_negative_revenue()
    test_601138_scenario()

    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
