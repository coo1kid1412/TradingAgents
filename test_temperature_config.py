#!/usr/bin/env python3
"""Test script to verify temperature configuration is working correctly."""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradingagents.default_config import DEFAULT_CONFIG

def test_temperature_config():
    """Test that temperature configurations are correctly set."""
    print("=" * 60)
    print("Temperature 配置测试")
    print("=" * 60)
    
    expected_temperatures = {
        "temperature_market": 0.5,
        "temperature_sentiment": 0.5,
        "temperature_news": 0.5,
        "temperature_fundamentals": 0.2,
        "temperature_trader": 0.3,
    }
    
    all_passed = True
    for key, expected_value in expected_temperatures.items():
        actual_value = DEFAULT_CONFIG.get(key)
        status = "✓" if actual_value == expected_value else "✗"
        print(f"{status} {key}: {actual_value} (期望: {expected_value})")
        if actual_value != expected_value:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ 所有 temperature 配置正确！")
    else:
        print("✗ 部分配置不正确，请检查")
        sys.exit(1)
    
    # Test that the configuration can be used
    print("\n测试 LLM 客户端创建...")
    try:
        from tradingagents.llm_clients import create_llm_client
        
        # Test creating a client with temperature
        client = create_llm_client(
            provider="openai",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            temperature=0.3,
        )
        llm = client.get_llm()
        print(f"✓ LLM 客户端创建成功，temperature={llm.temperature}")
        
    except Exception as e:
        print(f"✗ LLM 客户端创建失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_temperature_config()
