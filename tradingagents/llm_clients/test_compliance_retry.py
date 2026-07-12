"""LLM 合规重试的纯函数测试（输入审核净化 + 输入/输出审核检测）。

运行：python tradingagents/llm_clients/test_compliance_retry.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.llm_clients.openai_client import (
    _is_input_sensitive_error,
    _is_output_sensitive_error,
    _sanitize_text_for_compliance,
    _sanitize_text_for_output_compliance,
    _sanitize_input_for_compliance,
    _sanitize_input_for_output_compliance,
)


class _E(Exception):
    pass


def test_input_vs_output_sensitive_detection():
    """1026=输入层、1027=输出层，互斥分流（item7）。"""
    assert _is_input_sensitive_error(_E("422 ... 1026 input new_sensitive")) is True
    assert _is_input_sensitive_error(_E("input new_sensitive")) is True
    # 1026 不能被误判成输出层（否则走措辞补丁→对输入审核无效→死循环失败）
    assert _is_output_sensitive_error(_E("... 422 1026 ...")) is False
    # 1027 输出层
    assert _is_output_sensitive_error(_E("422 1027 new_sensitive")) is True
    assert _is_input_sensitive_error(_E("422 1027")) is False
    # 普通错误两者皆 False
    assert _is_input_sensitive_error(_E("500 timeout")) is False
    assert _is_output_sensitive_error(_E("500 timeout")) is False


def test_sanitize_text_replaces_sensitive_keeps_numbers():
    """敏感地缘政治措辞替换为中性等义词，定量数字/其余措辞不动。"""
    t = _sanitize_text_for_compliance("受芯片战与制裁影响，国产替代加速，营收 +50%")
    assert "芯片战" not in t and "制裁" not in t and "国产替代" not in t
    assert "半导体产业竞争" in t and "出口管制" in t and "国产化" in t
    assert "+50%" in t  # 数字不动
    # 无敏感词 → 原样
    assert _sanitize_text_for_compliance("营收增长 50%，毛利率提升") == "营收增长 50%，毛利率提升"


def test_sanitize_input_preserves_message_structure():
    """消息列表净化：逐条替换 content，结构/条数/角色不变。"""
    msgs = _sanitize_input_for_compliance(
        [SystemMessage(content="分析芯片战格局"), HumanMessage(content="正常内容")])
    assert len(msgs) == 2
    assert msgs[0].content == "分析半导体产业竞争格局"
    assert msgs[1].content == "正常内容"
    # 纯字符串输入
    assert _sanitize_input_for_compliance("脱钩风险") == "供应链调整风险"


def test_output_sanitize_replaces_trading_phrasing():
    """1027 重试：净化容易诱导模型复述的口语化风险措辞。"""
    t = _sanitize_text_for_output_compliance("接飞刀、恐慌抛售、主力出货、death cross")
    assert "接飞刀" not in t
    assert "恐慌抛售" not in t
    assert "主力出货" not in t
    assert "death cross" not in t
    assert "逆势承接风险" in t
    assert "集中卖压释放" in t
    assert "主力减仓" in t


def test_output_sanitize_preserves_message_structure():
    """1027 重试净化消息列表时保持 role/条数不变。"""
    msgs = _sanitize_input_for_output_compliance(
        [SystemMessage(content="避免接飞刀"), HumanMessage(content="正常内容")])
    assert len(msgs) == 2
    assert msgs[0].content == "避免逆势承接风险"
    assert msgs[1].content == "正常内容"


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
