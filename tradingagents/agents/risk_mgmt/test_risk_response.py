from tradingagents.agents.risk_mgmt.risk_response import invoke_risk_response


class _Reply:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Reply(reply)


def test_truncated_risk_response_retries_for_structured_tail():
    llm = _FakeLLM([
        "正文在表格中被截断",
        """```yaml
RISK_VIEW:
  role: event
  severity: high
  cap_pct: 3
  cap_basis: RM 方案中的官方事件窗口
  evidence_ids: [RM-PLAN]
  data_supported: true
```""",
    ])

    response = invoke_risk_response(llm, "原始提示", role="event")

    assert len(llm.calls) == 2
    assert "RISK_VIEW:" in response.content
    assert "正文在表格中被截断" in response.content


def test_missing_risk_view_fails_closed_after_one_retry():
    llm = _FakeLLM(["第一次截断", "第二次仍然截断"])

    response = invoke_risk_response(llm, "原始提示", role="tail")

    assert len(llm.calls) == 2
    assert "role: tail" in response.content
    assert "severity: unknown" in response.content
    assert "cap_pct: null" in response.content
    assert "data_supported: false" in response.content


def test_unclosed_first_yaml_fence_cannot_swallow_valid_retry_tail():
    llm = _FakeLLM([
        "正文\n```yaml\nRISK_VIEW:\n  role: event",
        """```yaml
RISK_VIEW:
  role: event
  severity: medium
  cap_pct: 4
  cap_basis: 官方事件证据
  evidence_ids: [NEWS-CAT-01]
  data_supported: true
```""",
    ])
    response = invoke_risk_response(llm, "原始提示", role="event")
    assert "cap_pct: 4" in response.content
    assert response.content.count("```yaml") == 1


def test_retry_exception_returns_fail_closed_view_instead_of_aborting():
    llm = _FakeLLM(["第一次截断", RuntimeError("provider timeout")])
    response = invoke_risk_response(llm, "原始提示", role="liquidity")
    assert "role: liquidity" in response.content
    assert "severity: unknown" in response.content
    assert "data_supported: false" in response.content


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
