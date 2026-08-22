import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from main import _send_decision_to_feishu_as_file


class _JsonResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_feishu_upload_uses_mobile_report_and_preserves_full_decision():
    responses = iter([
        {"code": 0, "tenant_access_token": "test-token"},
        {"code": 0, "data": {"file_key": "test-file-key"}},
        {"code": 0},
    ])
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return _JsonResponse(next(responses))

    with tempfile.TemporaryDirectory() as directory:
        report_path = Path(directory) / "300502_新易盛_20260821_042408"
        portfolio_path = report_path / "5_portfolio"
        portfolio_path.mkdir(parents=True)
        decision_text = """# 完整 PM 决策

audit-only-content

```yaml
PM_SUMMARY:
  pm_rating: OVERWEIGHT
```
"""
        mobile_text = "# 移动摘要\n\nmobile-only-content\n"
        decision_file = portfolio_path / "decision.md"
        decision_file.write_text(decision_text, encoding="utf-8")
        (report_path / "complete_report.md").write_text(mobile_text, encoding="utf-8")

        credentials = {
            "FEISHU_APP_ID": "test-app",
            "FEISHU_APP_SECRET": "test-secret",
            "FEISHU_USER_OPEN_ID": "test-open-id",
        }
        with patch.dict(os.environ, credentials), patch(
            "urllib.request.urlopen", side_effect=fake_urlopen,
        ):
            _send_decision_to_feishu_as_file(report_path)

        preserved_decision = decision_file.read_text(encoding="utf-8")

    assert len(requests) == 3
    upload_body = requests[1].data
    assert b"mobile-only-content" in upload_body
    assert b"audit-only-content" not in upload_body
    assert preserved_decision == decision_text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
