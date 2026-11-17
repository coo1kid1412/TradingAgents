"""Offline reporting and delivery contract; every HTTP request is intercepted."""

import copy
import hashlib
import html
import importlib
import json
import os
import socket
import traceback
import unittest
from unittest.mock import Mock, patch

import requests


MODULE = "tradingagents.strategies.small_cap.reporting"
WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/test-target"
CREDS = {
    "FEISHU_APP_ID": "app-sensitive",
    "FEISHU_APP_SECRET": "secret-sensitive",
    "FEISHU_USER_OPEN_ID": "open-id-sensitive",
    "FEISHU_MARKET_RISK_WEBHOOK": "https://other.invalid/private-hook",
}


def result_fixture(status="SUCCESS"):
    return {
        "status": status,
        "data_date": "2026-11-17",
        "as_of_at": "2026-11-17T19:31:22+08:00",
        "target_trade_date": "2026-11-18",
        "ranking_complete": True,
        "strategy_mode": "RESEARCH_CANDIDATES",
        "new_position_policy": "NOT_EVALUATED",
        "regime": "UNKNOWN",
        "rebalance_due": False,
        "coverage": {"expected": 5000, "observed": 4999},
        "warnings": ["边界外缺失已核验"],
        "changes": None,
        "top10": [{
            "rank": 1, "ts_code": "600001.SH", "name": "样例股份",
            "industry": "制造业", "industry_basis": "current_observed",
            "total_market_cap_yuan": 1250000000, "close": 12.34,
            "listing_days": 800, "financial_period": "2025-12-31",
            "financial_available_at": "2026-04-30T00:00:00+08:00",
            "profit": 20000000, "revenue": 230000000,
            "evidence": {"annual": "snapshot/income/row-1"},
            "flags": ["涨停，成交受限"],
        }],
    }


def response(payload=None, status=200, json_error=None):
    value = Mock()
    value.status_code = status
    value.json.side_effect = json_error
    value.json.return_value = payload
    return value


class ReportingTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        try:
            self.reporting = importlib.import_module(MODULE)
        except ModuleNotFoundError:
            self.fail("small_cap.reporting has not been implemented")
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.http = Mock()
        self.session_patch = patch.object(self.reporting.requests, "Session", return_value=self.http)
        self.session_patch.start()
        self.addCleanup(self.session_patch.stop)

    def app_delivery(self, result=None, markdown=None):
        os.environ.update(CREDS)
        result = result or result_fixture()
        return self.reporting.send_feishu(
            result, markdown if markdown is not None else self.reporting.render_markdown(result),
            "stable-run-uuid",
        )

    def auth_ok(self):
        return response({"code": 0, "tenant_access_token": "token-sensitive", "expire": 7200})

    def sent_ok(self):
        return response({"code": 0, "data": {"message_id": "om_actual"}})

    def card_from_app(self):
        return json.loads(self.http.post.call_args.kwargs["json"]["content"])

    def test_mobile_report_has_times_evidence_and_no_table_or_orders(self):
        source = result_fixture()
        before = copy.deepcopy(source)
        text = self.reporting.render_markdown(source)
        for expected in ("候选不是买入指令", "2026-11-17", "2026-11-18", "19:31:22",
                         "600001.SH", "样例股份", "12.50", "12.34", "0.20", "2.30",
                         "2025-12-31", "2026-04-30", "800", "current_observed",
                         "snapshot/income/row-1", "成交受限", "UNKNOWN", "首次/不可比较",
                         "周二", "未执行", "开盘前", "550", "375", "自然日"):
            self.assertIn(expected, html.unescape(text))
        self.assertNotIn("|---", text)
        self.assertNotIn("建议持有", text)
        self.assertEqual(source, before)

    def test_official_identity_cross_check_is_disclosed_without_changing_market_data_source(self):
        value = result_fixture()
        value["identity_verification"] = {"official_szse_resolved_count": 1, "scope":"current_forward_as_of_only"}
        text = self.reporting.render_markdown(value)
        self.assertIn("深交所官方", text)
        self.assertIn("行情、市值和财报仍来自 Tushare", text)

    def test_seasonal_target_date_cannot_be_overridden_by_missing_mode(self):
        for target in ("2027-01-04", "2027-04-01"):
            with self.subTest(target=target):
                result = result_fixture()
                result["target_trade_date"] = target
                result["data_date"] = "2026-12-31"
                text = self.reporting.render_markdown(result)
                self.assertIn("仅研究观察", text)
                self.assertIn("不新开股票仓位", text)
                self.assertIn("BLOCK", text)

    def test_insufficient_preserves_actual_count_and_unknown_optional_fields(self):
        result = result_fixture("INSUFFICIENT")
        result["top10"] = [{"rank": 1, "ts_code": "600002.SH", "name": "缺字段样例"}]
        text = self.reporting.render_markdown(result)
        self.assertIn("INSUFFICIENT", text)
        self.assertIn("1 只", text)
        self.assertIn("未知", text)
        self.assertNotIn("2. ", text)
        self.assertNotIn("0.00", text)

    def test_blocked_error_closed_and_incomplete_hide_all_draft_candidates(self):
        for status, complete in (("BLOCKED", False), ("ERROR", True),
                                 ("CLOSED", True), ("SUCCESS", False), ("INSUFFICIENT", False)):
            with self.subTest(status=status, complete=complete):
                result = result_fixture(status)
                result["ranking_complete"] = complete
                result["changes"] = {"added": ["600001.SH 样例股份"]}
                result["warnings"] = ["draft: 600001.SH 样例股份"]
                result["coverage"] = {"draft": result["top10"]}
                text = self.reporting.render_markdown(result)
                self.assertNotIn("600001", text)
                self.assertNotIn("样例股份", text)
                self.assertIn("未发布", text)

    def test_parameters_cannot_inject_markdown_links_mentions_or_new_sections(self):
        result = result_fixture()
        malicious = '[evil](https://evil.invalid)\n# 假指令 <at id="all">all</at> **buy**'
        result["top10"][0]["name"] = malicious
        result["warnings"] = [malicious]
        result["changes"] = {"added": [malicious]}
        text = self.reporting.render_markdown(result)
        for unsafe in ("[evil](", '<at id="all">', "\n# 假指令", "**buy**"):
            self.assertNotIn(unsafe, text)
        self.assertIn("&lt;at", text)

    def test_changes_and_coverage_are_visible_without_inventing_positions(self):
        result = result_fixture()
        result["changes"] = {"added": ["600002.SH"], "removed": ["600003.SH"],
                             "rank_changes": {"600001.SH": {"previous": 2, "current": 1}}}
        text = self.reporting.render_markdown(result)
        for expected in ("600002.SH", "600003.SH", "4999", "5000", "移出不等于卖出",
                         "新增：", "移出：", "排名变化："):
            self.assertIn(expected, text)
        self.assertNotIn("买入前 3", text)

    def test_blocked_report_retains_safe_diagnostic_codes_not_freeform_drafts(self):
        result = result_fixture("BLOCKED")
        result["warnings"] = ["RANKING_INCOMPLETE", {"code": "QUOTE_MISSING", "draft": "600001.SH"},
                              "600001.SH 样例股份 diagnostic draft"]
        text = html.unescape(self.reporting.render_markdown(result))
        self.assertIn("RANKING_INCOMPLETE", text)
        self.assertIn("QUOTE_MISSING", text)
        self.assertNotIn("600001.SH", text)
        self.assertNotIn("样例股份", text)

    def test_selector_revision_and_readable_labels_and_evidence_are_preserved(self):
        result = result_fixture()
        row = result["top10"][0]
        row.update(financial_revision_id="sealed-revision-123", industry_basis="current_snapshot",
                   evidence={"quote_trade_date": "2026-11-17", "observed_at": "2026-11-17T19:30:11+08:00",
                             "financial_revision_id": "sealed-revision-123"}, flags=["limit_up_at_close"])
        text = html.unescape(self.reporting.render_markdown(result))
        for expected in ("版本：sealed-revision-123", "本次采集口径", "未知/未评估",
                         "行情日期：2026-11-17", "采集时点：2026-11-17T19:30:11+08:00", "涨停"):
            self.assertIn(expected, text)
        self.assertNotIn('"quote_trade_date"', text)
        row["flags"] = []
        self.assertIn("未提供成交限制标记", self.reporting.render_markdown(result))

    def test_coverage_is_small_count_summary_not_a_blob(self):
        result = result_fixture()
        result["coverage"] = {"complete": True, "expected": 5000, "observed": 4999,
                              "unknown_codes": ["600002.SH", "600003.SH"],
                              "missing_fields": {"600002.SH": ["close"]},
                              "endpoints": {"large-internal-blob": "z" * 4000}}
        text = self.reporting.render_markdown(result)
        for expected in ("覆盖完整：是", "应覆盖：5000", "已覆盖：4999", "未知证券：2", "缺字段证券：1"):
            self.assertIn(expected, text)
        self.assertNotIn("large-internal-blob", text)
        self.assertNotIn("600002.SH", text)

    def test_blocked_lowercase_codes_are_allowlisted_without_identifier_suffixes(self):
        result = result_fixture("BLOCKED")
        result["warnings"] = ["ranking_incomplete", "snapshot_calendar_or_point_in_time_invalid",
                              "duplicate_security:600001.SH", "arbitrary_draft_candidate"]
        text = html.unescape(self.reporting.render_markdown(result))
        for expected in ("ranking_incomplete", "snapshot_calendar_or_point_in_time_invalid", "duplicate_security"):
            self.assertIn(expected, text)
        self.assertNotIn("arbitrary_draft_candidate", text)
        self.assertNotIn("600001.SH", text)

    def test_app_delivery_uses_explicit_user_card_uuid_and_actual_message_id(self):
        self.http.post.side_effect = [self.auth_ok(), self.sent_ok()]
        sent = self.app_delivery()
        self.assertEqual(sent, {"status": "sent", "message_id": "om_actual",
                                "recipient_hash": hashlib.sha256(b"open-id-sensitive").hexdigest()})
        auth, delivery = self.http.post.call_args_list
        self.assertEqual(auth.args[0], "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal")
        self.assertEqual(auth.kwargs["json"], {"app_id": "app-sensitive", "app_secret": "secret-sensitive"})
        self.assertEqual(delivery.args[0], "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id")
        body = delivery.kwargs["json"]
        self.assertEqual(body["receive_id"], "open-id-sensitive")
        self.assertEqual(body["uuid"], "stable-run-uuid")
        self.assertEqual(body["msg_type"], "interactive")
        self.assertEqual(delivery.kwargs["headers"]["Authorization"], "Bearer token-sensitive")
        self.assertFalse(delivery.kwargs["allow_redirects"])
        self.assertTrue(delivery.kwargs["timeout"])
        self.assertTrue(any(element["tag"] == "markdown" for element in self.card_from_app()["elements"]))

    def test_small_cap_user_override_and_hash_ignore_rotated_credentials(self):
        os.environ.update(CREDS, FEISHU_SMALL_CAP_USER_OPEN_ID="small-user")
        expected = hashlib.sha256(b"small-user").hexdigest()
        self.assertEqual(self.reporting.recipient_identity(), expected)
        os.environ["FEISHU_APP_SECRET"] = "rotated"
        self.assertEqual(self.reporting.recipient_identity(), expected)
        self.http.post.side_effect = [self.auth_ok(), self.sent_ok()]
        sent = self.app_delivery()
        self.assertEqual(sent["recipient_hash"], expected)
        self.assertEqual(self.http.post.call_args.kwargs["json"]["receive_id"], "small-user")

    def test_explicit_webhook_wins_and_ack_does_not_fabricate_message_id(self):
        os.environ.update(CREDS, FEISHU_SMALL_CAP_WEBHOOK=WEBHOOK)
        self.http.post.return_value = response({"StatusCode": 0, "StatusMessage": "success"})
        sent = self.reporting.send_feishu(result_fixture(), "report", "stable-run-uuid")
        self.assertEqual(self.http.post.call_count, 1)
        self.assertEqual(self.http.post.call_args.args[0], WEBHOOK)
        self.assertEqual(sent["recipient_hash"], hashlib.sha256(WEBHOOK.encode()).hexdigest())
        self.assertIsNone(sent["message_id"])
        self.assertNotIn("Authorization", self.http.post.call_args.kwargs.get("headers", {}))

    def test_webhook_checks_all_business_code_variants(self):
        os.environ["FEISHU_SMALL_CAP_WEBHOOK"] = WEBHOOK
        for payload in ({"code": 123, "msg": "secret-sensitive"},
                        {"status_code": 12}, {"StatusCode": 12},
                        {"code": 0, "status_code": 12}, {"code": 0, "StatusCode": 12}):
            with self.subTest(payload=payload):
                self.http.post.return_value = response(payload)
                with self.assertRaises(self.reporting.DeliveryError) as caught:
                    self.reporting.send_feishu(result_fixture(), "report", "stable-run-uuid")
                self.assertNotIsInstance(caught.exception, self.reporting.DeliveryUnknown)
                self.assertNotIn("secret-sensitive", str(caught.exception))
        for payload in ({"code": 0}, {"status_code": 0}, {"StatusCode": 0}):
            self.http.post.return_value = response(payload)
            self.assertEqual(self.reporting.send_feishu(result_fixture(), "report", "id")["status"], "sent")

    def test_missing_credentials_and_market_risk_only_never_send(self):
        for env in ({}, {"FEISHU_MARKET_RISK_WEBHOOK": "https://other.invalid/hook"},
                    {"FEISHU_APP_ID": "app", "FEISHU_USER_OPEN_ID": "user"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(self.reporting.DeliveryError):
                    self.reporting.send_feishu(result_fixture(), "report", "uuid")
                with self.assertRaises(self.reporting.DeliveryError):
                    self.reporting.recipient_identity()
        self.http.post.assert_not_called()

    def test_auth_http_business_and_malformed_errors_stop_before_send(self):
        for reply in (response({"code": 400, "msg": "secret-sensitive"}),
                      response({}, status=401), response({"code": 0}),
                      response(None, json_error=ValueError("secret-sensitive")),
                      response({"code": 0, "tenant_access_token": None})):
            with self.subTest(reply=reply):
                self.http.post.reset_mock(side_effect=True)
                self.http.post.return_value = reply
                with self.assertRaises(self.reporting.DeliveryError) as caught:
                    self.app_delivery()
                self.assertNotIsInstance(caught.exception, self.reporting.DeliveryUnknown)
                self.assertEqual(self.http.post.call_count, 1)

    def test_send_business_and_client_http_rejections_are_definite_failures(self):
        for reply in (response({"code": 230001, "msg": "token-sensitive"}),
                      response({}, status=400), response({}, status=429)):
            self.http.post.side_effect = [self.auth_ok(), reply]
            with self.assertRaises(self.reporting.DeliveryError) as caught:
                self.app_delivery()
            self.assertNotIsInstance(caught.exception, self.reporting.DeliveryUnknown)

    def test_send_timeouts_invalid_response_and_missing_id_are_unknown_no_retry(self):
        for outcome in (requests.Timeout("secret-sensitive"), requests.ConnectionError("secret-sensitive"),
                        response(None, json_error=ValueError("secret-sensitive")),
                        response({"code": 0, "data": {}}), response({}), response([]),
                        response({}, status=503), response({}, status=302)):
            with self.subTest(outcome=outcome):
                self.http.post.reset_mock(side_effect=True)
                self.http.post.side_effect = [self.auth_ok(), outcome]
                with self.assertRaises(self.reporting.DeliveryUnknown):
                    self.app_delivery()
                self.assertEqual(self.http.post.call_count, 2)

    def test_sensitive_exceptions_are_redacted_in_traceback_and_output(self):
        self.http.post.side_effect = [self.auth_ok(), requests.Timeout("secret-sensitive token-sensitive " + WEBHOOK)]
        with patch("sys.stdout") as stdout, patch("sys.stderr") as stderr:
            try:
                self.app_delivery()
            except self.reporting.DeliveryUnknown as error:
                rendered = "".join(traceback.format_exception(error))
                self.assertTrue(error.code)
            else:
                self.fail("uncertain delivery must not be reported as sent")
        for secret in ("secret-sensitive", "token-sensitive", "open-id-sensitive", WEBHOOK):
            self.assertNotIn(secret, rendered)
        stdout.write.assert_not_called()
        stderr.write.assert_not_called()

    def test_blocked_send_ignores_supplied_draft_markdown(self):
        for status in ("BLOCKED", "ERROR"):
            self.http.post.side_effect = [self.auth_ok(), self.sent_ok()]
            self.app_delivery(result_fixture(status), "正式Top10: 600001.SH 样例股份")
            card = json.dumps(self.card_from_app(), ensure_ascii=False)
            self.assertNotIn("600001.SH", card)
            self.assertNotIn("样例股份", card)
            self.assertIn(status, card)

    def test_long_report_is_segmented_bounded_and_explicitly_truncated_in_one_message(self):
        self.http.post.side_effect = [self.auth_ok(), self.sent_ok()]
        self.app_delivery(markdown="候选不是买入指令\n\n" + ('长文本\\"' * 20000))
        card = self.card_from_app()
        serialized = json.dumps(card, ensure_ascii=False).encode()
        self.assertLessEqual(len(serialized), 28000)
        self.assertGreater(len(card["elements"]), 1)
        self.assertLessEqual(len(card["elements"]), 40)
        for element in card["elements"]:
            self.assertLessEqual(len(element["content"].encode()), 4000)
        self.assertIn("截断", serialized.decode())
        self.assertEqual(self.http.post.call_count, 2)

    def test_invalid_uuid_does_not_send(self):
        os.environ.update(CREDS)
        for value in ("", "x" * 51, "id\nwith-injection"):
            with self.assertRaises(self.reporting.DeliveryError):
                self.reporting.send_feishu(result_fixture(), "report", value)
        self.http.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
