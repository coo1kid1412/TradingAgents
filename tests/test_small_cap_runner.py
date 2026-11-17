"""Offline archive, execution and delivery contracts."""

import importlib
import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from test_small_cap_selector import snapshot


class RunnerTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "tradingagents/strategies/small_cap/runner.py"
        self.assertTrue(path.exists(), "small-cap runner is not implemented")
        self.module = importlib.import_module("tradingagents.strategies.small_cap.runner")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = Mock()
        self.source.collect.return_value = snapshot()
        self.sender = Mock(return_value={"status": "sent", "message_id": "test-message",
                                         "recipient_hash": "test-recipient"})

    def run_daily(self, **kwargs):
        return self.module.run_daily(root=self.root, source=self.source, sender=self.sender,
                                     recipient="test-recipient", clock=lambda: datetime.fromisoformat(
                                         "2026-11-17T19:41:00+08:00"), **kwargs)

    def test_run_archives_snapshot_result_and_mobile_markdown_without_send(self):
        result = self.run_daily(notify=False, data_date="2026-11-17")
        self.assertEqual(result["status"], "SUCCESS")
        self.sender.assert_not_called()
        folder = Path(result["report_dir"])
        self.assertEqual(json.loads((folder / "result.json").read_text())["selection_hash"], result["selection_hash"])
        self.assertIn("候选不是买入指令", (folder / "report.md").read_text())
        self.assertTrue(Path(result["snapshot_path"]).is_file())

    def test_idempotent_same_day_does_not_recollect_or_redeliver(self):
        first = self.run_daily(notify=True, data_date="2026-11-17")
        second = self.run_daily(notify=True, data_date="2026-11-17")
        self.assertEqual(first["selection_hash"], second["selection_hash"])
        self.assertEqual(self.source.collect.call_count, 1)
        self.assertEqual(self.sender.call_count, 1)
        self.assertEqual(second["delivery_status"], "sent")

    def test_no_notify_followed_by_notify_reuses_sealed_inputs(self):
        self.run_daily(notify=False, data_date="2026-11-17")
        self.run_daily(notify=True, data_date="2026-11-17")
        self.assertEqual(self.source.collect.call_count, 1)
        self.assertEqual(self.sender.call_count, 1)

    def test_closed_day_is_not_sent_or_misrepresented_as_new_selection(self):
        self.source.collect.return_value = {"status": "CLOSED", "data_date": "2026-11-21"}
        result = self.run_daily(notify=True, data_date="2026-11-21")
        self.assertEqual(result["status"], "CLOSED")
        self.sender.assert_not_called()

    def test_blocked_report_sends_honest_error_without_draft_candidates(self):
        self.source.collect.return_value["coverage"]["complete"] = False
        result = self.run_daily(notify=True, data_date="2026-11-17")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["top10"], [])
        self.assertEqual(self.sender.call_count, 1)

    def test_ambiguous_delivery_is_not_automatically_retried(self):
        from tradingagents.strategies.small_cap.reporting import DeliveryUnknown
        self.sender.side_effect = DeliveryUnknown("SEND_TRANSPORT_ERROR")
        first = self.run_daily(notify=True, data_date="2026-11-17")
        second = self.run_daily(notify=True, data_date="2026-11-17")
        self.assertEqual(first["delivery_status"], "unknown")
        self.assertEqual(second["delivery_status"], "unknown")
        self.assertEqual(self.sender.call_count, 1)

    def test_definite_failure_can_retry_same_uuid_without_reselection(self):
        from tradingagents.strategies.small_cap.reporting import DeliveryError
        self.sender.side_effect = [DeliveryError("AUTH_BUSINESS_ERROR"), self.sender.return_value]
        first = self.run_daily(notify=True, data_date="2026-11-17")
        second = self.run_daily(notify=True, data_date="2026-11-17")
        self.assertEqual(first["delivery_status"], "failed")
        self.assertEqual(second["delivery_status"], "sent")
        self.assertEqual(self.source.collect.call_count, 1)
        self.assertEqual(self.sender.call_args_list[0].args[2], self.sender.call_args_list[1].args[2])

    def test_snapshot_replay_is_offline_and_identical_and_future_asof_cannot_be_invented(self):
        first = self.run_daily(notify=False, data_date="2026-11-17")
        before = Path(first["snapshot_path"]).read_bytes()
        result = self.module.replay_snapshot(first["snapshot_path"])
        self.assertEqual(result["selection_hash"], first["selection_hash"])
        self.assertEqual(Path(first["snapshot_path"]).read_bytes(), before)
        with self.assertRaises(ValueError):
            self.module.replay_snapshot(first["snapshot_path"], as_of="2026-11-18T19:40:00+08:00")

    def test_snapshot_tampering_detected_on_cached_run(self):
        first = self.run_daily(notify=False, data_date="2026-11-17")
        Path(first["snapshot_path"]).write_text('{}')
        with self.assertRaises(ValueError):
            self.run_daily(notify=False, data_date="2026-11-17")

    def test_concurrent_run_fails_without_collecting(self):
        with self.module.run_lock(self.root):
            with self.assertRaises(self.module.RunBusy):
                self.run_daily(notify=False, data_date="2026-11-17")
        self.source.collect.assert_not_called()

    def test_result_contains_auditable_provenance_and_rule_defaults(self):
        result = self.run_daily(notify=False, data_date="2026-11-17")
        for key in ("mode", "generated_at", "code_version", "rule_assumptions", "exclusions_ref"):
            self.assertTrue(result.get(key), key)
        row = result["top10"][0]
        self.assertEqual(row["exchange"], "SSE")
        self.assertEqual(row["board"], "主板")
        self.assertTrue(all(value == "PASS" for value in row["rule_checks"].values()))

    def test_refresh_identical_snapshot_cannot_bypass_sent_or_unknown_dedup(self):
        from tradingagents.strategies.small_cap.reporting import DeliveryUnknown
        for outcome in (self.sender.return_value, DeliveryUnknown("SEND_TRANSPORT_ERROR")):
            with tempfile.TemporaryDirectory() as root:
                sender = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
                for refresh in (False, True):
                    self.module.run_daily(root=root, source=self.source, sender=sender, recipient="test-recipient",
                        data_date="2026-11-17", notify=True, refresh=refresh,
                        clock=lambda: datetime.fromisoformat("2026-11-17T19:41:00+08:00"))
                self.assertEqual(sender.call_count, 1)

    def test_replay_checks_content_addressed_snapshot_hash(self):
        first = self.run_daily(notify=False, data_date="2026-11-17")
        value = json.loads(Path(first["snapshot_path"]).read_text())
        value["records"][0]["close"] = 1
        Path(first["snapshot_path"]).write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.module.replay_snapshot(first["snapshot_path"])

    def test_delivery_rechecks_current_time_after_slow_collection(self):
        ticks = iter([datetime.fromisoformat("2026-11-17T23:59:00+08:00"),
                      datetime.fromisoformat("2026-11-18T00:01:00+08:00")])
        result = self.module.run_daily(root=self.root, source=self.source, sender=self.sender,
                    recipient="test-recipient", data_date="2026-11-17", notify=True, clock=lambda: next(ticks))
        self.assertEqual(result["delivery_status"], "expired")
        self.sender.assert_not_called()

    def test_failed_report_bundle_is_not_visible_as_published_success(self):
        original = self.module._write_new
        def fail_markdown(path, value):
            if Path(path).name == "report.md":
                raise OSError("disk error")
            original(path, value)
        with patch.object(self.module, "_write_new", side_effect=fail_markdown):
            with self.assertRaises(OSError):
                self.run_daily(notify=False, data_date="2026-11-17")
        self.assertEqual(list((self.root / "reports").rglob("result.json")), [])

    def test_recovered_collection_is_not_stuck_on_failed_daily_cache(self):
        self.source.collect.side_effect = [RuntimeError("transport"), snapshot()]
        first = self.run_daily(notify=False, data_date="2026-11-17")
        second = self.run_daily(notify=False, data_date="2026-11-17")
        self.assertEqual(first["status"], "ERROR")
        self.assertEqual(second["status"], "SUCCESS")

    def test_replay_output_bundle_is_atomic_and_can_retry_after_write_failure(self):
        first = self.run_daily(notify=False, data_date="2026-11-17")
        output = self.root / "replay-output"
        original = self.module._write_new
        def fail_markdown(path, value):
            if Path(path).name == "report.md":
                raise OSError("disk error")
            original(path, value)
        with patch.object(self.module, "_write_new", side_effect=fail_markdown):
            with self.assertRaises(OSError):
                self.module.replay_snapshot(first["snapshot_path"], output_dir=output)
        self.assertFalse(output.exists())
        self.module.replay_snapshot(first["snapshot_path"], output_dir=output)
        self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
