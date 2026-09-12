"""Offline regression tests for small-sample quality reviews."""

import datetime as dt
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tradingagents.harness import db, weekly_review


class QualityReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = db.init_db(self.root / "harness.db")
        self.today = dt.date(2026, 9, 12)
        self.output_capture = redirect_stdout(io.StringIO())
        self.output_capture.__enter__()
        self.addCleanup(self.output_capture.__exit__, None, None, None)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def run_row(self, ident=1, ticker="300308", day="2026-09-09", action="WAIT", size=0):
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO runs (id,ticker,trade_date,report_timestamp,report_window,"
                "report_dir,git_commit) VALUES (?,?,?,?,?,?,?)",
                (ident, ticker, day, day + "T10:00:00", "morning", f"reports/run-{ident}", "abc123"),
            )
            conn.execute(
                "INSERT INTO predictions (run_id,current_price,pm_rating,pm_action_keyword,"
                "pm_size_low_pct,pm_size_high_pct,rm_yaml_parsed,pm_yaml_parsed,short_term_trend) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ident, 100, "OVERWEIGHT", action, size, size, 1, 1, "数据不足"),
            )

    def outcome(self, ident=1, horizon="T", status="fetched", target="2026-09-10", benchmark="510300"):
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO outcomes (run_id,horizon,reference_price,fetch_status,target_date,"
                "fetched_at,direction_predicted,realized_return_pct,benchmark_ticker) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ident, horizon, 100, status, target, "2026-09-10 15:00:00", "long", 3, benchmark),
            )

    def snapshot(self):
        from tradingagents.harness import quality_review
        return quality_review.build_snapshot(self.path, today=self.today)

    def test_weekly_entry_uses_quality_counts_not_horizon_sum(self):
        self.run_row()
        self.run_row(2)
        self.outcome()
        self.outcome(horizon="T+1")
        self.outcome(2)
        result = weekly_review.extract_summary(db_path=self.path, today=self.today)
        self.assertEqual(result["counts"]["reports"], 2)
        self.assertEqual(result["counts"]["ticker_dates"], 1)
        self.assertEqual(result["counts"]["observations"], 3)

    def test_missing_database_is_not_created_or_reported_healthy(self):
        from tradingagents.harness import quality_review
        missing = self.root / "missing.db"
        with self.assertRaises((FileNotFoundError, sqlite3.OperationalError)):
            quality_review.build_snapshot(missing, today=self.today)
        self.assertFalse(missing.exists())

    def test_priced_report_without_tracking_is_not_healthy(self):
        self.run_row()
        snapshot = self.snapshot()
        self.assertFalse(snapshot["data_healthy"])
        self.assertIn("untracked_report", {i["id"] for i in snapshot["issues"]})

    def test_unknown_short_term_text_is_not_an_explicit_forecast(self):
        self.run_row()
        with self.connection() as conn:
            conn.execute("UPDATE predictions SET short_term_trend='数据不足（置信度中）'")
        self.assertEqual(self.snapshot()["counts"]["short_term_views"], 0)

    def test_future_result_is_not_counted_as_completed(self):
        self.run_row()
        self.outcome(target="2026-09-15")
        snapshot = self.snapshot()
        self.assertEqual(snapshot["counts"]["observations"], 0)
        self.assertEqual(snapshot["tracking"]["invalid"], 1)

    def test_empty_database_cannot_be_called_healthy(self):
        self.assertFalse(self.snapshot()["data_healthy"])

    def test_evaluated_stock_dates_exclude_failed_unscored_archives(self):
        self.run_row()
        self.run_row(2, ticker="000001")
        self.outcome()
        self.assertEqual(self.snapshot()["counts"]["evaluated_ticker_dates"], 1)

    def test_read_only_and_no_invented_pnl_or_short_term_prediction(self):
        self.run_row()
        self.outcome()
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = self.snapshot()
        self.assertEqual(result["counts"]["short_term_views"], 0)
        self.assertEqual(result["counts"]["wait_reports"], 1)
        self.assertNotIn("pnl", result)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        self.assertIn("wait_scored_as_position", {i["id"] for i in result["issues"]})

    def test_wait_with_positive_new_position_is_a_contract_issue(self):
        self.run_row(size=5)
        issue = next(i for i in self.snapshot()["issues"] if i["id"] == "wait_position_conflict")
        self.assertEqual(issue["run_ids"], [1])
        self.assertIn("验收", issue["acceptance"])

    def test_seven_day_window_excludes_old_and_future_reports(self):
        for ident, day in enumerate(["2026-09-05", "2026-09-06", "2026-09-12", "2026-09-13"], 1):
            self.run_row(ident, day=day)
        result = self.snapshot()
        self.assertEqual(result["counts"]["reports"], 3)
        self.assertEqual(result["counts"]["new_reports"], 2)

    def test_sqlite_utc_fetched_time_uses_shanghai_week(self):
        self.run_row(day="2026-08-20")
        self.outcome()
        with self.connection() as conn:
            conn.execute("UPDATE outcomes SET fetched_at='2026-09-05 16:30:00',target_date='2026-09-04'")
        self.assertEqual(self.snapshot()["counts"]["new_observations"], 1)

    def test_unknown_due_date_does_not_mean_healthy_not_due(self):
        self.run_row()
        self.outcome(status="not_due", target=None)
        result = self.snapshot()
        self.assertEqual(result["tracking"]["due_unknown"], 1)
        self.assertEqual(result["tracking"]["not_due"], 0)
        self.assertFalse(result["data_healthy"])

    def test_overdue_today_and_future_are_distinct(self):
        self.run_row()
        self.outcome(status="not_due", target="2026-09-11")
        self.outcome(horizon="T+1", status="pending", target="2026-09-12")
        self.outcome(horizon="T+5", status="not_due", target="2026-09-15")
        tracking = self.snapshot()["tracking"]
        self.assertEqual(tracking["overdue"], 1)
        self.assertEqual(tracking["due_today"], 1)
        self.assertEqual(tracking["not_due"], 1)

    def test_us_benchmark_warning_does_not_change_a_share_records(self):
        self.run_row()
        self.run_row(2, ticker="NVDA")
        self.outcome()
        self.outcome(2)
        issue = next(i for i in self.snapshot()["issues"] if i["id"] == "foreign_benchmark")
        self.assertEqual(issue["run_ids"], [2])

    def test_old_archive_failure_is_not_a_new_runtime_failure(self):
        self.run_row(day="2026-05-20")
        with self.connection() as conn:
            conn.execute("UPDATE runs SET archive_status='failed'")
        result = self.snapshot()
        self.assertEqual(result["counts"]["new_archive_failures"], 0)
        self.assertEqual(result["counts"]["archive_failures"], 1)

    def test_evidence_diff_ignores_report_date_and_fetch_timestamp_only(self):
        from tradingagents.harness import quality_review
        self.run_row()
        self.outcome()
        previous = self.snapshot()
        self.today += dt.timedelta(days=1)
        with self.connection() as conn:
            conn.execute("UPDATE outcomes SET fetched_at='2026-09-12 01:00:00'")
        result = quality_review.compare_snapshot(self.snapshot(), previous)
        self.assertFalse(result["review_required"])

    def test_changed_price_and_new_issue_require_review(self):
        from tradingagents.harness import quality_review
        self.run_row()
        previous = self.snapshot()
        with self.connection() as conn:
            conn.execute("UPDATE predictions SET current_price=NULL")
        delta = quality_review.compare_snapshot(self.snapshot(), previous)
        self.assertTrue(delta["review_required"])
        self.assertIn("missing_price", delta["changed_issue_ids"])

    def test_resolved_issue_is_reported(self):
        from tradingagents.harness import quality_review
        self.run_row(size=5)
        previous = self.snapshot()
        with self.connection() as conn:
            conn.execute("UPDATE predictions SET pm_size_low_pct=0,pm_size_high_pct=0")
        delta = quality_review.compare_snapshot(self.snapshot(), previous)
        self.assertIn("wait_position_conflict", delta["resolved_issue_ids"])

    def test_mobile_report_avoids_legacy_win_rates_and_threshold_advice(self):
        self.run_row()
        result = self.snapshot()
        message = weekly_review.build_feishu_message(True, "任务日志正常", result)
        self.assertIn("质量复盘", message)
        self.assertIn("不自动调参", message)
        self.assertNotIn("|---", message)
        self.assertNotIn("命中率：", message)
        self.assertNotIn("本周改哪些", message)

    def test_dry_run_never_notifies_or_advances_delivered_state(self):
        self.run_row()
        output = self.root / "reviews"
        args = ["--no-notify", "--db-path", str(self.path), "--output-dir", str(output), "--date", "2026-09-12"]
        with patch.object(weekly_review, "send_feishu") as send:
            self.assertEqual(weekly_review.main(args), 0)
        send.assert_not_called()
        self.assertTrue((output / "quality_2026-09-12.md").exists())
        self.assertFalse((output / "quality_last_delivered.json").exists())

    def test_delivery_failure_preserves_change_baseline(self):
        self.run_row()
        output = self.root / "reviews"
        args = ["--db-path", str(self.path), "--output-dir", str(output), "--date", "2026-09-12"]
        with patch.object(weekly_review, "send_feishu", return_value=(False, "offline")):
            self.assertEqual(weekly_review.main(args), 1)
        self.assertFalse((output / "quality_last_delivered.json").exists())
        with patch.object(weekly_review, "send_feishu", return_value=(True, "ok")):
            self.assertEqual(weekly_review.main(args), 0)
        self.assertEqual(json.loads((output / "quality_last_delivered.json").read_text())["counts"]["reports"], 1)

    def test_repeated_delivery_reports_no_new_evidence(self):
        self.run_row()
        output = self.root / "reviews"
        args = ["--db-path", str(self.path), "--output-dir", str(output), "--date", "2026-09-12"]
        with patch.object(weekly_review, "send_feishu", return_value=(True, "ok")):
            self.assertEqual(weekly_review.main(args), 0)
            self.assertEqual(weekly_review.main(args), 0)
        saved = json.loads((output / "quality_last_delivered.json").read_text())
        self.assertFalse(saved["delta"]["review_required"])

    def test_malformed_baseline_does_not_block_review(self):
        self.run_row()
        output = self.root / "reviews"
        output.mkdir()
        state = output / "quality_last_delivered.json"
        state.write_text(json.dumps({"schema_version": "quality-v1", "evidence": "broken"}))
        args = ["--no-notify", "--db-path", str(self.path), "--output-dir", str(output), "--date", "2026-09-12"]
        self.assertEqual(weekly_review.main(args), 0)

    def test_manual_legacy_report_explicitly_disclaims_real_pnl(self):
        from tradingagents.harness import review
        insights = review.generate_insights(snapshot_date="2026-09-12", db_path=self.path)
        report = review.render_markdown(insights, db_path=self.path)
        self.assertIn("不是实际交易收益", report)
        self.assertNotIn("显著样本量", report)
        self.assertNotIn("未发现显著偏差", report)


if __name__ == "__main__":
    unittest.main()
