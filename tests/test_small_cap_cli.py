"""The independent entry point never imports the stock-analysis graph."""

import contextlib
import importlib
import io
import unittest
from unittest.mock import patch


class CliTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("tradingagents.strategies.small_cap.__main__")

    def test_default_run_is_preview_and_notify_is_explicit(self):
        for args, notify in ((["run"], False), (["run", "--no-notify"], False), (["run", "--notify"], True)):
            with patch.object(self.module, "run_daily", return_value={"status": "SUCCESS", "top10": []}) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.module.main(args), 0)
                self.assertIs(run.call_args.kwargs["notify"], notify)

    def test_replay_routes_to_offline_path_only(self):
        with patch.object(self.module, "replay_snapshot", return_value={"status": "BLOCKED"}) as replay:
            with patch.object(self.module, "run_daily", side_effect=AssertionError("must not run")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(self.module.main(["replay", "--snapshot", "/tmp/missing.json"]), 2)
            self.assertEqual(replay.call_count, 1)

    def test_transport_error_reports_safe_diagnostic_not_secret(self):
        from tradingagents.strategies.small_cap.data_source import DataSourceError
        error = DataSourceError("PERMISSION_DENIED", "secret-must-not-appear", endpoint="income_vip")
        with patch.object(self.module, "run_daily", side_effect=error):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(self.module.main(["run"]), 2)
        self.assertIn("PERMISSION_DENIED", output.getvalue())
        self.assertIn("income_vip", output.getvalue())
        self.assertNotIn("secret-must-not-appear", output.getvalue())


if __name__ == "__main__":
    unittest.main()
