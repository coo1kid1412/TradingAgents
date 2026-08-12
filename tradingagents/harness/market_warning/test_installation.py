from __future__ import annotations

import io
import tempfile
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from unittest import TestCase, main, mock
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.install_market_warning_rule_v1 import (
    activate_gate_engine,
    activate_notify_engine,
    deactivate_notify_engine,
    main as installer_main,
    _read_crontab,
    render_crontab,
    validate_gate_activation_preconditions,
    validate_install_preconditions,
)
from scripts.probe_market_warning_data import run_data_probe
from tradingagents.dataflows.intraday_quote import IntradayQuote
from tradingagents.harness.market_warning.adapters.tushare_realtime_breadth import (
    PremarketBreadthBaseline,
    RealtimePermissionProbe,
)
from tradingagents.harness.market_warning.domain import Market
from tradingagents.harness.market_warning.rule_policy import load_rule_manifest


SHANGHAI = ZoneInfo("Asia/Shanghai")
AS_OF = datetime(2026, 8, 3, 9, 40, tzinfo=SHANGHAI)
MANIFEST_PATH = Path(__file__).with_name("rule_manifest_v1.json")


def _ready_result() -> dict[str, object]:
    manifest = load_rule_manifest(MANIFEST_PATH)
    return {
        "ready": True,
        "mode": "rule_v1/notify",
        "engine_version": manifest.engine_version,
        "manifest_sha256": manifest.manifest_sha256,
        "failures": [],
    }


class InstallerGuardTests(TestCase):
    def test_install_requires_explicit_notify_mode_and_all_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "warning.db"
            env = {
                "FEISHU_APP_ID": "app",
                "FEISHU_APP_SECRET": "secret",
                "FEISHU_USER_OPEN_ID": "user",
            }
            with mock.patch(
                "scripts.install_market_warning_rule_v1._db_path_writable",
                return_value=True,
            ):
                self.assertTrue(
                    validate_install_preconditions(
                        None, _ready_result(), db_path, env, MANIFEST_PATH
                    )
                )
                self.assertTrue(
                    validate_install_preconditions(
                        "rule_v1/gate", _ready_result(), db_path, env, MANIFEST_PATH
                    )
                )
                self.assertEqual(
                    validate_install_preconditions(
                        "rule_v1/notify", _ready_result(), db_path, env, MANIFEST_PATH
                    ),
                    [],
                )

    def test_install_rejects_readiness_checksum_database_and_feishu_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "warning.db"
            readiness = _ready_result()
            readiness["ready"] = False
            readiness["manifest_sha256"] = "f" * 64
            readiness["failures"] = ["data smoke is not ready"]
            with mock.patch(
                "scripts.install_market_warning_rule_v1._db_path_writable",
                return_value=False,
            ):
                failures = validate_install_preconditions(
                    "rule_v1/notify",
                    readiness,
                    db_path,
                    {},
                    MANIFEST_PATH,
                )

        self.assertTrue(any("readiness" in item for item in failures))
        self.assertTrue(any("checksum" in item for item in failures))
        self.assertTrue(any("database" in item for item in failures))
        self.assertTrue(any("Feishu" in item for item in failures))

    def test_crontab_install_is_idempotent_and_uninstall_preserves_unrelated_entries(self) -> None:
        root = Path("/tmp/trading-agents")
        unrelated = "0 7 * * * /usr/bin/other-job"

        installed = render_crontab(unrelated + "\n", root, uninstall=False)
        repeated = render_crontab(installed, root, uninstall=False)
        removed = render_crontab(repeated, root, uninstall=True)

        self.assertEqual(installed, repeated)
        self.assertIn("*/5 * * * 1-5", installed)
        self.assertIn("--mode rule_v1", installed)
        self.assertEqual(removed.strip(), unrelated)

    def test_crontab_refuses_unbalanced_or_nested_v1_markers(self) -> None:
        malformed = (
            "# BEGIN TradingAgents market-warning rule-v1\n"
            "*/5 * * * 1-5 broken\n"
            "0 7 * * * /usr/bin/other-job\n"
        )
        nested = (
            "# BEGIN TradingAgents market-warning rule-v1\n"
            "# BEGIN TradingAgents market-warning rule-v1\n"
            "# END TradingAgents market-warning rule-v1\n"
            "# END TradingAgents market-warning rule-v1\n"
        )

        with self.assertRaisesRegex(ValueError, "marker"):
            render_crontab(malformed, Path("/tmp/trading-agents"), uninstall=True)
        with self.assertRaisesRegex(ValueError, "marker"):
            render_crontab(nested, Path("/tmp/trading-agents"), uninstall=True)

    def test_crontab_read_only_treats_explicit_no_crontab_as_empty(self) -> None:
        no_crontab = mock.Mock(
            returncode=1,
            stdout="",
            stderr="no crontab for test-user\n",
        )
        denied = mock.Mock(
            returncode=1,
            stdout="",
            stderr="permission denied\n",
        )

        with mock.patch("subprocess.run", return_value=no_crontab):
            self.assertEqual(_read_crontab(), "")
        with mock.patch("subprocess.run", return_value=denied):
            with self.assertRaisesRegex(RuntimeError, "permission denied"):
                _read_crontab()
        with mock.patch(
            "subprocess.run", side_effect=PermissionError("operation not permitted")
        ):
            with self.assertRaisesRegex(RuntimeError, "operation not permitted"):
                _read_crontab()

    def test_activation_can_only_enable_notify(self) -> None:
        class Repository:
            def __init__(self):
                self.calls = []

            def register_rule_engine(self, record):
                self.calls.append(("register", record))

            def activate_rule_engine(self, version, mode):
                self.calls.append(("activate", version, mode))

        repository = Repository()

        activate_notify_engine(repository, MANIFEST_PATH, _ready_result())

        self.assertEqual(repository.calls[-1], ("activate", "rule-v1.0.0", "notify"))
        self.assertFalse(any(call[-1] == "gate" for call in repository.calls))

    def test_uninstall_deactivates_notify_and_gate(self) -> None:
        repository = mock.Mock()

        deactivate_notify_engine(repository)

        repository.deactivate_rule_engine.assert_has_calls(
            (
                mock.call(Market.A_SHARE, "notify"),
                mock.call(Market.A_SHARE, "gate"),
            )
        )

    def test_gate_activation_requires_explicit_gate_mode_and_ten_session_readiness(
        self,
    ) -> None:
        readiness = _ready_result()
        readiness.update(
            {
                "mode": "rule_v1/gate",
                "ready": True,
                "soak_sessions": 10,
                "soak_audit": {
                    "scan_success_rate": 1.0,
                    "duplicate_runs": 0,
                    "overlap_skipped": 0,
                    "stale_misjudgments": 0,
                },
            }
        )

        self.assertTrue(
            validate_gate_activation_preconditions(
                "rule_v1/notify", readiness, MANIFEST_PATH
            )
        )
        self.assertEqual(
            validate_gate_activation_preconditions(
                "rule_v1/gate", readiness, MANIFEST_PATH
            ),
            [],
        )

        repository = mock.Mock()
        activate_gate_engine(repository, MANIFEST_PATH, readiness)
        repository.activate_rule_engine.assert_called_once_with(
            "rule-v1.0.0", "gate"
        )

    def test_gate_cli_dry_run_never_writes_crontab_or_activates(self) -> None:
        readiness = _ready_result()
        readiness.update(
            {
                "mode": "rule_v1/gate",
                "ready": True,
                "soak_sessions": 10,
                "soak_audit": {
                    "scan_success_rate": 1.0,
                    "duplicate_runs": 0,
                    "overlap_skipped": 0,
                    "stale_misjudgments": 0,
                },
            }
        )
        repository = mock.Mock()
        with mock.patch(
            "scripts.install_market_warning_rule_v1._read_crontab", return_value=""
        ), mock.patch(
            "scripts.install_market_warning_rule_v1._write_crontab"
        ) as write_crontab, mock.patch(
            "scripts.install_market_warning_rule_v1.SQLiteWarningRepository",
            return_value=repository,
        ), mock.patch(
            "scripts.install_market_warning_rule_v1.check_production_readiness",
            return_value=readiness,
        ), redirect_stdout(io.StringIO()):
            code = installer_main(
                [
                    "--mode",
                    "rule_v1/gate",
                    "--activate-gate",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        write_crontab.assert_not_called()
        repository.activate_rule_engine.assert_not_called()

    def test_uninstall_reports_partial_completion_when_registry_cleanup_fails(
        self,
    ) -> None:
        existing = render_crontab("", Path("/tmp/trading-agents"), uninstall=False)
        repository = mock.Mock()
        repository.deactivate_rule_engine.side_effect = OSError("database unavailable")
        output = io.StringIO()
        with mock.patch(
            "scripts.install_market_warning_rule_v1._read_crontab",
            return_value=existing,
        ), mock.patch(
            "scripts.install_market_warning_rule_v1._write_crontab"
        ) as write_crontab, mock.patch(
            "scripts.install_market_warning_rule_v1.SQLiteWarningRepository",
            return_value=repository,
        ), redirect_stdout(output):
            code = installer_main(["--uninstall"])

        self.assertEqual(code, 3)
        write_crontab.assert_called_once()
        self.assertIn("partially_uninstalled", output.getvalue())


class DataProbeTests(TestCase):
    def _baseline(self) -> PremarketBreadthBaseline:
        frame = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600000.SH"],
                "pre_close": [10.0, 20.0],
                "ma20": [9.5, 19.0],
                "low_20d": [8.0, 17.0],
                "industry": ["bank", "bank"],
                "down_limit": [9.0, 18.0],
            }
        )
        return PremarketBreadthBaseline(
            trade_date=date(2026, 8, 3),
            completed_trade_date=date(2026, 7, 31),
            universe_size=2,
            frame=frame,
        )

    def test_probe_reports_ready_only_with_current_index_and_full_cross_section(self) -> None:
        quote = IntradayQuote(
            symbol="sh000001",
            name="index",
            trade_date="2026-08-03",
            quote_time=datetime(2026, 8, 3, 9, 39, tzinfo=SHANGHAI),
            open=3500,
            high=3510,
            low=3490,
            last=3502,
            pre_close=3500,
            volume=100,
            amount=1000,
            source="tushare_rt_k",
        )
        cross = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600000.SH"],
                "last": [10.1, 19.8],
                "pre_close": [10.0, 20.0],
                "data_time": [quote.quote_time, quote.quote_time],
                "down_limit": [9.0, 18.0],
                "source": ["tushare_rt_k", "tushare_rt_k"],
            }
        )
        cross.attrs["universe_size"] = 2
        cross.index = [0, 0]

        result = run_data_probe(
            object(),
            AS_OF,
            quote_loader=lambda **_kwargs: quote,
            permission_probe=lambda *_args, **_kwargs: RealtimePermissionProbe("available", 2),
            baseline_builder=lambda *_args, **_kwargs: self._baseline(),
            cross_section_loader=lambda *_args, **_kwargs: cross,
            previous_session=lambda _day: date(2026, 7, 31),
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["rt_k_permission"], "available")
        self.assertEqual(result["realtime_breadth_coverage_pct"], 100.0)
        self.assertLessEqual(result["realtime_breadth_staleness_minutes"], 5.0)
        self.assertTrue(result["stk_limit_available"])
        self.assertEqual(result["latest_completed_trade_date"], "2026-07-31")

    def test_permission_failure_is_explicit_and_never_calls_full_market_loader(self) -> None:
        forbidden = mock.Mock(side_effect=AssertionError("must not load full market"))

        result = run_data_probe(
            object(),
            AS_OF,
            quote_loader=lambda **_kwargs: None,
            permission_probe=lambda *_args, **_kwargs: RealtimePermissionProbe(
                "permission_denied", 0, "PermissionError"
            ),
            baseline_builder=lambda *_args, **_kwargs: self._baseline(),
            cross_section_loader=forbidden,
            previous_session=lambda _day: date(2026, 7, 31),
        )

        self.assertFalse(result["ready"])
        self.assertEqual(result["rt_k_permission"], "permission_denied")
        self.assertIn("rt_k_permission", result["failures"])
        forbidden.assert_not_called()

    def test_probe_excludes_stale_rows_from_realtime_coverage(self) -> None:
        baseline = self._baseline()
        cross = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "600000.SH"],
                "last": [10.1, 19.8],
                "pre_close": [10.0, 20.0],
                "data_time": [
                    datetime(2026, 8, 3, 9, 39, tzinfo=SHANGHAI),
                    datetime(2026, 7, 31, 15, 0, tzinfo=SHANGHAI),
                ],
                "down_limit": [9.0, 18.0],
                "source": ["tushare_rt_k", "tushare_daily"],
            }
        )
        cross.attrs["universe_size"] = 2
        cross.index = [0, 0]
        quote = IntradayQuote(
            symbol="sh000001",
            name="index",
            trade_date="2026-08-03",
            quote_time=datetime(2026, 8, 3, 9, 39, tzinfo=SHANGHAI),
            open=3500,
            high=3510,
            low=3490,
            last=3502,
            pre_close=3500,
            volume=100,
            amount=1000,
            source="tushare_rt_k",
        )

        result = run_data_probe(
            object(),
            AS_OF,
            quote_loader=lambda **_kwargs: quote,
            permission_probe=lambda *_args, **_kwargs: RealtimePermissionProbe(
                "available", 2
            ),
            baseline_builder=lambda *_args, **_kwargs: baseline,
            cross_section_loader=lambda *_args, **_kwargs: cross,
            previous_session=lambda _day: date(2026, 7, 31),
        )

        self.assertFalse(result["ready"])
        self.assertEqual(result["realtime_breadth_coverage_pct"], 50.0)
        self.assertIn("realtime_breadth_coverage", result["failures"])


if __name__ == "__main__":
    main()
