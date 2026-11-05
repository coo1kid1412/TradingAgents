import contextlib
import fcntl
import importlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class AnalysisRunnerTests(unittest.TestCase):
    def setUp(self):
        try:
            self.runner = importlib.import_module("tradingagents.analysis_runner")
        except ModuleNotFoundError:
            self.fail("TA needs an observable runner independent of shell background children")
        self.temp = tempfile.TemporaryDirectory(prefix="ta-job-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fake_main(self, implementation):
        (self.root / "main.py").write_text(
            "import os\nfrom pathlib import Path\n"
            "def _build_config(): return {}\n"
            "def analyze_single_stock(ticker, analysis_date, config):\n"
            + "\n".join("    " + line for line in implementation.splitlines())
            + "\n",
            encoding="utf-8",
        )

    def run_job(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.runner.run_job("688507", "2026-11-04", 3, project_root=self.root)

    def latest(self):
        return self.runner.read_status("688507", project_root=self.root)

    def test_real_worker_finishes_and_does_not_edit_main_configuration(self):
        self.fake_main(
            "assert ticker == '688507'\n"
            "assert analysis_date == '2026-11-04'\n"
            "assert config['max_debate_rounds'] == 3\n"
            "print('Phase 13/13: Portfolio Manager complete', flush=True)\n"
            "report = Path('reports/test/5_portfolio')\n"
            "report.mkdir(parents=True)\n"
            "(report / 'decision.md').write_text('# HOLD')\n"
            "return ticker, True, str(report.parent.resolve())"
        )
        before = (self.root / "main.py").read_bytes()
        self.assertEqual(self.run_job(), 0)
        status = self.latest()
        self.assertEqual(status["status"], "succeeded")
        self.assertEqual(status["returncode"], 0)
        self.assertEqual(status["delivery_status"], "not_confirmed")
        self.assertTrue(Path(status["report_path"]).is_dir())
        self.assertIn("Phase 13/13", Path(status["log_path"]).read_text())
        self.assertEqual((self.root / "main.py").read_bytes(), before)

    def test_report_success_does_not_imply_feishu_success(self):
        self.fake_main(
            "report = Path('reports/test/5_portfolio')\n"
            "report.mkdir(parents=True)\n"
            "(report / 'decision.md').write_text('# HOLD')\n"
            "print('[fixture] ✓ 飞书已发送', flush=True)\n"
            "return ticker, True, str(report.parent.resolve())"
        )
        self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.latest()["delivery_status"], "sent")

    def test_child_sigkill_is_recorded_without_diagnosing_oom(self):
        self.fake_main("os.kill(os.getpid(), 9)")
        with patch.object(self.runner, "_notify_abnormal_failure") as notify:
            self.assertEqual(self.run_job(), 1)
        status = self.latest()
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["returncode"], -signal.SIGKILL)
        self.assertEqual(status["exit_reason"], "signal:SIGKILL")
        self.assertNotIn("OOM", status["exit_reason"])
        notify.assert_called_once()

    def test_bootstrap_failure_has_traceback_and_failed_state(self):
        (self.root / "main.py").write_text("raise RuntimeError('startup fixture')\n")
        with patch.object(self.runner, "_notify_abnormal_failure") as notify:
            self.assertEqual(self.run_job(), 1)
        status = self.latest()
        self.assertEqual(status["status"], "failed")
        self.assertIn("startup fixture", Path(status["log_path"]).read_text())
        notify.assert_called_once()

    def test_ordinary_pipeline_failure_does_not_send_duplicate_notification(self):
        self.fake_main("return ticker, False, 'pipeline failure'")
        with patch.object(self.runner, "_notify_abnormal_failure") as notify:
            self.assertEqual(self.run_job(), 1)
        self.assertEqual(self.latest()["status"], "failed")
        notify.assert_not_called()

    def test_missing_decision_file_cannot_be_a_success(self):
        self.fake_main("return ticker, True, str(Path('missing-report').resolve())")
        with patch.object(self.runner, "_notify_abnormal_failure"):
            self.assertEqual(self.run_job(), 1)
        self.assertEqual(self.latest()["status"], "failed")

    def test_active_job_blocks_a_second_analysis(self):
        run_root = self.root / "logs" / "analysis_runs"
        run_root.mkdir(parents=True)
        with (run_root / ".active.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(self.runner.AnalysisAlreadyRunning):
                self.run_job()
        self.assertEqual(list(run_root.glob("*/status.json")), [])

    def test_dead_supervisor_is_reported_as_interrupted(self):
        run_dir = self.root / "logs" / "analysis_runs" / "old"
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps({
            "ticker": "688507", "status": "running", "run_id": "old",
            "log_path": str(run_dir / "output.log"),
        }))
        status = self.latest()
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["exit_reason"], "supervisor_and_worker_not_running")

    def test_running_status_uses_job_lock_instead_of_pid_reuse(self):
        run_dir = self.root / "logs" / "analysis_runs" / "live"
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps({
            "ticker": "688507", "status": "running", "run_id": "live", "pid": 999999,
            "log_path": str(run_dir / "output.log"),
        }))
        with (run_dir / ".running.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.latest()["status"], "running")

    def test_broken_supervisor_output_still_waits_for_worker(self):
        self.fake_main(
            "report = Path('reports/test/5_portfolio')\n"
            "report.mkdir(parents=True)\n"
            "(report / 'decision.md').write_text('# HOLD')\n"
            "return ticker, True, str(report.parent.resolve())"
        )
        with patch('builtins.print', side_effect=BrokenPipeError), patch.object(
            self.runner, '_notify_abnormal_failure',
        ) as notify:
            self.assertEqual(self.run_job(), 0)
        self.assertEqual(self.latest()['status'], 'succeeded')
        notify.assert_not_called()

    def test_finished_worker_result_recovers_a_lost_supervisor(self):
        run_dir = self.root / 'logs' / 'analysis_runs' / 'recovered'
        run_dir.mkdir(parents=True)
        report = self.root / 'reports' / 'recovered' / '5_portfolio'
        report.mkdir(parents=True)
        (report / 'decision.md').write_text('# HOLD')
        (run_dir / 'status.json').write_text(json.dumps({
            'ticker': '688507', 'status': 'running', 'run_id': 'recovered',
            'log_path': str(run_dir / 'output.log'),
        }))
        (run_dir / 'result.json').write_text(json.dumps({
            'success': True, 'report_path': str(report.parent),
        }))
        status = self.latest()
        self.assertEqual(status['status'], 'succeeded')
        self.assertEqual(status['report_path'], str(report.parent.resolve()))
        self.assertIsNone(status['returncode'])

    def test_sent_report_survives_worker_death_without_false_failure_notice(self):
        self.fake_main(
            "report = Path('reports/test/5_portfolio')\n"
            "report.mkdir(parents=True)\n"
            "(report / 'decision.md').write_text('# HOLD')\n"
            "print(f'[{ticker}] 报告已保存至: {report.parent.resolve()}', flush=True)\n"
            "print('[fixture] ✓ 飞书已发送', flush=True)\n"
            "os.kill(os.getpid(), 9)"
        )
        with patch.object(self.runner, '_notify_abnormal_failure') as notify:
            self.assertEqual(self.run_job(), 1)
        status = self.latest()
        self.assertEqual(status['status'], 'partial')
        self.assertEqual(status['delivery_status'], 'sent')
        self.assertTrue(Path(status['report_path']).is_dir())
        notify.assert_not_called()

    def test_closed_real_stdout_pipe_does_not_change_runner_exit_code(self):
        self.fake_main(
            "report = Path('reports/test/5_portfolio')\n"
            "report.mkdir(parents=True)\n"
            "(report / 'decision.md').write_text('# HOLD')\n"
            "return ticker, True, str(report.parent.resolve())"
        )
        process = subprocess.Popen([
            sys.executable, str(Path(self.runner.__file__).resolve()), 'run',
            '--ticker', '688507', '--date', '2026-11-04', '--project-root', str(self.root),
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(process.stderr.close)
        process.stdout.close()
        self.assertEqual(process.wait(timeout=10), 0)
        self.assertNotIn('BrokenPipeError', process.stderr.read().decode())
        self.assertEqual(self.latest()['status'], 'succeeded')

    def test_progress_is_not_lost_when_recent_output_exceeds_tail_window(self):
        run_dir = self.root / 'logs' / 'analysis_runs' / 'long-log'
        run_dir.mkdir(parents=True)
        (run_dir / 'status.json').write_text(json.dumps({
            'ticker': '688507', 'status': 'running', 'run_id': 'long-log',
            'log_path': str(run_dir / 'output.log'),
        }))
        (run_dir / 'output.log').write_text('[688507] ✓ Phase 3/13: complete\n' + 'tool output\n' * 5000)
        with (run_dir / '.running.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.latest()['last_completed_phase'], 3)


if __name__ == "__main__":
    unittest.main()
