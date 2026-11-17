"""Offline contract tests: no vendor, network, credentials or notification access."""

import copy
import importlib
import json
import logging
import os
import socket
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import requests


NOW = datetime(2026, 11, 17, 8, 49, tzinfo=timezone.utc)
DAY = "20261117"


def identity(code="600001.SH", **changes):
    row = dict(ts_code=code, name="Example", exchange="SSE", market="主板",
               curr_type="CNY", list_date="20000101", delist_date=None,
               industry="Manufacturing", list_status="L")
    row.update(changes)
    return row


def annual(code="600001.SH", period="20251231", **changes):
    row = dict(ts_code=code, end_date=period, ann_date="20260301",
               f_ann_date="20260302", report_type="1",
               n_income_attr_p=20_000_000, revenue=200_000_000)
    row.update(changes)
    return row


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.rows = {
            "trade_cal": [dict(cal_date=DAY, is_open=1),
                          dict(cal_date="20261118", is_open=1)],
            "stock_basic": [identity()],
            "bak_basic": [dict(ts_code="600001.SH", trade_date=DAY,
                               name="Example", industry="Manufacturing")],
            "daily": [dict(ts_code="600001.SH", trade_date=DAY,
                           close=12.5, vol=123)],
            "daily_basic": [dict(ts_code="600001.SH", trade_date=DAY,
                                 total_mv=200_000, close=12.5)],
            "stock_st": [], "suspend_d": [], "stk_limit": [],
            "income_vip": [annual(), annual(period="20241231")],
        }
        self.errors = {}
        self.repeat = set()

    def query(self, name, **params):
        self.calls.append((name, copy.deepcopy(params)))
        error = self.errors.get(name)
        if error is not None:
            if isinstance(error, list):
                if error:
                    raise error.pop(0)
            elif isinstance(error, Exception):
                raise error
            else:
                return copy.deepcopy(error)
        rows = copy.deepcopy(self.rows[name])
        if name == "stock_basic":
            rows = [r for r in rows if r.get("list_status") == params["list_status"]]
        if name == "income_vip":
            rows = [r for r in rows if r["end_date"] == params["period"]
                    and str(r.get("report_type")) == str(params.get("report_type", "1"))]
            if params.get("start_date"):
                rows = [r for r in rows if params["start_date"] <= str(r.get("ann_date", "")) <= params["end_date"]]
        offset = 0 if name in self.repeat else params.get("offset", 0)
        return rows[offset:offset + params.get("limit", 100000)]


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, "connect",
                                    side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        try:
            self.module = importlib.import_module("tradingagents.strategies.small_cap.data_source")
        except ModuleNotFoundError:
            self.fail("TushareSource has not been implemented")
        logger = logging.getLogger(self.module.__name__)
        previous_level = logger.level
        logger.setLevel(logging.CRITICAL)
        self.addCleanup(logger.setLevel, previous_level)
        self.api = FakeAPI()

    def source(self, clock=lambda: NOW):
        return self.module.TushareSource(api=self.api, clock=clock)

    def assert_code(self, code, callback):
        with self.assertRaises(self.module.DataSourceError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_normalized_snapshot_units_and_audit_evidence(self):
        snapshot = self.source().collect()
        self.assertEqual(snapshot["schema_version"], "small-cap-snapshot-v1")
        self.assertEqual(snapshot["mode"], "forward")
        self.assertEqual(snapshot["data_date"], "2026-11-17")
        self.assertEqual(snapshot["target_trade_date"], "2026-11-18")
        self.assertTrue(snapshot["coverage"]["complete"])
        self.assertEqual(snapshot["as_of_at"], "2026-11-17T16:49:00+08:00")
        self.assertTrue(snapshot["calendar"])
        row = snapshot["records"][0]
        self.assertEqual(row["total_market_cap_yuan"], 2_000_000_000)
        self.assertEqual(row["close"], 12.5)
        self.assertEqual(row["volume"], 12300)
        self.assertEqual(row["security_status"], "NORMAL")
        self.assertEqual(row["status_effective_date"], "2026-11-17")
        self.assertFalse(row["suspended"])
        self.assertEqual(row["trade_date"], "2026-11-17")
        self.assertEqual(row["industry_basis"], "current_snapshot")
        self.assertEqual(row["financials"][0]["n_income_attr_p"], 20_000_000)
        self.assertEqual(row["financials"][0]["revenue"], 200_000_000)
        self.assertEqual(row["financials"][0]["f_ann_date"], "20260302")
        self.assertTrue(row["financials"][0]["revision_id"])
        for value in [snapshot["observed_at"], row["observed_at"],
                      row["financials"][0]["observed_at"]]:
            self.assertEqual(datetime.fromisoformat(value), NOW)
        page = snapshot["raw"]["daily"][0]
        self.assertEqual(page["params"]["trade_date"], DAY)
        self.assertEqual(page["rows"][0]["vol"], 123)
        self.assertEqual(datetime.fromisoformat(page["observed_at"]), NOW)
        json.dumps(snapshot, allow_nan=False)

    def test_future_and_historical_dates_rejected_without_requests(self):
        for date in ["2026-11-18", "2026-11-16", "20261118"]:
            self.assert_code("FORWARD_DATE_REQUIRED", lambda: self.source().collect(date))
        self.assertEqual(self.api.calls, [])

    def test_host_timezone_does_not_change_shanghai_day(self):
        instant = datetime(2026, 11, 16, 18, 49,
                           tzinfo=timezone(timedelta(hours=-14)))
        self.assertEqual(self.source(lambda: instant).collect()["data_date"], "2026-11-17")

    def test_naive_clock_is_rejected(self):
        self.assert_code("INVALID_CLOCK", lambda: self.source(lambda: NOW.replace(tzinfo=None)).collect())

    def test_before_close_is_rejected(self):
        instant = datetime(2026, 11, 17, 6, 59, tzinfo=timezone.utc)
        self.assert_code("BEFORE_CLOSE", lambda: self.source(lambda: instant).collect())
        self.assertTrue(all(n == "trade_cal" for n, _ in self.api.calls))

    def test_holiday_returns_closed_without_security_calls(self):
        self.api.rows["trade_cal"][0]["is_open"] = 0
        snapshot = self.source().preflight("2026-11-17")
        self.assertEqual(snapshot["status"], "CLOSED")
        self.assertEqual(snapshot["records"], [])
        self.assertTrue(all(n == "trade_cal" for n, _ in self.api.calls))

    def test_calendar_confirms_next_actual_open_not_next_date(self):
        self.api.rows["trade_cal"] = [dict(cal_date=DAY, is_open=1)] + [
            dict(cal_date=f"202611{d}", is_open=int(d == 23)) for d in range(18, 24)]
        self.assertEqual(self.source().collect()["target_trade_date"], "2026-11-23")

    def test_missing_today_or_next_calendar_is_not_guessed(self):
        for rows in [[dict(cal_date=DAY, is_open=1)],
                     [dict(cal_date="20261118", is_open=1)],
                     [dict(cal_date=DAY, is_open=1), dict(cal_date="20261119", is_open=1)]]:
            self.api.rows["trade_cal"] = rows
            self.assert_code("CALENDAR_INCOMPLETE", self.source().collect)

    def test_crossing_target_midnight_cannot_seal(self):
        calls = 0

        def clock():
            nonlocal calls
            calls += 1
            return NOW if calls == 1 else datetime(2026, 11, 17, 16, tzinfo=timezone.utc)

        self.assert_code("CUTOFF_EXPIRED", self.source(clock).collect)

    def test_quotes_must_match_exact_data_date(self):
        for endpoint in ["daily", "daily_basic", "bak_basic", "stock_st", "suspend_d"]:
            api = FakeAPI()
            api.rows[endpoint] = [dict(ts_code="600001.SH", trade_date="20261116", close=1)]
            self.api = api
            self.assert_code("DATE_MISMATCH", self.source().collect)

    def test_missing_quotes_are_unknown_not_suspension_or_zero(self):
        self.api.rows["stock_basic"].append(identity("600002.SH"))
        snapshot = self.source().collect()
        row = next(r for r in snapshot["records"] if r["ts_code"] == "600002.SH")
        self.assertIsNone(row["close"])
        self.assertIsNone(row["volume"])
        self.assertIsNone(row["suspended"])
        self.assertIsNone(row["total_market_cap_yuan"])
        self.assertEqual(row["financials"], [])
        self.assertIn("600002.SH", snapshot["coverage"]["unknown_codes"])

    def test_only_explicit_suspension_or_official_zero_volume_excludes(self):
        self.api.rows["suspend_d"] = [dict(ts_code="600002.SH", trade_date=DAY, suspend_type="S")]
        self.api.rows["stock_basic"].extend([identity("600002.SH"), identity("600003.SH")])
        self.api.rows["daily"].append(dict(ts_code="600003.SH", trade_date=DAY, close=3, vol=0))
        rows = {r["ts_code"]: r for r in self.source().collect()["records"]}
        self.assertTrue(rows["600002.SH"]["suspended"])
        self.assertTrue(rows["600003.SH"]["suspended"])

    def test_missing_financials_preserves_low_cap_unknown_without_zeroes(self):
        self.api.rows["income_vip"] = []
        snapshot = self.source().collect()
        self.assertEqual(len(snapshot["records"]), 1)
        self.assertEqual(snapshot["records"][0]["financials"], [])
        self.assertIn("600001.SH", snapshot["coverage"]["unknown_codes"])

    def test_two_complete_years_and_all_consolidated_versions_are_retained(self):
        self.api.rows["income_vip"] += [annual(report_type="4", revenue=None),
                                         annual(report_type="5"), annual(report_type="2")]
        snapshot = self.source().collect()
        rows = snapshot["records"][0]["financials"]
        self.assertEqual({r["end_date"] for r in rows}, {"20251231", "20241231"})
        self.assertEqual({str(r["report_type"]) for r in rows}, {"1", "4", "5"})
        self.assertTrue(any(r["revenue"] is None for r in rows))
        self.assertEqual({p["period"] for n, p in self.api.calls if n == "income_vip"},
                         {"20251231", "20241231"})
        self.assertEqual({(p["period"], p.get("report_type")) for n, p in self.api.calls if n == "income_vip"},
                         {(year, report_type) for year in ("20251231", "20241231")
                          for report_type in ("1", "4", "5")})
        self.assertTrue(all("ts_code" not in p for n, p in self.api.calls if n == "income_vip"))
        required = {"ts_code", "end_date", "ann_date", "f_ann_date", "report_type",
                    "n_income_attr_p", "revenue"}
        self.assertTrue(all(required <= set(p.get("fields", "").split(","))
                            for n, p in self.api.calls if n == "income_vip"))

    def test_financial_available_at_is_d_plus_one_not_observation_backfill(self):
        self.api.rows["income_vip"] = [annual(ann_date=DAY, f_ann_date=DAY)]
        row = self.source().collect()["records"][0]["financials"][0]
        self.assertEqual(row["available_at"], "2026-11-18T00:00:00+08:00")
        self.assertEqual(datetime.fromisoformat(row["observed_at"]), NOW)

    def test_original_and_corrected_financial_versions_have_distinct_hashes(self):
        self.api.rows["income_vip"] = [annual(), annual(n_income_attr_p=-123)]
        rows = self.source().collect()["records"][0]["financials"]
        self.assertEqual(len({r["revision_id"] for r in rows}), 2)

    def test_unidentified_quote_cannot_disappear_from_universe(self):
        self.api.rows["daily_basic"].append(dict(ts_code="600099.SH", trade_date=DAY, total_mv=100_000))
        rows = {r["ts_code"]: r for r in self.source().collect()["records"]}
        self.assertIn("600099.SH", rows)
        self.assertEqual(rows["600099.SH"]["security_status"], "UNKNOWN")
        self.assertIsNone(rows["600099.SH"]["market"])

    def test_current_st_list_name_and_effective_delisting_are_used(self):
        self.api.rows["stock_st"] = [dict(ts_code="600001.SH", trade_date=DAY, name="ST Example")]
        self.api.rows["stock_basic"] += [identity("600002.SH", list_status="D", delist_date="20261116")]
        rows = {r["ts_code"]: r for r in self.source().collect()["records"]}
        self.assertEqual(rows["600001.SH"]["security_status"], "ST")
        self.assertEqual(rows["600001.SH"]["name"], "ST Example")
        self.assertEqual(rows["600002.SH"]["security_status"], "DELISTING")
        self.assertEqual({p["list_status"] for n, p in self.api.calls if n == "stock_basic"}, {"L", "D", "P"})
        self.assertTrue(all(p["limit"] <= 1000 and p["trade_date"] == DAY
                            for n, p in self.api.calls if n == "stock_st"))

    def test_future_name_or_delisting_does_not_replace_t_day_status(self):
        self.api.rows["stock_basic"] = [identity(name="*ST Future", list_status="D", delist_date="20261118")]
        row = self.source().collect()["records"][0]
        self.assertEqual(row["security_status"], "NORMAL")
        self.assertEqual(row["name"], "Example")

    def test_missing_t_day_name_cannot_assert_normal(self):
        self.api.rows["bak_basic"] = []
        row = self.source().collect()["records"][0]
        self.assertEqual(row["security_status"], "UNKNOWN")

    def test_pagination_collects_all_pages(self):
        self.api.rows["daily"] += [dict(ts_code=f"60000{n}.SH", trade_date=DAY, close=2, vol=1)
                                   for n in [2, 3]]
        with patch.dict(self.module.TushareSource.PAGE_LIMITS, {"daily": 2}):
            snapshot = self.source().collect()
        self.assertEqual(len(snapshot["raw"]["daily"]), 2)
        self.assertEqual(len(snapshot["records"]), 3)
        self.assertEqual([p["offset"] for n, p in self.api.calls if n == "daily"], [0, 2])

    def test_repeated_page_is_a_hard_failure(self):
        self.api.repeat.add("daily")
        with patch.dict(self.module.TushareSource.PAGE_LIMITS, {"daily": 1}):
            self.assert_code("PAGINATION_REPEATED", self.source().collect)

    def test_full_last_allowed_page_is_not_assumed_complete(self):
        self.api.rows["daily"] += [dict(ts_code="600002.SH", trade_date=DAY, close=2, vol=1)]
        with patch.dict(self.module.TushareSource.PAGE_LIMITS, {"daily": 1}), \
                patch.object(self.module.TushareSource, "MAX_PAGES", 2):
            self.assert_code("PAGINATION_TRUNCATED", self.source().collect)

    def test_important_api_failure_fails_closed_without_credential_leaks(self):
        secret = "sensitive-test-token"
        for endpoint in ["trade_cal", "stock_basic", "bak_basic", "daily", "daily_basic",
                         "stock_st", "suspend_d", "income_vip"]:
            self.api = FakeAPI()
            self.api.errors[endpoint] = {"code": 2002, "msg": f"permission denied {secret}"}
            with patch.dict(os.environ, {"TUSHARE_TOKEN": secret}):
                error = self.assert_code("PERMISSION_DENIED", self.source().collect)
            self.assertNotIn(secret, str(error))
            self.assertNotIn(secret, repr(error))
            self.assertIsNone(error.__cause__)

    def test_optional_limits_failure_is_warned_without_poisoning_raw(self):
        self.api.errors["stk_limit"] = {"code": -1, "msg": "sensitive provider error"}
        snapshot = self.source().collect()
        self.assertTrue(snapshot["coverage"]["complete"])
        self.assertTrue(snapshot["warnings"])
        self.assertNotIn("sensitive provider error", json.dumps(snapshot))

    def test_snapshot_redacts_token_keys_and_embedded_credentials(self):
        secret = "sensitive-test-token"
        self.api.rows["daily"][0].update(token=secret, detail={"api_key": secret},
                                          notes=f"token={secret}")
        with patch.dict(os.environ, {"TUSHARE_TOKEN": secret}):
            snapshot = self.source().collect()
        serialized = json.dumps(snapshot)
        self.assertNotIn(secret, serialized)
        self.assertNotIn('"token"', serialized)
        self.assertNotIn('"api_key"', serialized)

    def test_transient_transport_retries_are_bounded_and_can_recover(self):
        self.api.errors["daily"] = [requests.Timeout("do not expose")]
        self.assertTrue(self.source().collect()["coverage"]["complete"])
        self.assertEqual(sum(n == "daily" for n, _ in self.api.calls), 2)
        self.api = FakeAPI()
        self.api.errors["daily"] = requests.Timeout("sensitive-test-token")
        error = self.assert_code("TRANSPORT_ERROR", self.source().collect)
        self.assertNotIn("sensitive-test-token", str(error))
        self.assertEqual(sum(n == "daily" for n, _ in self.api.calls), 2)

    def test_business_errors_are_classified_and_not_retried(self):
        for message, code in [("每分钟最多访问该接口", "RATE_LIMIT"),
                              ("您的token不对", "AUTHENTICATION_FAILED"),
                              ("provider internal failure", "API_ERROR")]:
            self.api = FakeAPI()
            self.api.errors["daily"] = dict(code=-2001, msg=message)
            self.assert_code(code, self.source().collect)
            self.assertEqual(sum(n == "daily" for n, _ in self.api.calls), 1)

    def test_malformed_structured_response_is_not_empty_success(self):
        self.api.errors["daily"] = {"code": 0, "data": {"fields": ["ts_code"], "items": [[1, 2]]}}
        self.assert_code("MALFORMED_RESPONSE", self.source().collect)

    def test_real_default_adapter_uses_https_structured_post_and_timeout(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"code":0,"data":{"fields":["cal_date","is_open"],"items":[["20261117",0]]}}'
        with patch.dict(os.environ, {"TUSHARE_TOKEN": "offline-only-secret"}), \
                patch.object(requests, "post", return_value=response) as post:
            snapshot = self.module.TushareSource(clock=lambda: NOW).collect()
        self.assertEqual(snapshot["status"], "CLOSED")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.tushare.pro")
        self.assertEqual(kwargs["json"]["api_name"], "trade_cal")
        self.assertEqual(kwargs["json"]["token"], "offline-only-secret")
        self.assertIn("params", kwargs["json"])
        self.assertIn("fields", kwargs["json"])
        self.assertTrue(kwargs["timeout"])
        self.assertNotIn("offline-only-secret", json.dumps(snapshot))

    def test_real_default_adapter_requires_explicit_environment_token(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assert_code("TOKEN_MISSING", lambda: self.module.TushareSource(clock=lambda: NOW).collect())

    def test_missing_t_name_does_not_leak_future_identity_name_into_record(self):
        self.api.rows["bak_basic"] = []
        self.api.rows["stock_basic"][0]["name"] = "*ST Future"
        row = self.source().collect()["records"][0]
        self.assertIsNone(row["name"])
        self.assertEqual(row["security_status"], "UNKNOWN")

    def test_observation_times_are_actual_per_fetch_not_one_frozen_start(self):
        ticks = iter(NOW + timedelta(seconds=n) for n in range(100))
        snapshot = self.source(lambda: next(ticks)).collect()
        earlier = snapshot["raw"]["daily"][0]["observed_at"]
        later = snapshot["records"][0]["financials"][0]["observed_at"]
        self.assertLess(datetime.fromisoformat(earlier), datetime.fromisoformat(later))
        self.assertLess(datetime.fromisoformat(later), datetime.fromisoformat(snapshot["as_of_at"]))
        self.assertLessEqual(datetime.fromisoformat(snapshot["observed_at"]),
                             datetime.fromisoformat(snapshot["as_of_at"]))

    def test_last_moment_before_cutoff_succeeds_but_exact_midnight_fails(self):
        before = datetime(2026, 11, 17, 15, 59, 59, tzinfo=timezone.utc)
        self.assertEqual(self.source(lambda: before).collect()["status"], "OK")
        ticks = iter([NOW] * 14 + [datetime(2026, 11, 17, 16, tzinfo=timezone.utc)] * 100)
        self.assert_code("CUTOFF_EXPIRED", self.source(lambda: next(ticks)).collect)

    def test_conflicting_same_day_quotes_are_not_arbitrarily_chosen(self):
        self.api.rows["daily"].append(dict(ts_code="600001.SH", trade_date=DAY, close=99, vol=123))
        self.assert_code("AMBIGUOUS_DATA", self.source().collect)

    def test_malformed_required_suspension_rows_fail_closed(self):
        self.api.rows["suspend_d"] = [dict(trade_date=DAY, suspend_type="S")]
        self.assert_code("MALFORMED_RESPONSE", self.source().collect)

    def test_missing_financial_fields_are_audited_not_defaulted_to_zero(self):
        self.api.rows["income_vip"] = [annual(revenue=None, n_income_attr_p=None)]
        snapshot = self.source().collect()
        self.assertIn("600001.SH", snapshot["coverage"]["unknown_codes"])
        row = snapshot["records"][0]["financials"][0]
        self.assertIsNone(row["revenue"])
        self.assertIsNone(row["n_income_attr_p"])

    def test_overflow_in_unit_conversion_is_unknown_not_invalid_json(self):
        self.api.rows["daily_basic"][0]["total_mv"] = 1e308
        self.api.rows["daily"][0]["vol"] = 1e308
        snapshot = self.source().collect()
        self.assertIsNone(snapshot["records"][0]["total_market_cap_yuan"])
        self.assertIsNone(snapshot["records"][0]["volume"])
        json.dumps(snapshot, allow_nan=False)

    def test_default_http_permissions_and_transport_failure_are_safe(self):
        with patch.dict(os.environ, {"TUSHARE_TOKEN": "offline-only-secret"}):
            for status, expected in [(401, "AUTHENTICATION_FAILED"), (403, "PERMISSION_DENIED"),
                                     (429, "RATE_LIMIT")]:
                response = requests.Response()
                response.status_code = status
                with patch.object(requests, "post", return_value=response) as post:
                    self.assert_code(expected, self.module.TushareSource(clock=lambda: NOW).collect)
                    self.assertEqual(post.call_count, 1)
            with patch.object(requests, "post", side_effect=requests.Timeout("offline-only-secret")) as post:
                self.assert_code("TRANSPORT_ERROR", self.module.TushareSource(clock=lambda: NOW).collect)
                self.assertEqual(post.call_count, 2)

    def test_source_never_follows_redirects_with_token_body(self):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"code":0,"data":{"fields":["cal_date","is_open"],"items":[["20261117",0]]}}'
        with patch.dict(os.environ, {"TUSHARE_TOKEN": "offline-only-secret"}), \
                patch.object(requests, "post", return_value=response) as post:
            self.module.TushareSource(clock=lambda: NOW).collect()
        self.assertIs(post.call_args.kwargs.get("allow_redirects"), False)

    def test_redaction_removes_api_key_spelling_variants(self):
        self.api.rows["daily"][0]["metadata"] = {"apiKey": "not-an-env-token"}
        snapshot = self.source().collect()
        self.assertNotIn("not-an-env-token", json.dumps(snapshot))

    def test_unjoined_codes_from_resumption_limits_or_financials_are_not_dropped(self):
        self.api.rows["suspend_d"] = [dict(ts_code="600002.SH", trade_date=DAY, suspend_type="R")]
        self.api.rows["stk_limit"] = [dict(ts_code="600003.SH", trade_date=DAY, up_limit=3)]
        self.api.rows["income_vip"].append(annual(code="600004.SH"))
        snapshot = self.source().collect()
        rows = {r["ts_code"]: r for r in snapshot["records"]}
        for code in ("600002.SH", "600003.SH", "600004.SH"):
            self.assertIn(code, rows)
            self.assertIsNone(rows[code]["total_market_cap_yuan"])
            self.assertIsNone(rows[code]["suspended"])
            self.assertIn(code, snapshot["coverage"]["unknown_codes"])

    def test_runner_safe_code_alias_exposes_only_classified_error(self):
        self.api.errors["daily"] = requests.Timeout("do-not-expose-private-error")
        error = self.assert_code("TRANSPORT_ERROR", self.source().collect)
        self.assertEqual(error.safe_code, "TRANSPORT_ERROR")
        self.assertEqual(error.safe_code, error.code)
        self.assertNotIn("do-not-expose-private-error", str(error))

    def test_endpoint_progress_is_info_and_redacted_on_success_and_failure(self):
        private = "provider-private-text-not-for-logs"
        self.api.errors["daily"] = requests.Timeout(private)
        with self.assertLogs(self.module.__name__, level="INFO") as captured:
            error = self.assert_code("TRANSPORT_ERROR", self.source().collect)
        self.assertEqual(error.endpoint, "daily")
        logs = "\n".join(captured.output)
        self.assertIn("endpoint=trade_cal", logs)
        self.assertIn("endpoint=daily", logs)
        self.assertIn("attempt=2", logs)
        self.assertIn("code=TRANSPORT_ERROR", logs)
        self.assertNotIn(private, logs)
        self.assertNotIn("Traceback", logs)

    def test_business_and_pagination_errors_expose_failing_endpoint(self):
        self.api.errors["income_vip"] = dict(code=2002, msg="permission denied private details")
        error = self.assert_code("PERMISSION_DENIED", self.source().collect)
        self.assertEqual(error.endpoint, "income_vip")
        self.api = FakeAPI()
        self.api.repeat.add("daily")
        with patch.dict(self.module.TushareSource.PAGE_LIMITS, {"daily": 1}):
            error = self.assert_code("PAGINATION_REPEATED", self.source().collect)
        self.assertEqual(error.endpoint, "daily")

    def test_post_fetch_validation_error_also_exposes_endpoint(self):
        self.api.rows["daily"].append(dict(ts_code="600001.SH", trade_date=DAY, close=99, vol=123))
        error = self.assert_code("AMBIGUOUS_DATA", self.source().collect)
        self.assertEqual(error.endpoint, "daily")

    def test_unexpected_api_failure_is_safely_raised_not_swallowed(self):
        self.api.errors["stock_st"] = RuntimeError("private-provider-error")
        error = self.assert_code("API_ERROR", self.source().collect)
        self.assertEqual(error.endpoint, "stock_st")
        self.assertEqual(sum(n == "stock_st" for n, _ in self.api.calls), 1)
        self.assertNotIn("private-provider-error", str(error))

    def test_default_progress_goes_to_stderr_without_contaminating_json_stdout(self):
        code = '''
import json
import socket
from datetime import datetime
from unittest.mock import patch
from tradingagents.strategies.small_cap.data_source import TushareSource
api = type("OfflineAPI", (), {"query": lambda self, name, **kw:
    [{"cal_date": "20261117", "is_open": 0}]})()
with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")):
    snapshot = TushareSource(api=api, clock=lambda:
        datetime.fromisoformat("2026-11-17T16:49:00+08:00")).collect()
print(json.dumps({"status": snapshot["status"]}))
'''
        environment = dict(os.environ, TUSHARE_TOKEN="offline-private-token")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                                text=True, env=environment, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout), {"status": "CLOSED"})
        self.assertIn("INFO", result.stderr)
        self.assertIn("endpoint=trade_cal", result.stderr)
        self.assertIn("rows=1", result.stderr)
        self.assertNotIn("offline-private-token", result.stdout + result.stderr)

    def test_stock_st_requests_explicit_minimal_fields(self):
        self.source().collect()
        params = next(p for n, p in self.api.calls if n == "stock_st")
        self.assertEqual(set(params.get("fields", "").split(",")), {"ts_code", "name", "trade_date"})
        self.assertEqual(params["trade_date"], DAY)

    def test_monotonic_collect_budget_is_shared_across_endpoints(self):
        elapsed = [0.0]
        original = self.api.query

        def slow_query(name, **params):
            elapsed[0] += 120
            return original(name, **params)

        with patch.object(self.module, "monotonic", side_effect=lambda: elapsed[0], create=True), \
                patch.object(self.api, "query", side_effect=slow_query):
            error = self.assert_code("COLLECT_TIMEOUT", self.source().collect)
        self.assertEqual(error.endpoint, "stock_basic")
        self.assertEqual(len(self.api.calls), 3)

    def test_expired_collect_budget_prevents_transport_retry(self):
        elapsed = [0.0]
        calls = []

        def slow_timeout(name, **params):
            calls.append(name)
            elapsed[0] = 301
            raise requests.Timeout("private transport details")

        with patch.object(self.module, "monotonic", side_effect=lambda: elapsed[0], create=True), \
                patch.object(self.api, "query", side_effect=slow_timeout):
            error = self.assert_code("COLLECT_TIMEOUT", self.source().collect)
        self.assertEqual(error.endpoint, "trade_cal")
        self.assertEqual(calls, ["trade_cal"])

    def test_budget_expired_during_final_assembly_cannot_return_complete(self):
        elapsed = [0.0]
        source = self.source()
        original = source._coverage

        def slow_coverage():
            elapsed[0] = 301
            return original()

        with patch.object(self.module, "monotonic", side_effect=lambda: elapsed[0], create=True), \
                patch.object(source, "_coverage", side_effect=slow_coverage):
            self.assert_code("COLLECT_TIMEOUT", source.collect)

    def test_default_https_timeouts_are_capped_by_remaining_budget(self):
        elapsed = [0.0]

        def clock():
            elapsed[0] = 298
            return NOW

        response = requests.Response()
        response.status_code = 200
        response._content = b'{"code":0,"data":{"fields":["cal_date","is_open"],"items":[["20261117",0]]}}'
        with patch.dict(os.environ, {"TUSHARE_TOKEN": "offline-private-token"}), \
                patch.object(self.module, "monotonic", side_effect=lambda: elapsed[0], create=True), \
                patch.object(requests, "post", return_value=response) as post:
            self.module.TushareSource(clock=clock).collect()
        for call in post.call_args_list:
            connect, read = call.kwargs["timeout"]
            self.assertGreater(connect, 0)
            self.assertGreater(read, 0)
            self.assertLessEqual(connect, 2)
            self.assertLessEqual(read, 2)
            self.assertNotIn("deadline", call.kwargs["json"]["params"])

    def test_confirmed_no_quote_halt_has_dated_suspension_evidence(self):
        self.api.rows["stock_basic"].append(identity("600002.SH"))
        self.api.rows["suspend_d"] = [dict(ts_code="600002.SH", trade_date=DAY, suspend_type="S")]
        snapshot = self.source().collect()
        row = next(r for r in snapshot["records"] if r["ts_code"] == "600002.SH")
        self.assertTrue(row["suspended"])
        self.assertEqual(row["suspension_date"], "2026-11-17")
        self.assertIsNone(row["trade_date"])

    def test_same_day_s_and_r_with_positive_volume_is_not_a_full_halt(self):
        self.api.rows["suspend_d"] = [
            dict(ts_code="600001.SH", trade_date=DAY, suspend_type="S"),
            dict(ts_code="600001.SH", trade_date=DAY, suspend_type="R"),
        ]
        snapshot = self.source().collect()
        row = snapshot["records"][0]
        self.assertIsNone(row["suspended"])
        self.assertIsNone(row["suspension_date"])
        self.assertIn("suspended", snapshot["coverage"]["missing_fields"]["600001.SH"])
        self.assertTrue(any(w.get("code") == "SUSPENSION_UNRESOLVED" for w in snapshot["warnings"]))

    def test_s_event_and_positive_volume_without_ordering_is_unknown(self):
        self.api.rows["suspend_d"] = [dict(ts_code="600001.SH", trade_date=DAY, suspend_type="S")]
        row = self.source().collect()["records"][0]
        self.assertIsNone(row["suspended"])

    def test_dated_zero_volume_still_proves_untradeability_with_resume_event(self):
        self.api.rows["daily"][0]["vol"] = 0
        self.api.rows["suspend_d"] = [dict(ts_code="600001.SH", trade_date=DAY, suspend_type="R")]
        row = self.source().collect()["records"][0]
        self.assertTrue(row["suspended"])
        self.assertEqual(row["suspension_date"], "2026-11-17")

    def test_timed_intraday_s_event_without_quote_cannot_prove_full_day_halt(self):
        self.api.rows["stock_basic"].append(identity("600002.SH"))
        self.api.rows["suspend_d"] = [dict(ts_code="600002.SH", trade_date=DAY,
                                               suspend_type="S", suspend_timing="09:30-10:00")]
        row = next(r for r in self.source().collect()["records"] if r["ts_code"] == "600002.SH")
        self.assertIsNone(row["suspended"])

    def test_financial_full_range_splits_disjoint_dates_without_offset(self):
        self.api.rows["income_vip"] = [
            annual(ann_date="20260105"),
            annual(code="600002.SH", ann_date="20260505"),
            annual(code="600003.SH", ann_date="20261005"),
        ]
        original = self.api.query

        def poison_full_parent(name, **params):
            rows = original(name, **params)
            if name == "income_vip" and params.get("start_date") == "20260101" and params.get("end_date") == DAY:
                for row in rows:
                    row["revenue"] = 999_999_999
            return rows

        with patch.object(self.api, "query", side_effect=poison_full_parent), \
                patch.dict(self.module.TushareSource.PAGE_LIMITS, {"income_vip": 2}):
            snapshot = self.source().collect()
        pages = [p for p in snapshot["raw"]["income_vip"]
                 if p["params"]["period"] == "20251231" and p["params"]["report_type"] == "1"]
        self.assertEqual(pages[0]["selection_disposition"], "split")
        self.assertEqual(pages[0]["rows"][0]["revenue"], 999_999_999)
        leaves = [p for p in pages if p["selection_disposition"] == "leaf"]
        ranges = sorted((p["params"]["start_date"], p["params"]["end_date"]) for p in leaves)
        self.assertEqual(ranges[0][0], "20260101")
        self.assertEqual(ranges[-1][1], DAY)
        for previous, current in zip(ranges, ranges[1:]):
            self.assertEqual((datetime.strptime(current[0], "%Y%m%d") -
                              datetime.strptime(previous[1], "%Y%m%d")).days, 1)
        self.assertTrue(all(p["params"]["offset"] == 0 for p in pages))
        self.assertEqual({r["ts_code"] for r in snapshot["records"]},
                         {"600001.SH", "600002.SH", "600003.SH"})
        self.assertTrue(all(f["revenue"] == 200_000_000 for r in snapshot["records"] for f in r["financials"]))

    def test_financial_same_day_full_range_fails_closed_instead_of_offset(self):
        self.api.rows["income_vip"] = [annual(), annual(code="600002.SH")]
        with patch.dict(self.module.TushareSource.PAGE_LIMITS, {"income_vip": 2}):
            error = self.assert_code("PAGINATION_TRUNCATED", self.source().collect)
        self.assertEqual(error.endpoint, "income_vip")
        self.assertTrue(all(p["offset"] == 0 for n, p in self.api.calls if n == "income_vip"))

    def test_financial_leaf_exact_duplicates_are_audited_but_deduplicated(self):
        self.api.rows["income_vip"] = [annual(), annual()]
        snapshot = self.source().collect()
        self.assertEqual(len(snapshot["records"][0]["financials"]), 1)
        page = snapshot["raw"]["income_vip"][0]
        self.assertEqual(len(page["rows"]), 2)
        self.assertEqual(page["selection_disposition"], "leaf")

    def test_financial_range_query_count_is_bounded(self):
        self.api.rows["income_vip"] = [annual(), annual(code="600002.SH")]
        with patch.dict(self.module.TushareSource.PAGE_LIMITS, {"income_vip": 2}), \
                patch.object(self.module.TushareSource, "MAX_FINANCIAL_QUERIES", 2, create=True):
            self.assert_code("PAGINATION_TRUNCATED", self.source().collect)
        self.assertEqual(sum(n == "income_vip" for n, _ in self.api.calls), 2)

    def test_financial_rows_outside_ann_date_range_fail_closed(self):
        self.api.errors["income_vip"] = {"code": 0, "data": {"fields": list(annual()),
                                       "items": [list(annual(ann_date="20251231").values())]}}
        error = self.assert_code("DATE_MISMATCH", self.source().collect)
        self.assertEqual(error.endpoint, "income_vip")

    def test_collect_timeout_in_optional_limits_is_not_swallowed_as_warning(self):
        elapsed = [0.0]
        original = self.api.query

        def timeout_on_limits(name, **params):
            if name == "stk_limit":
                elapsed[0] = 301
            return original(name, **params)

        with patch.object(self.module, "monotonic", side_effect=lambda: elapsed[0], create=True), \
                patch.object(self.api, "query", side_effect=timeout_on_limits):
            error = self.assert_code("COLLECT_TIMEOUT", self.source().collect)
        self.assertEqual(error.endpoint, "stk_limit")
        self.assertFalse(any(n == "income_vip" for n, _ in self.api.calls))

    def test_same_day_daily_and_valuation_close_conflict_fails_closed(self):
        self.api.rows["daily_basic"][0]["close"] = 12.51
        error = self.assert_code("AMBIGUOUS_DATA", self.source().collect)
        self.assertEqual(error.endpoint, "daily_basic")

    def seed_official_fallback_case(self):
        self.api.rows["stock_basic"].append(identity("000001.SZ", exchange="SZSE"))
        self.api.rows["bak_basic"].append(dict(ts_code="000001.SZ", trade_date=DAY, name="SZ Example"))
        self.api.rows["daily"].append(dict(ts_code="000001.SZ", trade_date=DAY, close=10, vol=1))
        self.api.rows["daily_basic"].append(dict(ts_code="000001.SZ", trade_date=DAY, close=10, total_mv=200000))
        self.api.rows["income_vip"] += [annual(code="000001.SZ"), annual(code="001999.SZ")]

    def test_official_master_is_captured_only_for_finance_only_szse_orphan(self):
        from test_small_cap_official import response, workbook
        self.seed_official_fallback_case()
        payload = workbook(rows=[["主板", "000001", "SZ Example", "2000-01-01"]])
        with patch.object(requests, "get", return_value=response(payload)) as get:
            snapshot = self.source().collect()
        self.assertEqual(get.call_count, 1)
        page = snapshot["raw"]["szse_official_a_share_master"][0]
        self.assertTrue(page["coverage_complete"])
        self.assertTrue(page["cross_check_complete"])
        self.assertEqual(page["rows"][0]["ts_code"], "000001.SZ")
        orphan = next(r for r in snapshot["records"] if r["ts_code"] == "001999.SZ")
        self.assertEqual(orphan["security_status"], "UNKNOWN")
        self.assertTrue(orphan["financials"])
        self.assertIsNone(orphan["market"])
        self.assertEqual(snapshot["coverage"]["endpoints"]["szse_official_a_share_master"]["rows"], 1)

    def test_no_official_request_for_no_orphan_or_for_shanghai_orphans(self):
        self.api.rows["income_vip"].append(annual(code="600099.SH"))
        with patch.object(requests, "get", side_effect=AssertionError("Unnecessary download")) as get:
            self.source().collect()
        get.assert_not_called()

    def test_any_positive_tushare_feed_defeats_finance_only_fallback_trigger(self):
        for endpoint in ("stock_basic", "bak_basic", "daily", "daily_basic", "stock_st", "suspend_d", "stk_limit"):
            self.api = FakeAPI()
            self.seed_official_fallback_case()
            row = identity("001999.SZ", exchange="SZSE") if endpoint == "stock_basic" else dict(
                ts_code="001999.SZ", trade_date=DAY, name="Example", close=10, vol=1,
                total_mv=200000, suspend_type="R")
            self.api.rows[endpoint].append(row)
            with self.subTest(endpoint=endpoint), patch.object(requests, "get") as get:
                self.source().collect()
                get.assert_not_called()

    def test_official_failure_preserves_orphan_and_core_acquisition_with_safe_warning(self):
        self.seed_official_fallback_case()
        with patch.object(requests, "get", side_effect=requests.Timeout("private-provider-details")):
            snapshot = self.source().collect()
        self.assertEqual(snapshot["status"], "OK")
        self.assertTrue(snapshot["coverage"]["complete"])
        self.assertNotIn("szse_official_a_share_master", snapshot["raw"])
        self.assertTrue(any(w.get("code") == "OFFICIAL_TRANSPORT_ERROR" for w in snapshot["warnings"]))
        self.assertIn("001999.SZ", snapshot["coverage"]["unknown_codes"])
        self.assertNotIn("private-provider-details", json.dumps(snapshot))

    def test_partial_official_master_is_not_captured_as_verified(self):
        from test_small_cap_official import response, workbook
        self.seed_official_fallback_case()
        payload = workbook(rows=[["主板", "000002", "Other", "2000-01-01"]])
        with patch.object(requests, "get", return_value=response(payload)):
            snapshot = self.source().collect()
        self.assertNotIn("szse_official_a_share_master", snapshot["raw"])
        self.assertTrue(any(w.get("code") == "OFFICIAL_CROSS_CHECK_FAILED" for w in snapshot["warnings"]))


if __name__ == "__main__":
    unittest.main()
