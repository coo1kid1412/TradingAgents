"""Official census tests use in-memory XLSX files and never access the network."""

import hashlib
import importlib
from io import BytesIO
import socket
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import requests

from tradingagents.strategies.small_cap.config import digest


NOW = datetime(2026, 11, 17, 8, 49, tzinfo=timezone.utc)
FIELDS = ["板块", "A股代码", "A股简称", "A股上市日期"]
URL = "https://www.szse.cn/api/report/ShowReport"


def workbook(rows=None, columns=None):
    if rows is None:
        rows = [["主板", "000001", "Example", "1991-04-03"],
                ["创业板", "300001", "ST Example", "2009-10-30"]]
    output = BytesIO()
    pd.DataFrame(rows, columns=columns or FIELDS).to_excel(output, engine="openpyxl", index=False)
    return output.getvalue()


def response(body=None, status=200, url=URL):
    result = requests.Response()
    result.status_code = status
    result.url = url
    result._content = workbook() if body is None else body
    result._content_consumed = True
    return result


class OfficialMasterTests(unittest.TestCase):
    def setUp(self):
        no_network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        no_network.start()
        self.addCleanup(no_network.stop)
        try:
            self.module = importlib.import_module("tradingagents.strategies.small_cap.official_master")
        except ModuleNotFoundError:
            self.fail("Official SZSE census helper is not implemented")

    def fetch(self, expected=("000001.SZ", "300001.SZ"), clock=lambda: NOW):
        return self.module.fetch_szse_a_share_master("2026-11-17", expected, clock=clock)

    def assert_error(self, code, action):
        with self.assertRaises(self.module.OfficialMasterError) as caught:
            action()
        self.assertEqual(caught.exception.safe_code, code)
        return caught.exception

    def test_official_https_get_has_fixed_params_tls_timeout_and_hashes(self):
        payload = workbook()
        with patch.object(requests, "get", return_value=response(payload)) as get:
            page = self.fetch()
        self.assertEqual(get.call_args.args, (URL,))
        kwargs = get.call_args.kwargs
        self.assertEqual(kwargs["params"], {"SHOWTYPE": "xlsx", "CATALOGID": "1110", "TABKEY": "tab1"})
        self.assertIs(kwargs["verify"], True)
        self.assertIs(kwargs["allow_redirects"], False)
        self.assertTrue(all(0 < x <= 20 for x in kwargs["timeout"]))
        self.assertEqual(page["source_url"], URL)
        self.assertTrue(page["coverage_complete"])
        self.assertTrue(page["cross_check_complete"])
        self.assertEqual(page["observed_at"], "2026-11-17T16:49:00+08:00")
        self.assertEqual(page["data_date"], "2026-11-17")
        self.assertEqual(page["rows_hash"], digest(page["rows"]))
        self.assertEqual(page["source_hash"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(page["rows"][0], dict(ts_code="000001.SZ", name="Example", market="主板", list_date="1991-04-03"))

    def test_missing_known_active_stock_means_partial_census(self):
        with patch.object(requests, "get", return_value=response(workbook(rows=[["主板", "000001", "Example", "1991-04-03"]]))):
            self.assert_error("OFFICIAL_CROSS_CHECK_FAILED", self.fetch)

    def test_duplicate_code_cannot_prove_complete_membership(self):
        payload = workbook(rows=[["主板", "000001", "A", "1991-04-03"],
                                 ["主板", "000001", "B", "1991-04-03"]])
        with patch.object(requests, "get", return_value=response(payload)):
            self.assert_error("OFFICIAL_INVALID_SCHEMA", lambda: self.fetch(expected=("000001.SZ",)))

    def test_required_schema_is_exact_and_empty_census_is_rejected(self):
        for payload in [workbook(columns=["板块", "证券代码", "A股简称", "A股上市日期"]), workbook(rows=[])]:
            with patch.object(requests, "get", return_value=response(payload)):
                self.assert_error("OFFICIAL_INVALID_SCHEMA", self.fetch)

    def test_invalid_future_or_missing_listing_dates_are_rejected(self):
        for value in ["2026-11-18", "not-a-date", None]:
            payload = workbook(rows=[["主板", "000001", "Example", value]])
            with patch.object(requests, "get", return_value=response(payload)):
                self.assert_error("OFFICIAL_INVALID_SCHEMA", lambda: self.fetch(expected=("000001.SZ",)))

    def test_numeric_excel_code_is_normalized_and_native_excel_date_is_supported(self):
        payload = workbook(rows=[["主板", 1, "Example", datetime(1991, 4, 3)]])
        with patch.object(requests, "get", return_value=response(payload)):
            page = self.fetch(expected=("000001.SZ",))
        self.assertEqual(page["rows"][0]["ts_code"], "000001.SZ")
        self.assertEqual(page["rows"][0]["list_date"], "1991-04-03")

    def test_historical_date_or_naive_clock_is_rejected_without_download(self):
        with patch.object(requests, "get") as get:
            self.assert_error("OFFICIAL_DATE_MISMATCH", lambda: self.module.fetch_szse_a_share_master(
                "2026-11-16", {"000001.SZ"}, clock=lambda: NOW))
            self.assert_error("OFFICIAL_DATE_MISMATCH", lambda: self.fetch(clock=lambda: NOW.replace(tzinfo=None)))
        get.assert_not_called()

    def test_crossing_shanghai_midnight_cannot_create_current_t_evidence(self):
        ticks = iter([NOW, datetime(2026, 11, 17, 16, tzinfo=timezone.utc)])
        with patch.object(requests, "get", return_value=response()):
            self.assert_error("OFFICIAL_DATE_MISMATCH", lambda: self.fetch(clock=lambda: next(ticks)))

    def test_redirect_nonofficial_response_and_bad_workbook_fail_safely(self):
        for result, code in [(response(status=302), "OFFICIAL_HTTP_ERROR"),
                             (response(url="https://example.com/file.xlsx"), "OFFICIAL_HTTP_ERROR"),
                             (response(body=b"not an xlsx"), "OFFICIAL_INVALID_SCHEMA")]:
            with patch.object(requests, "get", return_value=result):
                self.assert_error(code, self.fetch)

    def test_transport_failure_is_safe_bounded_and_does_not_leak_details(self):
        with patch.object(requests, "get", side_effect=requests.Timeout("private-token-error")) as get:
            error = self.assert_error("OFFICIAL_TRANSPORT_ERROR", self.fetch)
        self.assertEqual(get.call_count, 1)
        self.assertNotIn("private-token-error", str(error))

    def test_empty_cross_check_anchor_is_not_a_completeness_proof(self):
        with patch.object(requests, "get") as get:
            self.assert_error("OFFICIAL_CROSS_CHECK_FAILED", lambda: self.fetch(expected=()))
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
