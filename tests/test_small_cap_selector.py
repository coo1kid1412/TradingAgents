"""Pure selection and point-in-time contracts; no network or LLM access."""

from copy import deepcopy
import importlib
from pathlib import Path
import unittest


def financial(**changes):
    row = {"end_date": "20251231", "ann_date": "20260330", "f_ann_date": "20260330",
           "report_type": "1", "n_income_attr_p": 20_000_000, "revenue": 300_000_000,
           "observed_at": "2026-11-17T19:31:00+08:00", "revision_id": "original"}
    row.update(changes)
    return row


def record(index=0, **changes):
    row = {"ts_code": f"{600000 + index:06d}.SH", "name": f"样本{index}",
           "exchange": "SSE", "market": "主板", "curr_type": "CNY",
           "list_date": "20200101", "delist_date": None,
           "industry": "机械", "industry_basis": "current_snapshot",
           "security_status": "NORMAL", "status_effective_date": "2026-11-17",
           "observed_at": "2026-11-17T19:31:00+08:00", "trade_date": "20261117",
           "close": 10, "total_market_cap_yuan": 1_000_000_000 + index * 10_000_000,
           "volume": 100000, "suspended": False, "financials": [financial()]}
    row.update(changes)
    return row


def snapshot(rows=None, **changes):
    result = {"schema_version": "small-cap-snapshot-v1", "mode": "forward",
              "data_date": "2026-11-17", "as_of_at": "2026-11-17T19:40:00+08:00",
              "target_trade_date": "2026-11-18",
              "calendar": [{"cal_date": "20261117", "is_open": 1},
                           {"cal_date": "20261118", "is_open": 1}],
              "records": rows if rows is not None else [record(i) for i in range(12)],
              "coverage": {"complete": True}, "raw": {}}
    result.update(changes)
    return result


class SelectorTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "tradingagents/strategies/small_cap/selector.py"
        self.assertTrue(path.exists(), "small-cap pure selector is not implemented")
        self.module = importlib.import_module("tradingagents.strategies.small_cap.selector")

    def select(self, value=None):
        return self.module.select_top10(value or snapshot())

    def test_rank_is_smallest_total_market_cap_then_code_and_maximum_ten(self):
        value = snapshot(list(reversed(snapshot()["records"])))
        result = self.select(value)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["ranking_complete"])
        self.assertEqual([r["ts_code"] for r in result["top10"]],
                         [record(i)["ts_code"] for i in range(10)])
        self.assertEqual([r["rank"] for r in result["top10"]], list(range(1, 11)))
        self.assertEqual(result["regime"], "UNKNOWN")
        self.assertEqual(result["new_position_policy"], "NOT_EVALUATED")
        self.assertFalse(result["rebalance_due"])

    def test_fewer_than_ten_does_not_invent_replacements(self):
        result = self.select(snapshot([record()]))
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertEqual(len(result["top10"]), 1)

    def test_thresholds_are_exact_and_missing_values_are_not_zero(self):
        cases = [
            ({"total_market_cap_yuan": 1_000_000_000}, True),
            ({"total_market_cap_yuan": 10_000_000_000}, True),
            ({"total_market_cap_yuan": 999_999_999}, False),
            ({"total_market_cap_yuan": 10_000_000_001}, False),
            ({"close": 50}, True), ({"close": 50.01}, False),
            ({"close": -1}, None), ({"close": float("nan")}, None),
            ({"close": None}, None), ({"total_market_cap_yuan": None}, None),
            ({"list_date": "20251107"}, True),  # exactly 375 calendar days
            ({"list_date": "20251108"}, False),
        ]
        for changes, passed in cases:
            with self.subTest(changes=changes):
                result = self.select(snapshot([record(**changes)]))
                self.assertEqual(bool(result["top10"]), passed is True)
                self.assertEqual(result["ranking_complete"], passed is not None)

    def test_financial_filters_use_same_annual_consolidated_version(self):
        for change in ({"n_income_attr_p": 0}, {"revenue": 100_000_000},
                       {"n_income_attr_p": -1}, {"revenue": -1}):
            with self.subTest(change=change):
                result = self.select(snapshot([record(financials=[financial(**change)])]))
                self.assertEqual(result["status"], "INSUFFICIENT")
                self.assertEqual(result["top10"], [])
        for change in ({"n_income_attr_p": None}, {"revenue": None}):
            with self.subTest(change=change):
                self.assertEqual(self.select(snapshot([record(financials=[financial(**change)])]))["status"], "BLOCKED")

    def test_date_only_announcement_is_not_visible_on_announcement_date(self):
        value = snapshot([record(financials=[financial(ann_date="20261117", f_ann_date="20261117")])])
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_future_financial_rows_do_not_change_prior_selection_hash(self):
        value = snapshot()
        before = self.select(value)
        value["records"][0]["financials"].append(financial(
            end_date="20261231", ann_date="20270330", f_ann_date="20270330",
            observed_at="2027-03-31T19:30:00+08:00", n_income_attr_p=-999,
        ))
        self.assertEqual(self.select(value)["selection_hash"], before["selection_hash"])

    def test_financial_restatement_keeps_its_own_visibility(self):
        value = snapshot()
        before = self.select(value)
        value["records"][0]["financials"].append(financial(
            ann_date="20261118", f_ann_date="20261118", report_type="4",
            observed_at="2026-11-18T20:00:00+08:00", n_income_attr_p=-1,
        ))
        self.assertEqual(self.select(value)["selection_hash"], before["selection_hash"])

    def test_later_observation_cannot_backfill_an_old_snapshot(self):
        value = snapshot()
        value["records"][0]["financials"][0]["observed_at"] = "2026-11-18T19:30:00+08:00"
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_newer_annual_period_wins_over_later_old_year_correction(self):
        value = snapshot([record(financials=[financial(), financial(
            end_date="20241231", ann_date="20261101", f_ann_date="20261101",
            report_type="4", n_income_attr_p=-1,
        )])])
        self.assertEqual(self.select(value)["top10"][0]["financial_period"], "2025-12-31")

    def test_latest_annual_missing_fields_cannot_fall_back_to_older_profit(self):
        value = snapshot([record(financials=[financial(revenue=None), financial(
            end_date="20241231", ann_date="20250330", f_ann_date="20250330",
        )])])
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_parent_only_or_quarterly_financial_rows_are_not_annual_facts(self):
        for change in ({"report_type": "2"}, {"end_date": "20260930"}):
            with self.subTest(change=change):
                self.assertEqual(self.select(snapshot([record(financials=[financial(**change)])]))["status"], "BLOCKED")

    def test_old_annual_financials_are_unknown_not_profitable_evidence(self):
        row = financial(end_date="20241231", ann_date="20250330", f_ann_date="20250330")
        self.assertEqual(self.select(snapshot([record(financials=[row])]))["status"], "BLOCKED")

    def test_bad_market_or_security_status_is_excluded(self):
        for change in ({"market": "创业板"}, {"exchange": "BSE"}, {"curr_type": "USD"},
                       {"security_status": "ST"}, {"security_status": "DELISTING"},
                       {"delist_date": "20261116"}, {"suspended": True}, {"volume": 0}):
            with self.subTest(change=change):
                self.assertEqual(self.select(snapshot([record(**change)]))["status"], "INSUFFICIENT")

    def test_future_delisting_does_not_remove_earlier_stock(self):
        value = snapshot()
        before = self.select(value)
        value["records"][0]["delist_date"] = "20271117"
        self.assertEqual(self.select(value)["selection_hash"], before["selection_hash"])

    def test_unknown_candidate_ahead_of_boundary_blocks_but_later_one_does_not(self):
        for index, expected in ((0, "BLOCKED"), (11, "SUCCESS")):
            value = snapshot()
            value["records"][index]["financials"] = []
            with self.subTest(index=index):
                result = self.select(value)
                self.assertEqual(result["status"], expected)
                if expected == "BLOCKED":
                    self.assertEqual(result["top10"], [])

    def test_tied_unknown_uses_full_sort_key(self):
        value = snapshot([record(i, total_market_cap_yuan=2_000_000_000) for i in range(11)])
        value["records"][-1]["financials"] = []
        self.assertEqual(self.select(value)["status"], "SUCCESS")
        value["records"][0]["financials"] = []
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_t_minus_one_quote_is_not_today_and_api_gap_is_not_a_halt(self):
        value = snapshot()
        value["records"][0]["trade_date"] = "20261116"
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_global_truncation_blocks_even_with_ten_valid_rows(self):
        self.assertEqual(self.select(snapshot(coverage={"complete": False}))["status"], "BLOCKED")

    def test_provider_optional_data_warnings_survive_into_the_report_truth(self):
        warning = {"endpoint":"stk_limit", "code":"PERMISSION_DENIED"}
        value = snapshot(warnings=[warning])
        result = self.select(value)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn(warning, result["warnings"])

    def test_target_date_uses_calendar_and_seasonal_month_uses_target(self):
        value = snapshot()
        value.update(data_date="2026-12-31", as_of_at="2026-12-31T19:40:00+08:00",
                     target_trade_date="2027-01-04", calendar=[
                         {"cal_date": "20261231", "is_open": 1},
                         {"cal_date": "20270101", "is_open": 0},
                         {"cal_date": "20270104", "is_open": 1}])
        for row in value["records"]:
            row.update(trade_date="20261231", observed_at="2026-12-31T19:30:00+08:00",
                       status_effective_date="2026-12-31")
            row["financials"][0]["observed_at"] = row["observed_at"]
        result = self.select(value)
        self.assertEqual(result["new_position_policy"], "BLOCK")
        self.assertEqual(result["strategy_mode"], "SEASONAL_STOCK_AVOID")
        self.assertEqual(len(result["top10"]), 10)

    def test_missing_calendar_and_expired_cutoff_fail_closed(self):
        for changes in ({"calendar": []}, {"target_trade_date": "2026-11-19"},
                        {"as_of_at": "2026-11-18T00:00:00+08:00"},
                        {"as_of_at": "2026-11-17T19:40:00"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.select(snapshot(**changes))["status"], "BLOCKED")

    def test_historical_source_refetch_is_not_strict_pit(self):
        self.assertEqual(self.select(snapshot(mode="historical_unverified"))["status"], "BLOCKED")

    def test_does_not_mutate_frozen_input(self):
        value = snapshot()
        before = deepcopy(value)
        result = self.select(value)
        self.assertEqual(value, before)
        self.assertEqual(result["selection_hash"], self.select(value)["selection_hash"])

    def test_stale_overpriced_or_big_cap_does_not_exclude_unknown_today(self):
        for changes in ({"close": 60, "trade_date": "20261116"},
                        {"total_market_cap_yuan": 20_000_000_000, "valuation_date": "20261116"}):
            value = snapshot()
            value["records"][0].update(changes)
            self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_future_observation_cannot_prove_an_exclusion_or_sort_boundary(self):
        value = snapshot()
        value["records"][0].update(observed_at="2026-11-18T20:00:00+08:00",
                                    total_market_cap_yuan=20_000_000_000)
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_later_announcement_timestamp_is_the_conservative_visibility(self):
        value = snapshot([record(financials=[financial(ann_date="20261117", f_ann_date="20260330")])])
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_stale_zero_volume_cannot_prove_current_suspension(self):
        value = snapshot()
        value["records"][0].update(trade_date="20261116", volume=0)
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_newer_annual_with_unknown_publication_cannot_fall_back_to_old_profit(self):
        value = snapshot([record(financials=[financial(ann_date=None, f_ann_date=None),
                         financial(end_date="20241231", ann_date="20250330", f_ann_date="20250330")])])
        value.update(data_date="2026-05-18", as_of_at="2026-05-18T19:40:00+08:00", target_trade_date="2026-05-19",
                     calendar=[{"cal_date":"20260518","is_open":1},{"cal_date":"20260519","is_open":1}])
        value["records"][0].update(trade_date="20260518", observed_at="2026-05-18T19:30:00+08:00",
                                    status_effective_date="20260518")
        for row in value["records"][0]["financials"]:
            row["observed_at"] = "2026-05-18T19:30:00+08:00"
        self.assertEqual(self.select(value)["status"], "BLOCKED")
        value["records"][0]["financials"][0].update(ann_date="20260330", f_ann_date="20260330", report_type=None)
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_later_fetch_of_original_version_cannot_replace_latest_report(self):
        value = snapshot([record(financials=[financial(report_type="1"), financial(
            report_type="5", n_income_attr_p=-1, observed_at="2026-11-17T19:32:00+08:00")])])
        self.assertEqual(self.select(value)["status"], "INSUFFICIENT")
        self.assertEqual(self.select(value)["top10"][0]["profit"], 20_000_000)

    def test_conflicting_latest_versions_are_not_ordered_by_request_time(self):
        value = snapshot([record(financials=[financial(), financial(
            n_income_attr_p=-1, observed_at="2026-11-17T19:32:00+08:00")])])
        self.assertEqual(self.select(value)["status"], "BLOCKED")

    def official_census_fixture(self):
        value = snapshot()
        orphan = record(99, ts_code="001999.SZ", exchange=None, market=None, curr_type=None,
                        list_date=None, security_status="UNKNOWN", status_effective_date=None,
                        close=None, volume=None, suspended=None, trade_date=None, total_market_cap_yuan=None)
        value["records"].append(orphan)
        value["raw"] = {
            "stock_basic": [{"params":{"list_status":"L", "limit":5000,"offset":0,"exchange":""}, "rows":[{
                "ts_code":"000001.SZ", "list_status":"L", "exchange":"SZSE", "curr_type":"CNY", "list_date":"19910403"}],
                "observed_at":"2026-11-17T19:31:00+08:00"},
                {"params":{"list_status":"D","limit":5000,"offset":0,"exchange":""},"rows":[],"observed_at":"2026-11-17T19:31:00+08:00"},
                {"params":{"list_status":"P","limit":5000,"offset":0,"exchange":""},"rows":[],"observed_at":"2026-11-17T19:31:00+08:00"}],
            "szse_official_a_share_master": [{
                "source_url":"https://www.szse.cn/api/report/ShowReport",
                "params":{"SHOWTYPE":"xlsx","CATALOGID":"1110","TABKEY":"tab1"},
                "mode":"forward","data_date":"2026-11-17","scope":"current_as_of_listed_a_shares",
                "observed_at":"2026-11-17T19:35:00+08:00", "coverage_complete":True,
                "cross_check_complete":True, "rows":[{"ts_code":"000001.SZ", "name":"平安银行",
                                                       "market":"主板", "list_date":"19910403"}]}]}
        page = value["raw"]["szse_official_a_share_master"][0]
        page["rows_hash"] = self.module.digest(page["rows"])
        return value

    def test_finance_only_identifier_outside_both_verified_censuses_is_not_a_stock_candidate(self):
        result = self.select(self.official_census_fixture())
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(next(row for row in result["exclusions"] if row["ts_code"]=="001999.SZ")["reason"],
                         "outside_verified_listed_security_universe")

    def test_official_negative_membership_cannot_hide_positive_or_incomplete_evidence(self):
        for mutation in ("future", "partial", "positive", "present", "wrong_source", "wrong_day", "primary_future", "tampered_hash",
                         "filtered_primary", "truncated_primary", "historical_mode", "historical_date", "historical_scope"):
            value = self.official_census_fixture()
            page = value["raw"]["szse_official_a_share_master"][0]
            if mutation == "future": page["observed_at"] = "2026-11-18T19:35:00+08:00"
            if mutation == "partial": page["rows"] = []
            if mutation == "positive": value["raw"]["daily"] = [{"rows":[{"ts_code":"001999.SZ"}]}]
            if mutation == "present": page["rows"].append({"ts_code":"001999.SZ","list_date":"20200101","market":"主板"})
            if mutation == "wrong_source": page["source_url"] = "https://example.com/master"
            if mutation == "wrong_day": page["observed_at"] = "2026-11-16T19:35:00+08:00"
            if mutation == "primary_future": value["raw"]["stock_basic"][0]["observed_at"] = "2026-11-18T19:31:00+08:00"
            if mutation == "tampered_hash": page["rows_hash"] = "0" * 64
            if mutation == "filtered_primary": value["raw"]["stock_basic"][0]["params"]["ts_code"] = "000001.SZ"
            if mutation == "truncated_primary": value["raw"]["stock_basic"][0]["params"]["limit"] = 1
            if mutation == "historical_mode": page["mode"] = "historical"
            if mutation == "historical_date": page["data_date"] = "2026-11-16"
            if mutation == "historical_scope": page["scope"] = "historical"
            with self.subTest(mutation=mutation):
                self.assertEqual(self.select(value)["status"], "BLOCKED")

    def test_future_listing_is_not_a_current_official_census_anchor(self):
        value = self.official_census_fixture()
        value["raw"]["stock_basic"][0]["rows"].append({"ts_code":"001998.SZ", "list_status":"L",
            "exchange":"SZSE","curr_type":"CNY","list_date":"20261118"})
        self.assertEqual(self.select(value)["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
