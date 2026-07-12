"""主归档逻辑：扫描 reports/ → 写入 SQLite。"""

from __future__ import annotations

import datetime as _dt
import logging
import subprocess
from pathlib import Path
from typing import Iterable

from tradingagents.harness import db as _db
from tradingagents.harness.extractor import (
    ExtractResult,
    classify_window,
    extract_from_report,
    parse_report_dir_name,
)

logger = logging.getLogger(__name__)

# 预测表的所有 NULL-able 字段（用于批量插入）
_PRED_FIELDS = [
    # 共用
    "current_price", "style", "theme_stage", "composite_score", "momentum_score",
    # RM
    "rm_rating", "rm_conviction",
    "target_price_low", "target_price_mid", "target_price_high",
    "bull_target", "bull_prob",
    "base_target", "base_prob",
    "bear_target", "bear_prob",
    "base_case_expected_return_pct",
    "deviation_pct", "threshold_dn_pct", "threshold_up_pct",
    # RM 评级链审计（2026-06 P0：回测分腿归因）
    "valuation_regime", "regime_legs", "rating_raw", "peg_confidence",
    "overlay_style_adj", "overlay_vote_adj", "overlay_catalyst_adj",
    # PM
    "pm_rating", "pm_conviction_stars",
    "pm_invest_judgment", "pm_entry_judgment", "pm_action_keyword",
    "pm_size_low_pct", "pm_size_high_pct",
    "pm_entry_low", "pm_entry_high",
    "pm_tp1", "pm_tp2", "pm_tp3",
    "pm_sl_soft", "pm_sl_hard",
    "pm_horizon_months_low", "pm_horizon_months_high",
    "pm_rating_adjusted_from_rm",
    "market_risk_level", "market_entry_gate", "market_position_cap_pct",
    "short_term_trend", "short_term_confidence", "theme_outlook_12m",
]


def _get_git_commit() -> str | None:
    """获取当前 git HEAD commit hash（用于代码版本追溯）。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception as e:
        logger.debug("get_git_commit 失败: %s", e)
    return None


def resolve_trade_date(parsed: dict, extract: ExtractResult) -> str:
    """优先使用 PM_SUMMARY 的分析日，目录时间只代表报告生成时刻。"""
    candidate = (extract.pm_summary or {}).get("trade_date") or (extract.rm_summary or {}).get("trade_date")
    try:
        return _dt.date.fromisoformat(str(candidate)).isoformat()
    except (TypeError, ValueError):
        return parsed["trade_date"]


def _merge_pred_fields(extract: ExtractResult) -> dict:
    """把 RM_SUMMARY + PM_SUMMARY 合并成 predictions 表所需的字段字典。

    冲突字段（如 current_price 两边都有）以 PM 为准（PM 时间靠后，更新）。
    """
    out: dict = {f: None for f in _PRED_FIELDS}

    rm = extract.rm_summary or {}
    pm = extract.pm_summary or {}

    # 共用字段
    out["current_price"] = pm.get("current_price") or rm.get("current_price")
    out["style"] = rm.get("style")  # 仅 RM 输出
    out["theme_stage"] = rm.get("theme_stage")
    out["composite_score"] = rm.get("composite_score")
    out["momentum_score"] = rm.get("momentum_score")

    # RM 字段
    for k in ("rm_rating", "rm_conviction",
              "target_price_low", "target_price_mid", "target_price_high",
              "bull_target", "bull_prob", "base_target", "base_prob",
              "bear_target", "bear_prob", "base_case_expected_return_pct",
              "deviation_pct", "threshold_dn_pct", "threshold_up_pct",
              "valuation_regime", "rating_raw", "peg_confidence",
              "overlay_style_adj", "overlay_vote_adj", "overlay_catalyst_adj"):
        out[k] = rm.get(k)

    # regime_legs：LLM 可能照抄成 YAML flow mapping（dict）或带引号字符串，统一存 TEXT
    legs = rm.get("regime_legs")
    out["regime_legs"] = str(legs) if legs is not None else None

    # PM 字段
    for k in ("pm_rating", "pm_conviction_stars",
              "pm_invest_judgment", "pm_entry_judgment", "pm_action_keyword",
              "pm_size_low_pct", "pm_size_high_pct",
              "pm_entry_low", "pm_entry_high",
              "pm_tp1", "pm_tp2", "pm_tp3",
              "pm_sl_soft", "pm_sl_hard",
              "pm_horizon_months_low", "pm_horizon_months_high",
              "market_risk_level", "market_entry_gate", "market_position_cap_pct",
              "short_term_trend", "short_term_confidence", "theme_outlook_12m"):
        out[k] = pm.get(k)

    # bool → int
    adj = pm.get("pm_rating_adjusted_from_rm")
    if isinstance(adj, bool):
        out["pm_rating_adjusted_from_rm"] = 1 if adj else 0
    elif isinstance(adj, (int, float)):
        out["pm_rating_adjusted_from_rm"] = 1 if adj else 0
    else:
        out["pm_rating_adjusted_from_rm"] = None

    return out


def archive_run(report_dir: Path, db_path=None) -> int | None:
    """归档单个报告目录。

    Returns:
        run_id（成功插入）/ None（跳过：已存在 或 目录名不规范）
    """
    if not report_dir.is_dir():
        return None

    parsed = parse_report_dir_name(report_dir.name)
    if parsed is None:
        logger.warning("跳过：目录名不规范 %s", report_dir.name)
        return None

    if _db.run_exists(str(report_dir), db_path):
        logger.debug("跳过：已归档 %s", report_dir.name)
        return None

    extract = extract_from_report(report_dir)
    parsed["trade_date"] = resolve_trade_date(parsed, extract)
    window = classify_window(parsed["report_timestamp"])
    pred_fields = _merge_pred_fields(extract)

    # 判定归档状态
    if extract.rm_parsed and extract.pm_parsed:
        archive_status = "archived"
    elif extract.rm_parsed or extract.pm_parsed:
        archive_status = "partial"
    else:
        archive_status = "failed"

    git_commit = _get_git_commit()

    with _db.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (ticker, company_name, trade_date, report_timestamp,
                              report_window, report_dir, git_commit, archive_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parsed["ticker"], parsed["company_name"], parsed["trade_date"],
                parsed["report_timestamp"], window, str(report_dir),
                git_commit, archive_status,
            ),
        )
        run_id = cur.lastrowid

        # predictions 表插入（含解析状态）
        cols = ["run_id"] + _PRED_FIELDS + [
            "rm_yaml_parsed", "pm_yaml_parsed", "parse_warnings",
        ]
        vals = (
            [run_id]
            + [pred_fields[f] for f in _PRED_FIELDS]
            + [
                1 if extract.rm_parsed else 0,
                1 if extract.pm_parsed else 0,
                " | ".join(extract.warnings) if extract.warnings else None,
            ]
        )
        placeholders = ", ".join(["?"] * len(cols))
        col_list = ", ".join(cols)
        conn.execute(
            f"INSERT INTO predictions ({col_list}) VALUES ({placeholders})",
            vals,
        )

        # 注册 4 个 horizon 的真值采集占位
        ref_price = pred_fields["current_price"]
        if ref_price is not None and ref_price > 0:
            for horizon in ("T", "T+1", "T+5", "T+30"):
                conn.execute(
                    """
                    INSERT INTO outcomes (run_id, horizon, reference_price,
                                          fetch_status, direction_predicted)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (run_id, horizon, ref_price, _derive_direction(pred_fields)),
                )
        else:
            logger.warning(
                "run %d 缺 current_price，不创建 outcomes 占位（archive_status=%s）",
                run_id, archive_status,
            )

    logger.info(
        "归档 run %d: %s @ %s [%s] (status=%s)",
        run_id, parsed["ticker"], parsed["report_timestamp"],
        window, archive_status,
    )
    return run_id


def _derive_direction(pred_fields: dict) -> str | None:
    """根据 PM/RM 评级推导 direction_predicted（long / short / neutral）。"""
    rating = pred_fields.get("pm_rating") or pred_fields.get("rm_rating")
    if rating is None:
        return None
    upper = str(rating).upper()
    if upper in ("BUY", "OVERWEIGHT"):
        return "long"
    if upper in ("SELL", "UNDERWEIGHT"):
        return "short"
    if upper == "HOLD":
        return "neutral"
    return None


def archive_all_reports(reports_root: Path, db_path=None) -> dict:
    """全量扫描 reports/ 目录，归档新增的。

    Returns:
        统计字典：{total, archived, partial, failed, skipped}
    """
    stats = {"total": 0, "archived": 0, "partial": 0, "failed": 0, "skipped": 0}

    if not reports_root.is_dir():
        logger.error("reports 根目录不存在: %s", reports_root)
        return stats

    for d in sorted(reports_root.iterdir()):
        if not d.is_dir():
            continue
        stats["total"] += 1
        run_id = archive_run(d, db_path)
        if run_id is None:
            stats["skipped"] += 1
            continue
        # 查 archive_status
        with _db.connect(db_path) as conn:
            row = conn.execute(
                "SELECT archive_status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                stats[row["archive_status"]] = stats.get(row["archive_status"], 0) + 1

    return stats


def main():
    """CLI 入口：python -m tradingagents.harness.archive"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    reports_root = Path(__file__).resolve().parents[2] / "reports"
    print(f"扫描 reports 根目录: {reports_root}")
    _db.init_db()
    stats = archive_all_reports(reports_root)
    print(f"\n归档完成统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
