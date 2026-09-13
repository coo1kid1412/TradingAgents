"""Read-only evidence audits, without trading-performance claims or LLM calls."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from contextlib import closing
from zoneinfo import ZoneInfo

from tradingagents.harness import db

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_RULES = {
    "missing_price": ("报告数据", "参考价格缺失或无效", "核对行情来源和归档提取，不补造价格。", "验收：缺失/非正价格明确降级，不进入收益评价。"),
    "summary_incomplete": ("报告契约", "管理层摘要未完整归档", "对照原报告检查 YAML 提取，区分历史格式和真实生成失败。", "验收：已知报告可完整提取；损坏摘要明确失败，不猜评级。"),
    "wait_position_conflict": ("报告契约", "等待与新建仓位冲突", "核对 PM 动作、仓位及归档语义。", "验收：WAIT 不得同时建议正的新建仓位。"),
    "untracked_report": ("数据跟踪", "有参考价的报告未登记结果跟踪", "核对归档后的跟踪注册，不因暂无结果就显示健康。", "验收：有效报告有跟踪记录或明确的排除原因。"),
    "overdue_results": ("数据跟踪", "已过目标日仍未完成", "检查行情缓存和真值抓取，不能继续解释为尚未到期。", "验收：到期缺数据和真正未到期分开。"),
    "missing_target_date": ("数据跟踪", "待跟踪结果缺少到期日", "补交易日到期计算；本版仅标未知，不猜到期日。", "验收：每个待跟踪结果有可验证到期日或明确不支持原因。"),
    "fetch_failed": ("数据跟踪", "存在未完成的采集失败", "核对失败记录与数据源，保留历史与本周区分。", "验收：错误可定位，修复后可受控重试。"),
    "data_missing": ("数据跟踪", "已到期但行情缺失或不完整", "按目标交易日补齐行情；不以之后的价格代替。", "验收：完整交易日序列及有效 OHLC 到齐后才评价。"),
    "retry_exhausted": ("数据跟踪", "行情重试达到上限", "先检查数据源权限和历史覆盖，再人工重新入队。", "验收：最多自动尝试3次、间隔至少24小时；不无限刷取。"),
    "schedule_error": ("数据跟踪", "交易日目标无法确定", "核对市场、报告时间和交易日历覆盖；不猜节假日。", "验收：修复支持范围后再重新入队。"),
    "invalid_result": ("数据跟踪", "已采集结果缺日期或超出审计时点", "核对目标日和抓取记录，暂不计为已完成。", "验收：未来/缺日期结果不能进入当前已完成统计。"),
    "wait_scored_as_position": ("旧评价口径", "旧方向评分包含 WAIT 报告", "停止将方向评分解释为实际交易输赢，保留原值供审计。", "验收：WAIT、新建仓和已有持仓分别评价，不虚构成交。"),
    "foreign_benchmark": ("旧评价口径", "非 A 股记录使用 A 股 ETF 基准", "独立修复市场日历、基准及收益时点后再比较。", "验收：美股不会与沪深 300 混作同市场基准。"),
}


def _date(value, *, utc=False):
    if not value:
        return None
    try:
        stamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc if utc else _SHANGHAI)
        return stamp.astimezone(_SHANGHAI).date()
    except ValueError:
        return None


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_snapshot(db_path=None, *, today=None) -> dict:
    today = today or dt.datetime.now(_SHANGHAI).date()
    cutoff = today - dt.timedelta(days=6)
    path = db.get_db_path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("BEGIN")
            raw_runs = [dict(r) for r in conn.execute("SELECT r.*, p.* FROM runs r LEFT JOIN predictions p ON p.run_id=r.id ORDER BY r.report_timestamp,r.id")]
            outcomes = [dict(r) for r in conn.execute("SELECT * FROM outcomes ORDER BY run_id,horizon")]

    runs = {}
    for row in raw_runs:
        day = _date(row["report_timestamp"]) or _date(row["trade_date"])
        if day is not None and day <= today:
            runs[row["id"]] = {**row, "report_day": day.isoformat()}
    recent = [r for r in runs.values() if r["report_day"] >= cutoff.isoformat()]
    issue_refs: dict[str, dict] = {}

    def flag(rule, row, observation=None):
        refs = issue_refs.setdefault(rule, {"run_ids": set(), "observation_ids": set()})
        refs["run_ids"].add(row["id"])
        if observation:
            refs["observation_ids"].add(observation)

    tracked_ids = {o["run_id"] for o in outcomes}
    for row in runs.values():
        price = _number(row.get("current_price"))
        if price is None or price <= 0:
            flag("missing_price", row)
        elif row["id"] not in tracked_ids:
            flag("untracked_report", row)
        if not row.get("rm_yaml_parsed") or not row.get("pm_yaml_parsed"):
            flag("summary_incomplete", row)
        if str(row.get("pm_action_keyword") or "").strip().upper() == "WAIT":
            if any((_number(row.get(k)) or 0) > 0 for k in ("pm_size_low_pct", "pm_size_high_pct")):
                flag("wait_position_conflict", row)

    tracking = dict.fromkeys(("completed", "not_due", "due_today", "overdue", "due_unknown", "failed", "invalid", "data_missing", "retry_exhausted", "schedule_error"), 0)
    horizon_counts = Counter()
    observation_evidence = {}
    completed_runs = set()
    new_observations = 0
    for outcome in outcomes:
        row = runs.get(outcome["run_id"])
        if row is None:
            continue
        key = f"{outcome['run_id']}:{outcome['horizon']}"
        housekeeping = {"fetched_at", "attempt_count", "next_retry_at"}
        housekeeping.update(k for k in ("schedule_version", "due_at") if outcome.get(k) is None)
        observation_evidence[key] = _digest({k: v for k, v in outcome.items() if k not in housekeeping})
        target = _date(outcome.get("target_date"))
        status = outcome["fetch_status"]
        if status == "fetched":
            if target is None or target > today:
                tracking["invalid"] += 1
                flag("invalid_result", row, key)
                continue
            tracking["completed"] += 1
            completed_runs.add(row["id"])
            horizon_counts[outcome["horizon"]] += 1
            fetched = _date(outcome.get("fetched_at"), utc=True)
            if fetched is not None and cutoff <= fetched <= today:
                new_observations += 1
            if str(row.get("pm_action_keyword") or "").strip().upper() == "WAIT" and outcome.get("direction_predicted"):
                flag("wait_scored_as_position", row, key)
            ticker = str(row["ticker"]).upper()
            if not re.fullmatch(r"\d{6}(?:\.(?:SH|SZ|BJ))?", ticker):
                benchmark = str(outcome.get("benchmark_ticker") or "").split(".")[0]
                if benchmark in {"510300", "510500", "588000", "159915", "512760", "159928"}:
                    flag("foreign_benchmark", row, key)
        elif status == "failed":
            tracking["failed"] += 1
            flag("fetch_failed", row, key)
        elif status in {"data_missing", "retry_exhausted", "schedule_error"}:
            tracking[status] += 1
            flag(status, row, key)
        elif status in {"pending", "not_due"}:
            target = _date(outcome.get("due_at"), utc=True) or target
            if target is None:
                tracking["due_unknown"] += 1
                flag("missing_target_date", row, key)
            elif target < today:
                tracking["overdue"] += 1
                flag("overdue_results", row, key)
            elif target == today:
                tracking["due_today"] += 1
            else:
                tracking["not_due"] += 1
        else:
            tracking["invalid"] += 1
            flag("invalid_result", row, key)

    issues = []
    for rule, (kind, title, action, acceptance) in _RULES.items():
        if rule not in issue_refs:
            continue
        refs = issue_refs[rule]
        ids = sorted(refs["run_ids"])
        issues.append({
            "id": rule, "kind": kind, "title": title,
            "run_ids": ids, "observation_ids": sorted(refs["observation_ids"]),
            "recent_reports": sum(runs[i]["report_day"] >= cutoff.isoformat() for i in ids),
            "latest_report_date": max(runs[i]["report_day"] for i in ids),
            "action": action, "acceptance": acceptance,
            "examples": [{"run_id": i, "ticker": runs[i]["ticker"], "report_dir": runs[i]["report_dir"]} for i in sorted(ids, key=lambda i: (runs[i]["report_day"], i), reverse=True)[:3]],
        })
    # Current contract failures precede historical backlogs; neither is a strategy verdict.
    issues.sort(key=lambda issue: (not bool(issue["recent_reports"]), issue["kind"] == "旧评价口径"))
    short_views = sum(str(r.get("short_term_trend") or "").strip() in {"上涨", "下跌", "震荡", "震荡偏涨", "震荡偏跌"} for r in runs.values())
    latest = next(reversed(runs.values()), {})
    recorded_version = latest.get("git_commit")
    counts = {
        "reports": len(runs), "new_reports": len(recent),
        "ticker_dates": len({(str(r["ticker"]).upper(), r["trade_date"]) for r in runs.values()}),
        "observations": tracking["completed"], "evaluated_reports": len(completed_runs),
        "evaluated_ticker_dates": len({(str(runs[i]["ticker"]).upper(), runs[i]["trade_date"]) for i in completed_runs}),
        "new_observations": new_observations,
        "archive_failures": sum(r["archive_status"] == "failed" for r in runs.values()),
        "new_archive_failures": sum(r["archive_status"] == "failed" for r in recent),
        "wait_reports": sum(str(r.get("pm_action_keyword") or "").strip().upper() == "WAIT" for r in runs.values()),
        "short_term_views": short_views,
        "latest_recorded_version_reports": sum(r.get("git_commit") == recorded_version for r in runs.values()) if recorded_version else 0,
    }
    return {
        "schema_version": "quality-v1", "snapshot_date": today.isoformat(), "window_start": cutoff.isoformat(),
        "counts": counts, "tracking": tracking, "horizon_counts": dict(horizon_counts), "issues": issues,
        "latest_recorded_version": recorded_version,
        "data_healthy": bool(runs) and not any(i["kind"] != "旧评价口径" for i in issues),
        "evidence": {
            "reports": {str(i): _digest({k: v for k, v in r.items() if k != "created_at"}) for i, r in runs.items()},
            "observations": observation_evidence,
        },
    }


def compare_snapshot(current, previous=None) -> dict:
    previous = previous or {}
    if not isinstance(previous, dict):
        previous = {}
    evidence = previous.get("evidence", {})
    issues = previous.get("issues", [])
    valid_evidence = isinstance(evidence, dict) and all(isinstance(evidence.get(kind, {}), dict) for kind in ("reports", "observations"))
    valid_issues = isinstance(issues, list) and all(
        isinstance(issue, dict) and isinstance(issue.get("id"), str)
        and isinstance(issue.get("run_ids"), list) and isinstance(issue.get("observation_ids"), list)
        for issue in issues
    )
    if not valid_evidence or not valid_issues:
        previous = {}
    old_evidence = previous.get("evidence", {})
    changed = {}
    for kind, values in current["evidence"].items():
        old = old_evidence.get(kind, {})
        changed[kind] = sorted(k for k in values.keys() | old.keys() if values.get(k) != old.get(k))
    old_issues = {i["id"]: (i["run_ids"], i["observation_ids"]) for i in previous.get("issues", [])}
    new_issues = {i["id"]: (i["run_ids"], i["observation_ids"]) for i in current["issues"]}
    changed_issues = [key for key, refs in new_issues.items() if old_issues.get(key) != refs]
    resolved = sorted(old_issues.keys() - new_issues.keys())
    return {
        "review_required": bool(any(changed.values()) or changed_issues or resolved),
        "baseline": not bool(previous), "changed_report_ids": changed["reports"],
        "changed_observation_ids": changed["observations"],
        "changed_issue_ids": changed_issues, "resolved_issue_ids": resolved,
    }


def render_summary(snapshot, cron_healthy, cron_desc) -> str:
    counts, tracking = snapshot["counts"], snapshot["tracking"]
    delta = snapshot.get("delta", compare_snapshot(snapshot))
    lines = [
        f"【TradingAgents 少样本质量复盘】{snapshot['snapshot_date']}", "",
        f"任务日志：{'通过' if cron_healthy else '需检查'}；{cron_desc}",
        f"归档/跟踪数据：{'未见本版规则异常' if snapshot['data_healthy'] else '需核查或覆盖不足'}；不等于投资能力验证。", "",
        f"最近7天：归档报告 {counts['new_reports']} 份，其中归档失败 {counts['new_archive_failures']} 份。",
        f"近7天采集记录：{counts['new_observations']} 条（按最近采集时间，可能含重采）。",
        f"累计归档记录 {counts['reports']} 份（含历史失败 {counts['archive_failures']} 份）；股票日期组合 {counts['ticker_dates']} 个。",
        f"已有结果覆盖 {counts['evaluated_reports']} 份报告、{counts['evaluated_ticker_dates']} 个股票日期组合，共 {counts['observations']} 条跨期限结果（仍非独立样本）。",
        f"最近报告归档版本：{snapshot.get('latest_recorded_version') or '未知'}，同标签 {counts['latest_recorded_version_reports']} 份；生成时实际配置尚未验证。", "",
        f"结果跟踪：已完成 {tracking['completed']}；未到期 {tracking['not_due']}；今日待完成 {tracking['due_today']}。",
        f"需检查：已过期 {tracking['overdue']}；到期日未知 {tracking['due_unknown']}；采集失败 {tracking['failed']}；无效记录 {tracking['invalid']}。", "",
        f"采集阻塞：到期缺行情 {tracking['data_missing']}；重试耗尽 {tracking['retry_exhausted']}；日历不支持/时间错误 {tracking['schedule_error']}。", "",
        "本次优先问题：",
    ]
    for index, issue in enumerate(snapshot["issues"][:3], 1):
        examples = "、".join(f"{e['ticker']}（#{e['run_id']}）" for e in issue["examples"])
        lines.extend([
            f"{index}. {issue['title']}：关联 {len(issue['run_ids'])} 份报告，其中近7天 {issue['recent_reports']} 份。",
            f"案例：{examples}。处理：{issue['action']}",
        ])
    if not snapshot["issues"]:
        lines.append("暂无本版规则命中的问题；不代表已验证策略有效。")
    lines += ["", f"本次变化：{len(delta['changed_issue_ids'])} 类问题新增/变化，{len(delta['resolved_issue_ids'])} 类问题已消失。"]
    lines.append("复盘安排：有新证据，可按案例检查；本任务不调用 LLM。" if delta["review_required"] else "复盘安排：无新证据，不重复深度复盘。")
    lines += [
        "参数结论：不自动调参；少样本只支持问题定位，不支持胜率改善承诺。",
        "评价边界：WAIT 不等于持仓；旧命中率不是交易收益；未归档任务不在本统计内。",
        "3日/一年专用评价尚未接入，本版不以 T+5/T+30 替代。",
    ]
    return "\n".join(lines)


def render_markdown(snapshot, cron_healthy, cron_desc) -> str:
    lines = ["# Harness 质量复盘", "", render_summary(snapshot, cron_healthy, cron_desc), "", "## 问题与验收清单", ""]
    for issue in snapshot["issues"]:
        lines += [f"### {issue['title']}", f"规则：`{issue['id']}`；类型：{issue['kind']}", "", issue["action"], issue["acceptance"], ""]
        for example in issue["examples"]:
            lines.append(f"- {example['ticker']}，报告 #{example['run_id']}；本机目录：`{example['report_dir']}`")
        lines.append("")
    lines += ["## 已有期限覆盖", ""]
    lines.extend(f"- {horizon}：{count} 条原有结果，仅跟踪覆盖，不报告投资命中率。" for horizon, count in sorted(snapshot["horizon_counts"].items()))
    lines += ["", "本报告仅基于当前归档数据库审计，不是历史时点可重放回测；原始报告、评级和结果值均未修改。", ""]
    return "\n".join(lines)
