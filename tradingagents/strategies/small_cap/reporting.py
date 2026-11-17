"""Deterministic research reports and explicit, single-attempt Feishu delivery.

No environment loading, HTTP, or delivery occurs at import/render time. CLOSED
suppression and persistent idempotency belong to the runner. App messages use
the caller's uuid; bot webhooks cannot deduplicate by uuid and normally return
no message_id (represented by None, never a fabricated identifier).
"""

import hashlib
import html
import json
import os
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

import requests


_AUTH_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
_MESSAGE_URL = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id"
_STATUSES = {"SUCCESS", "INSUFFICIENT", "BLOCKED", "ERROR", "CLOSED"}
_CODE_KEYS = ("code", "status_code", "StatusCode")
_TRUNCATED = "\n[内容已截断；完整结果请查看本地归档报告。]"
_DIAGNOSTICS = {
    "ranking_incomplete": "排名完整性不足",
    "snapshot_calendar_or_point_in_time_invalid": "快照日历或信息时点校验未通过",
    "duplicate_security": "证券记录重复",
    "unknown_candidates_proven_behind_top10": "未知证券已证明不影响前十边界",
}
_LABELS = {"current_snapshot": "本次采集口径", "UNKNOWN": "未知/未评估（UNKNOWN）",
           "NOT_EVALUATED": "未评估（NOT_EVALUATED）",
           "RESEARCH_CANDIDATES": "研究候选（RESEARCH_CANDIDATES）",
           "limit_up_at_close": "收盘涨停，成交受限", "limit_down_at_close": "收盘跌停，成交受限"}


class DeliveryError(RuntimeError):
    """Definite failure; code is a local identifier, never remote error text."""

    def __init__(self, code: str):
        self.code = code if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) else "DELIVERY_ERROR"
        self.safe_code = self.code
        super().__init__(self.code)


class DeliveryUnknown(DeliveryError):
    """Sending may have succeeded. Do not automatically retry this delivery."""


def _text(value, limit=350):
    if value is None or value == "":
        return "未知"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    value = re.sub(r"\s+", " ", str(value)).strip()
    value = "".join(char for char in value if ord(char) >= 32)
    if len(value) > limit:
        value = value[:limit] + " [已截断]"
    value = html.escape(value, quote=True)
    # Entity escaping also neutralizes Feishu's links and <at> markup; source
    # values cannot add headings, emphasis, mentions, or multiline instructions.
    return re.sub(r"[\\`*_{}\[\]()#!|]", lambda match: f"&#{ord(match[0])};", value)


def _number(value, divisor=1, places=2):
    if value is None or isinstance(value, bool):
        return "未知"
    try:
        number = Decimal(str(value)) / Decimal(divisor)
        if not number.is_finite() or abs(number) > Decimal("1e25"):
            return "未知"
        return f"{number:.{places}f}"
    except (InvalidOperation, ValueError, TypeError):
        return "未知"


def _publishable(result):
    return result.get("status") in {"SUCCESS", "INSUFFICIENT"} and result.get("ranking_complete") is True


def _label(value):
    return _text(_LABELS.get(value, value) if isinstance(value, str) else value)


def _diagnostic(warning):
    code = warning.get("code") if isinstance(warning, dict) else warning
    if not isinstance(code, str):
        return None
    base = code.split(":", 1)[0]
    if base in _DIAGNOSTICS:
        return f"{_DIAGNOSTICS[base]}（{_text(base)}）"
    if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code):
        return _text(code)
    return None


def _coverage(value):
    if not isinstance(value, dict):
        return "未知；详见归档结果。"
    complete = value.get("complete")
    parts = [f"覆盖完整：{'是' if complete is True else '否' if complete is False else '未知'}"]
    for key, label in (("expected", "应覆盖"), ("observed", "已覆盖"), ("security_count", "证券数")):
        if key in value:
            parts.append(f"{label}：{_number(value[key], places=0)}")
    for key, label in (("unknown_codes", "未知证券"), ("missing_fields", "缺字段证券"), ("endpoints", "接口数")):
        if isinstance(value.get(key), (dict, list)):
            parts.append(f"{label}：{len(value[key])}")
    return "；".join(parts) + "；明细见归档。"


def _evidence(value):
    labels = {"quote_trade_date": "行情日期", "observed_at": "采集时点", "financial_revision_id": "财报版本"}
    if isinstance(value, dict) and any(key in value for key in labels):
        return "；".join(f"{label}：{_text(value[key])}" for key, label in labels.items() if key in value)
    return _text(value) if value else "未提供证据引用；请核对归档。"


def _flags(value):
    if value == []:
        return "未提供成交限制标记；不代表可以成交。"
    if isinstance(value, list):
        return "；".join(_label(flag) for flag in value[:10])
    return _label(value)


def _seasonal(result):
    try:
        month = date.fromisoformat(str(result.get("target_trade_date"))).month
    except ValueError:
        month = None
    return (month in {1, 4} or result.get("strategy_mode") == "SEASONAL_STOCK_AVOID"
            or result.get("new_position_policy") == "BLOCK")


def render_markdown(result: dict) -> str:
    """Render only supplied facts; never promote diagnostic candidates."""
    status = result.get("status") if result.get("status") in _STATUSES else "ERROR"
    publishable = _publishable(result)
    seasonal = _seasonal(result)
    lines = [
        f"# 小市值研究候选 Top10｜行情 {_text(result.get('data_date'))}",
        f"面向下一交易日：{_text(result.get('target_trade_date'))}",
        "", "**候选不是买入指令**", "",
        f"状态：{status}；榜单完整性：{'完整' if publishable else '未发布'}",
        f"数据截止：{_text(result.get('as_of_at'))}（按原始时区显示；策略时区 Asia/Shanghai）",
    ]
    identity = result.get("identity_verification") or {}
    if isinstance(identity, dict) and identity.get("official_szse_resolved_count"):
        lines += ["证券身份额外核对了深交所官方完整 A 股名录，仅用于当前身份核验；行情、市值和财报仍来自 Tushare。"]
    if seasonal:
        lines += ["**一/四月避险窗口：仅研究观察，不新开股票仓位（BLOCK）。**",
                  "策略模式：SEASONAL_STOCK_AVOID；不自动购买货币 ETF。"]
    else:
        lines += [f"策略模式：{_label(result.get('strategy_mode'))}",
                  f"新仓策略：{_label(result.get('new_position_policy'))}；筛选通过不等于获准买入。"]
    lines += [f"强弱市：{_label(result.get('regime'))}；V1 未定义强弱市，不给出当前持仓数量建议。",
              "原帖情景为强市前 3、弱市前 6；仅作规则说明，不是当前操作建议。",
              f"下一交易日周二调仓标记：{_text(result.get('rebalance_due'))}；V1 调仓未执行，不自动节假日顺延。",
              ""]
    if not publishable:
        lines += ["**正式 Top10 未发布；不展示草稿候选。**",
                  "休市不生成新榜单，默认不推送。" if status == "CLOSED"
                  else "数据异常或完整性尚未确认，请核对归档诊断后再发布。",
                  "覆盖、缺失与警告详见归档结果；诊断字段未作为正式名单发布。"]
        warnings = result.get("warnings") or []
        warnings = warnings if isinstance(warnings, list) else [warnings]
        for warning in warnings[:20]:
            diagnostic = _diagnostic(warning)
            if diagnostic:
                lines.append(f"诊断码：{diagnostic}")
    else:
        rows = result.get("top10")
        rows = rows if isinstance(rows, list) else []
        rows = [row for row in rows[:10] if isinstance(row, dict)]
        lines += [f"## 研究候选（{len(rows)} 只）"]
        if status == "INSUFFICIENT":
            lines += ["合格数量不足 10，只列真实候选，不放宽条件、不补齐。"]
        for row in rows:
            profit = row.get("profit", row.get("n_income_attr_p"))
            lines += [
                "",
                f"**{_text(row.get('rank'))}. {_text(row.get('ts_code'))} {_text(row.get('name'))}**",
                f"总市值 {_number(row.get('total_market_cap_yuan'), 100000000)} 亿元；未复权收盘 {_number(row.get('close'))} 元",
                f"行业：{_text(row.get('industry'))}（口径：{_label(row.get('industry_basis'))}）；上市 {_number(row.get('listing_days'), places=0)} 自然日",
                f"年度合并财报：{_text(row.get('financial_period'))}；归母净利润 {_number(profit, 100000000)} 亿元；营收 {_number(row.get('revenue'), 100000000)} 亿元",
                f"财报可用：{_text(row.get('financial_available_at'))}；版本：{_text(row.get('financial_revision_id', row.get('financial_version', row.get('revision_id'))))}",
                f"证据：{_evidence(row.get('evidence'))}",
                f"风险/成交限制：{_flags(row.get('flags'))}",
            ]
        lines += ["", "## 变化与覆盖"]
        changes = result.get("changes")
        if isinstance(changes, dict) and any(key in changes for key in ("added", "removed", "rank_changes")):
            for key, label in (("added", "新增"), ("removed", "移出"), ("rank_changes", "排名变化")):
                lines.append(f"{label}：{_text(changes.get(key), 1200)}")
        else:
            lines.append(f"名单变化：{_text(changes, 2000)}" if changes else "名单变化：首次/不可比较。")
        lines += ["移出不等于卖出；排名变化不是调仓指令。",
                  f"数据覆盖：{_coverage(result.get('coverage'))}"]
        warnings = result.get("warnings") or []
        warnings = warnings if isinstance(warnings, list) else [warnings]
        lines += ["缺失与待补证："] + [f"- {_diagnostic(warning) or _text(warning)}" for warning in warnings[:20]]
        if not warnings:
            lines += ["未提供额外警告；不代表不存在风险。"]
        elif len(warnings) > 20:
            lines += ["警告已截断；详见归档结果。"]
    lines += ["", "工程默认：上市年龄按自然日且至少 375 天；最新年度合并财报，陈旧上限 550 自然日，不是 TTM。",
              "开盘前须重核交易资格、新公告及临时停牌；截止时点后的变化未覆盖。",
              "日内卖出/补仓、止盈止损及真实持仓管理未实现；未下单，不承诺收益。"]
    return "\n".join(lines) + "\n"


def _destination():
    webhook = os.environ.get("FEISHU_SMALL_CAP_WEBHOOK", "").strip()
    if webhook:
        try:
            url = urlsplit(webhook)
            valid = (url.scheme == "https" and url.hostname == "open.feishu.cn"
                     and url.port in (None, 443) and not url.username and not url.password
                     and url.path.startswith("/open-apis/bot/v2/hook/")
                     and bool(url.path.rsplit("/", 1)[-1]) and not url.fragment and not url.query)
        except ValueError:
            valid = False
        if not valid:
            raise DeliveryError("INVALID_WEBHOOK")
        # Signed webhooks are deliberately unsupported rather than sent without
        # a required signature. Ordinary explicitly configured bots need no SDK.
        if os.environ.get("FEISHU_SMALL_CAP_WEBHOOK_SECRET", "").strip():
            raise DeliveryError("WEBHOOK_SIGNING_UNSUPPORTED")
        return "webhook", webhook
    target = (os.environ.get("FEISHU_SMALL_CAP_USER_OPEN_ID", "").strip()
              or os.environ.get("FEISHU_USER_OPEN_ID", "").strip())
    if not (target and os.environ.get("FEISHU_APP_ID", "").strip()
            and os.environ.get("FEISHU_APP_SECRET", "").strip()):
        raise DeliveryError("MISSING_CONFIG")
    return "app", target


def recipient_identity() -> str:
    """Stable SHA-256 of the selected target only; no credential/ID logging."""
    _, target = _destination()
    return hashlib.sha256(target.encode("utf-8")).hexdigest()


def _card(result, markdown):
    status = result.get("status") if result.get("status") in _STATUSES else "ERROR"
    card = {
        "config": {"wide_screen_mode": False},
        "header": {"template": "green" if _publishable(result) else "red",
                   "title": {"tag": "plain_text", "content": f"小市值研究候选｜{status}"}},
        "elements": [],
    }
    text = "**候选不是买入指令**\n\n" + markdown
    encoded = text.encode("utf-8")
    truncated = len(encoded) > 18000
    text = encoded[:18000].decode("utf-8", errors="ignore")
    while text:
        chunk = text.encode("utf-8")[:3000].decode("utf-8", errors="ignore")
        if len(chunk) < len(text):
            newline = chunk.rfind("\n")
            if newline > len(chunk) // 2:
                chunk = chunk[:newline + 1]
        card["elements"].append({"tag": "markdown", "content": chunk})
        text = text[len(chunk):]
    # Count serialized UTF-8 bytes, including JSON escaping and card overhead.
    while len(json.dumps(card, ensure_ascii=False).encode("utf-8")) > 24000:
        card["elements"].pop()
        truncated = True
    if truncated:
        card["elements"].append({"tag": "markdown", "content": _TRUNCATED})
    return card


def _post(session, url, payload, phase, headers=None):
    try:
        reply = session.post(url, json=payload, headers=headers or {},
                             timeout=(5, 20), allow_redirects=False)
    except requests.RequestException:
        error = DeliveryUnknown if phase == "SEND" else DeliveryError
        raise error(f"{phase}_TRANSPORT_ERROR") from None
    if not 200 <= reply.status_code < 300:
        uncertain = phase == "SEND" and (reply.status_code >= 500 or reply.status_code < 400
                                         or reply.status_code == 408)
        error = DeliveryUnknown if uncertain else DeliveryError
        raise error(f"{phase}_HTTP_ERROR")
    error = DeliveryUnknown if phase == "SEND" else DeliveryError
    try:
        body = reply.json()
    except (ValueError, requests.RequestException):
        raise error(f"{phase}_INVALID_JSON") from None
    if not isinstance(body, dict):
        raise error(f"{phase}_INVALID_RESPONSE")
    codes = [body[key] for key in _CODE_KEYS if key in body]
    if not codes or any(type(code) is not int for code in codes):
        raise error(f"{phase}_MISSING_CODE")
    if any(code != 0 for code in codes):
        raise DeliveryError(f"{phase}_BUSINESS_ERROR")
    return body


def send_feishu(result: dict, markdown: str, uuid: str) -> dict:
    """Send one card, once. Only a confirmed API success returns status=sent.

    DeliveryUnknown is intentionally distinct from a retryable explicit failure.
    The runner must reconcile uncertainty before attempting another delivery.
    """
    if not isinstance(uuid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", uuid):
        raise DeliveryError("INVALID_UUID")
    kind, target = _destination()
    recipient_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()
    # Never trust supplied draft Markdown for error, closed or incomplete runs.
    if not _publishable(result):
        markdown = render_markdown(result)
    if not isinstance(markdown, str):
        raise DeliveryError("INVALID_MARKDOWN")
    card = _card(result, markdown)
    session = requests.Session()
    try:
        if kind == "webhook":
            body = _post(session, target, {"msg_type": "interactive", "card": card}, "SEND")
            data = body.get("data")
            message_id = (data.get("message_id") if isinstance(data, dict) else None) or body.get("message_id")
            if not isinstance(message_id, str) or not message_id.strip():
                message_id = None
        else:
            auth = _post(session, _AUTH_URL, {
                "app_id": os.environ["FEISHU_APP_ID"].strip(),
                "app_secret": os.environ["FEISHU_APP_SECRET"].strip(),
            }, "AUTH")
            token = auth.get("tenant_access_token")
            if not isinstance(token, str) or not token.strip() or "\n" in token or "\r" in token:
                raise DeliveryError("AUTH_MISSING_TOKEN")
            body = _post(session, _MESSAGE_URL, {
                "receive_id": target, "msg_type": "interactive",
                "content": json.dumps(card, ensure_ascii=False), "uuid": uuid,
            }, "SEND", {"Authorization": f"Bearer {token}"})
            data = body.get("data")
            message_id = data.get("message_id") if isinstance(data, dict) else None
            if not isinstance(message_id, str) or not message_id.strip():
                raise DeliveryUnknown("SEND_MISSING_MESSAGE_ID")
        return {"status": "sent", "message_id": message_id, "recipient_hash": recipient_hash}
    finally:
        session.close()
