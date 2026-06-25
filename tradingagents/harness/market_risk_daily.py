"""开盘前市场风险快照任务。

运行：python -m tradingagents.harness.market_risk_daily
cron 可为 A 股和美股各配置一个当地开盘前时间；同一日期重复执行会覆盖快照。
"""

from __future__ import annotations

import datetime as _dt
import argparse
import json as _json
import logging
import math
import os
from pathlib import Path
from typing import Callable
import urllib.request as _ur

import pandas as pd

from tradingagents.harness import price_cache as _pcache
from tradingagents.harness.market_risk import compute_market_risk_snapshot, save_market_risk_snapshot

logger = logging.getLogger(__name__)

_CALENDAR_NAMES = {"a_share": "XSHG", "us": "XNYS"}
_MARKET_TICKERS = {
    "a_share": ("510300", "159915", "588000"),
    "us": ("^GSPC", "^IXIC", "^RUT"),
}

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    logger.debug("无法加载项目 .env，继续使用当前进程环境变量", exc_info=True)


def is_market_trading_day(market: str, as_of_date: str) -> bool:
    """以交易所日历判断市场是否开市，避免周末/节假日制造伪预测。"""
    calendar_name = _CALENDAR_NAMES.get(market)
    if calendar_name is None:
        return False
    try:
        import exchange_calendars as xcals

        return bool(xcals.get_calendar(calendar_name).is_session(pd.Timestamp(as_of_date)))
    except Exception as exc:  # 日历库不可用时保守地只在工作日运行
        logger.warning("市场日历不可用（%s），回退工作日判断: %s", market, exc)
        return _dt.date.fromisoformat(as_of_date).weekday() < 5


def _records_from_df(df) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    return [{"close": float(v)} for v in pd.to_numeric(df["Close"], errors="coerce").dropna()]


def _realized_volatility_pct(records: list[dict]) -> float | None:
    closes = [r["close"] for r in records]
    if len(closes) < 21:
        return None
    returns = [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(returns) < 20:
        return None
    mean = sum(returns[-20:]) / 20
    variance = sum((v - mean) ** 2 for v in returns[-20:]) / 20
    return round(math.sqrt(variance) * math.sqrt(252) * 100, 2)


def _breadth_proxy_pct(all_records: list[list[dict]]) -> float | None:
    votes = []
    for records in all_records:
        closes = [r["close"] for r in records]
        if len(closes) >= 20:
            votes.append(closes[-1] >= sum(closes[-20:]) / 20)
    return round(sum(votes) / len(votes) * 100, 2) if votes else None


def _default_fetch_prices(ticker: str, start: str, end: str):
    return _records_from_df(_pcache.fetch_with_cache(ticker, start, end))


def _format_message(snapshot: dict) -> str:
    reasons = "；".join(snapshot["reasons"])
    return (
        f"【{snapshot['market']} 开盘前风险】{snapshot['as_of_date']}\n"
        f"风险：{snapshot['risk_level']}（分数 {snapshot['risk_score']}） | T+1：{snapshot['t_plus_1_bias']}\n"
        f"动作：{snapshot['entry_gate']} | 新增仓位上限：{snapshot['position_cap_pct']}%\n"
        f"数据：{snapshot['data_status']} | 因子：{reasons}"
    )


def _send_feishu_message(message: str) -> None:
    """发送飞书文本消息。

    优先使用自定义机器人 webhook；未配置 webhook 时，复用项目已有的
    FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_USER_OPEN_ID 走 Open API 私聊。
    """
    webhook = os.getenv("FEISHU_MARKET_RISK_WEBHOOK")
    if webhook:
        import requests

        response = requests.post(webhook, json={"msg_type": "text", "content": {"text": message}}, timeout=20)
        response.raise_for_status()
        return

    app_id = os.getenv("FEISHU_APP_ID")
    app_secret = os.getenv("FEISHU_APP_SECRET")
    open_id = os.getenv("FEISHU_USER_OPEN_ID")
    if not (app_id and app_secret and open_id):
        raise RuntimeError(
            "飞书凭证未配置：需要 FEISHU_MARKET_RISK_WEBHOOK，"
            "或 FEISHU_APP_ID/FEISHU_APP_SECRET/FEISHU_USER_OPEN_ID"
        )

    req = _ur.Request(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        data=_json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = _json.loads(_ur.urlopen(req, timeout=10).read())
    if resp.get("code") != 0:
        raise RuntimeError(f"获取飞书 tenant_access_token 失败: {resp}")
    token = resp["tenant_access_token"]

    req = _ur.Request(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
        data=_json.dumps({
            "receive_id": open_id,
            "msg_type": "text",
            "content": _json.dumps({"text": message}, ensure_ascii=False),
        }, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    resp = _json.loads(_ur.urlopen(req, timeout=20).read())
    if resp.get("code") != 0:
        raise RuntimeError(f"发送飞书消息失败: {resp}")


def run_market_risk_daily(
    as_of_date: str | None = None,
    db_path=None,
    fetch_prices: Callable[[str, str, str], list[dict]] | None = None,
    send_message: Callable[[str], None] | None = None,
    dry_run: bool = False,
    markets: tuple[str, ...] = ("a_share", "us"),
) -> dict[str, dict]:
    """为 A 股和美股生成当天快照；推送失败不影响落库。"""
    as_of_date = as_of_date or _dt.date.today().isoformat()
    fetch_prices = fetch_prices or _default_fetch_prices
    send_message = send_message or _send_feishu_message
    start = (_dt.date.fromisoformat(as_of_date) - _dt.timedelta(days=120)).isoformat()
    results: dict[str, dict] = {}

    for market in markets:
        if market not in _MARKET_TICKERS:
            results[market] = {"status": "failed", "push_status": "skipped", "error": "unsupported market"}
            continue
        tickers = _MARKET_TICKERS[market]
        if not is_market_trading_day(market, as_of_date):
            results[market] = {"status": "closed", "push_status": "skipped"}
            continue
        try:
            all_records = [fetch_prices(ticker, start, as_of_date) for ticker in tickers]
            primary = all_records[0]
            snapshot = compute_market_risk_snapshot(
                market, primary, breadth_pct=_breadth_proxy_pct(all_records),
                volatility_pct=_realized_volatility_pct(primary), as_of_date=as_of_date,
            )
            if dry_run:
                results[market] = {"status": "dry_run", "push_status": "skipped", "snapshot": snapshot}
                continue
            save_market_risk_snapshot(snapshot, db_path)
            push_status = "sent"
            try:
                send_message(_format_message(snapshot))
            except Exception as exc:
                logger.warning("%s 风险快照已保存，但飞书推送失败: %s", market, exc)
                push_status = "failed"
            results[market] = {"status": "saved", "push_status": push_status, "snapshot": snapshot}
        except Exception as exc:
            logger.exception("%s 市场风险任务失败", market)
            results[market] = {"status": "failed", "push_status": "skipped", "error": str(exc)}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 A 股与美股开盘前市场风险快照")
    parser.add_argument("--date", help="快照日期 YYYY-MM-DD，默认今天")
    parser.add_argument("--dry-run", action="store_true", help="只计算，不落库、不推送")
    parser.add_argument("--market", choices=("a_share", "us"), help="仅运行指定市场；定时任务按当地开盘前调用")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S")
    markets = (args.market,) if args.market else ("a_share", "us")
    results = run_market_risk_daily(as_of_date=args.date, dry_run=args.dry_run, markets=markets)
    for market, result in results.items():
        print(f"{market}: {result['status']} / push={result['push_status']}")
    return 0 if all(r["status"] in ("saved", "closed", "dry_run") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
