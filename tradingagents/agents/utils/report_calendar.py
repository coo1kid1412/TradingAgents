"""Deterministic reporting-period and legal-window normalization."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def _year(title: str, event_date: str, source_date: str) -> int | None:
    for value in (title, event_date, source_date):
        match = re.search(r"20\d{2}", str(value or ""))
        if match:
            return int(match.group(0))
    return None


_OFFICIAL_HOSTS = ("cninfo.com.cn", "sse.com.cn", "szse.cn", "bse.cn")


def has_traceable_official_reference(
    *, source_tier: str | None, verification_status: str | None,
    source_url: str | None, document_id: str | None,
) -> bool:
    if str(source_tier or "").lower() not in {"official", "regulatory"}:
        return False
    if str(verification_status or "").lower() != "verified":
        return False
    host = (urlparse(str(source_url or "")).hostname or "").lower()
    trusted = any(host == item or host.endswith(f".{item}") for item in _OFFICIAL_HOSTS)
    document = str(document_id or "").strip()
    return trusted and bool(document) and document.lower() in str(source_url or "").lower()


def _traceable_official_reservation(
    *, event_date_basis: str | None, source_tier: str | None,
    verification_status: str | None, source_url: str | None, document_id: str | None,
) -> bool:
    return (
        str(event_date_basis or "").lower() == "official_reservation"
        and has_traceable_official_reference(
            source_tier=source_tier,
            verification_status=verification_status,
            source_url=source_url,
            document_id=document_id,
        )
    )


def normalize_reporting_event(
    title: str,
    event_date: str,
    source_date: str,
    *,
    event_date_basis: str | None = None,
    source_tier: str | None = None,
    verification_status: str | None = None,
    source_url: str | None = None,
    document_id: str | None = None,
) -> dict:
    """Normalize H1/Q3 reporting events without inventing reservation dates."""
    title = str(title or "")
    event_date = str(event_date or "未知").strip() or "未知"
    source_date = str(source_date or "未知").strip() or "未知"
    year = _year(title, event_date, source_date)

    if any(word in title for word in (
        "业绩预告", "业绩快报", "预增公告", "预减公告", "业绩预增", "业绩预减",
    )):
        return {
            "reporting_period": (
                f"{year}H1"
                if year and any(word in title.upper() for word in ("半年度", "中报", "半年", "H1"))
                else None
            ),
            "event_date": source_date,
            "date_basis": "publication_date",
        }

    if year and any(word in title for word in ("半年度报告", "中期报告", "中报")):
        exact = bool(re.fullmatch(r"20\d{2}-\d{2}-\d{2}", event_date))
        deadline = f"{year}-08-31"
        reserved = exact and event_date != deadline and _traceable_official_reservation(
            event_date_basis=event_date_basis, source_tier=source_tier,
            verification_status=verification_status, source_url=source_url,
            document_id=document_id,
        )
        return {
            "reporting_period": f"{year}H1",
            "event_date": event_date if reserved else deadline,
            "date_basis": "exact_reservation" if reserved else "legal_deadline",
        }
    if year and any(word in title for word in ("第三季度报告", "三季报", "Q3报告")):
        exact = bool(re.fullmatch(r"20\d{2}-\d{2}-\d{2}", event_date))
        deadline = f"{year}-10-31"
        reserved = exact and event_date != deadline and _traceable_official_reservation(
            event_date_basis=event_date_basis, source_tier=source_tier,
            verification_status=verification_status, source_url=source_url,
            document_id=document_id,
        )
        return {
            "reporting_period": f"{year}Q3",
            "event_date": event_date if reserved else deadline,
            "date_basis": "exact_reservation" if reserved else "legal_deadline",
        }
    return {"reporting_period": None, "event_date": event_date, "date_basis": "reported"}
