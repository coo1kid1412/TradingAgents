"""Versioned, explicit engineering defaults for the small-cap screen."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

RULE_ASSUMPTIONS = {
    "universe": {"value": "SSE/SZSE mainboard CNY ordinary shares", "origin": "source_post"},
    "cap_basis": {"value": "T-day total market capitalization, CNY", "origin": "engineering_default"},
    "financial_basis": {"value": "latest visible annual consolidated n_income_attr_p/revenue", "origin": "engineering_default"},
    "listing_age_unit": {"value": "calendar_days", "origin": "engineering_default"},
    "annual_age_limit": {"value": "550 calendar days", "origin": "engineering_default"},
    "date_only_visibility": {"value": "announcement date plus one day at 00:00 Shanghai", "origin": "engineering_default"},
    "screen_thresholds": {"value": "10-100 yi cap, profit>0, revenue>1 yi, price<=50, age>=375", "origin": "source_post"},
    "strong_weak_regime": {"value": "UNKNOWN", "origin": "source_post_undefined"},
    "selection_not_execution": {"value": True, "origin": "v1_boundary"},
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    rule_version: str = "small-cap-v1.0"
    timezone: str = "Asia/Shanghai"
    top_n: int = 10
    minimum_listing_days: int = 375
    minimum_cap_yuan: int = 1_000_000_000
    maximum_cap_yuan: int = 10_000_000_000
    maximum_price: int = 50
    minimum_revenue_yuan: int = 100_000_000
    maximum_annual_age_days: int = 550

    @property
    def config_hash(self):
        return digest(asdict(self))
