"""Data models for metal price monitoring."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional, List


@dataclass
class PriceObservation:
    """A single price observation from a data source."""
    date: str                                    # YYYY-MM-DD
    source: str                                  # metal_com, shmet, shfe, gfex, sge
    commodity: str                               # lithium_carbonate, nickel_sulfate, etc.
    grade: str                                   # battery_grade, industrial_grade, etc.
    price_cny: float                             # CNY per tonne
    price_usd: Optional[float] = None            # USD per tonne (converted)
    fx_rate: float = 7.25                        # USD/CNY rate used
    unit: str = "yuan/t"                         # yuan/t, $/t, $/lb
    contract_month: Optional[str] = None         # For futures
    change_pct: float = 0.0                      # Day-over-day % change
    change_type: str = "flat"                    # rise, fall, flat
    scrape_status: str = "success"               # success, partial, failed
    timestamp: str = ""                          # ISO 8601

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PriceObservation:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PriceSummary:
    """Weekly summary matching Mysteel report format."""
    commodity: str
    grade: str
    latest_price_cny: float
    latest_price_usd: Optional[float] = None
    wow_change_pct: float = 0.0
    mom_change_pct: float = 0.0
    yoy_change_pct: Optional[float] = None
    source: str = ""
    date: str = ""
    trend: str = "stable"                        # rising, falling, stable, volatile

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AIAnalysis:
    """AI-generated market analysis from Qwen."""
    commodity: str
    date: str
    market_commentary: str = ""
    outlook: str = ""
    key_drivers: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    recommendation: str = ""
    confidence: float = 0.0
    model_used: str = "qwen3.6-flash"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def key_drivers_json(self) -> str:
        return json.dumps(self.key_drivers, ensure_ascii=False)

    @property
    def risk_factors_json(self) -> str:
        return json.dumps(self.risk_factors, ensure_ascii=False)


@dataclass
class PriceAlert:
    """Price alert triggered by the alert engine."""
    id: str = ""
    commodity: str = ""
    alert_type: str = ""                         # anomaly, threshold, scrape_failure
    severity: str = ""                           # critical, high, medium, low
    message: str = ""
    price: float = 0.0
    threshold: float = 0.0
    current_value: float = 0.0
    triggered_at: str = ""
    acknowledged: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.triggered_at:
            self.triggered_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WeeklySummary:
    """Mysteel-format weekly summary for database storage."""
    date: str
    product: str
    unit: str
    avg_price: float
    wow_change_pct: float = 0.0
    mom_change_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CommodityInfo:
    """Metadata about a tracked commodity."""
    name: str
    name_cn: str
    unit: str
    grades: List[str]
    sources: List[str]
    category: str  # battery_material, precious_metal, base_metal

    def to_dict(self) -> dict:
        return asdict(self)


# Registry of all tracked commodities
COMMODITY_REGISTRY: List[CommodityInfo] = [
    CommodityInfo(
        name="lithium_carbonate", name_cn="碳酸锂",
        unit="yuan/t", grades=["battery_grade", "industrial_grade"],
        sources=["gfex", "shmet", "metal_com"],
        category="battery_material",
    ),
    CommodityInfo(
        name="lithium_hydroxide", name_cn="氢氧化锂",
        unit="yuan/t", grades=["battery_grade"],
        sources=["shmet", "metal_com"],
        category="battery_material",
    ),
    CommodityInfo(
        name="nickel", name_cn="电解镍",
        unit="yuan/t", grades=["#1_electrolytic", "most_traded_shfe"],
        sources=["shfe", "lme", "metal_com"],
        category="base_metal",
    ),
    CommodityInfo(
        name="nickel_sulfate", name_cn="硫酸镍",
        unit="yuan/t", grades=["battery_grade"],
        sources=["shmet", "metal_com"],
        category="battery_material",
    ),
    CommodityInfo(
        name="cobalt", name_cn="电解钴",
        unit="yuan/t", grades=["electrolytic"],
        sources=["metal_com", "shmet"],
        category="battery_material",
    ),
    CommodityInfo(
        name="cobalt_sulfate", name_cn="硫酸钴",
        unit="yuan/t", grades=["battery_grade"],
        sources=["shmet"],
        category="battery_material",
    ),
    CommodityInfo(
        name="manganese_sulfate", name_cn="硫酸锰",
        unit="yuan/t", grades=["battery_grade"],
        sources=["shmet"],
        category="battery_material",
    ),
    CommodityInfo(
        name="gold", name_cn="黄金",
        unit="yuan/g", grades=["benchmark"],
        sources=["sge"],
        category="precious_metal",
    ),
    CommodityInfo(
        name="silver", name_cn="白银",
        unit="yuan/kg", grades=["benchmark"],
        sources=["sge"],
        category="precious_metal",
    ),
    CommodityInfo(
        name="copper", name_cn="电解铜",
        unit="yuan/t", grades=["front_month_futures"],
        sources=["shfe"],
        category="base_metal",
    ),
]


def get_commodity_info(name: str) -> Optional[CommodityInfo]:
    """Look up commodity metadata by name."""
    for c in COMMODITY_REGISTRY:
        if c.name == name:
            return c
    return None
