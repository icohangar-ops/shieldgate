"""SHMET mock scraper — Shanghai Metals Market prices (Chinese domestic)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import List, Optional

from .base import BaseScraper
from ..models import PriceObservation


_BASE_PRICES = {
    ("lithium_carbonate", "battery_grade"): 157500,
    ("lithium_carbonate", "industrial_grade"): 153000,
    ("lithium_hydroxide", "battery_grade"): 179000,
    ("nickel_sulfate", "battery_grade"): 31500,
    ("cobalt", "electrolytic"): 268000,
    ("cobalt_sulfate", "battery_grade"): 97000,
    ("manganese_sulfate", "battery_grade"): 6600,
}

_FX_RATE = 7.25


class SHMETScraper(BaseScraper):
    """Mock scraper for SHMET (Shanghai Metals Market)."""

    source_name = "shmet"
    target_commodities = [
        "lithium_carbonate", "lithium_hydroxide",
        "nickel_sulfate", "cobalt", "cobalt_sulfate", "manganese_sulfate",
    ]

    async def scrape(self, date: Optional[str] = None) -> List[PriceObservation]:
        """Generate mock SHMET prices."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        results: List[PriceObservation] = []
        for (commodity, grade), base_price in _BASE_PRICES.items():
            price = self._generate_price(base_price, date, commodity, grade)
            price_usd = round(price / _FX_RATE, 2)

            prev_price = self._generate_price(
                base_price, self._prev_date(date), commodity, grade
            )
            change_pct, change_type = self._detect_change(price, prev_price)

            results.append(PriceObservation(
                date=date,
                source=self.source_name,
                commodity=commodity,
                grade=grade,
                price_cny=price,
                price_usd=price_usd,
                fx_rate=_FX_RATE,
                unit="yuan/t",
                change_pct=change_pct,
                change_type=change_type,
                scrape_status="success",
            ))

        return results

    def _generate_price(self, base: float, date_str: str,
                        commodity: str, grade: str) -> float:
        seed_str = f"{date_str}-{commodity}-{grade}-shmet"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        return self._random_walk(base, volatility=0.012, seed=seed)

    @staticmethod
    def _prev_date(date_str: str) -> str:
        d = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        return d.strftime("%Y-%m-%d")
