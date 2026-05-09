"""Metal.com mock scraper — returns realistic global metal prices."""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta
from typing import List, Optional

from .base import BaseScraper
from ..models import PriceObservation


# Realistic base prices from Mysteel reports (CNY/t)
_BASE_PRICES = {
    ("lithium_carbonate", "battery_grade"): 157000,
    ("lithium_carbonate", "industrial_grade"): 152000,
    ("lithium_hydroxide", "battery_grade"): 178000,
    ("nickel", "#1_electrolytic"): 128000,
    ("nickel_sulfate", "battery_grade"): 31000,
    ("cobalt", "electrolytic"): 265000,
    ("cobalt", "cobalt_sulfate"): 96500,
    ("cobalt_sulfate", "battery_grade"): 96500,
    ("manganese_sulfate", "battery_grade"): 6500,
}

_FX_RATE = 7.25


class MetalComScraper(BaseScraper):
    """Mock scraper for Metal.com global prices."""

    source_name = "metal_com"
    target_commodities = [
        "lithium_carbonate", "lithium_hydroxide",
        "nickel", "nickel_sulfate",
        "cobalt", "cobalt_sulfate", "manganese_sulfate",
    ]

    async def scrape(self, date: Optional[str] = None) -> List[PriceObservation]:
        """Generate mock prices with realistic daily variation."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        results: List[PriceObservation] = []
        for (commodity, grade), base_price in _BASE_PRICES.items():
            price = self._generate_price(base_price, date, commodity, grade)
            price_usd = round(price / _FX_RATE, 2)

            # Simulate change from previous day
            prev_price = self._generate_price(base_price, self._prev_date(date), commodity, grade)
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
        """Deterministic price generation based on date + commodity."""
        seed_str = f"{date_str}-{commodity}-{grade}-metal_com"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        return self._random_walk(base, volatility=0.015, seed=seed)

    @staticmethod
    def _prev_date(date_str: str) -> str:
        d = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        return d.strftime("%Y-%m-%d")
