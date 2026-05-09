"""SHFE mock scraper — Shanghai Futures Exchange futures prices."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import List, Optional

from .base import BaseScraper
from ..models import PriceObservation


_BASE_PRICES = {
    ("nickel", "most_traded_shfe"): 128500,
    ("copper", "front_month_futures"): 78000,
}

_FX_RATE = 7.25


class SHFEScraper(BaseScraper):
    """Mock scraper for SHFE (Shanghai Futures Exchange)."""

    source_name = "shfe"
    target_commodities = ["nickel", "copper"]

    async def scrape(self, date: Optional[str] = None) -> List[PriceObservation]:
        """Generate mock SHFE futures prices."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        # Determine contract month
        now = datetime.strptime(date, "%Y-%m-%d")
        contract_month = now.strftime("%Y%m")
        if now.day > 15:
            # Move to next month if past 15th
            next_month = now.replace(day=1)
            if next_month.month == 12:
                next_month = next_month.replace(year=next_month.year + 1, month=1)
            else:
                next_month = next_month.replace(month=next_month.month + 1)
            contract_month = next_month.strftime("%Y%m")

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
                contract_month=contract_month,
                change_pct=change_pct,
                change_type=change_type,
                scrape_status="success",
            ))

        return results

    def _generate_price(self, base: float, date_str: str,
                        commodity: str, grade: str) -> float:
        seed_str = f"{date_str}-{commodity}-{grade}-shfe"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        return self._random_walk(base, volatility=0.018, seed=seed)

    @staticmethod
    def _prev_date(date_str: str) -> str:
        d = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        return d.strftime("%Y-%m-%d")
