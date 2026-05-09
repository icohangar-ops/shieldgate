"""Base scraper class."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import PriceObservation


class BaseScraper(ABC):
    """Abstract base class for metal price scrapers."""

    source_name: str = "base"
    target_commodities: List[str] = []

    @abstractmethod
    async def scrape(self, date: Optional[str] = None) -> List[PriceObservation]:
        """Scrape prices for all target commodities.

        Args:
            date: Target date (YYYY-MM-DD). Defaults to today.

        Returns:
            List of PriceObservation objects.
        """
        raise NotImplementedError

    def _detect_change(self, current: float, previous: float) -> tuple[float, str]:
        """Calculate percentage change and direction."""
        if previous == 0:
            return 0.0, "flat"
        change = ((current - previous) / previous) * 100
        change_type = "rise" if change > 0.1 else ("fall" if change < -0.1 else "flat")
        return round(change, 2), change_type

    @staticmethod
    def _random_walk(price: float, volatility: float = 0.02,
                     seed: Optional[int] = None) -> float:
        """Generate a price using random walk with given volatility."""
        if seed is not None:
            random.seed(seed)
        factor = 1.0 + random.gauss(0, volatility)
        return round(price * factor, 2)
