"""Alert engine for price monitoring.

Monitors price observations and triggers alerts on anomalies, threshold breaches,
and scrape failures.
"""

from __future__ import annotations

from typing import List, Optional

from ..models import PriceObservation, PriceAlert
from ..database import (
    get_latest_price,
    get_previous_price,
    insert_alert,
)


# Alert thresholds by commodity (percentage for anomaly detection)
_DEFAULT_THRESHOLDS = {
    "lithium_carbonate": {"critical": 10.0, "high": 5.0, "medium": 2.0},
    "lithium_hydroxide": {"critical": 10.0, "high": 5.0, "medium": 2.0},
    "nickel": {"critical": 8.0, "high": 4.0, "medium": 1.5},
    "nickel_sulfate": {"critical": 10.0, "high": 5.0, "medium": 2.0},
    "cobalt": {"critical": 10.0, "high": 5.0, "medium": 2.0},
    "cobalt_sulfate": {"critical": 10.0, "high": 5.0, "medium": 2.0},
    "manganese_sulfate": {"critical": 12.0, "high": 6.0, "medium": 2.5},
    "gold": {"critical": 5.0, "high": 2.5, "medium": 1.0},
    "silver": {"critical": 8.0, "high": 4.0, "medium": 1.5},
    "copper": {"critical": 6.0, "high": 3.0, "medium": 1.0},
}


class AlertEngine:
    """Monitors price observations and triggers alerts on anomalies."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.thresholds = _DEFAULT_THRESHOLDS

    def check_observations(
        self, observations: List[PriceObservation]
    ) -> List[PriceAlert]:
        """Check a batch of new observations for anomalies.

        Args:
            observations: Newly scraped price observations.

        Returns:
            List of triggered alerts.
        """
        alerts: List[PriceAlert] = []
        for obs in observations:
            alert = self._check_single(obs)
            if alert:
                alerts.append(alert)
                insert_alert(alert, db_path=self.db_path)

        return alerts

    def check_scrape_failure(
        self, commodity: str, source: str, error: str
    ) -> PriceAlert:
        """Create an alert for a scrape failure."""
        alert = PriceAlert(
            commodity=commodity,
            alert_type="scrape_failure",
            severity="high",
            message=f"Failed to scrape {commodity} from {source}: {error}",
            price=0.0,
            threshold=0.0,
            current_value=0.0,
        )
        insert_alert(alert, db_path=self.db_path)
        return alert

    def _check_single(
        self, obs: PriceObservation
    ) -> Optional[PriceAlert]:
        """Check a single observation for price anomalies."""
        if obs.scrape_status != "success":
            return None

        # Get the previous day's price for this commodity/grade/source
        previous = get_previous_price(
            obs.commodity, obs.grade, obs.date,
            source=obs.source, db_path=self.db_path,
        )
        if previous is None:
            return None

        # Calculate change
        if previous.price_cny == 0:
            return None

        change_pct = abs(
            ((obs.price_cny - previous.price_cny) / previous.price_cny) * 100
        )

        # Get thresholds for this commodity
        commodity_thresholds = self.thresholds.get(obs.commodity, {
            "critical": 10.0, "high": 5.0, "medium": 2.0,
        })

        # Determine severity
        severity = None
        threshold = 0.0
        if change_pct >= commodity_thresholds["critical"]:
            severity = "critical"
            threshold = commodity_thresholds["critical"]
        elif change_pct >= commodity_thresholds["high"]:
            severity = "high"
            threshold = commodity_thresholds["high"]
        elif change_pct >= commodity_thresholds["medium"]:
            severity = "medium"
            threshold = commodity_thresholds["medium"]

        if severity is None:
            return None

        direction = "rise" if obs.price_cny > previous.price_cny else "fall"
        direction_cn = "上涨" if direction == "rise" else "下跌"

        alert = PriceAlert(
            commodity=obs.commodity,
            alert_type="anomaly",
            severity=severity,
            message=(
                f"{obs.commodity.replace('_', ' ').title()} "
                f"价格{direction_cn} {change_pct:.2f}% "
                f"({previous.price_cny:,.0f} → {obs.price_cny:,.0f} CNY/t) "
                f"[{obs.source}]"
            ),
            price=obs.price_cny,
            threshold=threshold,
            current_value=change_pct,
        )

        return alert

    def evaluate_alert(
        self, commodity: str, price: float, custom_threshold: float = 5.0
    ) -> Optional[PriceAlert]:
        """Manually evaluate a price against a threshold.

        Args:
            commodity: Commodity name.
            price: Current price.
            custom_threshold: Percentage threshold.

        Returns:
            Alert if threshold is exceeded, None otherwise.
        """
        latest = get_latest_price(commodity, db_path=self.db_path)
        if not latest:
            return None

        change_pct = abs(
            ((price - latest.price_cny) / latest.price_cny) * 100
        )
        if change_pct < custom_threshold:
            return None

        severity = "high" if change_pct >= 5.0 else "medium"

        return PriceAlert(
            commodity=commodity,
            alert_type="threshold",
            severity=severity,
            message=(
                f"{commodity.replace('_', ' ').title()} exceeded "
                f"{custom_threshold}% threshold: {change_pct:.2f}% move"
            ),
            price=price,
            threshold=custom_threshold,
            current_value=change_pct,
        )
