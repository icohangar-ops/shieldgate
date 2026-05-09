"""Tests for alert engine."""

import pytest
import tempfile
import os

from src.metal_monitor.database import init_db, upsert_observations
from src.metal_monitor.models import PriceObservation, PriceAlert
from src.metal_monitor.alerts.engine import AlertEngine, _DEFAULT_THRESHOLDS


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


@pytest.fixture
def engine(db_path):
    return AlertEngine(db_path=db_path)


class TestAlertThresholds:
    def test_all_commodities_have_thresholds(self):
        expected = [
            "lithium_carbonate", "lithium_hydroxide", "nickel",
            "nickel_sulfate", "cobalt", "cobalt_sulfate",
            "manganese_sulfate", "gold", "silver", "copper",
        ]
        for commodity in expected:
            assert commodity in _DEFAULT_THRESHOLDS, f"Missing threshold for {commodity}"

    def test_threshold_hierarchy(self):
        for commodity, thresholds in _DEFAULT_THRESHOLDS.items():
            assert thresholds["critical"] > thresholds["high"] > thresholds["medium"]


class TestAlertEngine:
    def test_no_alert_for_small_change(self, engine, db_path):
        """Small changes should not trigger alerts."""
        # Seed previous day
        prev = PriceObservation(
            date="2025-01-14", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        upsert_observations([prev], db_path)

        # 0.5% change — below medium threshold of 2%
        current = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157785.0,  # +0.5%
            change_pct=0.5, change_type="rise",
        )
        alerts = engine.check_observations([current])
        assert len(alerts) == 0

    def test_medium_alert(self, engine, db_path):
        """3% change should trigger medium alert."""
        prev = PriceObservation(
            date="2025-01-14", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        upsert_observations([prev], db_path)

        # 3% change — medium threshold (2-5%)
        current = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=161710.0,  # +3%
            change_pct=3.0, change_type="rise",
        )
        alerts = engine.check_observations([current])
        assert len(alerts) == 1
        assert alerts[0].severity == "medium"
        assert alerts[0].alert_type == "anomaly"
        assert "lithium carbonate" in alerts[0].message.lower()

    def test_high_alert(self, engine, db_path):
        """6% change should trigger high alert."""
        prev = PriceObservation(
            date="2025-01-14", source="metal_com",
            commodity="nickel", grade="#1_electrolytic",
            price_cny=128000.0,
        )
        upsert_observations([prev], db_path)

        # 6% change — high threshold (4-8% for nickel)
        current = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="nickel", grade="#1_electrolytic",
            price_cny=135680.0,  # +6%
            change_pct=6.0, change_type="rise",
        )
        alerts = engine.check_observations([current])
        assert len(alerts) == 1
        assert alerts[0].severity == "high"

    def test_critical_alert(self, engine, db_path):
        """12% change should trigger critical alert."""
        prev = PriceObservation(
            date="2025-01-14", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        upsert_observations([prev], db_path)

        # 12% change — critical threshold (>10%)
        current = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=175840.0,  # +12%
            change_pct=12.0, change_type="rise",
        )
        alerts = engine.check_observations([current])
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_fall_alert(self, engine, db_path):
        """Price drops should also trigger alerts."""
        prev = PriceObservation(
            date="2025-01-14", source="shmet",
            commodity="cobalt_sulfate", grade="battery_grade",
            price_cny=96500.0,
        )
        upsert_observations([prev], db_path)

        current = PriceObservation(
            date="2025-01-15", source="shmet",
            commodity="cobalt_sulfate", grade="battery_grade",
            price_cny=86850.0,  # -10%
            change_pct=-10.0, change_type="fall",
        )
        alerts = engine.check_observations([current])
        assert len(alerts) == 1
        assert "下跌" in alerts[0].message or "fall" in alerts[0].message.lower()

    def test_no_previous_data_no_alert(self, engine):
        """No alert if there's no previous price to compare."""
        current = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        alerts = engine.check_observations([current])
        assert len(alerts) == 0

    def test_failed_scrape_no_alert(self, engine, db_path):
        """Failed scrapes should not trigger anomaly alerts."""
        prev = PriceObservation(
            date="2025-01-14", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        upsert_observations([prev], db_path)

        current = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=175000.0,  # Big change
            scrape_status="failed",
        )
        alerts = engine.check_observations([current])
        assert len(alerts) == 0

    def test_scrape_failure_alert(self, engine, db_path):
        from src.metal_monitor.database import get_active_alerts
        alert = engine.check_scrape_failure("lithium_carbonate", "metal_com", "Connection timeout")
        assert alert.alert_type == "scrape_failure"
        assert alert.severity == "high"
        assert "metal_com" in alert.message

        active = get_active_alerts(db_path)
        assert len(active) == 1

    def test_evaluate_alert_below_threshold(self, engine, db_path):
        from src.metal_monitor.database import upsert_observations
        obs = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        upsert_observations([obs], db_path)

        result = engine.evaluate_alert("lithium_carbonate", 158000.0, custom_threshold=5.0)
        assert result is None

    def test_evaluate_alert_above_threshold(self, engine, db_path):
        from src.metal_monitor.database import upsert_observations
        obs = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        upsert_observations([obs], db_path)

        # 10% above = 172,700
        result = engine.evaluate_alert("lithium_carbonate", 172700.0, custom_threshold=5.0)
        assert result is not None
        assert result.alert_type == "threshold"
        assert result.severity == "high"

    def test_multiple_observations_batch(self, engine, db_path):
        """Multiple observations in one batch should all be checked."""
        prev_obs = [
            PriceObservation(
                date="2025-01-14", source="metal_com",
                commodity="lithium_carbonate", grade="battery_grade",
                price_cny=157000.0,
            ),
            PriceObservation(
                date="2025-01-14", source="metal_com",
                commodity="nickel_sulfate", grade="battery_grade",
                price_cny=31000.0,
            ),
        ]
        upsert_observations(prev_obs, db_path)

        current_obs = [
            PriceObservation(
                date="2025-01-15", source="metal_com",
                commodity="lithium_carbonate", grade="battery_grade",
                price_cny=166490.0,  # +6% — high alert
                change_pct=6.0, change_type="rise",
            ),
            PriceObservation(
                date="2025-01-15", source="metal_com",
                commodity="nickel_sulfate", grade="battery_grade",
                price_cny=32240.0,  # +4% — high alert
                change_pct=4.0, change_type="rise",
            ),
        ]
        alerts = engine.check_observations(current_obs)
        assert len(alerts) == 2
