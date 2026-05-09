"""Tests for data models."""

import json
import pytest
from datetime import datetime

from src.metal_monitor.models import (
    PriceObservation,
    PriceSummary,
    AIAnalysis,
    PriceAlert,
    WeeklySummary,
    CommodityInfo,
    COMMODITY_REGISTRY,
    get_commodity_info,
)


class TestPriceObservation:
    """Test PriceObservation model."""

    def test_creation_minimal(self):
        obs = PriceObservation(
            date="2025-01-15",
            source="metal_com",
            commodity="lithium_carbonate",
            grade="battery_grade",
            price_cny=157000.0,
        )
        assert obs.date == "2025-01-15"
        assert obs.source == "metal_com"
        assert obs.commodity == "lithium_carbonate"
        assert obs.price_cny == 157000.0
        assert obs.price_usd is None
        assert obs.change_pct == 0.0
        assert obs.change_type == "flat"
        assert obs.scrape_status == "success"
        assert obs.timestamp != ""  # Auto-generated
        assert obs.fx_rate == 7.25

    def test_creation_full(self):
        obs = PriceObservation(
            date="2025-01-15",
            source="shfe",
            commodity="nickel",
            grade="#1_electrolytic",
            price_cny=128000.0,
            price_usd=17655.17,
            fx_rate=7.25,
            unit="yuan/t",
            contract_month="202502",
            change_pct=1.25,
            change_type="rise",
            scrape_status="success",
            timestamp="2025-01-15T10:00:00Z",
        )
        assert obs.price_usd == 17655.17
        assert obs.contract_month == "202502"
        assert obs.change_pct == 1.25
        assert obs.change_type == "rise"

    def test_to_dict(self):
        obs = PriceObservation(
            date="2025-01-15",
            source="metal_com",
            commodity="lithium_carbonate",
            grade="battery_grade",
            price_cny=157000.0,
        )
        d = obs.to_dict()
        assert d["date"] == "2025-01-15"
        assert d["price_cny"] == 157000.0
        assert d["price_usd"] is None
        assert isinstance(d, dict)

    def test_from_dict(self):
        data = {
            "date": "2025-01-15",
            "source": "shmet",
            "commodity": "cobalt_sulfate",
            "grade": "battery_grade",
            "price_cny": 96500.0,
            "price_usd": 13310.34,
            "fx_rate": 7.25,
            "unit": "yuan/t",
            "change_pct": -0.52,
            "change_type": "fall",
            "scrape_status": "success",
            "timestamp": "2025-01-15T10:00:00Z",
        }
        obs = PriceObservation.from_dict(data)
        assert obs.commodity == "cobalt_sulfate"
        assert obs.price_cny == 96500.0
        assert obs.change_type == "fall"

    def test_from_dict_ignores_extra_fields(self):
        data = {
            "date": "2025-01-15",
            "source": "metal_com",
            "commodity": "lithium_carbonate",
            "grade": "battery_grade",
            "price_cny": 157000.0,
            "extra_field": "should_be_ignored",
        }
        obs = PriceObservation.from_dict(data)
        assert obs.price_cny == 157000.0
        assert not hasattr(obs, "extra_field")


class TestPriceSummary:
    """Test PriceSummary model."""

    def test_creation(self):
        s = PriceSummary(
            commodity="lithium_carbonate",
            grade="battery_grade",
            latest_price_cny=157000.0,
            wow_change_pct=2.5,
            mom_change_pct=-1.2,
            trend="rising",
        )
        assert s.commodity == "lithium_carbonate"
        assert s.wow_change_pct == 2.5
        assert s.trend == "rising"

    def test_to_dict(self):
        s = PriceSummary(
            commodity="nickel_sulfate",
            grade="battery_grade",
            latest_price_cny=31000.0,
        )
        d = s.to_dict()
        assert d["commodity"] == "nickel_sulfate"
        assert d["latest_price_cny"] == 31000.0


class TestAIAnalysis:
    """Test AIAnalysis model."""

    def test_creation(self):
        a = AIAnalysis(
            commodity="lithium_carbonate",
            date="2025-01-15",
            market_commentary="Lithium prices continue to rise...",
            outlook="Bullish short-term",
            key_drivers=["EV demand surge", "Supply constraints"],
            risk_factors=["Policy changes"],
            recommendation="Buy",
            confidence=0.8,
        )
        assert a.commodity == "lithium_carbonate"
        assert len(a.key_drivers) == 2
        assert a.confidence == 0.8

    def test_key_drivers_json(self):
        a = AIAnalysis(
            commodity="nickel",
            date="2025-01-15",
            key_drivers=["Stainless steel demand", "Indonesian supply"],
        )
        j = a.key_drivers_json
        parsed = json.loads(j)
        assert parsed == ["Stainless steel demand", "Indonesian supply"]

    def test_risk_factors_json(self):
        a = AIAnalysis(
            commodity="cobalt",
            date="2025-01-15",
            risk_factors=["DRC political risk", "Recycling growth"],
        )
        j = a.risk_factors_json
        parsed = json.loads(j)
        assert "DRC political risk" in parsed

    def test_to_dict(self):
        a = AIAnalysis(
            commodity="copper",
            date="2025-01-15",
            recommendation="Hold",
            confidence=0.6,
        )
        d = a.to_dict()
        assert d["recommendation"] == "Hold"
        assert d["confidence"] == 0.6
        assert d["key_drivers"] == []


class TestPriceAlert:
    """Test PriceAlert model."""

    def test_creation_minimal(self):
        alert = PriceAlert(
            commodity="lithium_carbonate",
            alert_type="anomaly",
            severity="high",
            message="Price rose 5.2%",
        )
        assert alert.id != ""  # Auto-generated UUID prefix
        assert alert.commodity == "lithium_carbonate"
        assert alert.acknowledged is False
        assert alert.triggered_at != ""  # Auto-generated

    def test_creation_full(self):
        alert = PriceAlert(
            id="alert-001",
            commodity="nickel",
            alert_type="threshold",
            severity="critical",
            message="Price dropped 12%",
            price=128000.0,
            threshold=10.0,
            current_value=12.0,
            triggered_at="2025-01-15T10:00:00Z",
            acknowledged=True,
        )
        assert alert.id == "alert-001"
        assert alert.severity == "critical"
        assert alert.acknowledged is True

    def test_to_dict(self):
        alert = PriceAlert(
            commodity="cobalt",
            alert_type="scrape_failure",
            severity="medium",
            message="SHMET scrape failed",
        )
        d = alert.to_dict()
        assert d["commodity"] == "cobalt"
        assert d["alert_type"] == "scrape_failure"
        assert d["acknowledged"] is False


class TestWeeklySummary:
    """Test WeeklySummary model."""

    def test_creation(self):
        ws = WeeklySummary(
            date="2025-W03",
            product="lithium_carbonate",
            unit="yuan/t",
            avg_price=157500.0,
            wow_change_pct=1.5,
            mom_change_pct=-0.8,
        )
        assert ws.avg_price == 157500.0
        assert ws.wow_change_pct == 1.5

    def test_to_dict(self):
        ws = WeeklySummary(
            date="2025-W03",
            product="nickel_sulfate",
            unit="yuan/t",
            avg_price=31000.0,
        )
        d = ws.to_dict()
        assert d["product"] == "nickel_sulfate"
        assert d["wow_change_pct"] == 0.0


class TestCommodityRegistry:
    """Test commodity registry and lookup."""

    def test_registry_not_empty(self):
        assert len(COMMODITY_REGISTRY) >= 10

    def test_registry_has_required_commodities(self):
        names = [c.name for c in COMMODITY_REGISTRY]
        assert "lithium_carbonate" in names
        assert "nickel" in names
        assert "cobalt" in names
        assert "manganese_sulfate" in names
        assert "gold" in names
        assert "copper" in names

    def test_get_commodity_info_found(self):
        info = get_commodity_info("lithium_carbonate")
        assert info is not None
        assert info.name_cn == "碳酸锂"
        assert "battery_grade" in info.grades
        assert "shmet" in info.sources
        assert info.category == "battery_material"

    def test_get_commodity_info_not_found(self):
        info = get_commodity_info("nonexistent_metal")
        assert info is None

    def test_commodity_info_to_dict(self):
        info = get_commodity_info("nickel")
        d = info.to_dict()
        assert d["name"] == "nickel"
        assert isinstance(d["grades"], list)
        assert isinstance(d["sources"], list)
