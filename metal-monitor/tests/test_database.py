"""Tests for database operations."""

import os
import pytest
import tempfile
from datetime import datetime, timedelta

from src.metal_monitor.database import (
    init_db,
    upsert_observation,
    upsert_observations,
    get_latest_price,
    get_price_history,
    get_all_latest_prices,
    get_previous_price,
    upsert_analysis,
    get_latest_analysis,
    get_all_latest_analyses,
    insert_alert,
    get_active_alerts,
    acknowledge_alert,
    upsert_weekly_summary,
    get_weekly_summary,
    get_all_weekly_summaries,
    compute_price_summary,
    get_dashboard_data,
)
from src.metal_monitor.models import (
    PriceObservation,
    AIAnalysis,
    PriceAlert,
    WeeklySummary,
)


@pytest.fixture
def db_path():
    """Create a temporary database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_observations():
    """Create sample price observations."""
    return [
        PriceObservation(
            date="2025-01-13",
            source="metal_com",
            commodity="lithium_carbonate",
            grade="battery_grade",
            price_cny=155000.0,
            change_pct=-1.0,
            change_type="fall",
        ),
        PriceObservation(
            date="2025-01-14",
            source="metal_com",
            commodity="lithium_carbonate",
            grade="battery_grade",
            price_cny=156000.0,
            change_pct=0.65,
            change_type="rise",
        ),
        PriceObservation(
            date="2025-01-15",
            source="metal_com",
            commodity="lithium_carbonate",
            grade="battery_grade",
            price_cny=157000.0,
            change_pct=0.64,
            change_type="rise",
        ),
        PriceObservation(
            date="2025-01-15",
            source="shmet",
            commodity="nickel_sulfate",
            grade="battery_grade",
            price_cny=31000.0,
        ),
    ]


class TestDatabaseInit:
    def test_init_creates_tables(self, db_path):
        """Database should be initialized with all tables."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "price_observations" in tables
        assert "ai_analyses" in tables
        assert "price_alerts" in tables
        assert "weekly_summaries" in tables


class TestPriceObservationsCRUD:
    def test_upsert_single(self, db_path):
        obs = PriceObservation(
            date="2025-01-15",
            source="metal_com",
            commodity="lithium_carbonate",
            grade="battery_grade",
            price_cny=157000.0,
        )
        row_id = upsert_observation(obs, db_path)
        assert row_id >= 1

    def test_upsert_conflict_updates(self, db_path):
        obs1 = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        obs2 = PriceObservation(
            date="2025-01-15", source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=158000.0,
        )
        upsert_observation(obs1, db_path)
        upsert_observation(obs2, db_path)
        latest = get_latest_price("lithium_carbonate", "battery_grade", "metal_com", db_path)
        assert latest.price_cny == 158000.0

    def test_bulk_upsert(self, db_path, sample_observations):
        count = upsert_observations(sample_observations, db_path)
        assert count == 4

    def test_get_latest_price(self, db_path, sample_observations):
        upsert_observations(sample_observations, db_path)
        latest = get_latest_price("lithium_carbonate", "battery_grade", "metal_com", db_path)
        assert latest is not None
        assert latest.price_cny == 157000.0
        assert latest.date == "2025-01-15"

    def test_get_latest_price_not_found(self, db_path):
        latest = get_latest_price("nonexistent", "battery_grade", db_path=db_path)
        assert latest is None

    def test_get_price_history(self, db_path):
        """Use dates relative to today so get_price_history can find them."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        two_days_ago = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
        obs = [
            PriceObservation(date=two_days_ago, source="metal_com",
                             commodity="lithium_carbonate", grade="battery_grade",
                             price_cny=155000.0),
            PriceObservation(date=yesterday, source="metal_com",
                             commodity="lithium_carbonate", grade="battery_grade",
                             price_cny=156000.0),
            PriceObservation(date=today, source="metal_com",
                             commodity="lithium_carbonate", grade="battery_grade",
                             price_cny=157000.0),
        ]
        upsert_observations(obs, db_path)
        history = get_price_history("lithium_carbonate", "battery_grade", days=30, source="metal_com", db_path=db_path)
        assert len(history) == 3
        assert history[0].date == two_days_ago
        assert history[2].date == today

    def test_get_all_latest_prices(self, db_path, sample_observations):
        upsert_observations(sample_observations, db_path)
        all_latest = get_all_latest_prices(db_path)
        assert len(all_latest) == 2  # 2 unique (commodity, grade, source) combos

    def test_get_previous_price(self, db_path, sample_observations):
        upsert_observations(sample_observations, db_path)
        prev = get_previous_price("lithium_carbonate", "battery_grade", "2025-01-15", "metal_com", db_path)
        assert prev is not None
        assert prev.price_cny == 156000.0

    def test_get_previous_price_none(self, db_path):
        prev = get_previous_price("lithium_carbonate", "battery_grade", "2025-01-10", db_path=db_path)
        assert prev is None


class TestAIAnalysesCRUD:
    def test_upsert_and_get(self, db_path):
        analysis = AIAnalysis(
            commodity="lithium_carbonate",
            date="2025-01-15",
            market_commentary="Prices are rising due to EV demand",
            outlook="Bullish",
            key_drivers=["EV demand", "Supply cuts"],
            risk_factors=["Policy changes"],
            recommendation="Buy",
            confidence=0.8,
        )
        row_id = upsert_analysis(analysis, db_path)
        assert row_id >= 1

        retrieved = get_latest_analysis("lithium_carbonate", db_path)
        assert retrieved is not None
        assert retrieved.commodity == "lithium_carbonate"
        assert retrieved.market_commentary == "Prices are rising due to EV demand"
        assert len(retrieved.key_drivers) == 2
        assert retrieved.confidence == 0.8

    def test_upsert_conflict(self, db_path):
        a1 = AIAnalysis(commodity="nickel", date="2025-01-15", recommendation="Buy", confidence=0.9)
        a2 = AIAnalysis(commodity="nickel", date="2025-01-15", recommendation="Hold", confidence=0.5)
        upsert_analysis(a1, db_path)
        upsert_analysis(a2, db_path)
        retrieved = get_latest_analysis("nickel", db_path)
        assert retrieved.recommendation == "Hold"
        assert retrieved.confidence == 0.5

    def test_get_all_latest(self, db_path):
        upsert_analysis(AIAnalysis(commodity="lithium_carbonate", date="2025-01-15"), db_path)
        upsert_analysis(AIAnalysis(commodity="nickel", date="2025-01-14"), db_path)
        all_latest = get_all_latest_analyses(db_path)
        assert len(all_latest) == 2

    def test_get_not_found(self, db_path):
        result = get_latest_analysis("nonexistent", db_path)
        assert result is None


class TestAlertsCRUD:
    def test_insert_and_get(self, db_path):
        alert = PriceAlert(
            commodity="lithium_carbonate",
            alert_type="anomaly",
            severity="high",
            message="Price rose 5.2%",
            price=157000.0,
            threshold=5.0,
            current_value=5.2,
        )
        row_id = insert_alert(alert, db_path)
        assert row_id >= 1

        active = get_active_alerts(db_path)
        assert len(active) == 1
        assert active[0].commodity == "lithium_carbonate"
        assert active[0].severity == "high"
        assert active[0].acknowledged is False

    def test_acknowledge(self, db_path):
        alert = PriceAlert(
            commodity="nickel", alert_type="anomaly", severity="critical",
            message="Price crash", price=128000.0,
        )
        row_id = insert_alert(alert, db_path)
        success = acknowledge_alert(row_id, db_path)
        assert success is True

        active = get_active_alerts(db_path)
        assert len(active) == 0

    def test_acknowledge_not_found(self, db_path):
        success = acknowledge_alert(9999, db_path)
        assert success is False


class TestWeeklySummariesCRUD:
    def test_upsert_and_get(self, db_path):
        ws = WeeklySummary(
            date="2025-W03",
            product="lithium_carbonate",
            unit="yuan/t",
            avg_price=157500.0,
            wow_change_pct=1.5,
            mom_change_pct=-0.8,
        )
        row_id = upsert_weekly_summary(ws, db_path)
        assert row_id >= 1

        retrieved = get_weekly_summary("lithium_carbonate", db_path)
        assert retrieved is not None
        assert retrieved.avg_price == 157500.0
        assert retrieved.wow_change_pct == 1.5

    def test_get_all(self, db_path):
        upsert_weekly_summary(WeeklySummary(date="2025-W03", product="lithium_carbonate", unit="yuan/t", avg_price=157500.0), db_path)
        upsert_weekly_summary(WeeklySummary(date="2025-W03", product="nickel_sulfate", unit="yuan/t", avg_price=31000.0), db_path)
        all_summaries = get_all_weekly_summaries(db_path)
        assert len(all_summaries) == 2


class TestPriceSummary:
    def test_compute_summary(self, db_path, sample_observations):
        upsert_observations(sample_observations, db_path)
        summary = compute_price_summary("lithium_carbonate", "battery_grade", "metal_com", db_path)
        assert summary is not None
        assert summary.latest_price_cny == 157000.0
        assert summary.commodity == "lithium_carbonate"
        assert summary.date == "2025-01-15"

    def test_compute_summary_no_data(self, db_path):
        summary = compute_price_summary("nonexistent", db_path=db_path)
        assert summary is None


class TestDashboard:
    def test_dashboard_data(self, db_path, sample_observations):
        upsert_observations(sample_observations, db_path)
        data = get_dashboard_data(db_path)
        assert "latest_prices" in data
        assert "ai_analyses" in data
        assert "active_alerts" in data
        assert "weekly_summaries" in data
        assert "commodities_tracked" in data
        assert data["commodities_tracked"] >= 10
