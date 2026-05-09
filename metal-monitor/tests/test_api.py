"""Tests for FastAPI endpoints."""

import os
import pytest
import tempfile

from fastapi.testclient import TestClient

from src.metal_monitor.api.main import create_app
from src.metal_monitor.database import init_db


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)


@pytest.fixture
def client(db_path):
    app = create_app(db_path=db_path, with_lifespan=False)
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "metal-monitor"


class TestCommoditiesEndpoint:
    def test_list_commodities(self, client):
        resp = client.get("/api/v1/commodities")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 10

    def test_commodity_has_required_fields(self, client):
        resp = client.get("/api/v1/commodities")
        commodities = resp.json()
        c = next(x for x in commodities if x["name"] == "lithium_carbonate")
        assert "name" in c
        assert "name_cn" in c
        assert "unit" in c
        assert "grades" in c
        assert "sources" in c
        assert "category" in c


class TestPricesEndpoints:
    def test_list_latest_prices_empty(self, client):
        resp = client.get("/api/v1/prices")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["prices"] == []

    def test_get_commodity_prices_empty(self, client):
        resp = client.get("/api/v1/prices/lithium_carbonate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_get_commodity_summary_empty(self, client):
        resp = client.get("/api/v1/prices/lithium_carbonate/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") is not None

    def test_scrape_endpoint(self, client):
        resp = client.post("/api/v1/prices/scrape")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["observations_scraped"] > 0
        assert data["errors"] is None

    def test_scrape_then_list(self, client):
        client.post("/api/v1/prices/scrape")
        resp = client.get("/api/v1/prices")
        data = resp.json()
        assert data["count"] > 0


class TestAnalysisEndpoints:
    def test_list_analyses_empty(self, client):
        resp = client.get("/api/v1/analysis")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_get_analysis_empty(self, client):
        resp = client.get("/api/v1/analysis/lithium_carbonate")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_generate_analysis_no_data(self, client):
        resp = client.post("/api/v1/analysis/generate?commodity=lithium_carbonate")
        assert resp.status_code == 200


class TestAlertsEndpoints:
    def test_list_alerts_empty(self, client):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_acknowledge_nonexistent(self, client):
        resp = client.post("/api/v1/alerts/9999/acknowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"


class TestDashboardEndpoint:
    def test_dashboard_empty(self, client):
        resp = client.get("/api/v1/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "latest_prices" in data
        assert "ai_analyses" in data
        assert "active_alerts" in data
        assert "weekly_summaries" in data
        assert "commodities_tracked" in data


class TestFullWorkflow:
    """Integration test: scrape → list → summary → alerts."""

    def test_full_workflow(self, db_path):
        from src.metal_monitor.database import init_db
        init_db(db_path)

        app = create_app(db_path=db_path, with_lifespan=False)
        with TestClient(app) as client:
            # Trigger scrape
            resp = client.post("/api/v1/prices/scrape")
            assert resp.status_code == 200
            assert resp.json()["observations_scraped"] > 0

            # List prices
            resp = client.get("/api/v1/prices")
            assert resp.json()["count"] > 0

            # Dashboard
            resp = client.get("/api/v1/dashboard")
            assert resp.status_code == 200
            assert resp.json()["commodities_tracked"] >= 10
