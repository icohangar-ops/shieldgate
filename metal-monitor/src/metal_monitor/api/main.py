"""FastAPI application — Metal Price Monitoring Agent."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..database import init_db, _get_db_path
from ..scraper.metal_com import MetalComScraper
from ..scraper.shmet import SHMETScraper
from ..scraper.shfe import SHFEScraper
from ..alerts.engine import AlertEngine
from .routes.prices import router as prices_router
from .routes.analysis import router as analysis_router
from .routes.alerts import router as alerts_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB and seed data."""
    db_path = os.environ.get("METAL_MONITOR_DB", None)
    init_db(db_path)
    # Seed data if empty
    from ..database import get_all_latest_prices
    prices = get_all_latest_prices(db_path)
    if not prices:
        await seed_historical_data(db_path)
    yield


def create_app(db_path: Optional[str] = None, with_lifespan: bool = True) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        db_path: Optional database path.
        with_lifespan: Whether to use the lifespan handler (disable for testing).
    """
    app = FastAPI(
        title="Metal Price Monitoring Agent",
        description="Qwen-powered battery materials intelligence platform",
        version="0.1.0",
        lifespan=lifespan if with_lifespan else None,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(prices_router, prefix="/api/v1")
    app.include_router(analysis_router, prefix="/api/v1")
    app.include_router(alerts_router, prefix="/api/v1")

    # Store db_path for use in routes
    app.state.db_path = db_path

    @app.get("/api/v1/health")
    async def health_check():
        return {"status": "ok", "service": "metal-monitor", "version": "0.1.0"}

    @app.get("/api/v1/dashboard")
    async def dashboard():
        from ..database import get_dashboard_data
        return get_dashboard_data(db_path)

    @app.get("/api/v1/commodities")
    async def list_commodities():
        from ..models import COMMODITY_REGISTRY
        return [c.to_dict() for c in COMMODITY_REGISTRY]

    return app


app = create_app()


async def seed_historical_data(db_path: Optional[str] = None):
    """Seed the database with 30 days of mock historical data."""
    from datetime import datetime, timedelta
    from ..database import upsert_observations

    scrapers = [MetalComScraper(), SHMETScraper(), SHFEScraper()]
    today = datetime.utcnow()

    total = 0
    for days_ago in range(30, -1, -1):
        date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        for scraper in scrapers:
            observations = await scraper.scrape(date)
            count = upsert_observations(observations, db_path)
            total += count

    print(f"[seed] Inserted {total} price observations over 31 days")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.metal_monitor.api.main:app", host="0.0.0.0", port=8000, reload=True)
