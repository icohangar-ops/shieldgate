"""Price endpoints — /api/v1/prices"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, Request

from ...database import (
    get_all_latest_prices,
    get_latest_price,
    get_price_history,
    get_weekly_summary,
    compute_price_summary,
    upsert_observations,
)
from ...scraper.metal_com import MetalComScraper
from ...scraper.shmet import SHMETScraper
from ...scraper.shfe import SHFEScraper
from ...alerts.engine import AlertEngine

router = APIRouter(prefix="/prices", tags=["prices"])


@router.get("")
async def list_latest_prices(request: Request):
    """Get latest prices for all commodities."""
    db_path = getattr(request.app.state, "db_path", None)
    prices = get_all_latest_prices(db_path)
    return {"count": len(prices), "prices": [p.to_dict() for p in prices]}


@router.get("/{commodity}")
async def get_commodity_prices(
    commodity: str,
    request: Request,
    grade: str = Query("battery_grade", description="Grade filter"),
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    source: Optional[str] = Query(None, description="Source filter"),
):
    """Get price history for a specific commodity."""
    db_path = getattr(request.app.state, "db_path", None)
    history = get_price_history(
        commodity, grade=grade, days=days, source=source, db_path=db_path
    )
    return {
        "commodity": commodity,
        "grade": grade,
        "days": days,
        "count": len(history),
        "prices": [p.to_dict() for p in history],
    }


@router.get("/{commodity}/summary")
async def get_commodity_summary(commodity: str, request: Request):
    """Get Mysteel-format weekly summary for a commodity."""
    db_path = getattr(request.app.state, "db_path", None)
    summary = compute_price_summary(commodity, db_path=db_path)
    weekly = get_weekly_summary(commodity, db_path)

    if not summary:
        return {"commodity": commodity, "error": "No data available"}

    result = summary.to_dict()
    if weekly:
        result["weekly"] = weekly.to_dict()
    return result


@router.post("/scrape")
async def trigger_scrape(request: Request):
    """Trigger manual price scraping from all sources."""
    db_path = getattr(request.app.state, "db_path", None)
    scrapers = [MetalComScraper(), SHMETScraper(), SHFEScraper()]

    total_observations = []
    errors = []

    for scraper in scrapers:
        try:
            observations = await scraper.scrape()
            total_observations.extend(observations)
            count = upsert_observations(observations, db_path)
            # Run alert engine on new data
            engine = AlertEngine(db_path=db_path)
            alerts = engine.check_observations(observations)
        except Exception as e:
            errors.append({"source": scraper.source_name, "error": str(e)})

    return {
        "status": "completed",
        "observations_scraped": len(total_observations),
        "errors": errors if errors else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
