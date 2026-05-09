"""AI analysis endpoints — /api/v1/analysis"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, Request

from ...ai.analyst import AIAnalyst
from ...database import (
    get_all_latest_analyses,
    get_latest_analysis,
    upsert_analysis,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("")
async def list_analyses(request: Request):
    """Get latest AI analysis for all commodities."""
    db_path = getattr(request.app.state, "db_path", None)
    analyses = get_all_latest_analyses(db_path)
    return {
        "count": len(analyses),
        "analyses": [a.to_dict() for a in analyses],
    }


@router.get("/{commodity}")
async def get_analysis(commodity: str, request: Request):
    """Get AI analysis for a specific commodity."""
    db_path = getattr(request.app.state, "db_path", None)
    analysis = get_latest_analysis(commodity, db_path)
    if not analysis:
        return {
            "commodity": commodity,
            "error": "No analysis available. Generate one with POST /analysis/generate.",
        }
    return analysis.to_dict()


@router.post("/generate")
async def generate_analysis(
    request: Request,
    commodity: Optional[str] = Query(None, description="Commodity to analyze (omit for all)"),
):
    """Generate new AI analysis using Qwen."""
    db_path = getattr(request.app.state, "db_path", None)
    analyst = AIAnalyst()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    if commodity:
        analysis = await analyst.generate_analysis(commodity, today, db_path)
        upsert_analysis(analysis, db_path)
        return {
            "status": "generated",
            "analysis": analysis.to_dict(),
        }

    # Generate for all battery materials
    from ...models import COMMODITY_REGISTRY
    battery_commodities = [
        c.name for c in COMMODITY_REGISTRY if c.category == "battery_material"
    ]

    results = []
    for name in battery_commodities:
        analysis = await analyst.generate_analysis(name, today, db_path)
        upsert_analysis(analysis, db_path)
        results.append(analysis.to_dict())

    return {
        "status": "generated",
        "count": len(results),
        "analyses": results,
    }
