"""Alert endpoints — /api/v1/alerts"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request

from ...database import get_active_alerts, acknowledge_alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
async def list_alerts(
    request: Request,
    severity: Optional[str] = None,
):
    """Get active (unacknowledged) alerts."""
    db_path = getattr(request.app.state, "db_path", None)
    alerts = get_active_alerts(db_path)

    if severity:
        alerts = [a for a in alerts if a.severity == severity]

    return {
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    }


@router.post("/{alert_id}/acknowledge")
async def ack_alert(alert_id: int, request: Request):
    """Acknowledge (dismiss) an alert."""
    db_path = getattr(request.app.state, "db_path", None)
    success = acknowledge_alert(alert_id, db_path)
    if success:
        return {"status": "acknowledged", "alert_id": alert_id}
    return {"status": "not_found", "alert_id": alert_id}
