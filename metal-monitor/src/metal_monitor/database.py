"""SQLite database persistence layer for metal price data."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

from .models import (
    PriceObservation,
    PriceSummary,
    AIAnalysis,
    PriceAlert,
    WeeklySummary,
    CommodityInfo,
    COMMODITY_REGISTRY,
    get_commodity_info,
)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "db", "metal_prices.db")


def _get_db_path() -> str:
    """Get the database path, creating parent directories if needed."""
    path = os.environ.get("METAL_MONITOR_DB", DEFAULT_DB_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(db_path or _get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS price_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    commodity TEXT NOT NULL,
    grade TEXT NOT NULL,
    price_cny REAL,
    price_usd REAL,
    fx_rate REAL DEFAULT 7.25,
    unit TEXT DEFAULT 'yuan/t',
    contract_month TEXT,
    change_pct REAL DEFAULT 0.0,
    change_type TEXT DEFAULT 'flat',
    scrape_status TEXT DEFAULT 'success',
    timestamp TEXT DEFAULT (datetime('now')),
    UNIQUE(date, source, commodity, grade)
);

CREATE TABLE IF NOT EXISTS ai_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity TEXT NOT NULL,
    date TEXT NOT NULL,
    market_commentary TEXT,
    outlook TEXT,
    key_drivers TEXT,
    risk_factors TEXT,
    recommendation TEXT,
    confidence REAL,
    model_used TEXT DEFAULT 'qwen3.6-flash',
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(commodity, date)
);

CREATE TABLE IF NOT EXISTS price_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT,
    price REAL,
    threshold REAL,
    current_value REAL,
    triggered_at TEXT DEFAULT (datetime('now')),
    acknowledged INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS weekly_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    product TEXT NOT NULL,
    unit TEXT NOT NULL,
    avg_price REAL,
    wow_change_pct REAL,
    mom_change_pct REAL,
    UNIQUE(date, product)
);
"""


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Initialize the database schema."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ── Price Observations CRUD ──────────────────────────────────────────────────

def upsert_observation(obs: PriceObservation, db_path: Optional[str] = None) -> int:
    """Insert or update a price observation. Returns the row id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO price_observations
               (date, source, commodity, grade, price_cny, price_usd, fx_rate,
                unit, contract_month, change_pct, change_type, scrape_status, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(date, source, commodity, grade)
               DO UPDATE SET price_cny=excluded.price_cny, price_usd=excluded.price_usd,
                 fx_rate=excluded.fx_rate, change_pct=excluded.change_pct,
                 change_type=excluded.change_type, scrape_status=excluded.scrape_status,
                 timestamp=excluded.timestamp""",
            (obs.date, obs.source, obs.commodity, obs.grade, obs.price_cny,
             obs.price_usd, obs.fx_rate, obs.unit, obs.contract_month,
             obs.change_pct, obs.change_type, obs.scrape_status, obs.timestamp),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def upsert_observations(observations: List[PriceObservation], db_path: Optional[str] = None) -> int:
    """Bulk insert/update observations. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        count = 0
        for obs in observations:
            conn.execute(
                """INSERT INTO price_observations
                   (date, source, commodity, grade, price_cny, price_usd, fx_rate,
                    unit, contract_month, change_pct, change_type, scrape_status, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(date, source, commodity, grade)
                   DO UPDATE SET price_cny=excluded.price_cny, price_usd=excluded.price_usd,
                     fx_rate=excluded.fx_rate, change_pct=excluded.change_pct,
                     change_type=excluded.change_type, scrape_status=excluded.scrape_status,
                     timestamp=excluded.timestamp""",
                (obs.date, obs.source, obs.commodity, obs.grade, obs.price_cny,
                 obs.price_usd, obs.fx_rate, obs.unit, obs.contract_month,
                 obs.change_pct, obs.change_type, obs.scrape_status, obs.timestamp),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def get_latest_price(commodity: str, grade: str = "battery_grade",
                     source: Optional[str] = None, db_path: Optional[str] = None) -> Optional[PriceObservation]:
    """Get the most recent price observation for a commodity."""
    conn = get_connection(db_path)
    try:
        query = """SELECT * FROM price_observations
                   WHERE commodity = ? AND grade = ?"""
        params: list = [commodity, grade]
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY date DESC, timestamp DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        if row:
            return PriceObservation.from_dict(dict(row))
        return None
    finally:
        conn.close()


def get_price_history(commodity: str, grade: str = "battery_grade",
                      days: int = 30, source: Optional[str] = None,
                      db_path: Optional[str] = None) -> List[PriceObservation]:
    """Get price history for a commodity over the last N days."""
    conn = get_connection(db_path)
    try:
        since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = """SELECT * FROM price_observations
                   WHERE commodity = ? AND grade = ? AND date >= ?"""
        params: list = [commodity, grade, since]
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY date ASC"
        rows = conn.execute(query, params).fetchall()
        return [PriceObservation.from_dict(dict(r)) for r in rows]
    finally:
        conn.close()


def get_all_latest_prices(db_path: Optional[str] = None) -> List[PriceObservation]:
    """Get the latest price for every (commodity, grade, source) combination."""
    conn = get_connection(db_path)
    try:
        query = """
            SELECT p.* FROM price_observations p
            INNER JOIN (
                SELECT commodity, grade, source, MAX(date) as max_date
                FROM price_observations
                GROUP BY commodity, grade, source
            ) latest ON p.commodity = latest.commodity
                     AND p.grade = latest.grade
                     AND p.source = latest.source
                     AND p.date = latest.max_date
            ORDER BY p.commodity, p.source
        """
        rows = conn.execute(query).fetchall()
        return [PriceObservation.from_dict(dict(r)) for r in rows]
    finally:
        conn.close()


def get_previous_price(commodity: str, grade: str, current_date: str,
                       source: Optional[str] = None,
                       db_path: Optional[str] = None) -> Optional[PriceObservation]:
    """Get the price observation immediately before current_date."""
    conn = get_connection(db_path)
    try:
        query = """SELECT * FROM price_observations
                   WHERE commodity = ? AND grade = ? AND date < ?"""
        params: list = [commodity, grade, current_date]
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY date DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        if row:
            return PriceObservation.from_dict(dict(row))
        return None
    finally:
        conn.close()


# ── AI Analyses CRUD ─────────────────────────────────────────────────────────

def upsert_analysis(analysis: AIAnalysis, db_path: Optional[str] = None) -> int:
    """Insert or update an AI analysis. Returns row id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO ai_analyses
               (commodity, date, market_commentary, outlook, key_drivers,
                risk_factors, recommendation, confidence, model_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(commodity, date)
               DO UPDATE SET market_commentary=excluded.market_commentary,
                 outlook=excluded.outlook, key_drivers=excluded.key_drivers,
                 risk_factors=excluded.risk_factors, recommendation=excluded.recommendation,
                 confidence=excluded.confidence, model_used=excluded.model_used""",
            (analysis.commodity, analysis.date, analysis.market_commentary,
             analysis.outlook, analysis.key_drivers_json,
             analysis.risk_factors_json, analysis.recommendation,
             analysis.confidence, analysis.model_used),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_latest_analysis(commodity: str, db_path: Optional[str] = None) -> Optional[AIAnalysis]:
    """Get the most recent AI analysis for a commodity."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM ai_analyses WHERE commodity = ? ORDER BY date DESC LIMIT 1",
            (commodity,),
        ).fetchone()
        if row:
            d = dict(row)
            return AIAnalysis(
                commodity=d["commodity"],
                date=d["date"],
                market_commentary=d["market_commentary"] or "",
                outlook=d["outlook"] or "",
                key_drivers=json.loads(d["key_drivers"]) if d["key_drivers"] else [],
                risk_factors=json.loads(d["risk_factors"]) if d["risk_factors"] else [],
                recommendation=d["recommendation"] or "",
                confidence=d["confidence"] or 0.0,
                model_used=d["model_used"] or "qwen3.6-flash",
            )
        return None
    finally:
        conn.close()


def get_all_latest_analyses(db_path: Optional[str] = None) -> List[AIAnalysis]:
    """Get latest AI analysis for every commodity."""
    conn = get_connection(db_path)
    try:
        query = """
            SELECT a.* FROM ai_analyses a
            INNER JOIN (
                SELECT commodity, MAX(date) as max_date
                FROM ai_analyses GROUP BY commodity
            ) latest ON a.commodity = latest.commodity AND a.date = latest.max_date
            ORDER BY a.commodity
        """
        rows = conn.execute(query).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            results.append(AIAnalysis(
                commodity=d["commodity"],
                date=d["date"],
                market_commentary=d["market_commentary"] or "",
                outlook=d["outlook"] or "",
                key_drivers=json.loads(d["key_drivers"]) if d["key_drivers"] else [],
                risk_factors=json.loads(d["risk_factors"]) if d["risk_factors"] else [],
                recommendation=d["recommendation"] or "",
                confidence=d["confidence"] or 0.0,
                model_used=d["model_used"] or "qwen3.6-flash",
            ))
        return results
    finally:
        conn.close()


# ── Price Alerts CRUD ────────────────────────────────────────────────────────

def insert_alert(alert: PriceAlert, db_path: Optional[str] = None) -> int:
    """Insert a new price alert. Returns row id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO price_alerts
               (commodity, alert_type, severity, message, price, threshold,
                current_value, triggered_at, acknowledged)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (alert.commodity, alert.alert_type, alert.severity, alert.message,
             alert.price, alert.threshold, alert.current_value,
             alert.triggered_at, int(alert.acknowledged)),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_active_alerts(db_path: Optional[str] = None) -> List[PriceAlert]:
    """Get all unacknowledged alerts, most recent first."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM price_alerts WHERE acknowledged = 0 ORDER BY triggered_at DESC"
        ).fetchall()
        return [
            PriceAlert(
                id=str(r["id"]),
                commodity=r["commodity"],
                alert_type=r["alert_type"],
                severity=r["severity"],
                message=r["message"] or "",
                price=r["price"] or 0.0,
                threshold=r["threshold"] or 0.0,
                current_value=r["current_value"] or 0.0,
                triggered_at=r["triggered_at"],
                acknowledged=bool(r["acknowledged"]),
            )
            for r in rows
        ]
    finally:
        conn.close()


def acknowledge_alert(alert_id: int, db_path: Optional[str] = None) -> bool:
    """Mark an alert as acknowledged. Returns True if found."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE price_alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ── Weekly Summaries CRUD ────────────────────────────────────────────────────

def upsert_weekly_summary(summary: WeeklySummary, db_path: Optional[str] = None) -> int:
    """Insert or update a weekly summary."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO weekly_summaries (date, product, unit, avg_price, wow_change_pct, mom_change_pct)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(date, product)
               DO UPDATE SET unit=excluded.unit, avg_price=excluded.avg_price,
                 wow_change_pct=excluded.wow_change_pct, mom_change_pct=excluded.mom_change_pct""",
            (summary.date, summary.product, summary.unit, summary.avg_price,
             summary.wow_change_pct, summary.mom_change_pct),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_weekly_summary(commodity: str, db_path: Optional[str] = None) -> Optional[WeeklySummary]:
    """Get the latest weekly summary for a commodity."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM weekly_summaries WHERE product = ? ORDER BY date DESC LIMIT 1",
            (commodity,),
        ).fetchone()
        if row:
            d = dict(row)
            return WeeklySummary(
                date=d["date"], product=d["product"], unit=d["unit"],
                avg_price=d["avg_price"],
                wow_change_pct=d["wow_change_pct"] or 0.0,
                mom_change_pct=d["mom_change_pct"] or 0.0,
            )
        return None
    finally:
        conn.close()


def get_all_weekly_summaries(db_path: Optional[str] = None) -> List[WeeklySummary]:
    """Get latest weekly summaries for all products."""
    conn = get_connection(db_path)
    try:
        query = """
            SELECT w.* FROM weekly_summaries w
            INNER JOIN (
                SELECT product, MAX(date) as max_date
                FROM weekly_summaries GROUP BY product
            ) latest ON w.product = latest.product AND w.date = latest.max_date
            ORDER BY w.product
        """
        rows = conn.execute(query).fetchall()
        return [
            WeeklySummary(
                date=r["date"], product=r["product"], unit=r["unit"],
                avg_price=r["avg_price"],
                wow_change_pct=r["wow_change_pct"] or 0.0,
                mom_change_pct=r["mom_change_pct"] or 0.0,
            )
            for r in rows
        ]
    finally:
        conn.close()


# ── Price Summary Computation ────────────────────────────────────────────────

def compute_price_summary(commodity: str, grade: str = "battery_grade",
                          source: Optional[str] = None,
                          db_path: Optional[str] = None) -> Optional[PriceSummary]:
    """Compute a PriceSummary (with WoW, MoM, trend) for a commodity."""
    latest = get_latest_price(commodity, grade, source, db_path)
    if not latest:
        return None

    # WoW: compare to 7 days ago
    wow_obs = get_price_at_offset(commodity, grade, 7, latest.date, source, db_path)
    wow_change_pct = _calc_pct(latest.price_cny, wow_obs.price_cny if wow_obs else None)

    # MoM: compare to 30 days ago
    mom_obs = get_price_at_offset(commodity, grade, 30, latest.date, source, db_path)
    mom_change_pct = _calc_pct(latest.price_cny, mom_obs.price_cny if mom_obs else None)

    # Trend detection from recent 5 observations
    trend = _detect_trend(commodity, grade, source, db_path)

    return PriceSummary(
        commodity=commodity,
        grade=grade,
        latest_price_cny=latest.price_cny,
        latest_price_usd=latest.price_usd,
        wow_change_pct=wow_change_pct,
        mom_change_pct=mom_change_pct,
        source=latest.source,
        date=latest.date,
        trend=trend,
    )


def get_price_at_offset(commodity: str, grade: str, days_offset: int,
                        ref_date: str, source: Optional[str] = None,
                        db_path: Optional[str] = None) -> Optional[PriceObservation]:
    """Get price N days before ref_date."""
    conn = get_connection(db_path)
    try:
        from datetime import datetime, timedelta
        target = (datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=days_offset)).strftime("%Y-%m-%d")
        query = """SELECT * FROM price_observations
                   WHERE commodity = ? AND grade = ? AND date <= ?"""
        params: list = [commodity, grade, target]
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY date DESC LIMIT 1"
        row = conn.execute(query, params).fetchone()
        if row:
            return PriceObservation.from_dict(dict(row))
        return None
    finally:
        conn.close()


def _calc_pct(current: float, previous: Optional[float]) -> float:
    if previous is None or previous == 0:
        return 0.0
    return round(((current - previous) / previous) * 100, 2)


def _detect_trend(commodity: str, grade: str, source: Optional[str] = None,
                  db_path: Optional[str] = None) -> str:
    """Detect price trend from last 5 data points."""
    history = get_price_history(commodity, grade, days=7, source=source, db_path=db_path)
    if len(history) < 3:
        return "stable"

    prices = [h.price_cny for h in history]
    recent = prices[-3:]

    changes = [(recent[i] - recent[i - 1]) / recent[i - 1] * 100
               for i in range(1, len(recent))]

    avg_change = sum(changes) / len(changes)
    volatility = max(changes) - min(changes)

    if volatility > 3.0:
        return "volatile"
    if avg_change > 0.5:
        return "rising"
    if avg_change < -0.5:
        return "falling"
    return "stable"


# ── Dashboard Data ───────────────────────────────────────────────────────────

def get_dashboard_data(db_path: Optional[str] = None) -> dict:
    """Aggregate dashboard data: prices, summaries, alerts."""
    prices = get_all_latest_prices(db_path)
    analyses = get_all_latest_analyses(db_path)
    alerts = get_active_alerts(db_path)
    summaries = get_all_weekly_summaries(db_path)

    return {
        "latest_prices": [p.to_dict() for p in prices],
        "ai_analyses": [a.to_dict() for a in analyses],
        "active_alerts": [a.to_dict() for a in alerts],
        "weekly_summaries": [s.to_dict() for s in summaries],
        "commodities_tracked": len(COMMODITY_REGISTRY),
    }
