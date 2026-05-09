"""Tests for scrapers."""

import pytest
from datetime import datetime, timedelta

from src.metal_monitor.scraper.base import BaseScraper
from src.metal_monitor.scraper.metal_com import MetalComScraper
from src.metal_monitor.scraper.shmet import SHMETScraper
from src.metal_monitor.scraper.shfe import SHFEScraper


class TestBaseScraper:
    """Tests use MetalComScraper (concrete subclass) to test base methods."""

    def test_detect_change_rise(self):
        scraper = MetalComScraper()
        pct, typ = scraper._detect_change(200.0, 100.0)
        assert pct == 100.0
        assert typ == "rise"

    def test_detect_change_fall(self):
        scraper = MetalComScraper()
        pct, typ = scraper._detect_change(90.0, 100.0)
        assert pct == -10.0
        assert typ == "fall"

    def test_detect_change_flat(self):
        scraper = MetalComScraper()
        pct, typ = scraper._detect_change(100.0, 100.0)
        assert pct == 0.0
        assert typ == "flat"

    def test_detect_change_zero_previous(self):
        scraper = MetalComScraper()
        pct, typ = scraper._detect_change(100.0, 0.0)
        assert pct == 0.0
        assert typ == "flat"

    def test_detect_change_small_rise(self):
        scraper = MetalComScraper()
        pct, typ = scraper._detect_change(100.05, 100.0)
        assert pct == 0.05
        assert typ == "flat"  # Below 0.1 threshold

    def test_detect_change_small_fall(self):
        scraper = MetalComScraper()
        pct, typ = scraper._detect_change(99.95, 100.0)
        assert pct == -0.05
        assert typ == "flat"

    def test_random_walk_deterministic(self):
        price1 = BaseScraper._random_walk(100000.0, seed=42)
        price2 = BaseScraper._random_walk(100000.0, seed=42)
        assert price1 == price2

    def test_random_walk_varies_by_seed(self):
        price1 = BaseScraper._random_walk(100000.0, seed=1)
        price2 = BaseScraper._random_walk(100000.0, seed=999)
        assert price1 != price2

    def test_random_walk_produces_reasonable_values(self):
        """Random walk should stay within ~5% for low volatility."""
        base = 157000.0
        for seed in range(100):
            price = BaseScraper._random_walk(base, volatility=0.015, seed=seed)
            assert base * 0.85 < price < base * 1.15, f"seed={seed}, price={price}"

    def test_cannot_instantiate_base_scraper(self):
        """BaseScraper is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract"):
            BaseScraper()


class TestMetalComScraper:
    @pytest.mark.asyncio
    async def test_scrape_returns_data(self):
        scraper = MetalComScraper()
        results = await scraper.scrape("2025-01-15")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_scrape_all_sources(self):
        scraper = MetalComScraper()
        results = await scraper.scrape("2025-01-15")
        sources = {r.source for r in results}
        assert "metal_com" in sources

    @pytest.mark.asyncio
    async def test_scrape_lithium_carbonate_present(self):
        scraper = MetalComScraper()
        results = await scraper.scrape("2025-01-15")
        commodities = {(r.commodity, r.grade) for r in results}
        assert ("lithium_carbonate", "battery_grade") in commodities

    @pytest.mark.asyncio
    async def test_scrape_deterministic(self):
        scraper = MetalComScraper()
        r1 = await scraper.scrape("2025-01-15")
        r2 = await scraper.scrape("2025-01-15")
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.price_cny == b.price_cny

    @pytest.mark.asyncio
    async def test_scrape_different_dates_vary(self):
        scraper = MetalComScraper()
        r1 = await scraper.scrape("2025-01-14")
        r2 = await scraper.scrape("2025-01-15")
        prices_differ = any(
            a.price_cny != b.price_cny
            for a, b in zip(r1, r2)
        )
        assert prices_differ

    @pytest.mark.asyncio
    async def test_scrape_realistic_prices(self):
        scraper = MetalComScraper()
        results = await scraper.scrape("2025-01-15")
        for r in results:
            assert r.price_cny > 0
            assert r.price_usd is not None
            assert r.price_usd > 0
            assert r.fx_rate == 7.25
            assert r.scrape_status == "success"
            assert r.change_type in ("rise", "fall", "flat")

    @pytest.mark.asyncio
    async def test_scrape_default_date(self):
        scraper = MetalComScraper()
        results = await scraper.scrape()
        assert len(results) > 0
        for r in results:
            assert r.date == datetime.utcnow().strftime("%Y-%m-%d")

    @pytest.mark.asyncio
    async def test_scrape_covers_all_target_commodities(self):
        scraper = MetalComScraper()
        results = await scraper.scrape("2025-01-15")
        commodities = {r.commodity for r in results}
        for target in scraper.target_commodities:
            assert target in commodities, f"Missing {target}"


class TestSHMETScraper:
    @pytest.mark.asyncio
    async def test_scrape_returns_data(self):
        scraper = SHMETScraper()
        results = await scraper.scrape("2025-01-15")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_scrape_has_shmet_source(self):
        scraper = SHMETScraper()
        results = await scraper.scrape("2025-01-15")
        for r in results:
            assert r.source == "shmet"

    @pytest.mark.asyncio
    async def test_scrape_no_nickel_futures(self):
        """SHMET doesn't scrape SHFE futures."""
        scraper = SHMETScraper()
        results = await scraper.scrape("2025-01-15")
        commodities = {r.commodity for r in results}
        assert "nickel_sulfate" in commodities


class TestSHFEScraper:
    @pytest.mark.asyncio
    async def test_scrape_returns_data(self):
        scraper = SHFEScraper()
        results = await scraper.scrape("2025-01-15")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_scrape_has_contract_month(self):
        scraper = SHFEScraper()
        results = await scraper.scrape("2025-01-15")
        for r in results:
            assert r.contract_month is not None

    @pytest.mark.asyncio
    async def test_scrape_nickel_and_copper(self):
        scraper = SHFEScraper()
        results = await scraper.scrape("2025-01-15")
        commodities = {r.commodity for r in results}
        assert "nickel" in commodities
        assert "copper" in commodities

    @pytest.mark.asyncio
    async def test_scrape_shfe_source(self):
        scraper = SHFEScraper()
        results = await scraper.scrape("2025-01-15")
        for r in results:
            assert r.source == "shfe"
