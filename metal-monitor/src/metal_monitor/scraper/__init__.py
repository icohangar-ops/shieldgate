"""Scraper package — metal price data scrapers."""

from .base import BaseScraper
from .metal_com import MetalComScraper
from .shmet import SHMETScraper
from .shfe import SHFEScraper

__all__ = ["BaseScraper", "MetalComScraper", "SHMETScraper", "SHFEScraper"]
