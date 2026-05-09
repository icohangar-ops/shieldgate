"""AI-powered market analyst using Qwen LLM.

Generates market commentary in the style of Mysteel Battery Materials Weekly.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from ..models import PriceObservation, PriceSummary, AIAnalysis
from ..database import get_price_history, compute_price_summary
from .qwen_client import QwenClient


SYSTEM_PROMPT = """你是一位资深的电池材料市场分析师，精通锂、镍、钴、锰等新能源金属的价格分析。
你的分析风格参考 Mysteel（我的钢铁网）电池材料周报，特点如下：
1. 简洁专业，数据驱动
2. 中英双语分析
3. 重点关注供需平衡、政策变化、宏观因素
4. 提供1-2周短期展望
5. 标注关键风险因素

请严格按照以下JSON格式输出分析结果（不要加markdown代码块标记）：
{
  "market_commentary": "中英双语的市場评论...",
  "outlook": "短期展望（1-2周）",
  "key_drivers": ["驱动因素1", "驱动因素2", "驱动因素3"],
  "risk_factors": ["风险因素1", "风险因素2"],
  "recommendation": "Buy/Hold/Sell",
  "confidence": 0.75
}"""


class AIAnalyst:
    """Generate market analysis using Qwen AI."""

    def __init__(self, client: Optional[QwenClient] = None):
        self.client = client or QwenClient()

    async def generate_analysis(
        self, commodity: str, date: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> AIAnalysis:
        """Generate AI analysis for a specific commodity.

        Args:
            commodity: Commodity identifier (e.g. 'lithium_carbonate').
            date: Analysis date (defaults to today).
            db_path: Optional database path.

        Returns:
            AIAnalysis object with Qwen-generated insights.
        """
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        # Gather price data
        summary = compute_price_summary(commodity, db_path=db_path)
        history = get_price_history(commodity, days=30, db_path=db_path)

        if not history:
            return AIAnalysis(
                commodity=commodity,
                date=date,
                market_commentary=f"No price data available for {commodity}.",
                outlook="Insufficient data",
                recommendation="N/A",
                confidence=0.0,
            )

        # Build prompt with price data
        prompt = self._build_prompt(commodity, summary, history, date)

        try:
            raw_response = await self.client.analyze(prompt, SYSTEM_PROMPT)
            return self._parse_response(commodity, date, raw_response)
        except Exception as e:
            return AIAnalysis(
                commodity=commodity,
                date=date,
                market_commentary=f"AI analysis failed: {str(e)}",
                outlook="Analysis unavailable",
                recommendation="N/A",
                confidence=0.0,
            )

    def _build_prompt(
        self,
        commodity: str,
        summary: Optional[PriceSummary],
        history: List[PriceObservation],
        date: str,
    ) -> str:
        """Build analysis prompt from price data."""
        pretty_name = commodity.replace("_", " ").title()

        lines = [
            f"请分析 {pretty_name}（{commodity}）在 {date} 的市场情况。",
            "",
            "## 最新价格数据",
        ]

        if summary:
            lines.append(
                f"- 最新价格: {summary.latest_price_cny:,.0f} CNY/t"
                f" (≈{summary.latest_price_usd:,.0f} USD/t)"
                if summary.latest_price_usd
                else f"- 最新价格: {summary.latest_price_cny:,.0f} CNY/t"
            )
            lines.append(f"- 周环比: {summary.wow_change_pct:+.2f}%")
            lines.append(f"- 月环比: {summary.mom_change_pct:+.2f}%")
            lines.append(f"- 趋势: {summary.trend}")
            lines.append(f"- 数据来源: {summary.source}")

        if len(history) >= 7:
            recent = history[-7:]
            prices = [f"{h.price_cny:,.0f}" for h in recent]
            dates = [h.date for h in recent]
            lines.append("")
            lines.append("## 近7天价格走势")
            for d, p in zip(dates, prices):
                lines.append(f"- {d}: {p} CNY/t")

            # Calculate volatility
            price_vals = [h.price_cny for h in recent]
            avg = sum(price_vals) / len(price_vals)
            variance = sum((p - avg) ** 2 for p in price_vals) / len(price_vals)
            std_dev = variance ** 0.5
            lines.append(f"- 7日标准差: {std_dev:,.0f} CNY/t")
            lines.append(f"- 7日均值: {avg:,.0f} CNY/t")

        lines.append("")
        lines.append("请基于以上数据提供市场分析，输出严格JSON格式。")

        return "\n".join(lines)

    def _parse_response(
        self, commodity: str, date: str, raw: str
    ) -> AIAnalysis:
        """Parse Qwen's JSON response into an AIAnalysis."""
        # Strip potential markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines if they are code fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: use raw text as commentary
            return AIAnalysis(
                commodity=commodity,
                date=date,
                market_commentary=raw,
                outlook="Unable to parse structured response",
                recommendation="N/A",
                confidence=0.3,
            )

        return AIAnalysis(
            commodity=commodity,
            date=date,
            market_commentary=data.get("market_commentary", ""),
            outlook=data.get("outlook", ""),
            key_drivers=data.get("key_drivers", []),
            risk_factors=data.get("risk_factors", []),
            recommendation=data.get("recommendation", "N/A"),
            confidence=float(data.get("confidence", 0.5)),
        )
