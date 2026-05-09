"""Tests for AI client and analyst."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from src.metal_monitor.ai.qwen_client import QwenClient
from src.metal_monitor.ai.analyst import AIAnalyst, SYSTEM_PROMPT


class TestQwenClient:
    def test_init_defaults(self):
        client = QwenClient()
        assert client.model == "qwen3.6-flash"
        assert "dashscope.aliyuncs.com" in client.base_url
        assert client.api_key.startswith("sk-")

    def test_init_custom(self):
        client = QwenClient(
            api_key="custom-key",
            base_url="https://custom.example.com/v1",
            model="qwen-max",
        )
        assert client.model == "qwen-max"
        assert client.api_key == "custom-key"
        assert client.base_url == "https://custom.example.com/v1"

    @pytest.mark.asyncio
    async def test_analyze_success(self):
        client = QwenClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Lithium prices are stable."}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.metal_monitor.ai.qwen_client.httpx.AsyncClient", return_value=mock_client):
            result = await client.analyze("Analyze lithium prices")
        assert result == "Lithium prices are stable."

    @pytest.mark.asyncio
    async def test_analyze_with_system_prompt(self):
        client = QwenClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("src.metal_monitor.ai.qwen_client.httpx.AsyncClient", return_value=mock_client):
            result = await client.analyze("test", system="Be concise")

        call_args = mock_client.post.call_args
        messages = call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = QwenClient()
        with patch.object(client, "analyze", new_callable=AsyncMock, return_value="OK"):
            result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        client = QwenClient()
        with patch.object(client, "analyze", new_callable=AsyncMock, side_effect=Exception("fail")):
            result = await client.health_check()
        assert result is False


class TestAIAnalyst:
    def _mock_client(self, return_text: str):
        mock = AsyncMock()
        mock.analyze = AsyncMock(return_value=return_text)
        return mock

    @pytest.mark.asyncio
    async def test_generate_analysis_no_data(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        from src.metal_monitor.database import init_db
        init_db(db_path)

        analyst = AIAnalyst()
        result = await analyst.generate_analysis("lithium_carbonate", db_path=db_path)
        assert "No price data" in result.market_commentary
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_generate_analysis_success(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        from src.metal_monitor.database import init_db, upsert_observations
        init_db(db_path)

        # Use today's date so get_price_history can find the data
        today = datetime.utcnow().strftime("%Y-%m-%d")
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        two_days_ago = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")

        from src.metal_monitor.models import PriceObservation
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

        response_json = json.dumps({
            "market_commentary": "碳酸锂价格本周小幅上涨，电池级碳酸锂均价157,000元/吨。下游正极材料厂采购需求有所恢复。",
            "outlook": "短期内预计价格将维持稳定，关注下游排产情况。",
            "key_drivers": ["下游需求回暖", "供应端减产", "库存低位"],
            "risk_factors": ["政策变化", "新增产能投放"],
            "recommendation": "Hold",
            "confidence": 0.75,
        })

        mock_client = self._mock_client(response_json)
        analyst = AIAnalyst(client=mock_client)
        result = await analyst.generate_analysis("lithium_carbonate", today, db_path)

        assert result.commodity == "lithium_carbonate"
        assert "碳酸锂" in result.market_commentary
        assert len(result.key_drivers) == 3
        assert result.recommendation == "Hold"
        assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_generate_analysis_api_failure(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        from src.metal_monitor.database import init_db, upsert_observations
        init_db(db_path)

        today = datetime.utcnow().strftime("%Y-%m-%d")
        from src.metal_monitor.models import PriceObservation
        obs = PriceObservation(
            date=today, source="metal_com",
            commodity="lithium_carbonate", grade="battery_grade",
            price_cny=157000.0,
        )
        upsert_observations([obs], db_path)

        mock_client = self._mock_client("bad response")
        mock_client.analyze.side_effect = Exception("API timeout")
        analyst = AIAnalyst(client=mock_client)

        result = await analyst.generate_analysis("lithium_carbonate", today, db_path)
        assert "failed" in result.market_commentary.lower()
        assert result.confidence == 0.0

    def test_parse_json_with_code_fences(self):
        response = '```json\n{"market_commentary": "test", "outlook": "bullish", "key_drivers": ["a"], "risk_factors": ["b"], "recommendation": "Buy", "confidence": 0.8}\n```'
        mock_client = self._mock_client(response)
        analyst = AIAnalyst(client=mock_client)

        result = analyst._parse_response("nickel", "2025-01-15", response)
        assert result.market_commentary == "test"
        assert result.recommendation == "Buy"
        assert result.confidence == 0.8

    def test_parse_invalid_json_fallback(self):
        response = "This is not JSON at all"
        mock_client = self._mock_client(response)
        analyst = AIAnalyst(client=mock_client)

        result = analyst._parse_response("cobalt", "2025-01-15", response)
        assert result.market_commentary == response
        assert result.confidence == 0.3

    def test_system_prompt_exists(self):
        assert SYSTEM_PROMPT is not None
        assert "Mysteel" in SYSTEM_PROMPT
        assert "电池材料" in SYSTEM_PROMPT
