import asyncio
from decimal import Decimal
import pytest
from httpx import ASGITransport, AsyncClient

from engine.exchange_rate_service import (
    SUPPORTED_CURRENCIES,
    CachedRates,
    ExchangeRateService,
    exchange_rate_service,
)
from main import app


@pytest.mark.asyncio
async def test_supported_currencies_structure():
    """Validates that all required ISO 4217 currencies are configured with symbols and precision."""
    required = ["INR", "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "SGD", "AED"]
    supported = await exchange_rate_service.get_supported_currencies()
    for code in required:
        assert code in supported
        assert "symbol" in supported[code]
        assert "name" in supported[code]
        assert "minor_units" in supported[code]
    assert supported["JPY"]["minor_units"] == 0
    assert supported["INR"]["symbol"] == "₹"
    assert supported["USD"]["symbol"] == "$"


@pytest.mark.asyncio
async def test_decimal_safe_conversion():
    """Validates decimal-safe conversion arithmetic and JPY rounding."""
    service = ExchangeRateService()
    rates_mock = {
        "INR": 1.0,
        "USD": 0.012,
        "EUR": 0.011,
        "JPY": 1.75,
    }

    # 1. Identity conversion
    res_identity = service.convert(1000.0, "INR", "INR", rates_mock)
    assert res_identity["is_identity"] is True
    assert res_identity["converted_amount"] == 1000.0

    # 2. INR -> USD conversion (10,000 * 0.012 = 120.00)
    res_usd = service.convert(10000.0, "INR", "USD", rates_mock)
    assert res_usd["is_identity"] is False
    assert res_usd["converted_amount"] == 120.0
    assert res_usd["target_currency"] == "USD"

    # 3. INR -> JPY conversion (10,000 * 1.75 = 17,500 with 0 minor units)
    res_jpy = service.convert(10000.0, "INR", "JPY", rates_mock)
    assert res_jpy["converted_amount"] == 17500.0
    assert isinstance(res_jpy["converted_amount"], float)


@pytest.mark.asyncio
async def test_unsupported_currency_conversion():
    """Asserts that unsupported currencies raise a clean ValueError."""
    service = ExchangeRateService()
    with pytest.raises(ValueError, match="Unsupported"):
        service.convert(100.0, "XYZ", "USD", {"XYZ": 1.0, "USD": 1.0})


@pytest.mark.asyncio
async def test_caching_and_staleness_fallback(monkeypatch):
    """Asserts that cached rates are reused within TTL and serve stale on failure."""
    service = ExchangeRateService()
    service._cache.clear()

    # Seed mock cache
    cached = CachedRates(
        base="USD",
        rates={"USD": 1.0, "INR": 83.5, "EUR": 0.92},
        provider="TestProvider",
        last_updated="2026-10-14 00:00:00 UTC",
    )
    service._cache["USD"] = cached

    # Within TTL, returns non-stale cached rates
    rates = await service.get_rates("USD", force_refresh=False)
    assert rates["is_stale"] is False
    assert rates["rates"]["INR"] == 83.5

    # Simulate network failure on force_refresh
    async def mock_fetch_fail(base):
        raise RuntimeError("Network Timeout Mock")

    monkeypatch.setattr(service, "_fetch_rates_with_fallback", mock_fetch_fail)

    # Force refresh triggers failure, but falls back gracefully to stale cache
    fallback = await service.get_rates("USD", force_refresh=True)
    assert fallback["is_stale"] is True
    assert "Network Timeout Mock" in fallback["stale_warning"]
    assert fallback["rates"]["INR"] == 83.5


@pytest.mark.asyncio
async def test_api_rates_endpoints():
    """Tests the /api/rates HTTP API endpoints using ASGITransport."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GET /api/rates/supported
        resp_sup = await client.get("/api/rates/supported")
        assert resp_sup.status_code == 200
        data_sup = resp_sup.json()
        assert "INR" in data_sup["currencies"]
        assert "AED" in data_sup["currencies"]

        # 2. GET /api/rates/latest?base=INR
        resp_latest = await client.get("/api/rates/latest?base=INR")
        assert resp_latest.status_code in (200, 503)
        if resp_latest.status_code == 200:
            data = resp_latest.json()
            assert data["base"] == "INR"
            assert "USD" in data["rates"]
            assert data["rates"]["INR"] == 1.0
            assert "provider" in data
            assert "last_updated" in data

        # 3. POST /api/rates/convert
        resp_conv = await client.post(
            "/api/rates/convert",
            json={"amount": 1000.0, "from_currency": "INR", "to_currency": "INR"},
        )
        assert resp_conv.status_code == 200
        data_conv = resp_conv.json()
        assert data_conv["converted_amount"] == 1000.0
        assert data_conv["is_identity"] is True
