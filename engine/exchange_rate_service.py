"""ChronoFi Dedicated Exchange Rate Service

Provides reliable, cached, multi-provider foreign exchange rates
with decimal-safe conversions, currency validation, request deduplication,
and graceful offline/stale fallbacks.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Supported Currencies (ISO 4217)
SUPPORTED_CURRENCIES: Dict[str, Dict[str, Any]] = {
    "INR": {"name": "Indian Rupee", "symbol": "₹", "minor_units": 2},
    "USD": {"name": "US Dollar", "symbol": "$", "minor_units": 2},
    "EUR": {"name": "Euro", "symbol": "€", "minor_units": 2},
    "GBP": {"name": "British Pound", "symbol": "£", "minor_units": 2},
    "JPY": {"name": "Japanese Yen", "symbol": "¥", "minor_units": 0},
    "CAD": {"name": "Canadian Dollar", "symbol": "CA$", "minor_units": 2},
    "AUD": {"name": "Australian Dollar", "symbol": "AU$", "minor_units": 2},
    "SGD": {"name": "Singapore Dollar", "symbol": "SG$", "minor_units": 2},
    "AED": {"name": "UAE Dirham", "symbol": "AED", "minor_units": 2},
}

CACHE_TTL_SECONDS = int(os.getenv("EXCHANGE_RATE_CACHE_TTL", "10800"))  # 3 hours default
MAX_STALENESS_SECONDS = 86400 * 7  # 7 days max offline grace period


class CachedRates:
    def __init__(
        self,
        base: str,
        rates: Dict[str, float],
        provider: str,
        last_updated: str,
        next_update: Optional[str] = None,
    ):
        self.base = base
        self.rates = rates
        self.provider = provider
        self.last_updated = last_updated
        self.next_update = next_update
        self.cached_at = time.time()

    def is_expired(self, ttl: int = CACHE_TTL_SECONDS) -> bool:
        return (time.time() - self.cached_at) > ttl

    def is_within_max_staleness(self) -> bool:
        return (time.time() - self.cached_at) <= MAX_STALENESS_SECONDS


class ExchangeRateService:
    """Singleton service for live and cached exchange rates with fallback providers."""

    _instance: Optional["ExchangeRateService"] = None

    def __new__(cls) -> "ExchangeRateService":
        if cls._instance is None:
            cls._instance = super(ExchangeRateService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self) -> None:
        self._cache: Dict[str, CachedRates] = {}
        self._lock = asyncio.Lock()
        self._in_flight: Dict[str, asyncio.Future] = {}

    async def get_supported_currencies(self) -> Dict[str, Dict[str, Any]]:
        return SUPPORTED_CURRENCIES

    async def get_rates(self, base_currency: str = "INR", force_refresh: bool = False) -> Dict[str, Any]:
        base = base_currency.upper().strip()
        if base not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported base currency: {base}. Supported: {list(SUPPORTED_CURRENCIES.keys())}")

        cached = self._cache.get(base)
        if cached and not cached.is_expired() and not force_refresh:
            return self._format_response(cached, is_stale=False)

        # Request deduplication: coalesce parallel requests for the same base
        async with self._lock:
            cached = self._cache.get(base)
            if cached and not cached.is_expired() and not force_refresh:
                return self._format_response(cached, is_stale=False)

            try:
                rates_data = await self._fetch_rates_with_fallback(base)
                new_cache = CachedRates(
                    base=base,
                    rates=rates_data["rates"],
                    provider=rates_data["provider"],
                    last_updated=rates_data["last_updated"],
                    next_update=rates_data.get("next_update"),
                )
                self._cache[base] = new_cache
                return self._format_response(new_cache, is_stale=False)
            except Exception as exc:
                logger.warning(f"Exchange rate fetch failed for {base}: {exc}")
                if cached and cached.is_within_max_staleness():
                    # Graceful fallback: return cached rate with stale flag and clear disclosure
                    return self._format_response(
                        cached,
                        is_stale=True,
                        stale_warning=f"Upstream provider unreachable ({exc}). Serving cached rates from {cached.last_updated}.",
                    )
                raise RuntimeError(
                    f"Exchange rate service unavailable for {base} and no valid cached rates exist: {exc}"
                )

    async def _fetch_rates_with_fallback(self, base: str) -> Dict[str, Any]:
        """Attempts primary provider (ExchangeRate-API open access), then Frankfurter."""
        async with httpx.AsyncClient(timeout=6.0) as client:
            # 1. Primary: Open ExchangeRate API
            try:
                resp = await client.get(f"https://open.er-api.com/v6/latest/{base}")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("result") == "success" and "rates" in data:
                        filtered_rates = {
                            curr: float(data["rates"][curr])
                            for curr in SUPPORTED_CURRENCIES
                            if curr in data["rates"]
                        }
                        # Ensure base is 1.0
                        filtered_rates[base] = 1.0
                        return {
                            "rates": filtered_rates,
                            "provider": "ExchangeRate-API (Daily Reference)",
                            "last_updated": data.get("time_last_update_utc", datetime.now(timezone.utc).isoformat()),
                            "next_update": data.get("time_next_update_utc"),
                        }
            except Exception as primary_err:
                logger.info(f"Primary exchange rate provider failed for {base}: {primary_err}")

            # 2. Secondary Fallback: Frankfurter (ECB reference rates)
            try:
                resp = await client.get(f"https://api.frankfurter.dev/v1/latest?base={base}")
                if resp.status_code == 200:
                    data = resp.json()
                    if "rates" in data:
                        rates_dict = {
                            curr: float(data["rates"][curr])
                            for curr in SUPPORTED_CURRENCIES
                            if curr in data["rates"]
                        }
                        rates_dict[base] = 1.0
                        # If AED is missing in Frankfurter, check if we have USD rate and peg AED at 3.6725
                        if "AED" not in rates_dict and "USD" in rates_dict:
                            rates_dict["AED"] = round(rates_dict["USD"] * 3.6725, 6)

                        return {
                            "rates": rates_dict,
                            "provider": "Frankfurter / ECB (Daily Reference)",
                            "last_updated": f"{data.get('date', 'today')} 16:00 CET",
                            "next_update": "Next ECB Working Day",
                        }
            except Exception as secondary_err:
                logger.info(f"Secondary exchange rate provider failed for {base}: {secondary_err}")

        raise RuntimeError("All configured upstream exchange rate providers failed to respond.")

    def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        rates_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """Performs a decimal-safe conversion using the supplied rates table."""
        from_curr = from_currency.upper().strip()
        to_curr = to_currency.upper().strip()

        if from_curr not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported source currency: {from_curr}")
        if to_curr not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported target currency: {to_curr}")

        if from_curr == to_curr:
            minor_units = SUPPORTED_CURRENCIES[to_curr]["minor_units"]
            dec_amount = Decimal(str(amount)).quantize(
                Decimal("1" if minor_units == 0 else f"0.{'0' * minor_units}"),
                rounding=ROUND_HALF_UP,
            )
            return {
                "original_amount": amount,
                "original_currency": from_curr,
                "converted_amount": float(dec_amount),
                "target_currency": to_curr,
                "exchange_rate": 1.0,
                "is_identity": True,
            }

        rate_from = rates_dict.get(from_curr)
        rate_to = rates_dict.get(to_curr)

        if not rate_from or not rate_to or rate_from <= 0:
            raise ValueError(f"Missing exchange rate data to convert between {from_curr} and {to_curr}")

        # cross_rate = rate_to / rate_from
        dec_from_rate = Decimal(str(rate_from))
        dec_to_rate = Decimal(str(rate_to))
        cross_rate = dec_to_rate / dec_from_rate

        dec_amount = Decimal(str(amount))
        converted = dec_amount * cross_rate

        minor_units = SUPPORTED_CURRENCIES[to_curr]["minor_units"]
        quantizer = Decimal("1" if minor_units == 0 else f"0.{'0' * minor_units}")
        final_amount = converted.quantize(quantizer, rounding=ROUND_HALF_UP)

        return {
            "original_amount": amount,
            "original_currency": from_curr,
            "converted_amount": float(final_amount),
            "target_currency": to_curr,
            "exchange_rate": float(cross_rate.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)),
            "is_identity": False,
        }

    def _format_response(
        self,
        cached: CachedRates,
        is_stale: bool = False,
        stale_warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "base": cached.base,
            "rates": cached.rates,
            "provider": cached.provider,
            "last_updated": cached.last_updated,
            "next_update": cached.next_update,
            "is_stale": is_stale,
            "stale_warning": stale_warning,
            "freshness": "Daily Reference Rate",
            "cached_at": datetime.fromtimestamp(cached.cached_at, tz=timezone.utc).isoformat(),
            "supported_currencies": SUPPORTED_CURRENCIES,
        }


exchange_rate_service = ExchangeRateService()
