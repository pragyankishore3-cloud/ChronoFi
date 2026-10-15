"""FastAPI Router for Live and Cached Foreign Exchange Rates."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from engine.exchange_rate_service import (
    SUPPORTED_CURRENCIES,
    exchange_rate_service,
)

router = APIRouter()


class ConvertRequest(BaseModel):
    amount: float = Field(..., ge=0.0, description="Amount to convert")
    from_currency: str = Field(..., description="Source currency ISO code (e.g. INR)")
    to_currency: str = Field(..., description="Target currency ISO code (e.g. USD)")


class ConvertResponse(BaseModel):
    original_amount: float
    original_currency: str
    converted_amount: float
    target_currency: str
    exchange_rate: float
    is_identity: bool
    provider: str
    last_updated: str
    is_stale: bool


@router.get("/supported")
async def get_supported_currencies() -> Dict[str, Any]:
    """Returns the list of supported currencies with display symbols and decimal precision."""
    return {"currencies": SUPPORTED_CURRENCIES}


@router.get("/latest")
async def get_latest_rates(
    base: str = Query("INR", description="Base currency code"),
    force_refresh: bool = Query(False, description="Bypass cache and fetch fresh rates"),
) -> Dict[str, Any]:
    """Retrieves live or cached exchange rates for the specified base currency."""
    try:
        return await exchange_rate_service.get_rates(base_currency=base, force_refresh=force_refresh)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.post("/convert", response_model=ConvertResponse)
async def convert_currency(payload: ConvertRequest) -> ConvertResponse:
    """Performs decimal-safe conversion between supported currencies using current reference rates."""
    from_curr = payload.from_currency.upper().strip()
    to_curr = payload.to_currency.upper().strip()

    if from_curr not in SUPPORTED_CURRENCIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported from_currency: {from_curr}",
        )
    if to_curr not in SUPPORTED_CURRENCIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported to_currency: {to_curr}",
        )

    try:
        rates_data = await exchange_rate_service.get_rates(base_currency=from_curr)
        res = exchange_rate_service.convert(
            amount=payload.amount,
            from_currency=from_curr,
            to_currency=to_curr,
            rates_dict=rates_data["rates"],
        )
        return ConvertResponse(
            original_amount=res["original_amount"],
            original_currency=res["original_currency"],
            converted_amount=res["converted_amount"],
            target_currency=res["target_currency"],
            exchange_rate=res["exchange_rate"],
            is_identity=res["is_identity"],
            provider=rates_data["provider"],
            last_updated=rates_data["last_updated"],
            is_stale=rates_data.get("is_stale", False),
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
