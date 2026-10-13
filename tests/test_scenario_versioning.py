"""
Tests for Financial Scenario Versioning and Historical Integrity.
Verifies Section 9 requirements:
1. Scenario modifications create new version (version 1 -> version 2)
2. Preserves predecessor scenario intact (historical results never overwritten)
3. Establishes parent_scenario_id linkage
4. Retrieval allows comparing 'what did we calculate last week' vs 'what does it do today'
"""

import uuid
import pytest
from sqlalchemy import select

from ai.scenario_tools import run_financial_scenario, retrieve_previous_scenarios
from core.checkpointer import init_checkpointer, close_checkpointer
import models.database as db_models
from models.database import ScenarioRecord
from seed.seed_db import seed_database


@pytest.fixture(autouse=True)
async def setup_test_data():
    await seed_database()
    await init_checkpointer(use_memory=True)
    yield
    await close_checkpointer()


@pytest.mark.asyncio
async def test_scenario_versioning_and_parent_linkage():
    """Verifies that updating a scenario creates version 2 with parent linkage without overwriting version 1."""
    user_id = 2  # Priya

    # Version 1: 5-year loan at 9%
    res_v1 = await run_financial_scenario.ainvoke({
        "scenario_type": "vehicle_loan",
        "loan_principal": 1000000.0,
        "annual_interest_rate": 0.09,
        "loan_tenure_months": 60,
        "user_id": user_id,
    })
    v1_id = res_v1["scenario_id"]
    assert res_v1["version"] == 1
    assert res_v1.get("parent_scenario_id") is None

    # Version 2: Modified tenure to 84 months (7 years) linked to Version 1
    res_v2 = await run_financial_scenario.ainvoke({
        "scenario_type": "vehicle_loan",
        "loan_principal": 1000000.0,
        "annual_interest_rate": 0.09,
        "loan_tenure_months": 84,
        "parent_scenario_id": v1_id,
        "user_id": user_id,
    })
    v2_id = res_v2["scenario_id"]
    assert res_v2["version"] == 2
    assert res_v2["parent_scenario_id"] == v1_id
    assert v2_id != v1_id

    # Verify both records exist distinctly in the database
    async with db_models.AsyncSessionLocal() as db:
        rec_v1 = (await db.execute(select(ScenarioRecord).where(ScenarioRecord.id == v1_id))).scalars().first()
        rec_v2 = (await db.execute(select(ScenarioRecord).where(ScenarioRecord.id == v2_id))).scalars().first()

        assert rec_v1 is not None
        assert rec_v2 is not None
        assert rec_v1.version == 1
        assert rec_v2.version == 2
        assert rec_v1.parameters["loan_tenure_months"] == 60
        assert rec_v2.parameters["loan_tenure_months"] == 84
        assert rec_v2.parent_scenario_id == v1_id

    # Verify memory retrieval returns both versions in audit trail
    history = await retrieve_previous_scenarios.ainvoke({
        "scenario_type": "vehicle_loan",
        "user_id": user_id,
        "limit": 10,
    })
    assert history["count"] >= 2
    ids = [s["scenario_id"] for s in history["scenarios"]]
    assert v1_id in ids
    assert v2_id in ids
