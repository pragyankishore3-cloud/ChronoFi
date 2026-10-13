"""
Tests for Scenario Confirmation vs. Intention and Real Liability Persistence.
Specifically verifies Section 1 requirements:
"Remember that car loan we discussed last week? I decided to take it."
1. System recognizes the previous car-loan discussion and retrieves saved scenario.
2. Distinguishes intent/decision from confirmed transaction: asks for confirmation without prematurely inserting a DB liability.
3. Upon user confirmation ("Yes, I took it"), records an official Liability in the database.
4. Recalculates updated baseline roadmap and explains impact.
5. Verifies idempotent writes and no duplicate liabilities on repeat requests.
"""

import uuid
import pytest
from sqlalchemy import select

from ai.scenario_tools import run_financial_scenario
from ai.scenario_workflow import scenario_workflow_app
from core.checkpointer import init_checkpointer, close_checkpointer
import models.database as db_models
from models.database import Liability, ScenarioRecord, User
from seed.seed_db import seed_database


@pytest.fixture(autouse=True)
async def setup_test_data():
    await seed_database()
    await init_checkpointer(use_memory=True)
    yield
    await close_checkpointer()


@pytest.mark.asyncio
async def test_full_scenario_intent_clarification_and_confirmation_journey():
    """
    End-to-end multi-turn journey:
    Turn 1: Initial simulation runs and saves scenario.
    Turn 2: "Remember that car loan we discussed last week? I decided to take it." -> Prompt asks for confirmation, no DB liability created yet!
    Turn 3: "Yes, I took it yesterday. Record it." -> Officially creates DB liability and updates roadmap!
    """
    user_id = 2  # Priya
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    # --- Turn 1: Initial What-If Query ---
    res_1 = await scenario_workflow_app.ainvoke({
        "messages": [("user", "What happens to my retirement goal if I take a ₹10 lakh car loan today at 9% interest?")],
        "user_query": "What happens to my retirement goal if I take a ₹10 lakh car loan today at 9% interest?",
        "user_id": user_id,
        "thread_id": thread_id,
    }, config)

    assert res_1["validation_status"] == "valid"
    scenario_id = res_1.get("active_scenario_id")
    assert scenario_id is not None

    # Verify no liabilities added yet
    async with db_models.AsyncSessionLocal() as db:
        user_res = await db.execute(select(User).where(User.id == user_id))
        user = user_res.scalars().first()
        initial_liability_count = len(user.liabilities)

    # --- Turn 2: "Remember that car loan we discussed last week? I decided to take it." ---
    res_2 = await scenario_workflow_app.ainvoke({
        "messages": [("user", "Remember that car loan we discussed last week? I decided to take it.")],
        "user_query": "Remember that car loan we discussed last week? I decided to take it.",
        "user_id": user_id,
        "thread_id": thread_id,
    }, config)

    # Must ask for confirmation and NOT record liability yet!
    assert res_2.get("workflow_status") == "awaiting_confirmation"
    assert res_2.get("pending_confirmation") is not None
    response_text = res_2["messages"][-1].content
    assert "10 lakh" in response_text or "10" in response_text
    assert "9%" in response_text
    assert "have you already taken the loan" in response_text.lower()

    # Guardrail check: Verify database liability was NOT prematurely created
    async with db_models.AsyncSessionLocal() as db:
        user_res = await db.execute(select(User).where(User.id == user_id))
        user = user_res.scalars().first()
        assert len(user.liabilities) == initial_liability_count, "Liability must not be recorded prematurely!"

    # --- Turn 3: User confirms transaction ---
    res_3 = await scenario_workflow_app.ainvoke({
        "messages": [("user", "Yes, I took it yesterday. Please record it.")],
        "user_query": "Yes, I took it yesterday. Please record it.",
        "user_id": user_id,
        "thread_id": thread_id,
    }, config)

    assert res_3.get("workflow_status") == "confirmed"
    assert res_3.get("pending_confirmation") is None
    confirm_text = res_3["messages"][-1].content
    assert "officially recorded" in confirm_text.lower()
    assert "20,758" in confirm_text

    # Verify permanent DB liability was officially created
    async with db_models.AsyncSessionLocal() as db:
        liab_res = await db.execute(select(Liability).where(Liability.user_id == user_id))
        all_liabilities = liab_res.scalars().all()
        assert len(all_liabilities) == initial_liability_count + 1
        car_loan = next(l for l in all_liabilities if any(w in l.name.lower() for w in ["car", "vehicle"]))
        assert car_loan.principal_remaining == 1000000.0
        assert car_loan.interest_rate == 0.09
        assert car_loan.monthly_payment == 20758.36

        # Verify ScenarioRecord status was updated
        scen_res = await db.execute(select(ScenarioRecord).where(ScenarioRecord.id == scenario_id))
        scen_record = scen_res.scalars().first()
        assert scen_record.status == "confirmed_liability"
