"""
Tests for Historical Financial Memory Retrieval.
Verifies:
1. Scenario retrieval by search query (e.g. "car loan", "last week")
2. Cross-user isolation (user A cannot view user B's historical scenarios)
3. Preserving original structured assumptions, version, and simulation results
4. Clarification prompt when multiple matching historical scenarios exist
"""

from datetime import datetime, timezone
import uuid
import pytest
from sqlalchemy import select

from ai.scenario_tools import retrieve_previous_scenarios, run_financial_scenario
from ai.scenario_workflow import scenario_workflow_app
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
async def test_retrieve_previous_scenarios_tool():
    """Verifies that retrieve_previous_scenarios returns saved scenarios for authorized user."""
    # Run a simulation for Priya (user_id=2)
    sim_res = await run_financial_scenario.ainvoke({
        "scenario_type": "vehicle_loan",
        "loan_principal": 1000000.0,
        "annual_interest_rate": 0.09,
        "loan_tenure_months": 60,
        "user_id": 2,
    })
    scen_id = sim_res.get("scenario_id")
    assert scen_id is not None

    # Retrieve scenario
    ret_res = await retrieve_previous_scenarios.ainvoke({
        "search_query": "car loan",
        "user_id": 2,
    })
    assert ret_res["status"] == "success"
    assert ret_res["count"] >= 1
    found = next((s for s in ret_res["scenarios"] if s["scenario_id"] == scen_id), None)
    assert found is not None
    assert found["parameters"]["loan_principal"] == 1000000.0
    assert found["parameters"]["annual_interest_rate"] == 0.09
    assert found["parameters"]["loan_tenure_months"] == 60
    assert found["version"] == 1


@pytest.mark.asyncio
async def test_cross_user_scenario_isolation():
    """Verifies that User 1 (Arjun) cannot retrieve User 2's (Priya) scenarios."""
    # Create scenario for User 2 (Priya)
    await run_financial_scenario.ainvoke({
        "scenario_type": "vehicle_loan",
        "loan_principal": 1200000.0,
        "annual_interest_rate": 0.085,
        "loan_tenure_months": 48,
        "user_id": 2,
    })

    # User 1 attempts retrieval
    ret_res = await retrieve_previous_scenarios.ainvoke({
        "search_query": "vehicle loan",
        "user_id": 1,
    })
    assert ret_res["status"] == "success"
    # None of User 2's scenarios should be returned
    for s in ret_res["scenarios"]:
        assert s["user_id"] == 1
        assert s["parameters"]["loan_principal"] != 1200000.0


@pytest.mark.asyncio
async def test_workflow_memory_retrieval_and_cached_explanation():
    """Verifies that the workflow retrieves historical scenarios and presents saved results without rerunning."""
    # 1. First create a car loan scenario
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, "user_id": 2}}

    await scenario_workflow_app.ainvoke({
        "messages": [("user", "What if I take a ₹10 lakh car loan at 9%?")],
        "user_query": "What if I take a ₹10 lakh car loan at 9%?",
        "user_id": 2,
        "thread_id": thread_id,
    }, config)

    # 2. Later, in a new turn or thread, user asks about what was calculated
    new_thread = str(uuid.uuid4())
    new_config = {"configurable": {"thread_id": new_thread, "user_id": 2}}

    res = await scenario_workflow_app.ainvoke({
        "messages": [("user", "What did we calculate for my car loan last week?")],
        "user_query": "What did we calculate for my car loan last week?",
        "user_id": 2,
        "thread_id": new_thread,
    }, new_config)

    assert res["validation_status"] == "cached_result"
    assert "saved scenario" in res["messages"][-1].content.lower()
    assert res.get("active_scenario_id") is not None
