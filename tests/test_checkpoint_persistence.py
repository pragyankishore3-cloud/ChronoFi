"""
Tests for LangGraph Checkpointer Persistence and Thread Isolation.
Verifies:
1. Resumption of state within the same thread
2. Complete isolation between separate threads
3. Durable persistence surviving process restarts (SQLite checkpointer)
4. Handling missing or expired checkpoints safely
"""

import os
import uuid
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from core.checkpointer import init_checkpointer, get_checkpointer, close_checkpointer
from ai.scenario_workflow import scenario_workflow_app
from seed.seed_db import seed_database


@pytest.fixture(autouse=True)
async def setup_test_users():
    """Seed test users (Arjun ID=1, Priya ID=2) for all checkpoint tests."""
    await seed_database()
    # Reset checkpointer to MemorySaver for clean test isolation
    await init_checkpointer(use_memory=True)
    yield
    await close_checkpointer()


@pytest.mark.asyncio
async def test_thread_resumption_and_continuity():
    """Verifies that subsequent invocations on the same thread_id retain previous state."""
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "user_id": 2}}

    # First turn: User states initial loan query
    input_1 = {
        "messages": [HumanMessage(content="What happens if I take a ₹10 lakh car loan at 9% interest?")],
        "user_query": "What happens if I take a ₹10 lakh car loan at 9% interest?",
        "user_id": 2,
        "thread_id": thread_id,
    }
    res_1 = await scenario_workflow_app.ainvoke(input_1, config)
    assert res_1["validation_status"] == "valid"
    assert res_1.get("active_scenario_id") is not None
    saved_scenario_id = res_1["active_scenario_id"]

    # Second turn on the same thread: User asks follow-up
    # Graph checkpointer should restore active_scenario_id and prior messages!
    input_2 = {
        "messages": [HumanMessage(content="Explain the trade-offs")],
        "user_query": "Explain the trade-offs",
        "user_id": 2,
        "thread_id": thread_id,
    }
    res_2 = await scenario_workflow_app.ainvoke(input_2, config)
    assert res_2.get("active_scenario_id") == saved_scenario_id
    assert len(res_2["messages"]) >= 2


@pytest.mark.asyncio
async def test_thread_isolation():
    """Verifies that two distinct threads do not leak conversation state."""
    thread_a = str(uuid.uuid4())
    thread_b = str(uuid.uuid4())

    config_a: RunnableConfig = {"configurable": {"thread_id": thread_a, "user_id": 2}}
    config_b: RunnableConfig = {"configurable": {"thread_id": thread_b, "user_id": 2}}

    # Run on thread A
    await scenario_workflow_app.ainvoke({
        "messages": [HumanMessage(content="Car loan of ₹10 lakh at 9%")],
        "user_query": "Car loan of ₹10 lakh at 9%",
        "user_id": 2,
        "thread_id": thread_a,
    }, config_a)

    # Check state of thread B (should be completely empty/fresh)
    cp = get_checkpointer()
    tuple_b = await cp.aget_tuple(config_b)
    assert tuple_b is None, "Thread B should have no saved checkpoint yet"


@pytest.mark.asyncio
async def test_sqlite_checkpointer_survives_restart(tmp_path):
    """Verifies that an AsyncSqliteSaver persists state to disk across simulated process restarts."""
    db_file = str(tmp_path / "persistent_checkpoints.db")
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "user_id": 2}}

    # Session 1: Initialize SQLite checkpointer, run graph, save state
    await init_checkpointer(db_path=db_file)
    try:
        input_1 = {
            "messages": [HumanMessage(content="Take a ₹10 lakh car loan at 9%")],
            "user_query": "Take a ₹10 lakh car loan at 9%",
            "user_id": 2,
            "thread_id": thread_id,
        }
        res_1 = await scenario_workflow_app.ainvoke(input_1, config)
        saved_scenario_id = res_1.get("active_scenario_id")
        assert saved_scenario_id is not None
    finally:
        await close_checkpointer()

    # Session 2: Simulated process restart (re-open same SQLite file)
    await init_checkpointer(db_path=db_file)
    try:
        cp = get_checkpointer()
        chk_tuple = await cp.aget_tuple(config)
        assert chk_tuple is not None, "Checkpoint must survive across process restarts"
        channel_values = chk_tuple.checkpoint.get("channel_values", {})
        assert channel_values.get("active_scenario_id") == saved_scenario_id
    finally:
        await close_checkpointer()


@pytest.mark.asyncio
async def test_missing_or_new_checkpoint_handling():
    """Verifies that queries with nonexistent thread IDs start clean without errors."""
    fresh_thread = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": fresh_thread, "user_id": 2}}

    res = await scenario_workflow_app.ainvoke({
        "messages": [HumanMessage(content="What if I borrow ₹5 lakh at 10%?")],
        "user_query": "What if I borrow ₹5 lakh at 10%?",
        "user_id": 2,
        "thread_id": fresh_thread,
    }, config)

    assert res["validation_status"] == "valid"
    assert res.get("active_scenario_id") is not None
