import pytest
from langchain_core.messages import HumanMessage
from core.security import get_password_hash
from models.database import Asset, Goal, Liability, User
from ai.scenario_tools import run_financial_scenario
from ai.scenario_workflow import (
    _parse_currency_amount,
    _parse_interest_rate,
    _parse_tenure_months,
    scenario_workflow_app,
)


@pytest.fixture
async def priya_scenario_user(db_session):
    """Seed Priya persona for LangGraph scenario tests."""
    user = User(
        name="Priya",
        email="priya_scenario@chronofi.demo",
        hashed_password=get_password_hash("password123"),
        age=35,
        risk_tolerance="Moderate",
        base_salary=250000.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    db_session.add(
        Asset(
            user_id=user.id,
            name="Mutual Funds",
            asset_type="Equity",
            current_value=2500000.0,
            expected_return_rate=0.12,
        )
    )
    db_session.add(
        Asset(
            user_id=user.id,
            name="Fixed Deposit",
            asset_type="Cash",
            current_value=1000000.0,
            expected_return_rate=0.07,
        )
    )
    db_session.add(
        Goal(
            user_id=user.id,
            name="Retirement",
            target_amount=50000000.0,
            target_years=25,
            priority="Essential",
            inflation_rate=0.06,
        )
    )
    await db_session.commit()
    from sqlalchemy.orm import selectinload
    from sqlalchemy import select
    res = await db_session.execute(
        select(User)
        .where(User.id == user.id)
        .options(selectinload(User.assets), selectinload(User.liabilities), selectinload(User.goals))
    )
    return res.scalars().first()


def test_parameter_parsing_units():
    """Test natural language parameter extraction helpers for diverse Indian financial expressions."""
    # 1. Currency
    assert _parse_currency_amount("10 lakh car loan") == 1000000.0
    assert _parse_currency_amount("₹10,00,000") == 1000000.0
    assert _parse_currency_amount("take a 15 Lac loan") == 1500000.0
    assert _parse_currency_amount("1.5 crore house") == 15000000.0
    assert _parse_currency_amount("50k expense") == 50000.0

    # 2. Rates
    assert _parse_interest_rate("at 9% interest") == 0.09
    assert _parse_interest_rate("at 8.5 %") == 0.085
    assert _parse_interest_rate("10.25 percent APR") == 0.1025

    # 3. Tenure
    assert _parse_tenure_months("for 5 years") == 60
    assert _parse_tenure_months("for 3 yrs") == 36
    assert _parse_tenure_months("for 48 months") == 48


@pytest.mark.asyncio
async def test_scenario_tool_unauthorized_rejection():
    """Verify tool rejects execution without authenticated user_id in RunnableConfig."""
    res = await run_financial_scenario.ainvoke(
        {"loan_principal": 1000000.0, "annual_interest_rate": 0.09},
        config={},
    )
    assert res.get("status") == "error"
    assert "Unauthorized" in res.get("error", "")


@pytest.mark.asyncio
async def test_scenario_tool_direct_execution(priya_scenario_user):
    """Test direct invocation of LangChain run_financial_scenario tool."""
    config = {"configurable": {"user_id": priya_scenario_user.id}}
    result = await run_financial_scenario.ainvoke(
        {
            "scenario_type": "vehicle_loan",
            "loan_principal": 1000000.0,
            "annual_interest_rate": 0.09,
            "loan_tenure_months": 60,
            "down_payment": 0.0,
            "user_id": priya_scenario_user.id,
        },
        config=config,
    )

    print("\nDEBUG RESULT:", result)
    assert result["status"] == "completed"
    assert result["loan_metrics"]["monthly_emi"] == 20758.36
    assert result["loan_metrics"]["total_interest"] == 245501.60
    assert result["baseline"]["projected_corpus"] > result["modified"]["projected_corpus"]
    assert len(result["alternatives"]) == 4


@pytest.mark.asyncio
async def test_workflow_missing_parameters_clarification():
    """Test workflow returns a polite clarification when material inputs are absent."""
    state = {
        "messages": [HumanMessage(content="What happens to my retirement if I take a loan?")],
        "user_query": "What happens to my retirement if I take a loan?",
        "user_id": 9999,  # placeholder
    }
    config = {"configurable": {"user_id": 9999}}

    # The workflow should stop at clarification_node
    result = await scenario_workflow_app.ainvoke(state, config=config)

    assert result.get("validation_status") == "needs_clarification"
    assert len(result["messages"]) > 0
    reply = result["messages"][-1].content
    assert "clarify" in reply.lower() or "loan amount" in reply.lower()


@pytest.mark.asyncio
async def test_full_scenario_workflow_execution(priya_scenario_user, db_session):
    """
    End-to-end test of the complete LangGraph workflow with the user's primary prompt:
    'What happens to my retirement goal if I take a ₹10 lakh car loan today at 9% interest?'
    """
    prompt = "What happens to my retirement goal if I take a ₹10 lakh car loan today at 9% interest?"
    state = {
        "messages": [HumanMessage(content=prompt)],
        "user_query": prompt,
        "user_id": priya_scenario_user.id,
    }
    config = {"configurable": {"user_id": priya_scenario_user.id}}

    result = await scenario_workflow_app.ainvoke(state, config=config)

    assert result["validation_status"] == "valid"
    assert result["simulation_result"] is not None

    sim = result["simulation_result"]
    assert sim["loan_metrics"]["monthly_emi"] == 20758.36
    assert sim["impact"]["corpus_difference"] < 0
    assert sim["impact"]["retirement_age_difference"] >= 0

    # Verify Final Presentation
    presentation = result["final_presentation"]
    assert presentation is not None
    assert "structured_data" in presentation
    assert len(presentation["alternatives"]) == 4
    assert len(presentation["trajectory"]) > 0

    # Verify AI Message Content
    final_message = result["messages"][-1].content
    assert "20,758" in final_message  # Exact EMI formatted
    assert "Retirement" in final_message
    assert "Alternatives" in final_message

    # Read-Only Verification: Database user assets & liabilities must remain unchanged
    from models.database import Asset, Liability
    from sqlalchemy import select
    assets_res = await db_session.execute(select(Asset).where(Asset.user_id == priya_scenario_user.id))
    user_assets = assets_res.scalars().all()
    liab_res = await db_session.execute(select(Liability).where(Liability.user_id == priya_scenario_user.id))
    user_liabs = liab_res.scalars().all()
    assert len(user_liabs) == 0  # No real liability created!
    assert len(user_assets) == 2  # Assets preserved exactly!
