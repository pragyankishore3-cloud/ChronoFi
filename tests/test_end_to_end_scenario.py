"""
End-to-end integration test verifying the complete user journey:
Natural language prompt -> Parameter extraction -> Context resolution ->
Core API / Engine calculation -> Paired Monte Carlo -> Result interpretation.
"""

import pytest
from core.security import create_access_token, get_password_hash
from models.database import Asset, Goal, User
from ai.scenario_workflow import scenario_workflow_app
from langchain_core.messages import HumanMessage


@pytest.fixture
async def e2e_priya(db_session):
    user = User(
        name="Priya",
        email="priya_e2e@chronofi.demo",
        hashed_password=get_password_hash("password123"),
        age=35,
        risk_tolerance="Moderate",
        base_salary=250000.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    db_session.add(Asset(user_id=user.id, name="Mutual Funds", asset_type="Equity", current_value=2500000.0, expected_return_rate=0.12))
    db_session.add(Asset(user_id=user.id, name="Fixed Deposit", asset_type="Cash", current_value=1000000.0, expected_return_rate=0.07))
    db_session.add(Goal(user_id=user.id, name="Retirement", target_amount=50000000.0, target_years=25, priority="Essential", inflation_rate=0.06))
    await db_session.commit()
    return user


@pytest.mark.asyncio
async def test_full_scenario_simulation_journey(client, e2e_priya):
    """
    Test user asking:
    'What happens to my retirement goal if I take a ₹10 lakh car loan today at 9% interest?'
    Verifies that:
    1. LangGraph extracts parameters correctly.
    2. Missing tenure is defaulted to documented 60 months assumption.
    3. Engine computes accurate amortizing loan math.
    4. Retirement corpus & age changes are calculated deterministically & via paired Monte Carlo.
    5. Actionable alternatives are offered.
    6. Numerical outputs in AI response match actual engine results.
    """
    token = create_access_token(data={"sub": str(e2e_priya.id)})
    headers = {"Authorization": f"Bearer {token}"}

    prompt = "What happens to my retirement goal if I take a ₹10 lakh car loan today at 9% interest?"

    # 1. Test via LangGraph Workflow
    state = {
        "messages": [HumanMessage(content=prompt)],
        "user_query": prompt,
        "user_id": e2e_priya.id,
    }
    config = {"configurable": {"user_id": e2e_priya.id}}
    agent_res = await scenario_workflow_app.ainvoke(state, config=config)

    assert agent_res["validation_status"] == "valid"
    sim_result = agent_res["simulation_result"]
    assert sim_result is not None

    # Verify extracted parameters
    extracted = agent_res["extracted_params"]
    assert extracted["loan_principal"] == 1000000.0
    assert extracted["annual_interest_rate"] == 0.09
    assert extracted["loan_tenure_months"] == 60

    # Verify documented assumptions
    assumptions = sim_result["effective_assumptions"]
    tenure_assump = next(a for a in assumptions if a["field"] == "loan_tenure_months")
    assert tenure_assump["source"] == "explicit_default"
    assert tenure_assump["value"] == 60

    # Verify engine loan metrics
    loan = sim_result["loan_metrics"]
    assert loan["monthly_emi"] == 20758.36
    assert loan["total_interest"] == 245501.60
    assert loan["total_payment"] == 1245501.60

    # Verify retirement baseline vs modified
    base = sim_result["baseline"]
    mod = sim_result["modified"]
    impact = sim_result["impact"]

    assert mod["monthly_investable_surplus"] == base["monthly_investable_surplus"] - 20758.36
    assert impact["monthly_cash_flow_difference"] == -20758.36
    assert mod["projected_corpus"] < base["projected_corpus"]
    assert impact["corpus_difference"] < 0
    assert mod["retirement_age"] >= base["retirement_age"]
    assert impact["retirement_age_difference"] >= 0
    assert mod["goal_success_probability"] <= base["goal_success_probability"]

    # Verify AI presentation text
    ai_text = agent_res["messages"][-1].content
    assert "20,758" in ai_text
    assert "Retirement" in ai_text
    assert "Alternatives" in ai_text

    # 2. Test via Core API Endpoint directly
    api_res = await client.post(
        "/api/v1/financial-scenarios/simulate",
        json={
            "scenario_type": "vehicle_loan",
            "changes": {
                "loan_principal": 1000000.0,
                "annual_interest_rate": 0.09,
                "loan_tenure_months": 60,
            },
        },
        headers=headers,
    )
    assert api_res.status_code == 200
    api_data = api_res.json()
    assert api_data["status"] == "completed"
    assert api_data["loan_metrics"]["monthly_emi"] == 20758.36
    assert len(api_data["alternatives"]) == 4
    assert len(api_data["trajectory"]) == 26
