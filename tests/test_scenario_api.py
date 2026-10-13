import pytest
from core.security import create_access_token, get_password_hash
from models.database import Asset, Goal, User


@pytest.fixture
async def priya_user(db_session):
    """Create Priya persona with realistic assets and retirement goal."""
    user = User(
        name="Priya",
        email="priya@chronofi.demo",
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
    return user


@pytest.fixture
async def arjun_user(db_session):
    """Create Arjun persona."""
    user = User(
        name="Arjun",
        email="arjun@chronofi.demo",
        hashed_password=get_password_hash("password123"),
        age=25,
        risk_tolerance="Aggressive",
        base_salary=80000.0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.mark.asyncio
async def test_simulate_scenario_unauthorized(client):
    """Test that simulation route requires valid JWT authentication."""
    response = await client.post(
        "/api/v1/financial-scenarios/simulate",
        json={
            "scenario_type": "vehicle_loan",
            "changes": {
                "loan_principal": 1000000.0,
                "annual_interest_rate": 0.09,
                "loan_tenure_months": 60,
            },
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_simulate_scenario_success(client, priya_user):
    """
    Test standard What-If scenario: Priya takes a ₹10 Lakh car loan at 9% for 60 months.
    Verifies accurate math, baseline & modified metrics, and alternatives.
    """
    token = create_access_token(data={"sub": str(priya_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "scenario_type": "vehicle_loan",
        "changes": {
            "loan_principal": 1000000.0,
            "annual_interest_rate": 0.09,
            "loan_tenure_months": 60,
            "down_payment": 0.0,
            "monthly_vehicle_expenses": 5000.0,
        },
        "simulation_settings": {
            "num_simulations": 500,
            "seed": 42,
        },
    }

    response = await client.post(
        "/api/v1/financial-scenarios/simulate",
        json=payload,
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "completed"
    assert data["scenario_type"] == "vehicle_loan"
    assert "scenario_id" in data

    # Verify Loan Metrics
    loan = data["loan_metrics"]
    assert loan["monthly_emi"] == 20758.36
    assert loan["principal"] == 1000000.0
    assert loan["total_interest"] == 245501.60

    # Verify Baseline vs Modified
    baseline = data["baseline"]
    modified = data["modified"]
    impact = data["impact"]

    assert modified["monthly_investable_surplus"] < baseline["monthly_investable_surplus"]
    assert impact["monthly_cash_flow_difference"] == -(20758.36 + 5000.0)
    assert modified["projected_corpus"] < baseline["projected_corpus"]
    assert impact["corpus_difference"] < 0
    assert impact["retirement_age_difference"] >= 0
    assert modified["goal_success_probability"] <= baseline["goal_success_probability"]

    # Verify Alternatives and Trajectory
    assert len(data["alternatives"]) == 4
    assert len(data["trajectory"]) > 0
    assert data["engine_version"] == "chronofi-sim-v1.0.0"


@pytest.mark.asyncio
async def test_simulate_scenario_validation_errors(client, priya_user):
    """Test validation rejection for invalid financial parameters."""
    token = create_access_token(data={"sub": str(priya_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    # Negative principal
    res1 = await client.post(
        "/api/v1/financial-scenarios/simulate",
        json={
            "changes": {"loan_principal": -50000.0, "annual_interest_rate": 0.09, "loan_tenure_months": 60}
        },
        headers=headers,
    )
    assert res1.status_code == 422

    # Invalid interest rate (> 1.0)
    res2 = await client.post(
        "/api/v1/financial-scenarios/simulate",
        json={
            "changes": {"loan_principal": 500000.0, "annual_interest_rate": 1.5, "loan_tenure_months": 60}
        },
        headers=headers,
    )
    assert res2.status_code == 422


@pytest.mark.asyncio
async def test_simulate_scenario_idempotency(client, priya_user):
    """Test idempotency key ensures reproducible and cached response."""
    token = create_access_token(data={"sub": str(priya_user.id)})
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "changes": {
            "loan_principal": 800000.0,
            "annual_interest_rate": 0.085,
            "loan_tenure_months": 48,
        },
        "idempotency_key": "test_idem_key_12345",
    }

    res1 = await client.post("/api/v1/financial-scenarios/simulate", json=payload, headers=headers)
    assert res1.status_code == 200
    data1 = res1.json()

    res2 = await client.post("/api/v1/financial-scenarios/simulate", json=payload, headers=headers)
    assert res2.status_code == 200
    data2 = res2.json()

    assert data1["scenario_id"] == data2["scenario_id"]
    assert data1["impact"] == data2["impact"]


@pytest.mark.asyncio
async def test_user_isolation_and_access_control(client, priya_user, arjun_user):
    """Test that users cannot read or simulate other users' financial data."""
    priya_token = create_access_token(data={"sub": str(priya_user.id)})
    arjun_token = create_access_token(data={"sub": str(arjun_user.id)})

    # Priya creates a scenario
    res = await client.post(
        "/api/v1/financial-scenarios/simulate",
        json={"changes": {"loan_principal": 1000000.0, "annual_interest_rate": 0.09, "loan_tenure_months": 60}},
        headers={"Authorization": f"Bearer {priya_token}"},
    )
    assert res.status_code == 200
    scenario_id = res.json()["scenario_id"]

    # Arjun tries to access Priya's scenario by ID
    res_arjun = await client.get(
        f"/api/v1/financial-scenarios/{scenario_id}",
        headers={"Authorization": f"Bearer {arjun_token}"},
    )
    assert res_arjun.status_code == 404

    # Priya can access her own scenario
    res_priya = await client.get(
        f"/api/v1/financial-scenarios/{scenario_id}",
        headers={"Authorization": f"Bearer {priya_token}"},
    )
    assert res_priya.status_code == 200
    assert res_priya.json()["scenario_id"] == scenario_id
