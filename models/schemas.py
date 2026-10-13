from enum import Enum

from pydantic import BaseModel


class PriorityLevel(str, Enum):
    ESSENTIAL = "Essential"
    IMPORTANT = "Important"
    ASPIRATIONAL = "Aspirational"

class AssetBase(BaseModel):
    name: str
    asset_type: str
    current_value: float
    expected_return_rate: float

class AssetResponse(AssetBase):
    id: int
    class Config:
        from_attributes = True

class LiabilityBase(BaseModel):
    name: str
    principal_remaining: float
    interest_rate: float
    monthly_payment: float

class LiabilityResponse(LiabilityBase):
    id: int
    class Config:
        from_attributes = True

class GoalBase(BaseModel):
    name: str
    target_amount: float
    target_years: int
    priority: PriorityLevel
    inflation_rate: float = 0.06

class GoalResponse(GoalBase):
    id: int
    class Config:
        from_attributes = True

class AssetCreate(AssetBase):
    pass

class LiabilityCreate(LiabilityBase):
    pass

class GoalCreate(GoalBase):
    pass

class UserBase(BaseModel):
    name: str
    email: str
    age: int
    risk_tolerance: str
    base_salary: float
    monthly_expenses: float = 0.0

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    risk_tolerance: Optional[str] = None
    base_salary: Optional[float] = None
    monthly_expenses: Optional[float] = None

class UserProfileResponse(UserBase):
    id: int
    assets: list[AssetResponse] = []
    liabilities: list[LiabilityResponse] = []
    goals: list[GoalResponse] = []
    net_worth: float = 0.0
    monthly_investable_surplus: float = 0.0
    months_of_runway: float = 0.0
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    user_id: int
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    reasoning_logs: list | None = None
    thread_id: Optional[str] = None

class ConversationThreadResponse(BaseModel):
    id: str
    user_id: int
    title: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class HistoricalScenarioResponse(BaseModel):
    scenario_id: str
    user_id: int
    thread_id: Optional[str] = None
    name: Optional[str] = None
    scenario_type: str
    parameters: dict[str, Any]
    assumptions: Optional[list[Any]] = None
    baseline_id: Optional[str] = None
    results_summary: Optional[dict[str, Any]] = None
    version: int = 1
    parent_scenario_id: Optional[str] = None
    status: str = "simulated"
    engine_version: str = "1.0.0-crn"
    created_at: str
    updated_at: Optional[str] = None
    is_current: bool = True

class ConfirmLiabilityRequest(BaseModel):
    scenario_id: str
    name: Optional[str] = "Car Loan"
    principal: Optional[float] = None
    interest_rate: Optional[float] = None
    tenure_months: Optional[int] = None
    emi: Optional[float] = None

class ConfirmLiabilityResponse(BaseModel):
    success: bool
    liability_id: int
    name: str
    principal: float
    interest_rate: float
    monthly_payment: float
    scenario_status: str
    message: str

# --- WHAT-IF SCENARIO SCHEMAS ---

from typing import Any, Optional

class ScenarioChanges(BaseModel):
    loan_principal: float = 0.0
    annual_interest_rate: float = 0.0
    loan_tenure_months: int = 60
    down_payment: float = 0.0
    monthly_vehicle_expenses: float = 0.0
    monthly_savings_change: float = 0.0
    income_change: float = 0.0
    expense_change: float = 0.0

class SimulationSettings(BaseModel):
    num_simulations: int = 1000
    seed: int = 42
    include_trajectory: bool = True
    horizon_years: Optional[int] = None

class AssumptionOverride(BaseModel):
    expected_annual_return: Optional[float] = None
    inflation_rate: Optional[float] = None
    estimated_monthly_expenses: Optional[float] = None

class SimulateScenarioRequest(BaseModel):
    baseline_plan_id: Optional[str] = None
    scenario_type: str = "vehicle_loan"
    changes: ScenarioChanges
    simulation_settings: SimulationSettings = SimulationSettings()
    assumptions_override: Optional[AssumptionOverride] = None
    idempotency_key: Optional[str] = None

class ScenarioAssumptionSchema(BaseModel):
    field: str
    value: Any
    source: str
    description: str

class LoanMetricsSchema(BaseModel):
    monthly_emi: float
    principal: float
    annual_interest_rate: float
    tenure_months: int
    total_payment: float
    total_interest: float
    is_zero_interest: bool

class ProjectionMetricsSchema(BaseModel):
    retirement_age: float
    projected_corpus: float
    goal_success_probability: float
    monthly_investable_surplus: float
    worst_case_10th_percentile: float
    expected_50th_percentile: float
    best_case_90th_percentile: float
    conditional_value_at_risk_cvar: float

class ScenarioImpactSchema(BaseModel):
    retirement_age_difference: float
    corpus_difference: float
    success_probability_difference: float
    monthly_cash_flow_difference: float
    total_interest_paid: float

class AlternativeScenarioSchema(BaseModel):
    name: str
    description: str
    loan_principal: float
    loan_tenure_months: int
    down_payment: float
    monthly_emi: float
    total_interest: float
    projected_corpus: float
    goal_success_probability: float
    retirement_age_difference: float
    trade_off_summary: str

class TrajectoryPointSchema(BaseModel):
    year: int
    age: int
    baseline_expected: float
    modified_expected: float
    baseline_worst: float
    modified_worst: float
    target_corpus: float

class GoalConflictItemSchema(BaseModel):
    goal_id: Optional[int] = None
    name: str
    priority: str
    target_amount: float
    target_years: int
    inflated_target: float
    monthly_sip_required: float
    baseline_monthly_allocated: float
    baseline_projected_corpus: float
    baseline_funding_ratio: float
    baseline_status: str
    modified_monthly_allocated: float
    modified_projected_corpus: float
    modified_funding_ratio: float
    modified_status: str
    monthly_allocation_change: float
    corpus_shortfall: float
    projected_delay_months: int
    trade_off_note: str

class SimulateScenarioResponse(BaseModel):
    scenario_id: str
    scenario_type: str
    status: str
    effective_assumptions: list[ScenarioAssumptionSchema]
    loan_metrics: Optional[LoanMetricsSchema] = None
    baseline: ProjectionMetricsSchema
    modified: ProjectionMetricsSchema
    impact: ScenarioImpactSchema
    alternatives: list[AlternativeScenarioSchema] = []
    trajectory: list[TrajectoryPointSchema] = []
    warnings: list[str] = []
    goal_impacts: list[GoalConflictItemSchema] = []
    engine_version: str = "chronofi-sim-v1.0.0"
