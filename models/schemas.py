from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PriorityLevel(str, Enum):
    ESSENTIAL = "Essential"
    IMPORTANT = "Important"
    ASPIRATIONAL = "Aspirational"

class AssetBase(BaseModel):
    name: str
    asset_type: str
    current_value: float = Field(..., ge=0.0)
    expected_return_rate: float
    currency: str = Field(default="INR")

class AssetResponse(AssetBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class LiabilityBase(BaseModel):
    name: str
    principal_remaining: float = Field(..., ge=0.0)
    interest_rate: float = Field(..., ge=0.0)
    monthly_payment: float = Field(..., ge=0.0)
    currency: str = Field(default="INR")

class LiabilityResponse(LiabilityBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class GoalBase(BaseModel):
    name: str
    target_amount: float = Field(..., ge=0.0)
    target_years: int = Field(..., ge=1)
    priority: PriorityLevel
    inflation_rate: float = 0.06
    currency: str = Field(default="INR")

class GoalResponse(GoalBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class AssetCreate(AssetBase):
    pass

class AssetUpdate(BaseModel):
    name: Optional[str] = None
    asset_type: Optional[str] = None
    current_value: Optional[float] = Field(None, ge=0.0)
    expected_return_rate: Optional[float] = None
    currency: Optional[str] = None

class LiabilityCreate(LiabilityBase):
    pass

class LiabilityUpdate(BaseModel):
    name: Optional[str] = None
    principal_remaining: Optional[float] = Field(None, ge=0.0)
    interest_rate: Optional[float] = Field(None, ge=0.0)
    monthly_payment: Optional[float] = Field(None, ge=0.0)
    currency: Optional[str] = None

class GoalCreate(GoalBase):
    pass

class GoalUpdate(BaseModel):
    name: Optional[str] = None
    target_amount: Optional[float] = Field(None, ge=0.0)
    target_years: Optional[int] = Field(None, ge=1)
    priority: Optional[PriorityLevel] = None
    inflation_rate: Optional[float] = None
    currency: Optional[str] = None

class UserBase(BaseModel):
    name: str
    email: str
    age: int = Field(..., ge=0)
    risk_tolerance: str
    base_salary: float = Field(..., ge=0.0)
    monthly_expenses: float = Field(default=0.0, ge=0.0)
    currency: str = Field(default="INR")
    occupation: Optional[str] = None
    date_of_birth: Optional[str] = None

class UserProfileUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0)
    risk_tolerance: Optional[str] = None
    base_salary: Optional[float] = Field(None, ge=0.0)
    monthly_expenses: Optional[float] = Field(None, ge=0.0)
    currency: Optional[str] = None
    occupation: Optional[str] = None
    date_of_birth: Optional[str] = None

class UserProfileResponse(UserBase):
    id: int
    assets: list[AssetResponse] = []
    liabilities: list[LiabilityResponse] = []
    goals: list[GoalResponse] = []
    net_worth: float = 0.0
    monthly_investable_surplus: float = 0.0
    months_of_runway: float = 0.0
    created_at: Optional[str] = "2026-01-01T00:00:00Z"
    
    model_config = ConfigDict(from_attributes=True)

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

    model_config = ConfigDict(from_attributes=True)

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
