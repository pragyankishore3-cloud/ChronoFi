from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

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

class UserBase(BaseModel):
    name: str
    email: str
    age: int
    risk_tolerance: str
    base_salary: float

class UserProfileResponse(UserBase):
    id: int
    assets: List[AssetResponse] = []
    liabilities: List[LiabilityResponse] = []
    goals: List[GoalResponse] = []
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    user_id: int
    message: str

class ChatResponse(BaseModel):
    response: str
    reasoning_logs: Optional[list] = None
