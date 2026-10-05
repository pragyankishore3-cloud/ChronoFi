from sqlalchemy import Column, Integer, String, Float, Enum, Date, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import enum
import os

Base = declarative_base()

# Use SQLite for easy MVP/Hackathon setup, easily swappable to Postgres
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chronofi.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class PriorityLevel(str, enum.Enum):
    ESSENTIAL = "Essential"
    IMPORTANT = "Important"
    ASPIRATIONAL = "Aspirational"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    risk_tolerance = Column(String) # Conservative, Moderate, Aggressive
    base_salary = Column(Float)
    
    assets = relationship("Asset", back_populates="user", cascade="all, delete-orphan")
    liabilities = relationship("Liability", back_populates="user", cascade="all, delete-orphan")
    goals = relationship("Goal", back_populates="user", cascade="all, delete-orphan")

class Asset(Base):
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    asset_type = Column(String) # Cash, Equity, Property
    current_value = Column(Float)
    expected_return_rate = Column(Float) # e.g., 0.10 for 10%
    
    user = relationship("User", back_populates="assets")

class Liability(Base):
    __tablename__ = "liabilities"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    principal_remaining = Column(Float)
    interest_rate = Column(Float) # e.g., 0.08 for 8%
    monthly_payment = Column(Float)
    
    user = relationship("User", back_populates="liabilities")

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    target_amount = Column(Float)
    target_years = Column(Integer) # Years from now
    priority = Column(Enum(PriorityLevel))
    inflation_rate = Column(Float, default=0.06)
    
    user = relationship("User", back_populates="goals")

def init_db():
    Base.metadata.create_all(bind=engine)
