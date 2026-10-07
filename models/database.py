from sqlalchemy import Column, Integer, String, Float, Enum, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
import enum
import os

Base = declarative_base()

# Defaulting to async sqlite for local dev without Docker, seamlessly swappable to postgres
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./chronofi.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

class PriorityLevel(str, enum.Enum):
    ESSENTIAL = "Essential"
    IMPORTANT = "Important"
    ASPIRATIONAL = "Aspirational"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    name = Column(String, index=True)
    age = Column(Integer)
    risk_tolerance = Column(String) 
    base_salary = Column(Float)
    
    # lazy="selectin" is crucial for async SQLAlchemy to avoid lazy-load blocking
    assets = relationship("Asset", back_populates="user", cascade="all, delete-orphan", lazy="selectin")
    liabilities = relationship("Liability", back_populates="user", cascade="all, delete-orphan", lazy="selectin")
    goals = relationship("Goal", back_populates="user", cascade="all, delete-orphan", lazy="selectin")

class Asset(Base):
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    asset_type = Column(String) 
    current_value = Column(Float)
    expected_return_rate = Column(Float) 
    
    user = relationship("User", back_populates="assets")

class Liability(Base):
    __tablename__ = "liabilities"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    principal_remaining = Column(Float)
    interest_rate = Column(Float) 
    monthly_payment = Column(Float)
    
    user = relationship("User", back_populates="liabilities")

class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    target_amount = Column(Float)
    target_years = Column(Integer) 
    priority = Column(Enum(PriorityLevel))
    inflation_rate = Column(Float, default=0.06)
    
    user = relationship("User", back_populates="goals")
import json
from sqlalchemy import JSON

class DocumentType(str, enum.Enum):
    TAX_RETURN = "Tax Return"
    LOAN_STATEMENT = "Loan Statement"
    INSURANCE = "Insurance"
    OTHER = "Other"
    UNKNOWN = "Unknown"

class DocumentStatus(str, enum.Enum):
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    READY = "Ready"
    FAILED = "Failed"

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    file_path = Column(String)
    mime_type = Column(String)
    size_bytes = Column(Integer)
    document_type = Column(String)
    financial_year = Column(String)
    issuer = Column(String)
    status = Column(Enum(DocumentStatus))
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan", lazy="selectin")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(String)
    page_number = Column(Integer)
    chunk_index = Column(Integer)
    embedding = Column(JSON)
    
    document = relationship("Document", back_populates="chunks")

class KnowledgeBaseChunk(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    topic = Column(String)
    embedding = Column(JSON)



