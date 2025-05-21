from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import enum

from backend.core.database import Base
from backend.models.user import SubscriptionTier

class StrategyType(str, enum.Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    PATTERN_RECOGNITION = "pattern_recognition"
    CUSTOM = "custom"


class CustomStrategy(Base):
    """SQLAlchemy model for custom trading strategies created by users"""
    __tablename__ = "custom_strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    strategy_type = Column(String, nullable=False)
    dsl_content = Column(Text, nullable=False)  # YAML-like DSL content
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)  # Whether strategy can be viewed by other users
    parameters = Column(JSON, nullable=True)  # Additional parameters as JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="custom_strategies")
    backtest_results = relationship("BacktestResult", back_populates="strategy", cascade="all, delete-orphan")


class BacktestResult(Base):
    """SQLAlchemy model for storing backtest results for strategies"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("custom_strategies.id"), nullable=False)
    asset_symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    win_rate = Column(Integer, nullable=True)  # Percentage
    profit_factor = Column(Integer, nullable=True)  # Percentage (profit/loss ratio * 100)
    max_drawdown = Column(Integer, nullable=True)  # Percentage
    total_trades = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=True)  # Additional metrics as JSON
    equity_curve = Column(JSON, nullable=True)  # JSON array of equity curve points
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    strategy = relationship("CustomStrategy", back_populates="backtest_results")


# Pydantic models for API requests/responses
class StrategyDSLBase(BaseModel):
    """Base model for strategy DSL validation"""
    strategy_name: str
    rules: List[Dict[str, Any]]
    risk: Dict[str, Any]


class StrategyCreate(BaseModel):
    """Model for creating a new custom strategy"""
    name: str
    description: Optional[str] = None
    strategy_type: StrategyType
    dsl_content: str
    is_public: bool = False
    parameters: Optional[Dict[str, Any]] = None


class StrategyUpdate(BaseModel):
    """Model for updating an existing custom strategy"""
    name: Optional[str] = None
    description: Optional[str] = None
    strategy_type: Optional[StrategyType] = None
    dsl_content: Optional[str] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None


class StrategyResponse(BaseModel):
    """Model for strategy response"""
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    strategy_type: str
    dsl_content: str
    is_active: bool
    is_public: bool
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class BacktestRequest(BaseModel):
    """Model for backtest request"""
    strategy_id: Optional[int] = None  # Optional for temporary strategies not saved yet
    dsl_content: str
    asset_symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime


class BacktestResponse(BaseModel):
    """Model for backtest response"""
    id: Optional[int] = None
    strategy_id: Optional[int] = None
    asset_symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    equity_curve: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class StrategyPreset(BaseModel):
    """Model for strategy presets"""
    id: str
    name: str
    description: str
    strategy_type: StrategyType
    dsl_content: str
    tier_required: SubscriptionTier
    parameters: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True
