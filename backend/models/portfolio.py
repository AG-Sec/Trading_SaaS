from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime
from typing import List, Optional

from backend.core.database import Base
from backend.models.user import User

class AssetClass(str, enum.Enum):
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    ETF = "ETF"
    
class TradeDirection(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    
class TradeStatus(str, enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"

class Portfolio(Base):
    """User portfolio for tracking assets and performance"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    
    # Owner relationship
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user = relationship("User", back_populates="portfolios")
    
    # Portfolio statistics and metrics
    starting_capital = Column(Float, nullable=False, default=10000.0)
    current_value = Column(Float, nullable=False, default=10000.0)
    cash_balance = Column(Float, nullable=False, default=10000.0)
    total_profit_loss = Column(Float, nullable=False, default=0.0)
    daily_profit_loss = Column(Float, nullable=False, default=0.0)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    win_rate = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Portfolio {self.name} (ID: {self.id})>"
    
class Position(Base):
    """Represents a current holding in a portfolio"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Asset information
    symbol = Column(String, nullable=False, index=True)
    asset_class = Column(Enum(AssetClass), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    
    # Position details
    quantity = Column(Float, nullable=False)
    average_entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    
    # Cost and value
    cost_basis = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    
    # Performance metrics
    unrealized_profit_loss = Column(Float, nullable=False, default=0.0)
    unrealized_profit_loss_percent = Column(Float, nullable=False, default=0.0)
    total_fees = Column(Float, nullable=False, default=0.0)
    
    # Risk management
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    max_risk_percent = Column(Float, nullable=True)
    position_size_percent = Column(Float, nullable=False, default=0.0)
    
    # Strategy information
    strategy_id = Column(Integer, ForeignKey("custom_strategies.id", ondelete="SET NULL"), nullable=True)
    strategy = relationship("CustomStrategy")
    
    # Timestamps
    opened_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    portfolio = relationship("Portfolio", back_populates="positions")
    trades = relationship("Trade", back_populates="position", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Position {self.symbol} {self.direction} {self.quantity} @ {self.average_entry_price} (ID: {self.id})>"
    
class Trade(Base):
    """Represents a trade execution in a portfolio"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Trade details
    symbol = Column(String, nullable=False, index=True)
    asset_class = Column(Enum(AssetClass), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    
    # Execution details
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Trade tracking
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN)
    fees = Column(Float, nullable=False, default=0.0)
    slippage = Column(Float, nullable=False, default=0.0)
    total_cost = Column(Float, nullable=False)
    
    # Trade calculations
    realized_profit_loss = Column(Float, nullable=True)
    realized_profit_loss_percent = Column(Float, nullable=True)
    
    # Trade metadata
    notes = Column(String, nullable=True)
    tags = Column(String, nullable=True)
    
    # Strategy information
    strategy_id = Column(Integer, ForeignKey("custom_strategies.id", ondelete="SET NULL"), nullable=True)
    strategy = relationship("CustomStrategy")
    
    # Trade relationships
    portfolio_id = Column(Integer, ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    portfolio = relationship("Portfolio", back_populates="trades")
    position_id = Column(Integer, ForeignKey("positions.id", ondelete="CASCADE"), nullable=True)
    position = relationship("Position", back_populates="trades")
    
    # If this trade is closing another one
    closing_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    closing_trades = relationship("Trade", backref="opening_trade", remote_side=[id])
    
    def __repr__(self):
        return f"<Trade {self.symbol} {self.direction} {self.quantity} @ {self.price} (ID: {self.id})>"
