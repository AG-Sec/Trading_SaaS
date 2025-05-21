from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class AssetSymbol(str, Enum):
    BTC_USD = "BTC-USD"
    ETH_USD = "ETH-USD"
    EUR_USD_FX = "EURUSD=X"  # Forex, =X suffix in yfinance
    SPY = "SPY"  # S&P 500 ETF

class Timeframe(str, Enum):
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"

class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"  # Stop Market
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    NEW = "new"
    PENDING_SUBMIT = "pending_submit" # Submitted to internal system, but not yet to broker
    SUBMITTED = "submitted" # Submitted to broker
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    PENDING_CANCEL = "pending_cancel"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"

class CandleModel(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp of the candle's open time (UTC)")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(..., description="Volume")

class HistoricalDataModel(BaseModel):
    asset: AssetSymbol
    timeframe: Timeframe
    candles: List[CandleModel] = Field(default_factory=list)

class TradingSignalModel(BaseModel):
    signal_id: str = Field(..., description="Unique identifier for the signal (e.g., UUID)")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when the signal was generated (UTC)")
    asset: AssetSymbol
    timeframe: Timeframe
    signal_type: SignalType
    entry_price: Optional[float] = Field(None, description="Suggested entry price based on signal logic")
    stop_loss: Optional[float] = Field(None, description="Suggested stop loss price based on signal logic")
    take_profit: Optional[float] = Field(None, description="Suggested take profit price based on signal logic")
    
    # Fields to be added/updated by RiskManagerAgent
    risk_reward_ratio: Optional[float] = Field(None, description="Calculated risk-to-reward ratio")
    position_size_usd: Optional[float] = Field(None, description="Calculated position size in USD based on risk parameters")
    position_size_asset: Optional[float] = Field(None, description="Calculated position size in asset units")
    risk_per_trade_usd: Optional[float] = Field(None, description="Calculated monetary risk for this trade in USD")
    
    # Optional field for strategy specific metadata or raw indicator values
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata, e.g., indicator values, strategy name")

class OrderModel(BaseModel):
    order_id: str = Field(..., description="Unique identifier for the order (client-generated UUID recommended before submission)")
    signal_id: Optional[str] = Field(None, description="Signal ID that triggered this order, if applicable")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when the order object was created locally (UTC)")
    asset: AssetSymbol
    order_type: OrderType
    side: OrderSide
    quantity_asset: float = Field(..., description="Order quantity in terms of the base asset (e.g., number of shares, amount of BTC)")
    limit_price: Optional[float] = Field(None, description="Limit price for LIMIT or STOP_LIMIT orders")
    stop_price: Optional[float] = Field(None, description="Stop price for STOP or STOP_LIMIT orders (trigger price for stop orders)")
    status: OrderStatus = Field(default=OrderStatus.NEW)
    # Broker-specific fields to be added upon submission/confirmation
    broker_order_id: Optional[str] = Field(None, description="Order ID provided by the broker upon successful submission")
    filled_quantity_asset: Optional[float] = Field(0.0, description="Quantity of the asset that has been filled")
    average_fill_price: Optional[float] = Field(None, description="Average price at which the order was filled")
    commission_usd: Optional[float] = Field(None, description="Commission paid for this order in USD")
    last_update_at: Optional[datetime] = Field(None, description="Timestamp of the last update from the broker (UTC)")

class TradeModel(BaseModel):
    trade_id: str = Field(..., description="Unique identifier for the trade (e.g., UUID)")
    # Typically, a trade results from one or more order fills. For simplicity, can link to an initiating order.
    originating_order_id: str = Field(..., description="The primary order ID that opened this trade")
    asset: AssetSymbol
    side: OrderSide # Same as the originating order's side
    entry_price: float = Field(..., description="Weighted average entry price of the trade")
    exit_price: Optional[float] = Field(None, description="Weighted average exit price of the trade (if closed)")
    quantity_asset: float = Field(..., description="Total quantity of the asset traded for this position")
    entry_time: datetime = Field(..., description="Timestamp of trade entry (first fill, UTC)")
    exit_time: Optional[datetime] = Field(None, description="Timestamp of trade exit (last fill for closing, UTC)")
    status: TradeStatus = Field(default=TradeStatus.OPEN)
    pnl_usd: Optional[float] = Field(None, description="Realized Profit or Loss in USD (if closed)")
    commission_usd: Optional[float] = Field(0.0, description="Total commission paid for entry and exit of this trade in USD")
    # Market regime information
    market_regime: Optional[str] = Field(None, description="Market regime type during trade entry")
    regime_strength: Optional[float] = Field(None, description="Strength of the market regime during trade entry")
    # Could add: stop_loss_at_entry, take_profit_at_entry, closing_reason, etc.

# For user-defined strategies (Phase 1 MVP - simple representation)
class StrategyParameterModel(BaseModel):
    name: str = Field(..., description="Name of the strategy parameter, e.g., 'EMA_Fast_Period'")
    value: Any = Field(..., description="Value of the parameter, can be int, float, str, bool")
    # data_type: str # Optional: 'int', 'float', 'str', 'bool' for UI generation

class UserStrategyConfigModel(BaseModel):
    config_id: str = Field(..., description="Unique identifier for this specific configuration (e.g., UUID)")
    strategy_template_id: str = Field(..., description="Identifier for the pre-made strategy recipe/template being used")
    user_given_name: Optional[str] = Field(None, description="User-friendly name for this strategy configuration")
    asset: AssetSymbol
    timeframe: Timeframe
    parameters: List[StrategyParameterModel] = Field(default_factory=list, description="List of parameters overriding template defaults")
    # Logic (e.g., AND/OR for combining conditions) would be part of the strategy_template_id's definition
    # or could be explicitly set here if the templates are highly flexible.
    is_active: bool = Field(default=False, description="Whether this strategy configuration is currently active for live scanning")
