from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Body
from sqlalchemy.orm import Session
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.core.middleware import requires_feature
from backend.models.user import User
from backend.models.portfolio import AssetClass, TradeDirection, Portfolio, Position, Trade
from backend.services.portfolio_service import PortfolioService
from backend.agents import RiskManagerAgent
from backend.agents.market_regime_detector import MarketRegimeDetector

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class PortfolioCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    starting_capital: float = Field(10000.0, gt=0)

class PortfolioUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

class PortfolioResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    starting_capital: float
    current_value: float
    cash_balance: float
    total_profit_loss: float
    daily_profit_loss: float
    win_rate: float
    profit_factor: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class PositionCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    asset_class: AssetClass
    direction: TradeDirection
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    strategy_id: Optional[int] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_risk_percent: Optional[float] = Field(None, gt=0, le=10)
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

class PositionResponse(BaseModel):
    id: int
    symbol: str
    asset_class: AssetClass
    direction: TradeDirection
    quantity: float
    average_entry_price: float
    current_price: float
    cost_basis: float
    market_value: float
    unrealized_profit_loss: float
    unrealized_profit_loss_percent: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    position_size_percent: float
    opened_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class PositionClose(BaseModel):
    quantity: Optional[float] = None
    exit_price: float = Field(..., gt=0)
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

class TradeResponse(BaseModel):
    id: int
    symbol: str
    asset_class: AssetClass
    direction: TradeDirection
    quantity: float
    price: float
    timestamp: datetime
    status: str
    fees: float
    total_cost: float
    realized_profit_loss: Optional[float]
    realized_profit_loss_percent: Optional[float]
    notes: Optional[str]
    tags: Optional[str]
    
    class Config:
        orm_mode = True

# Routes for portfolios
@router.post("/", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
@requires_feature("portfolio_management")
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new portfolio.
    Available to Pro and Enterprise subscription tiers.
    """
    try:
        portfolio = await PortfolioService.create_portfolio(
            db, 
            current_user, 
            portfolio_data.name, 
            portfolio_data.description, 
            portfolio_data.starting_capital
        )
        return portfolio
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating portfolio: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating portfolio: {str(e)}"
        )

@router.get("/", response_model=List[PortfolioResponse])
async def get_user_portfolios(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all portfolios for the current user.
    Available to all subscription tiers.
    """
    # For development, return mock data
    if os.getenv("ENV", "development") == "development":
        logger.info("Using mock portfolio data for development")
        mock_portfolios = [
            {
                "id": 1,
                "name": "Crypto Portfolio",
                "description": "Digital asset investments focusing on Bitcoin and Ethereum",
                "starting_capital": 10000.0,
                "current_value": 12500.0,
                "cash_balance": 5000.0,
                "total_profit_loss": 2500.0,
                "daily_profit_loss": 150.0,
                "win_rate": 0.65,
                "profit_factor": 2.1,
                "created_at": datetime.now() - timedelta(days=30),
                "updated_at": datetime.now()
            },
            {
                "id": 2,
                "name": "Stock Trading",
                "description": "US equities swing trading portfolio",
                "starting_capital": 25000.0,
                "current_value": 27300.0,
                "cash_balance": 15000.0,
                "total_profit_loss": 2300.0,
                "daily_profit_loss": 85.0,
                "win_rate": 0.58,
                "profit_factor": 1.7,
                "created_at": datetime.now() - timedelta(days=60),
                "updated_at": datetime.now()
            }
        ]
        return [PortfolioResponse(**portfolio) for portfolio in mock_portfolios]
    
    # For production, use the database
    return await PortfolioService.get_user_portfolios(db, current_user.id)

@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: int = Path(..., description="ID of the portfolio to retrieve"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific portfolio by ID.
    Users can only access their own portfolios.
    """
    return await PortfolioService.get_portfolio(db, portfolio_id, current_user.id)

@router.put("/{portfolio_id}", response_model=PortfolioResponse)
@requires_feature("portfolio_management")
async def update_portfolio(
    portfolio_data: PortfolioUpdate,
    portfolio_id: int = Path(..., description="ID of the portfolio to update"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a portfolio.
    Users can only update their own portfolios.
    Available to Pro and Enterprise subscription tiers.
    """
    return await PortfolioService.update_portfolio(
        db, 
        portfolio_id, 
        current_user.id, 
        portfolio_data.name, 
        portfolio_data.description
    )

@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
@requires_feature("portfolio_management")
async def delete_portfolio(
    portfolio_id: int = Path(..., description="ID of the portfolio to delete"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a portfolio.
    Users can only delete their own portfolios.
    Available to Pro and Enterprise subscription tiers.
    """
    await PortfolioService.delete_portfolio(db, portfolio_id, current_user.id)
    return None

# Routes for portfolio metrics and performance
@router.get("/{portfolio_id}/metrics", response_model=Dict[str, Any])
async def get_portfolio_metrics(
    portfolio_id: int = Path(..., description="ID of the portfolio"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Calculate performance metrics for a portfolio.
    Available to all subscription tiers.
    """
    return await PortfolioService.calculate_portfolio_metrics(db, portfolio_id, current_user.id)

@router.post("/{portfolio_id}/update-prices", response_model=Dict[str, Any])
@requires_feature("portfolio_management")
async def update_portfolio_prices(
    portfolio_id: int = Path(..., description="ID of the portfolio"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update current prices and metrics for all positions in a portfolio.
    Available to Pro and Enterprise subscription tiers.
    """
    return await PortfolioService.update_position_prices(db, portfolio_id, current_user.id)

# Routes for positions
@router.post("/{portfolio_id}/positions", response_model=PositionResponse, status_code=status.HTTP_201_CREATED)
@requires_feature("portfolio_management")
async def create_position(
    position_data: PositionCreate,
    portfolio_id: int = Path(..., description="ID of the portfolio"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new position in a portfolio by executing a trade.
    Available to Pro and Enterprise subscription tiers.
    """
    try:
        # Check if risk parameters are aligned with RiskManagerAgent rules
        if position_data.max_risk_percent is None:
            # Use default from RiskManagerAgent
            position_data.max_risk_percent = 2.0  # Default 2% risk per trade
        
        # Optional: Get market regime for additional context
        try:
            regime_detector = MarketRegimeDetector()
            market_regime = await regime_detector.detect_regime(
                position_data.symbol, 
                position_data.asset_class.value
            )
            logger.info(f"Current market regime for {position_data.symbol}: {market_regime}")
            
            # Could adjust risk parameters based on regime
            if "high_volatility" in market_regime or "bearish" in market_regime:
                # Reduce risk in volatile or bearish markets
                position_data.max_risk_percent = min(position_data.max_risk_percent, 1.5)
                logger.info(f"Adjusted risk parameters for {market_regime} regime")
        except Exception as e:
            logger.warning(f"Could not detect market regime: {str(e)}")
        
        position = await PortfolioService.create_position(
            db,
            portfolio_id,
            current_user.id,
            position_data.symbol,
            position_data.asset_class,
            position_data.direction,
            position_data.quantity,
            position_data.entry_price,
            position_data.strategy_id,
            position_data.stop_loss_price,
            position_data.take_profit_price,
            position_data.max_risk_percent,
            position_data.notes,
            position_data.tags
        )
        return position
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating position: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating position: {str(e)}"
        )

@router.get("/{portfolio_id}/positions", response_model=List[PositionResponse])
async def get_portfolio_positions(
    portfolio_id: int = Path(..., description="ID of the portfolio"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all positions in a portfolio.
    Users can only access positions in their own portfolios.
    """
    return await PortfolioService.get_portfolio_positions(db, portfolio_id, current_user.id)

@router.get("/positions/{position_id}", response_model=PositionResponse)
async def get_position(
    position_id: int = Path(..., description="ID of the position"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific position by ID.
    Users can only access positions in their own portfolios.
    """
    return await PortfolioService.get_position(db, position_id, current_user.id)

@router.post("/positions/{position_id}/close", response_model=TradeResponse)
@requires_feature("portfolio_management")
async def close_position(
    position_data: PositionClose,
    position_id: int = Path(..., description="ID of the position to close"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Close a position (fully or partially) by executing a sell trade.
    Available to Pro and Enterprise subscription tiers.
    """
    try:
        trade = await PortfolioService.close_position(
            db,
            position_id,
            current_user.id,
            position_data.exit_price,
            position_data.quantity,
            position_data.notes,
            position_data.tags
        )
        return trade
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error closing position: {str(e)}"
        )

# Routes for trades
@router.get("/{portfolio_id}/trades", response_model=List[TradeResponse])
async def get_portfolio_trades(
    portfolio_id: int = Path(..., description="ID of the portfolio"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get trades for a portfolio with pagination.
    Users can only access trades in their own portfolios.
    """
    return await PortfolioService.get_portfolio_trades(db, portfolio_id, current_user.id, limit, offset)

@router.get("/positions/{position_id}/trades", response_model=List[TradeResponse])
async def get_position_trades(
    position_id: int = Path(..., description="ID of the position"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all trades for a position.
    Users can only access trades for positions in their own portfolios.
    """
    return await PortfolioService.get_position_trades(db, position_id, current_user.id)

# Route for signal import
@router.post("/{portfolio_id}/import-signal", response_model=PositionResponse, status_code=status.HTTP_201_CREATED)
@requires_feature("portfolio_management")
async def import_trading_signal(
    signal_data: Dict[str, Any] = Body(...),
    portfolio_id: int = Path(..., description="ID of the portfolio"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Import a trading signal from SignalScannerAgent and create a position.
    The signal must be validated by RiskManagerAgent before import.
    Available to Enterprise subscription tier only.
    """
    try:
        # Create a RiskManagerAgent to validate the signal
        risk_manager = RiskManagerAgent()
        
        # Check if signal is already validated or needs validation
        if not signal_data.get("is_validated", False):
            # Validate the signal
            signal_data = await risk_manager.evaluate_signal(signal_data)
            
            if not signal_data.get("is_approved", False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Signal was rejected by risk management rules"
                )
        
        # Extract data from signal
        symbol = signal_data.get("asset", "")
        
        # Determine asset class
        asset_class_str = signal_data.get("asset_class", "")
        if asset_class_str.upper() in [e.value for e in AssetClass]:
            asset_class = AssetClass(asset_class_str.upper())
        else:
            # Try to determine from symbol
            if "-USD" in symbol:
                asset_class = AssetClass.CRYPTO
            elif "=" in symbol or symbol.endswith("USD"):
                asset_class = AssetClass.FOREX
            else:
                asset_class = AssetClass.STOCK
        
        # Determine direction
        signal_type = signal_data.get("signal_type", "").upper()
        if "LONG" in signal_type or "BUY" in signal_type:
            direction = TradeDirection.LONG
        elif "SHORT" in signal_type or "SELL" in signal_type:
            direction = TradeDirection.SHORT
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to determine trade direction from signal"
            )
        
        # Extract price and risk parameters
        entry_price = signal_data.get("price", 0)
        if not entry_price:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signal does not contain a valid price"
            )
        
        # Use risk parameters from signal if available
        position_size = signal_data.get("position_size", 0)
        stop_loss = signal_data.get("stop_loss", None)
        take_profit = signal_data.get("take_profit", None)
        max_risk_percent = signal_data.get("risk_percent", 2.0)
        
        # Get strategy ID if available
        strategy_id = signal_data.get("strategy_id", None)
        
        # Create notes from signal metadata
        notes = f"Imported signal from {signal_data.get('strategy_name', 'Trading SaaS')}. "
        if "regime" in signal_data:
            notes += f"Market regime: {signal_data.get('regime')}. "
        if "confidence" in signal_data:
            notes += f"Signal confidence: {signal_data.get('confidence')}%. "
        
        # Create tags from signal metadata
        tags = ["imported"]
        if "timeframe" in signal_data:
            tags.append(f"timeframe:{signal_data.get('timeframe')}")
        if "strategy_name" in signal_data:
            tags.append(f"strategy:{signal_data.get('strategy_name')}")
        
        # Calculate quantity from position size and entry price
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, current_user.id)
        position_value = portfolio.current_value * (position_size / 100) if position_size else portfolio.current_value * 0.02
        quantity = position_value / entry_price
        
        # Create position
        position = await PortfolioService.create_position(
            db,
            portfolio_id,
            current_user.id,
            symbol,
            asset_class,
            direction,
            quantity,
            entry_price,
            strategy_id,
            stop_loss,
            take_profit,
            max_risk_percent,
            notes,
            tags
        )
        
        return position
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error importing signal: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error importing signal: {str(e)}"
        )
