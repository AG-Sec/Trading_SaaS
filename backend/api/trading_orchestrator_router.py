from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.core.middleware import requires_feature
from backend.models.user import User
from backend.services.trading_orchestrator import TradingOrchestrator
from backend.services.portfolio_service import PortfolioService

import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize orchestrator
orchestrator = TradingOrchestrator()

# Pydantic models
class OrchestrationRequest(BaseModel):
    portfolio_id: Optional[int] = None
    auto_execute: bool = Field(False, description="Whether to automatically execute trades")
    timeframes: List[str] = Field(["1h", "4h", "1d"], description="Timeframes to scan")
    asset_classes: List[str] = Field(["STOCK", "CRYPTO", "ETF", "FOREX"], description="Asset classes to include")

class SignalResponse(BaseModel):
    signals: List[Dict[str, Any]]
    executed_count: int
    message: str

class AutomationSettingsRequest(BaseModel):
    enabled: bool = Field(..., description="Whether automation is enabled")
    portfolio_id: Optional[int] = Field(None, description="Portfolio to execute trades in")
    run_frequency: str = Field("daily", description="How often to run (hourly, daily, etc)")
    max_trades_per_day: int = Field(3, description="Maximum trades to execute per day")
    preferred_timeframes: List[str] = Field(["1h", "4h", "1d"], description="Preferred timeframes")
    preferred_assets: List[str] = Field(None, description="Preferred assets to trade")
    risk_profile: str = Field("moderate", description="Risk profile (conservative, moderate, aggressive)")

class AutomationSettingsResponse(BaseModel):
    settings: Dict[str, Any]
    message: str

@router.post("/scan", response_model=SignalResponse)
@requires_feature("signal_scanning")
async def scan_for_signals(
    request: OrchestrationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Scan for trading signals and optionally execute them in a portfolio.
    This is an on-demand scan triggered by the user.
    """
    try:
        # Process signals for the user
        signals = await orchestrator.process_signals_for_user(
            db=db,
            user=current_user,
            portfolio_id=request.portfolio_id,
            auto_execute=request.auto_execute
        )
        
        # Count executed signals
        executed_count = sum(1 for signal in signals if signal.get("signal_id", "") in 
                           orchestrator._signal_status and 
                           orchestrator._signal_status[signal.get("signal_id", "")] == "executed")
        
        # Prepare response message
        if len(signals) == 0:
            message = "No signals detected at this time."
        elif request.auto_execute and request.portfolio_id:
            message = f"Found {len(signals)} signals. {executed_count} were automatically executed in your portfolio."
        else:
            message = f"Found {len(signals)} signals. No automatic execution was performed."
        
        return {
            "signals": signals,
            "executed_count": executed_count,
            "message": message
        }
    except Exception as e:
        logger.error(f"Error scanning for signals: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/simulate", response_model=SignalResponse)
@requires_feature("strategy_simulation")
async def simulate_signals(
    request: OrchestrationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Simulate signal generation and execution without actually creating positions.
    Useful for testing strategies before putting real money at risk.
    """
    try:
        # Process signals but don't execute
        signals = await orchestrator.process_signals_for_user(
            db=db,
            user=current_user,
            auto_execute=False
        )
        
        # Calculate potential impact
        portfolio = None
        executed_count = 0
        
        if request.portfolio_id:
            try:
                portfolio = await PortfolioService.get_portfolio(db, request.portfolio_id, current_user.id)
                executed_count = len(signals)
            except Exception as e:
                logger.error(f"Error getting portfolio: {str(e)}")
        
        # Prepare response message
        if len(signals) == 0:
            message = "No signals detected for simulation."
        elif portfolio:
            message = f"Found {len(signals)} signals. Simulated impact on portfolio '{portfolio.name}' calculated."
        else:
            message = f"Found {len(signals)} signals. No portfolio selected for impact simulation."
        
        return {
            "signals": signals,
            "executed_count": executed_count,
            "message": message
        }
    except Exception as e:
        logger.error(f"Error simulating signals: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/settings", response_model=AutomationSettingsResponse)
@requires_feature("automated_trading")
async def get_automation_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the current automation settings for the user.
    """
    # TODO: Implement actual settings storage
    # For now, return default settings
    return {
        "settings": {
            "enabled": False,
            "portfolio_id": None,
            "run_frequency": "daily",
            "max_trades_per_day": 3,
            "preferred_timeframes": ["1h", "4h", "1d"],
            "preferred_assets": ["BTC-USD", "ETH-USD", "SPY", "AAPL"],
            "risk_profile": "moderate"
        },
        "message": "Retrieved automation settings."
    }

@router.post("/settings", response_model=AutomationSettingsResponse)
@requires_feature("automated_trading")
async def update_automation_settings(
    settings: AutomationSettingsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update the automation settings for the user.
    """
    # Validate portfolio if provided
    if settings.enabled and settings.portfolio_id:
        try:
            portfolio = await PortfolioService.get_portfolio(db, settings.portfolio_id, current_user.id)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid portfolio_id: {str(e)}"
            )
    
    # TODO: Implement actual settings storage
    # For now, just return the provided settings
    return {
        "settings": settings.dict(),
        "message": "Automation settings updated successfully."
    }

@router.post("/backfill/{portfolio_id}", response_model=Dict[str, Any])
@requires_feature("portfolio_management")
async def backfill_portfolio_data(
    portfolio_id: int = Path(..., description="ID of the portfolio"),
    days: int = Query(30, description="Number of days to backfill"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Backfill a portfolio with historical pricing data.
    Useful for updating portfolio metrics after adding positions.
    """
    try:
        # Check if portfolio exists and user has access
        portfolio = await PortfolioService.get_portfolio(db, portfolio_id, current_user.id)
        
        # Update prices for all positions
        result = await PortfolioService.update_position_prices(db, portfolio_id, current_user.id)
        
        return {
            "status": "success",
            "message": f"Successfully backfilled pricing data for portfolio '{portfolio.name}'",
            "updated_positions": len(result.get("positions", [])),
            "portfolio_value": portfolio.current_value
        }
    except Exception as e:
        logger.error(f"Error backfilling portfolio data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
