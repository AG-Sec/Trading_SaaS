from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from sqlalchemy.orm import Session
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.core.middleware import requires_feature
from backend.models.user import User
from backend.models.strategy import StrategyCreate, StrategyUpdate, StrategyResponse, BacktestResponse, StrategyPreset
from backend.services.strategy_service import StrategyService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/presets", response_model=List[Dict[str, Any]])
async def get_strategy_presets(current_user: User = Depends(get_current_user)):
    """
    Get available strategy presets based on user's subscription tier.
    All tiers have access to at least the basic presets.
    """
    return await StrategyService.get_strategy_presets(current_user)


@router.post("/", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
@requires_feature("strategy_customization")
async def create_strategy(
    strategy_data: StrategyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new custom trading strategy.
    Requires Enterprise subscription tier.
    """
    try:
        strategy = await StrategyService.create_strategy(db, current_user, strategy_data)
        return strategy
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating strategy: {str(e)}"
        )


@router.get("/", response_model=List[StrategyResponse])
async def get_user_strategies(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all custom strategies created by the current user.
    Available to all subscription tiers.
    """
    return await StrategyService.get_user_strategies(db, current_user.id)


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: int = Path(..., description="ID of the strategy to retrieve"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific custom strategy by ID.
    Users can only access their own strategies or public strategies.
    """
    return await StrategyService.get_strategy(db, strategy_id, current_user.id)


@router.put("/{strategy_id}", response_model=StrategyResponse)
@requires_feature("strategy_customization")
async def update_strategy(
    strategy_data: StrategyUpdate,
    strategy_id: int = Path(..., description="ID of the strategy to update"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a custom strategy.
    Users can only update their own strategies.
    Requires Enterprise subscription tier.
    """
    return await StrategyService.update_strategy(db, strategy_id, current_user.id, strategy_data)


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: int = Path(..., description="ID of the strategy to delete"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a custom strategy.
    Users can only delete their own strategies.
    """
    await StrategyService.delete_strategy(db, strategy_id, current_user.id)
    return None


@router.post("/validate", response_model=Dict[str, Any])
@requires_feature("strategy_customization")
async def validate_dsl(
    dsl_content: Dict[str, str],
    current_user: User = Depends(get_current_user)
):
    """
    Validate DSL content for syntax and semantic errors.
    Returns the parsed DSL if valid.
    Requires Enterprise subscription tier.
    """
    try:
        from backend.services.dsl_parser import DSLParser
        parsed_dsl = DSLParser.parse(dsl_content["content"])
        return {"valid": True, "parsed": parsed_dsl}
    except HTTPException as e:
        return {"valid": False, "error": e.detail}
    except Exception as e:
        logger.error(f"Error validating DSL: {str(e)}")
        return {"valid": False, "error": str(e)}


@router.post("/generate-code", response_model=Dict[str, str])
@requires_feature("strategy_customization")
async def generate_strategy_code(
    dsl_content: Dict[str, str],
    current_user: User = Depends(get_current_user)
):
    """
    Generate Python code from DSL content.
    Requires Enterprise subscription tier.
    """
    python_code = await StrategyService.generate_strategy_code(dsl_content["content"])
    return {"code": python_code}


@router.post("/backtest", response_model=BacktestResponse)
@requires_feature("strategy_customization")
async def backtest_strategy(
    backtest_params: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Backtest a strategy with historical data.
    Requires Enterprise subscription tier.
    """
    strategy_id = backtest_params.get("strategy_id")
    dsl_content = backtest_params.get("dsl_content", "")
    asset_symbol = backtest_params.get("asset_symbol")
    timeframe = backtest_params.get("timeframe")
    start_date = backtest_params.get("start_date")
    end_date = backtest_params.get("end_date")
    
    if not dsl_content and not strategy_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either strategy_id or dsl_content must be provided"
        )
    
    if not asset_symbol or not timeframe or not start_date or not end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="asset_symbol, timeframe, start_date, and end_date are required"
        )
    
    try:
        result = await StrategyService.backtest_strategy(
            db,
            current_user,
            strategy_id,
            dsl_content,
            asset_symbol,
            timeframe,
            start_date,
            end_date
        )
        return result
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error backtesting strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error backtesting strategy: {str(e)}"
        )
