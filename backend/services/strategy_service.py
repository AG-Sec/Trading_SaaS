from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
import logging
from typing import List, Dict, Any, Optional
import os
import importlib.util
import sys
from pathlib import Path
import tempfile
import uuid
import json

from backend.models.strategy import CustomStrategy, StrategyType, BacktestResult, StrategyCreate, StrategyUpdate
from backend.models.user import User, SubscriptionTier
from backend.services.dsl_parser import DSLParser
from backend.core.middleware import get_tier_limit

# Configure logging
logger = logging.getLogger(__name__)

class StrategyService:
    """Service for managing custom trading strategies"""
    
    # Strategy presets available for all users
    STRATEGY_PRESETS = [
        {
            "id": "preset_ema_crossover",
            "name": "EMA Crossover",
            "description": "Basic trend following strategy using EMA crossovers",
            "strategy_type": StrategyType.TREND_FOLLOWING,
            "tier_required": SubscriptionTier.BASIC,
            "dsl_content": """strategy_name: ema_crossover
rules:
  - if: ema(9) > ema(21)
    and: ema(9) > ema(9).shift(1)
    then: signal = "long"
  - if: ema(9) < ema(21)
    and: ema(9) < ema(9).shift(1)
    then: signal = "short"
risk:
  entry_size_pct: 1.0
  stop: atr(14) * 2.0
  take_profit: 3R"""
        },
        {
            "id": "preset_rsi_mean_reversion",
            "name": "RSI Mean Reversion",
            "description": "Mean reversion strategy using RSI oversold/overbought conditions",
            "strategy_type": StrategyType.MEAN_REVERSION,
            "tier_required": SubscriptionTier.BASIC,
            "dsl_content": """strategy_name: rsi_mean_reversion
rules:
  - if: rsi(14) < 30
    and: close > close.shift(1)
    then: signal = "long"
  - if: rsi(14) > 70
    and: close < close.shift(1)
    then: signal = "short"
risk:
  entry_size_pct: 1.0
  stop: atr(14) * 1.5
  take_profit: 2R"""
        },
        {
            "id": "preset_breakout",
            "name": "Breakout Strategy",
            "description": "Volatility breakout strategy using ATR and previous highs/lows",
            "strategy_type": StrategyType.BREAKOUT,
            "tier_required": SubscriptionTier.PRO,
            "dsl_content": """strategy_name: volatility_breakout
rules:
  - if: close > highest(20).shift(1)
    and: atr(14) > atr(14).rolling(5).mean()
    then: signal = "long"
  - if: close < lowest(20).shift(1)
    and: atr(14) > atr(14).rolling(5).mean()
    then: signal = "short"
risk:
  entry_size_pct: 1.0
  stop: atr(14) * 2.0
  take_profit: 2.5R"""
        },
        {
            "id": "preset_multi_timeframe",
            "name": "Multi-Timeframe Confirmation",
            "description": "Advanced strategy using multi-timeframe confirmation",
            "strategy_type": StrategyType.TREND_FOLLOWING,
            "tier_required": SubscriptionTier.ENTERPRISE,
            "dsl_content": """strategy_name: multi_timeframe_confirmation
rules:
  - if: ema(9) > ema(21)
    and: rsi(14) > 50
    and: higher_tf_trend == "bullish"
    then: signal = "long"
  - if: ema(9) < ema(21)
    and: rsi(14) < 50
    and: higher_tf_trend == "bearish"
    then: signal = "short"
risk:
  entry_size_pct: 1.0
  stop: atr(14) * 1.5
  take_profit: 3R
  max_trades: 3
  max_daily_drawdown_pct: 3.0"""
        }
    ]
    
    @staticmethod
    async def get_strategy_presets(user: User) -> List[Dict[str, Any]]:
        """Get available strategy presets for a user based on their subscription tier"""
        tier_values = {
            SubscriptionTier.BASIC: 1,
            SubscriptionTier.PRO: 2,
            SubscriptionTier.ENTERPRISE: 3
        }
        
        user_tier_value = tier_values.get(user.subscription_tier, 0)
        
        # Filter presets based on user's subscription tier
        available_presets = []
        for preset in StrategyService.STRATEGY_PRESETS:
            preset_tier_value = tier_values.get(preset["tier_required"], 3)
            if user_tier_value >= preset_tier_value:
                available_presets.append(preset)
                
        return available_presets
    
    @staticmethod
    async def create_strategy(db: Session, user: User, strategy_data: StrategyCreate) -> CustomStrategy:
        """Create a new custom strategy for a user"""
        # Check if user has reached their strategy limit
        user_strategies = db.query(CustomStrategy).filter(CustomStrategy.user_id == user.id).all()
        max_strategies = await get_tier_limit(user, "max_custom_strategies")
        
        if len(user_strategies) >= max_strategies:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You have reached your limit of {max_strategies} custom strategies. Upgrade your subscription to create more."
            )
        
        # Validate DSL content
        try:
            parsed_dsl = DSLParser.parse(strategy_data.dsl_content)
            
            # Create new strategy
            db_strategy = CustomStrategy(
                user_id=user.id,
                name=strategy_data.name,
                description=strategy_data.description,
                strategy_type=strategy_data.strategy_type,
                dsl_content=strategy_data.dsl_content,
                is_public=strategy_data.is_public,
                parameters=strategy_data.parameters or {}
            )
            
            db.add(db_strategy)
            db.commit()
            db.refresh(db_strategy)
            
            logger.info(f"Created custom strategy '{strategy_data.name}' for user ID {user.id}")
            return db_strategy
            
        except HTTPException as e:
            # Re-raise HTTP exceptions from parser
            raise e
        except Exception as e:
            logger.error(f"Error creating strategy: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating strategy: {str(e)}"
            )
    
    @staticmethod
    async def get_user_strategies(db: Session, user_id: int) -> List[CustomStrategy]:
        """Get all custom strategies for a user"""
        return db.query(CustomStrategy).filter(CustomStrategy.user_id == user_id).all()
    
    @staticmethod
    async def get_strategy(db: Session, strategy_id: int, user_id: int) -> CustomStrategy:
        """Get a specific custom strategy"""
        strategy = db.query(CustomStrategy).filter(
            CustomStrategy.id == strategy_id
        ).first()
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        # Check ownership or if strategy is public
        if strategy.user_id != user_id and not strategy.is_public:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to view this strategy"
            )
        
        return strategy
    
    @staticmethod
    async def update_strategy(db: Session, strategy_id: int, user_id: int, strategy_data: StrategyUpdate) -> CustomStrategy:
        """Update a custom strategy"""
        strategy = db.query(CustomStrategy).filter(
            CustomStrategy.id == strategy_id,
            CustomStrategy.user_id == user_id
        ).first()
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found or you do not have permission to update it"
            )
        
        # Update strategy fields if provided
        if strategy_data.name is not None:
            strategy.name = strategy_data.name
            
        if strategy_data.description is not None:
            strategy.description = strategy_data.description
            
        if strategy_data.strategy_type is not None:
            strategy.strategy_type = strategy_data.strategy_type
            
        if strategy_data.is_active is not None:
            strategy.is_active = strategy_data.is_active
            
        if strategy_data.is_public is not None:
            strategy.is_public = strategy_data.is_public
            
        if strategy_data.parameters is not None:
            strategy.parameters = strategy_data.parameters
            
        # If DSL content is provided, validate it
        if strategy_data.dsl_content is not None:
            try:
                parsed_dsl = DSLParser.parse(strategy_data.dsl_content)
                strategy.dsl_content = strategy_data.dsl_content
            except HTTPException as e:
                # Re-raise HTTP exceptions from parser
                raise e
        
        try:
            db.commit()
            db.refresh(strategy)
            
            logger.info(f"Updated strategy ID {strategy_id} for user ID {user_id}")
            return strategy
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating strategy: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating strategy: {str(e)}"
            )
    
    @staticmethod
    async def delete_strategy(db: Session, strategy_id: int, user_id: int) -> bool:
        """Delete a custom strategy"""
        strategy = db.query(CustomStrategy).filter(
            CustomStrategy.id == strategy_id,
            CustomStrategy.user_id == user_id
        ).first()
        
        if not strategy:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found or you do not have permission to delete it"
            )
        
        try:
            db.delete(strategy)
            db.commit()
            
            logger.info(f"Deleted strategy ID {strategy_id} for user ID {user_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting strategy: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting strategy: {str(e)}"
            )
    
    @staticmethod
    async def generate_strategy_code(dsl_content: str) -> str:
        """Generate Python code from DSL content"""
        try:
            parsed_dsl = DSLParser.parse(dsl_content)
            python_code = DSLParser.to_python_code(parsed_dsl)
            return python_code
        except HTTPException as e:
            # Re-raise HTTP exceptions from parser
            raise e
        except Exception as e:
            logger.error(f"Error generating strategy code: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating strategy code: {str(e)}"
            )
    
    @staticmethod
    async def backtest_strategy(
        db: Session, 
        user: User, 
        strategy_id: Optional[int], 
        dsl_content: str,
        asset_symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Backtest a strategy with historical data.
        
        If strategy_id is provided, the strategy will be loaded from the database.
        Otherwise, dsl_content must be provided.
        """
        try:
            # Get strategy if ID is provided
            if strategy_id:
                strategy = await StrategyService.get_strategy(db, strategy_id, user.id)
                dsl_content = strategy.dsl_content
            
            # Validate DSL content
            parsed_dsl = DSLParser.parse(dsl_content)
            
            # Generate Python code from DSL
            strategy_code = DSLParser.to_python_code(parsed_dsl)
            
            # Create a temporary module to load the strategy
            module_name = f"temp_strategy_{uuid.uuid4().hex}"
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"{module_name}.py")
            
            # Write code to temporary file
            with open(temp_file_path, 'w') as f:
                f.write(strategy_code)
            
            # Import the temporary module
            try:
                spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # Get the class
                    strategy_class_name = f"{parsed_dsl['strategy_name'].title().replace('_', '')}Strategy"
                    strategy_class = getattr(module, strategy_class_name)
                    
                    # Create instance of the strategy
                    strategy_instance = strategy_class()
                    
                    # Here you would call your actual backtesting engine
                    # For now, we'll mock the backtest results
                    backtest_results = StrategyService._mock_backtest_results(
                        strategy_id, 
                        asset_symbol,
                        timeframe,
                        start_date,
                        end_date
                    )
                    
                    # Save results to database if strategy_id is provided
                    if strategy_id:
                        db_backtest_result = BacktestResult(
                            strategy_id=strategy_id,
                            asset_symbol=asset_symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            win_rate=backtest_results["win_rate"],
                            profit_factor=backtest_results["profit_factor"],
                            max_drawdown=backtest_results["max_drawdown"],
                            total_trades=backtest_results["total_trades"],
                            metrics=backtest_results["metrics"],
                            equity_curve=backtest_results["equity_curve"]
                        )
                        
                        db.add(db_backtest_result)
                        db.commit()
                        db.refresh(db_backtest_result)
                        backtest_results["id"] = db_backtest_result.id
                    
                    return backtest_results
            finally:
                # Clean up temporary file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                
                # Clean up module from sys.modules
                if module_name in sys.modules:
                    del sys.modules[module_name]
        except HTTPException as e:
            # Re-raise HTTP exceptions
            raise e
        except Exception as e:
            logger.error(f"Error backtesting strategy: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error backtesting strategy: {str(e)}"
            )
    
    @staticmethod
    def _mock_backtest_results(
        strategy_id: Optional[int],
        asset_symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Mock backtest results for demonstration purposes.
        
        In a real implementation, this would call the actual backtesting engine.
        """
        import random
        from datetime import datetime, timedelta
        
        # Generate random realistic backtest results
        win_rate = random.uniform(40.0, 70.0)
        profit_factor = random.uniform(1.1, 2.5)
        max_drawdown = random.uniform(5.0, 25.0)
        total_trades = random.randint(20, 200)
        
        # Generate random equity curve
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        days = (end - start).days
        
        equity_curve = []
        equity = 10000.0  # Start with $10,000
        
        # Generate daily equity values
        current_date = start
        while current_date <= end:
            # Random daily change between -3% and +3%
            daily_change = random.uniform(-0.03, 0.03)
            equity *= (1 + daily_change)
            
            equity_curve.append({
                "date": current_date.isoformat(),
                "equity": round(equity, 2)
            })
            
            current_date += timedelta(days=1)
        
        # Additional metrics
        metrics = {
            "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
            "sortino_ratio": round(random.uniform(0.8, 3.0), 2),
            "average_gain": round(random.uniform(1.0, 5.0), 2),
            "average_loss": round(random.uniform(0.5, 2.0), 2),
            "win_loss_ratio": round(win_rate / (100 - win_rate), 2),
            "avg_holding_period": round(random.uniform(2.0, 10.0), 1),
            "largest_win": round(random.uniform(5.0, 15.0), 2),
            "largest_loss": round(random.uniform(2.0, 8.0), 2),
        }
        
        return {
            "strategy_id": strategy_id,
            "asset_symbol": asset_symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_drawdown, 2),
            "total_trades": total_trades,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "total_return": round((equity_curve[-1]["equity"] / equity_curve[0]["equity"] - 1) * 100, 2),
            "annualized_return": round(((equity_curve[-1]["equity"] / equity_curve[0]["equity"]) ** (365 / days) - 1) * 100, 2) if days > 0 else 0
        }
