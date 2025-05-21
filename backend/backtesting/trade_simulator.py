"""
Trade Simulator for Backtesting Module

This module simulates trade execution with realistic conditions including
slippage, commissions, liquidity constraints, and partial fills.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import random

from shared_types.models import (
    AssetSymbol, Timeframe, TradingSignalModel, 
    HistoricalDataModel, SignalType
)
from backend.backtesting.backtester import BacktestTrade, TradeStatus

logger = logging.getLogger(__name__)

class ExecutionModel(str, Enum):
    """Models for simulating execution quality."""
    PERFECT = "perfect"  # Perfect execution at requested price
    SIMPLE_SLIPPAGE = "simple_slippage"  # Fixed percentage slippage
    VOLUME_BASED = "volume_based"  # Slippage based on position size vs volume
    REALISTIC = "realistic"  # Combination of factors including volatility, spread, etc.


class LiquidityModel(str, Enum):
    """Models for simulating market liquidity."""
    INFINITE = "infinite"  # Always execute full size
    VOLUME_RATIO = "volume_ratio"  # Limit based on % of volume
    MARKET_IMPACT = "market_impact"  # Price impact based on order size


class TradeSimulator:
    """
    Simulates trade execution with realistic market conditions.
    
    This class provides functionality to:
    1. Apply realistic slippage to entries and exits
    2. Account for commissions and fees
    3. Model liquidity constraints and partial fills
    4. Simulate market impact for larger orders
    """
    
    def __init__(self, 
                execution_model: ExecutionModel = ExecutionModel.SIMPLE_SLIPPAGE,
                liquidity_model: LiquidityModel = LiquidityModel.VOLUME_RATIO,
                slippage_factor: float = 0.0005,  # 0.05% default slippage
                commission_rate: float = 0.001,  # 0.1% commission
                spread_factor: float = 0.0002,  # 0.02% spread
                max_volume_ratio: float = 0.1,  # Max 10% of volume for any order
                market_impact_factor: float = 0.2,  # Market impact coefficient
                random_seed: Optional[int] = None):
        """
        Initialize the TradeSimulator.
        
        Args:
            execution_model: Model for execution quality
            liquidity_model: Model for market liquidity
            slippage_factor: Base slippage as percentage
            commission_rate: Commission rate as percentage
            spread_factor: Bid-ask spread as percentage
            max_volume_ratio: Maximum ratio of order size to volume
            market_impact_factor: Coefficient for market impact calculation
            random_seed: Seed for random number generation
        """
        self.execution_model = execution_model
        self.liquidity_model = liquidity_model
        self.slippage_factor = slippage_factor
        self.commission_rate = commission_rate
        self.spread_factor = spread_factor
        self.max_volume_ratio = max_volume_ratio
        self.market_impact_factor = market_impact_factor
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def simulate_entry(self, 
                     signal: TradingSignalModel, 
                     candle: pd.Series, 
                     quantity: float) -> Tuple[BacktestTrade, float]:
        """
        Simulate entry execution for a signal.
        
        Args:
            signal: Trading signal to execute
            candle: Current price candle
            quantity: Position size to execute
            
        Returns:
            Tuple of (BacktestTrade, actual_quantity)
        """
        # Apply execution model to calculate actual entry price
        entry_price = self._calculate_execution_price(
            requested_price=signal.entry_price,
            signal_type=signal.signal_type,
            candle=candle,
            is_entry=True
        )
        
        # Apply liquidity model to calculate actual quantity
        actual_quantity = self._calculate_execution_quantity(
            requested_quantity=quantity,
            candle=candle,
            asset=signal.asset
        )
        
        # Create trade object
        trade = BacktestTrade(
            trade_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            asset=signal.asset,
            timeframe=signal.timeframe,
            signal_type=signal.signal_type,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=candle['timestamp'],
            quantity=actual_quantity,
            status=TradeStatus.OPEN,
            metadata=signal.metadata.copy() if signal.metadata else {}
        )
        
        return trade, actual_quantity
    
    def simulate_exit(self, 
                    trade: BacktestTrade, 
                    candle: pd.Series,
                    exit_type: TradeStatus) -> float:
        """
        Simulate exit execution for a trade.
        
        Args:
            trade: Trade to exit
            candle: Current price candle
            exit_type: Type of exit (take profit, stop loss, etc.)
            
        Returns:
            Actual exit price
        """
        # Determine requested exit price based on exit type
        if exit_type == TradeStatus.CLOSED_TP:
            requested_price = trade.take_profit
        elif exit_type == TradeStatus.CLOSED_SL:
            requested_price = trade.stop_loss
        else:
            # For manual or signal exits, use current price
            requested_price = candle['close']
        
        # Apply execution model to calculate actual exit price
        exit_price = self._calculate_execution_price(
            requested_price=requested_price,
            signal_type=trade.signal_type,
            candle=candle,
            is_entry=False
        )
        
        return exit_price
    
    def calculate_trade_pnl(self, 
                          trade: BacktestTrade, 
                          include_commission: bool = True) -> Tuple[float, float]:
        """
        Calculate P&L for a trade.
        
        Args:
            trade: Completed trade
            include_commission: Whether to include commission in the calculation
            
        Returns:
            Tuple of (absolute_pnl, percentage_pnl)
        """
        if trade.exit_price is None:
            return 0.0, 0.0
        
        # Calculate raw P&L
        if trade.signal_type.value == "long":
            pnl_absolute = trade.quantity * (trade.exit_price - trade.entry_price)
            pnl_percentage = (trade.exit_price / trade.entry_price - 1) * 100
        else:  # SHORT
            pnl_absolute = trade.quantity * (trade.entry_price - trade.exit_price)
            pnl_percentage = (trade.entry_price / trade.exit_price - 1) * 100
        
        # Apply commission if requested
        if include_commission:
            # Commission applied on both entry and exit
            commission_amount = (trade.quantity * trade.entry_price * self.commission_rate) + \
                               (trade.quantity * trade.exit_price * self.commission_rate)
            pnl_absolute -= commission_amount
            
            # Adjust percentage P&L for commission
            commission_percentage = self.commission_rate * 2  # Entry and exit
            pnl_percentage -= commission_percentage * 100
        
        return pnl_absolute, pnl_percentage
    
    def _calculate_execution_price(self, 
                                 requested_price: float, 
                                 signal_type: SignalType, 
                                 candle: pd.Series,
                                 is_entry: bool) -> float:
        """
        Calculate actual execution price based on the execution model.
        
        Args:
            requested_price: Requested execution price
            signal_type: Type of signal (long or short)
            candle: Current price candle
            is_entry: Whether this is an entry or exit
            
        Returns:
            Actual execution price
        """
        # Determine direction for slippage application
        is_buy = (signal_type.value == "long" and is_entry) or (signal_type.value == "short" and not is_entry)
        direction = 1 if is_buy else -1
        
        if self.execution_model == ExecutionModel.PERFECT:
            # Perfect execution at requested price
            return requested_price
        
        elif self.execution_model == ExecutionModel.SIMPLE_SLIPPAGE:
            # Apply simple percentage slippage in unfavorable direction
            slippage = requested_price * self.slippage_factor * direction
            return requested_price + slippage
        
        elif self.execution_model == ExecutionModel.VOLUME_BASED:
            # Slippage based on high-low range and volume
            price_range = candle['high'] - candle['low']
            avg_volume = candle['volume']  # Ideally should compare to average volume
            
            # Higher range and lower volume = more slippage
            volume_factor = 1.0
            if avg_volume > 0:
                volume_factor = min(1.0, candle['volume'] / avg_volume)
            
            adjusted_slippage = self.slippage_factor * (price_range / requested_price) / volume_factor
            slippage = requested_price * adjusted_slippage * direction
            
            return requested_price + slippage
        
        elif self.execution_model == ExecutionModel.REALISTIC:
            # Combine multiple factors for realistic execution
            
            # 1. Base slippage with randomness
            base_slippage = requested_price * self.slippage_factor * direction
            random_factor = 1.0 + random.uniform(-0.5, 1.0)  # Random between 0.5x and 2x
            base_slippage *= random_factor
            
            # 2. Spread component
            spread = requested_price * self.spread_factor * direction
            
            # 3. Volatility component
            if 'high' in candle and 'low' in candle:
                volatility = (candle['high'] - candle['low']) / candle['low']
                volatility_slippage = requested_price * volatility * 0.1 * direction  # 10% of the day's range
            else:
                volatility_slippage = 0
            
            # 4. Time of day factor (simplified)
            time_factor = 1.0  # Would be higher at open/close in a more detailed model
            
            # Combine all factors
            total_slippage = (base_slippage + spread + volatility_slippage) * time_factor
            
            # Limit maximum slippage to a reasonable value (e.g., 1%)
            max_slippage = requested_price * 0.01 * direction
            total_slippage = min(total_slippage, max_slippage) if direction > 0 else max(total_slippage, max_slippage)
            
            return requested_price + total_slippage
        
        else:
            # Default to simple slippage if model not recognized
            slippage = requested_price * self.slippage_factor * direction
            return requested_price + slippage
    
    def _calculate_execution_quantity(self, 
                                   requested_quantity: float, 
                                   candle: pd.Series,
                                   asset: AssetSymbol) -> float:
        """
        Calculate actual execution quantity based on the liquidity model.
        
        Args:
            requested_quantity: Requested position size
            candle: Current price candle
            asset: Asset being traded
            
        Returns:
            Actual execution quantity
        """
        if self.liquidity_model == LiquidityModel.INFINITE:
            # Always execute full size
            return requested_quantity
        
        elif self.liquidity_model == LiquidityModel.VOLUME_RATIO:
            # Limit based on percentage of volume
            if 'volume' not in candle or candle['volume'] <= 0:
                return requested_quantity
                
            max_quantity = candle['volume'] * self.max_volume_ratio
            
            # Convert to notional value for comparison
            notional_requested = requested_quantity * candle['close']
            notional_max = max_quantity * candle['close']
            
            if notional_requested > notional_max:
                # Scale down to maximum allowed
                actual_quantity = requested_quantity * (notional_max / notional_requested)
                return actual_quantity
            else:
                return requested_quantity
        
        elif self.liquidity_model == LiquidityModel.MARKET_IMPACT:
            # Start with volume ratio limit
            if 'volume' not in candle or candle['volume'] <= 0:
                max_quantity = requested_quantity
            else:
                max_quantity = candle['volume'] * self.max_volume_ratio
            
            # Convert to notional value
            notional_requested = requested_quantity * candle['close']
            notional_max = max_quantity * candle['close']
            
            # Calculate market impact
            market_impact = self._calculate_market_impact(
                notional_amount=notional_requested,
                asset=asset,
                candle=candle
            )
            
            # If market impact is too high, scale down the order
            max_impact = 0.01  # Maximum 1% price impact
            if market_impact > max_impact:
                scale_factor = max_impact / market_impact
                notional_adjusted = notional_requested * scale_factor
                
                # Take the minimum of volume-based and impact-based limits
                notional_actual = min(notional_adjusted, notional_max)
                
                actual_quantity = requested_quantity * (notional_actual / notional_requested)
                return actual_quantity
            else:
                # Take the minimum of requested and volume-based limit
                return min(requested_quantity, max_quantity)
        
        else:
            # Default to full execution
            return requested_quantity
    
    def _calculate_market_impact(self, 
                              notional_amount: float, 
                              asset: AssetSymbol,
                              candle: pd.Series) -> float:
        """
        Calculate estimated market impact of an order.
        
        Args:
            notional_amount: Notional value of the order
            asset: Asset being traded
            candle: Current price candle
            
        Returns:
            Estimated price impact as a fraction
        """
        # Simple square-root model for market impact
        # Impact = k * sqrt(order_size / ADV)
        
        if 'volume' not in candle or candle['volume'] <= 0:
            return 0.0
            
        # Calculate Daily Volume in notional terms
        daily_volume = candle['volume'] * candle['close']
        
        if daily_volume <= 0:
            return 0.0
            
        # Calculate impact
        impact = self.market_impact_factor * np.sqrt(notional_amount / daily_volume)
        
        return impact

    def generate_realistic_fills(self, 
                               trade: BacktestTrade, 
                               candle: pd.Series,
                               num_fills: int = 3) -> List[Dict[str, Any]]:
        """
        Generate realistic partial fills for a trade.
        
        Args:
            trade: Trade to generate fills for
            candle: Current price candle
            num_fills: Number of partial fills to generate
            
        Returns:
            List of fill dictionaries with timestamp, price, quantity
        """
        # Only used in detailed execution simulations
        fills = []
        
        # Divide quantity into random partial fills
        remaining = trade.quantity
        timestamps = []
        
        # Generate random times within the candle timeframe
        candle_start = candle['timestamp']
        
        # Assuming candle period from metadata or a default
        candle_period_minutes = 60  # Default to 1 hour
        if trade.timeframe.value == '15m':
            candle_period_minutes = 15
        elif trade.timeframe.value == '4h':
            candle_period_minutes = 240
        elif trade.timeframe.value == '1d':
            candle_period_minutes = 1440
        
        candle_end = candle_start + timedelta(minutes=candle_period_minutes)
        
        for i in range(num_fills):
            # Random time within candle
            fill_time = candle_start + (candle_end - candle_start) * random.random()
            timestamps.append(fill_time)
        
        # Sort timestamps
        timestamps.sort()
        
        # Generate fills with varying prices and sizes
        for i, timestamp in enumerate(timestamps):
            # Last fill takes all remaining quantity
            if i == num_fills - 1:
                fill_qty = remaining
            else:
                # Random portion of remaining
                portion = random.uniform(0.1, 0.5)
                fill_qty = remaining * portion
                remaining -= fill_qty
            
            # Varying price for each fill
            direction = 1 if trade.signal_type.value == "long" else -1
            price_variance = trade.entry_price * random.uniform(0, self.slippage_factor * 2) * direction
            fill_price = trade.entry_price + price_variance
            
            fills.append({
                'timestamp': timestamp,
                'price': fill_price,
                'quantity': fill_qty
            })
        
        return fills
