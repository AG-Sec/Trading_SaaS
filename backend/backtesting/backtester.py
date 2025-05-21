"""
Core Backtesting Engine for Trading SaaS

This module provides a comprehensive backtesting framework for evaluating
trading strategies across different market regimes and conditions.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

from shared_types.models import (
    AssetSymbol, Timeframe, TradingSignalModel, 
    HistoricalDataModel, SignalType
)
from backend.agents.market_regime_detector import MarketRegime, MarketRegimeType
from backend.strategies.base_strategy import BaseStrategy
from backend.adaptive_engine.adaptive_engine import AdaptiveEngine

logger = logging.getLogger(__name__)

class TradeStatus(str, Enum):
    """Status of a trade in the backtesting system."""
    OPEN = "open"
    CLOSED_TP = "closed_take_profit"
    CLOSED_SL = "closed_stop_loss"
    CLOSED_EXIT = "closed_exit_signal"
    CLOSED_MANUAL = "closed_manual"
    EXPIRED = "expired"


@dataclass
class BacktestTrade:
    """Represents a trade taken during backtesting."""
    trade_id: str
    signal_id: str
    asset: AssetSymbol
    timeframe: Timeframe
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    status: TradeStatus = TradeStatus.OPEN
    quantity: float = 0.0
    pnl_absolute: float = 0.0
    pnl_percentage: float = 0.0
    market_regime: str = ""
    regime_strength: float = 0.0
    strategy_name: str = ""
    strategy_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Contains results of a backtest."""
    trades: List[BacktestTrade]
    metrics: Dict[str, Any]
    equity_curve: pd.DataFrame
    regime_performance: Dict[str, Dict[str, Any]]
    drawdowns: pd.DataFrame
    settings: Dict[str, Any]
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    asset: AssetSymbol
    
    def __post_init__(self):
        """Calculate basic performance metrics after initialization."""
        if not self.metrics:
            self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from trades."""
        if not self.trades:
            self.metrics = {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0
            }
            return self.metrics
        
        # Basic metrics
        closed_trades = [t for t in self.trades if t.status != TradeStatus.OPEN]
        total_trades = len(closed_trades)
        
        if total_trades == 0:
            self.metrics = {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return": 0.0
            }
            return self.metrics
        
        winning_trades = [t for t in closed_trades if t.pnl_absolute > 0]
        losing_trades = [t for t in closed_trades if t.pnl_absolute <= 0]
        
        win_count = len(winning_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.pnl_absolute for t in winning_trades)
        total_loss = abs(sum(t.pnl_absolute for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Advanced metrics from equity curve if available
        max_drawdown = 0.0
        sharpe_ratio = 0.0
        
        if len(self.equity_curve) > 0:
            max_drawdown = self._calculate_max_drawdown(self.equity_curve)
            sharpe_ratio = self._calculate_sharpe_ratio(self.equity_curve)
        
        # Average trade metrics
        avg_trade = sum(t.pnl_absolute for t in closed_trades) / total_trades if total_trades > 0 else 0
        avg_win = sum(t.pnl_absolute for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(t.pnl_absolute for t in losing_trades) / len(losing_trades) if len(losing_trades) > 0 else 0
        
        # Max win/loss
        max_win = max([t.pnl_absolute for t in winning_trades]) if winning_trades else 0
        max_loss = min([t.pnl_absolute for t in losing_trades]) if losing_trades else 0
        
        total_return = (self.equity_curve['equity'].iloc[-1] / self.initial_capital - 1) * 100 if len(self.equity_curve) > 0 else 0
        
        self.metrics = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss
        }
        
        return self.metrics
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown from equity curve."""
        equity = equity_curve['equity'].values
        peak = equity[0]
        max_dd = 0.0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd * 100  # Return as percentage
    
    def _calculate_sharpe_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if len(equity_curve) < 2:
            return 0.0
            
        # Calculate daily returns
        equity_curve['daily_return'] = equity_curve['equity'].pct_change()
        
        # Calculate annualized Sharpe ratio (assuming 252 trading days)
        daily_returns = equity_curve['daily_return'].dropna()
        if len(daily_returns) < 1:
            return 0.0
            
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        if std_return == 0:
            return 0.0
            
        sharpe = (mean_return / std_return) * (252 ** 0.5)  # Annualized
        return sharpe


class Backtester:
    """
    Core backtesting engine for trading strategies.
    
    This class provides functionality to:
    1. Backtest strategies against historical data
    2. Analyze performance across different market regimes
    3. Optimize strategy parameters based on performance
    4. Generate performance metrics and visualizations
    """
    
    def __init__(self, 
                initial_capital: float = 10000.0,
                commission_rate: float = 0.001,  # 0.1% per trade
                slippage: float = 0.0005,  # 0.05% slippage
                risk_per_trade: float = 0.02,  # 2% risk per trade
                max_open_trades: int = 3,
                use_regime_detection: bool = True):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for the backtest
            commission_rate: Commission rate as percentage per trade
            slippage: Estimated slippage as percentage per trade
            risk_per_trade: Risk per trade as percentage of capital
            max_open_trades: Maximum number of open trades at any time
            use_regime_detection: Whether to use market regime detection
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.max_open_trades = max_open_trades
        self.use_regime_detection = use_regime_detection
        self.current_capital = initial_capital
        
        # State for tracking
        self.open_trades: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        self.daily_equity: Dict[datetime, float] = {}
        self.current_regimes: Dict[Tuple[AssetSymbol, Timeframe], MarketRegime] = {}
    
    def run_backtest(self, 
                    strategy: Union[BaseStrategy, AdaptiveEngine],
                    asset: AssetSymbol,
                    timeframe: Timeframe,
                    historical_data: HistoricalDataModel,
                    start_date: datetime,
                    end_date: datetime) -> BacktestResult:
        """
        Run a backtest using the given strategy and historical data.
        
        Args:
            strategy: Trading strategy or adaptive engine to test
            asset: Asset symbol to test on
            timeframe: Timeframe to test on
            historical_data: Historical price data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult object containing performance metrics and trades
        """
        # Reset state
        self.open_trades = []
        self.closed_trades = []
        self.daily_equity = {}
        self.current_regimes = {}
        self.current_capital = self.initial_capital
        
        # Convert historical data to DataFrame
        df = self._convert_historical_data_to_df(historical_data)
        
        # Filter data for backtest period
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        if len(df) == 0:
            logger.warning(f"No data found for {asset.value} {timeframe.value} in the specified date range")
            return self._create_empty_result(strategy, asset, start_date, end_date)
        
        # Track equity and trades for each day in the backtest
        equity_curve = []
        daily_regime_data = []
        
        # Main backtest loop - iterate through each candle
        for i in range(1, len(df)):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            current_date = current_candle['timestamp']
            
            # Get historical data up to current date for signal generation
            historical_slice = self._create_historical_slice(historical_data, current_date)
            
            # Detect market regime if enabled
            if self.use_regime_detection:
                self._update_regime(asset, timeframe, df.iloc[:i+1])
            
            # Check for closed trades (hit stop loss or take profit)
            self._process_trade_exits(current_candle)
            
            # Generate signals
            signals = []
            if isinstance(strategy, AdaptiveEngine):
                # For adaptive engine, use its signal generation method
                signals = strategy.generate_optimized_signals(
                    asset=asset,
                    timeframe=timeframe,
                    historical_data=historical_slice
                )
            elif isinstance(strategy, BaseStrategy):
                # For individual strategy, use its signal generation method
                signals = strategy.generate_signals(
                    asset=asset,
                    timeframe=timeframe,
                    historical_data=historical_slice
                )
            
            # Process signals and open new trades if appropriate
            for signal in signals:
                self._process_signal(signal, current_candle)
            
            # Update equity for the current day
            day_key = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            self.daily_equity[day_key] = self._calculate_current_equity(current_candle)
            
            # Save daily data for analysis
            equity_curve.append({
                'date': day_key,
                'equity': self.daily_equity[day_key],
                'market_regime': self.current_regimes.get((asset, timeframe), None)
            })
            
            # Add regime data if available
            if (asset, timeframe) in self.current_regimes:
                regime = self.current_regimes[(asset, timeframe)]
                daily_regime_data.append({
                    'date': day_key,
                    'regime_type': regime.regime_type.value,
                    'regime_strength': regime.strength,
                    'close_price': current_candle['close']
                })
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(equity_curve)
        equity_df = equity_df.drop_duplicates(subset=['date'])
        
        # Convert regime data to DataFrame
        regime_df = pd.DataFrame(daily_regime_data) if daily_regime_data else pd.DataFrame()
        
        # Calculate drawdowns
        drawdown_df = self._calculate_drawdowns(equity_df)
        
        # Calculate regime-specific performance
        regime_performance = self._calculate_regime_performance()
        
        # Create backtest result
        strategy_name = strategy.name if hasattr(strategy, 'name') else "Adaptive Engine"
        
        result = BacktestResult(
            trades=self.open_trades + self.closed_trades,
            metrics={},  # Will be calculated in post_init
            equity_curve=equity_df,
            regime_performance=regime_performance,
            drawdowns=drawdown_df,
            settings={
                "initial_capital": self.initial_capital,
                "commission_rate": self.commission_rate,
                "slippage": self.slippage,
                "risk_per_trade": self.risk_per_trade,
                "max_open_trades": self.max_open_trades,
                "use_regime_detection": self.use_regime_detection
            },
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            asset=asset
        )
        
        return result
    
    def _convert_historical_data_to_df(self, historical_data: HistoricalDataModel) -> pd.DataFrame:
        """Convert HistoricalDataModel to DataFrame."""
        candles = [
            {
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            }
            for candle in historical_data.candles
        ]
        
        df = pd.DataFrame(candles)
        return df
    
    def _create_historical_slice(self, historical_data: HistoricalDataModel, current_date: datetime) -> HistoricalDataModel:
        """Create a historical data slice up to the current date."""
        filtered_candles = [c for c in historical_data.candles if c.timestamp <= current_date]
        return HistoricalDataModel(
            asset=historical_data.asset,
            timeframe=historical_data.timeframe,
            candles=filtered_candles
        )
    
    def _update_regime(self, asset: AssetSymbol, timeframe: Timeframe, data_slice: pd.DataFrame):
        """Update the current market regime based on the data slice."""
        # Simplified regime detection for backtesting
        # In a real implementation, this would use the actual MarketRegimeDetector
        
        # Calculate a simple regime based on price action and volatility
        if len(data_slice) < 20:
            return
            
        # Calculate 20-day returns
        returns = data_slice['close'].pct_change(20).iloc[-1] * 100
        
        # Calculate 20-day volatility
        volatility = data_slice['close'].pct_change().rolling(20).std().iloc[-1] * 100
        
        # Determine regime type
        regime_type = MarketRegimeType.NEUTRAL_RANGING
        
        if returns > 5 and volatility < 3:
            regime_type = MarketRegimeType.BULLISH_TRENDING
        elif returns < -5 and volatility < 3:
            regime_type = MarketRegimeType.BEARISH_TRENDING
        elif volatility > 5:
            regime_type = MarketRegimeType.HIGH_VOLATILITY
        elif volatility < 1:
            regime_type = MarketRegimeType.LOW_VOLATILITY
        
        # Set regime strength based on the magnitude of returns and volatility
        strength = min(1.0, max(0.1, abs(returns) / 10))
        
        self.current_regimes[(asset, timeframe)] = MarketRegime(
            regime_type=regime_type,
            strength=strength
        )
    
    def _process_trade_exits(self, current_candle: pd.Series):
        """Check open trades for stop loss or take profit hits."""
        still_open_trades = []
        
        for trade in self.open_trades:
            # Check if trade hit stop loss
            if (trade.signal_type == SignalType.LONG and current_candle['low'] <= trade.stop_loss) or \
               (trade.signal_type == SignalType.SHORT and current_candle['high'] >= trade.stop_loss):
                # Trade hit stop loss
                trade.exit_price = trade.stop_loss
                trade.exit_time = current_candle['timestamp']
                trade.status = TradeStatus.CLOSED_SL
                
                # Calculate P&L
                self._calculate_trade_pnl(trade)
                self.closed_trades.append(trade)
                
            # Check if trade hit take profit
            elif (trade.signal_type == SignalType.LONG and current_candle['high'] >= trade.take_profit) or \
                 (trade.signal_type == SignalType.SHORT and current_candle['low'] <= trade.take_profit):
                # Trade hit take profit
                trade.exit_price = trade.take_profit
                trade.exit_time = current_candle['timestamp']
                trade.status = TradeStatus.CLOSED_TP
                
                # Calculate P&L
                self._calculate_trade_pnl(trade)
                self.closed_trades.append(trade)
                
            else:
                # Trade remains open
                still_open_trades.append(trade)
        
        self.open_trades = still_open_trades
    
    def _process_signal(self, signal: TradingSignalModel, current_candle: pd.Series):
        """Process a trading signal and potentially open a new trade."""
        # Check if we have room for a new trade
        if len(self.open_trades) >= self.max_open_trades:
            return
        
        # Check for opposing position on the same asset
        for trade in self.open_trades:
            if trade.asset == signal.asset and trade.signal_type != signal.signal_type:
                # Close the opposing trade
                trade.exit_price = current_candle['close']
                trade.exit_time = current_candle['timestamp']
                trade.status = TradeStatus.CLOSED_EXIT
                
                # Calculate P&L
                self._calculate_trade_pnl(trade)
                self.closed_trades.append(trade)
                
                # Remove from open trades
                self.open_trades = [t for t in self.open_trades if t.trade_id != trade.trade_id]
        
        # Calculate position size based on risk
        risk_amount = self.current_capital * self.risk_per_trade
        price_distance = abs(signal.entry_price - signal.stop_loss)
        
        if price_distance == 0:
            return  # Avoid division by zero
            
        position_size = risk_amount / price_distance
        
        # Apply slippage to entry price
        entry_price = signal.entry_price
        if signal.signal_type == SignalType.LONG:
            entry_price *= (1 + self.slippage)
        else:
            entry_price *= (1 - self.slippage)
        
        # Create new trade
        trade = BacktestTrade(
            trade_id=str(uuid.uuid4()),
            signal_id=signal.signal_id,
            asset=signal.asset,
            timeframe=signal.timeframe,
            signal_type=signal.signal_type,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=current_candle['timestamp'],
            quantity=position_size,
            status=TradeStatus.OPEN,
            metadata=signal.metadata.copy() if signal.metadata else {}
        )
        
        # Add market regime information if available
        if (signal.asset, signal.timeframe) in self.current_regimes:
            regime = self.current_regimes[(signal.asset, signal.timeframe)]
            trade.market_regime = regime.regime_type.value
            trade.regime_strength = regime.strength
        
        # Add strategy information from metadata
        if signal.metadata:
            if 'strategy_name' in signal.metadata:
                trade.strategy_name = signal.metadata['strategy_name']
            if 'strategy_type' in signal.metadata:
                trade.strategy_type = signal.metadata['strategy_type']
        
        # Add to open trades
        self.open_trades.append(trade)
    
    def _calculate_trade_pnl(self, trade: BacktestTrade):
        """Calculate P&L for a closed trade."""
        if trade.signal_type == SignalType.LONG:
            trade.pnl_absolute = trade.quantity * (trade.exit_price - trade.entry_price)
            trade.pnl_percentage = (trade.exit_price / trade.entry_price - 1) * 100
        else:  # SHORT
            trade.pnl_absolute = trade.quantity * (trade.entry_price - trade.exit_price)
            trade.pnl_percentage = (trade.entry_price / trade.exit_price - 1) * 100
        
        # Apply commission
        trade.pnl_absolute *= (1 - self.commission_rate)
        
        # Update current capital
        self.current_capital += trade.pnl_absolute
    
    def _calculate_current_equity(self, current_candle: pd.Series) -> float:
        """Calculate current equity including open positions at marked to market prices."""
        equity = self.current_capital
        
        # Add value of open positions
        for trade in self.open_trades:
            if trade.signal_type == SignalType.LONG:
                unrealized_pnl = trade.quantity * (current_candle['close'] - trade.entry_price)
            else:  # SHORT
                unrealized_pnl = trade.quantity * (trade.entry_price - current_candle['close'])
            
            # Apply estimated commission for eventual close
            unrealized_pnl *= (1 - self.commission_rate)
            
            equity += unrealized_pnl
        
        return equity
    
    def _calculate_drawdowns(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdowns from equity curve."""
        if 'equity' not in equity_df.columns or len(equity_df) < 2:
            return pd.DataFrame()
            
        df = equity_df.copy()
        df['equity_peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity_peak'] - df['equity']) / df['equity_peak'] * 100
        df['drawdown_duration'] = 0
        
        # Calculate drawdown duration
        current_dd_start = None
        
        for i, row in df.iterrows():
            if row['drawdown'] == 0:
                current_dd_start = None
            elif current_dd_start is None:
                current_dd_start = i
            
            if current_dd_start is not None:
                df.at[i, 'drawdown_duration'] = i - current_dd_start
        
        return df[['date', 'equity', 'equity_peak', 'drawdown', 'drawdown_duration']]
    
    def _calculate_regime_performance(self) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics by market regime."""
        regime_trades = {}
        
        # Group trades by regime
        for trade in self.closed_trades:
            if not trade.market_regime:
                continue
                
            if trade.market_regime not in regime_trades:
                regime_trades[trade.market_regime] = []
                
            regime_trades[trade.market_regime].append(trade)
        
        # Calculate metrics for each regime
        regime_performance = {}
        
        for regime, trades in regime_trades.items():
            if not trades:
                continue
                
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.pnl_absolute > 0]
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            total_profit = sum(t.pnl_absolute for t in winning_trades)
            total_loss = abs(sum(t.pnl_absolute for t in trades if t.pnl_absolute <= 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_return = sum(t.pnl_percentage for t in trades) / total_trades if total_trades > 0 else 0
            
            regime_performance[regime] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_return': avg_return,
                'total_pnl': sum(t.pnl_absolute for t in trades),
                'best_trade': max(t.pnl_absolute for t in trades) if trades else 0,
                'worst_trade': min(t.pnl_absolute for t in trades) if trades else 0
            }
        
        return regime_performance
    
    def _create_empty_result(self, strategy: Union[BaseStrategy, AdaptiveEngine], 
                           asset: AssetSymbol, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Create an empty result when no data is available."""
        strategy_name = strategy.name if hasattr(strategy, 'name') else "Adaptive Engine"
        
        return BacktestResult(
            trades=[],
            metrics={},
            equity_curve=pd.DataFrame(),
            regime_performance={},
            drawdowns=pd.DataFrame(),
            settings={
                "initial_capital": self.initial_capital,
                "commission_rate": self.commission_rate,
                "slippage": self.slippage,
                "risk_per_trade": self.risk_per_trade,
                "max_open_trades": self.max_open_trades,
                "use_regime_detection": self.use_regime_detection
            },
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            asset=asset
        )
