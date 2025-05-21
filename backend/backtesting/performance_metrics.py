"""
Performance Metrics for Backtesting Module

This module calculates various performance metrics for trading strategies
including standard financial metrics, risk-adjusted returns, and
regime-specific performance analysis.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from backend.backtesting.backtester import BacktestTrade, TradeStatus
from backend.agents.market_regime_detector import MarketRegimeType

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculates and provides performance metrics for trading backtests.
    """
    
    @staticmethod
    def calculate_basic_metrics(trades: List[BacktestTrade]) -> Dict[str, Any]:
        """
        Calculate basic performance metrics from a list of trades.
        
        Args:
            trades: List of backtest trades
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
            }
        
        # Filter to closed trades only
        closed_trades = [t for t in trades if t.status != TradeStatus.OPEN]
        
        if not closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
            }
        
        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl_absolute > 0]
        losing_trades = [t for t in closed_trades if t.pnl_absolute <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.pnl_absolute for t in winning_trades)
        total_loss = abs(sum(t.pnl_absolute for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        # Calculate average trade
        avg_trade = sum(t.pnl_absolute for t in closed_trades) / total_trades
        
        # Calculate best and worst trades
        best_trade = max([t.pnl_absolute for t in closed_trades]) if closed_trades else 0
        worst_trade = min([t.pnl_absolute for t in closed_trades]) if closed_trades else 0
        
        # Calculate average holding time
        holding_times = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in closed_trades if t.exit_time]
        avg_holding_time_hours = sum(holding_times) / len(holding_times) if holding_times else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
        
        # Calculate percentage of profitable trades by direction
        long_trades = [t for t in closed_trades if t.signal_type.value == "long"]
        short_trades = [t for t in closed_trades if t.signal_type.value == "short"]
        
        long_win_rate = len([t for t in long_trades if t.pnl_absolute > 0]) / len(long_trades) if long_trades else 0
        short_win_rate = len([t for t in short_trades if t.pnl_absolute > 0]) / len(short_trades) if short_trades else 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "avg_trade": avg_trade,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "avg_holding_time_hours": avg_holding_time_hours,
            "expectancy": expectancy,
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "long_trades_count": len(long_trades),
            "short_trades_count": len(short_trades)
        }
    
    @staticmethod
    def calculate_advanced_metrics(equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics from equity curve.
        
        Args:
            equity_curve: DataFrame with 'date' and 'equity' columns
            
        Returns:
            Dictionary of advanced metrics
        """
        if len(equity_curve) < 2 or 'equity' not in equity_curve.columns:
            return {
                "max_drawdown_pct": 0.0,
                "max_drawdown_duration": 0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "volatility_annual": 0.0,
                "cagr": 0.0
            }
        
        # Calculate returns
        equity_curve = equity_curve.copy()
        equity_curve['return'] = equity_curve['equity'].pct_change()
        
        # Remove NaN values
        equity_curve = equity_curve.dropna(subset=['return'])
        
        # Calculate metrics based on returns
        
        # Drawdown calculation
        equity_curve['equity_peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity_peak'] - equity_curve['equity']) / equity_curve['equity_peak']
        
        max_drawdown = equity_curve['drawdown'].max()
        
        # Drawdown duration
        in_drawdown = False
        current_dd_start = None
        drawdown_periods = []
        
        for i, row in equity_curve.iterrows():
            if row['drawdown'] == 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append(i - current_dd_start)
            elif row['drawdown'] > 0 and not in_drawdown:
                in_drawdown = True
                current_dd_start = i
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Risk metrics
        returns = equity_curve['return'].values
        
        # Annualized return (compound annual growth rate)
        start_equity = equity_curve['equity'].iloc[0]
        end_equity = equity_curve['equity'].iloc[-1]
        n_days = (equity_curve['date'].iloc[-1] - equity_curve['date'].iloc[0]).days
        
        if n_days > 0 and start_equity > 0:
            cagr = (end_equity / start_equity) ** (365 / n_days) - 1
        else:
            cagr = 0
        
        # Volatility (annualized standard deviation)
        volatility = np.std(returns) * np.sqrt(252)  # Assuming 252 trading days per year
        
        # Risk-adjusted metrics
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        avg_daily_return = np.mean(returns)
        sharpe_ratio = (avg_daily_return * 252) / volatility if volatility > 0 else 0
        
        # Sortino ratio (only considers downside risk)
        downside_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        
        sortino_ratio = (avg_daily_return * 252) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
        
        return {
            "max_drawdown_pct": max_drawdown * 100,
            "max_drawdown_duration": max_drawdown_duration,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "volatility_annual": volatility * 100,  # Convert to percentage
            "cagr": cagr * 100  # Convert to percentage
        }
    
    @staticmethod
    def calculate_regime_metrics(trades: List[BacktestTrade]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics grouped by market regime.
        
        Args:
            trades: List of backtest trades
            
        Returns:
            Dictionary of regime metrics
        """
        # Group trades by regime
        regime_trades = defaultdict(list)
        
        for trade in trades:
            if trade.status != TradeStatus.OPEN and trade.market_regime:
                regime_trades[trade.market_regime].append(trade)
        
        # Calculate metrics for each regime
        regime_metrics = {}
        
        for regime, regime_specific_trades in regime_trades.items():
            if not regime_specific_trades:
                continue
            
            metrics = PerformanceMetrics.calculate_basic_metrics(regime_specific_trades)
            
            # Add additional regime-specific metrics
            # Count trades by exit type
            tp_exits = sum(1 for t in regime_specific_trades if t.status == TradeStatus.CLOSED_TP)
            sl_exits = sum(1 for t in regime_specific_trades if t.status == TradeStatus.CLOSED_SL)
            signal_exits = sum(1 for t in regime_specific_trades if t.status == TradeStatus.CLOSED_EXIT)
            
            metrics.update({
                "regime": regime,
                "trade_count": len(regime_specific_trades),
                "total_pnl": sum(t.pnl_absolute for t in regime_specific_trades),
                "tp_exits": tp_exits,
                "sl_exits": sl_exits,
                "signal_exits": signal_exits,
                "tp_exit_rate": tp_exits / len(regime_specific_trades) if regime_specific_trades else 0,
                "sl_exit_rate": sl_exits / len(regime_specific_trades) if regime_specific_trades else 0
            })
            
            regime_metrics[regime] = metrics
        
        return regime_metrics
    
    @staticmethod
    def calculate_strategy_metrics(trades: List[BacktestTrade]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics grouped by strategy.
        
        Args:
            trades: List of backtest trades
            
        Returns:
            Dictionary of strategy metrics
        """
        # Group trades by strategy
        strategy_trades = defaultdict(list)
        
        for trade in trades:
            if trade.status != TradeStatus.OPEN and trade.strategy_name:
                strategy_trades[trade.strategy_name].append(trade)
        
        # Calculate metrics for each strategy
        strategy_metrics = {}
        
        for strategy, strategy_specific_trades in strategy_trades.items():
            if not strategy_specific_trades:
                continue
            
            metrics = PerformanceMetrics.calculate_basic_metrics(strategy_specific_trades)
            
            # Add additional strategy-specific metrics
            metrics.update({
                "strategy": strategy,
                "trade_count": len(strategy_specific_trades),
                "total_pnl": sum(t.pnl_absolute for t in strategy_specific_trades),
                # Add regime distribution for this strategy
                "regime_distribution": PerformanceMetrics._calculate_regime_distribution(strategy_specific_trades)
            })
            
            strategy_metrics[strategy] = metrics
        
        return strategy_metrics
    
    @staticmethod
    def _calculate_regime_distribution(trades: List[BacktestTrade]) -> Dict[str, float]:
        """Calculate the distribution of trades across different regimes."""
        regime_counts = defaultdict(int)
        
        for trade in trades:
            if trade.market_regime:
                regime_counts[trade.market_regime] += 1
        
        total_trades = len(trades)
        
        regime_distribution = {
            regime: count / total_trades
            for regime, count in regime_counts.items()
        } if total_trades > 0 else {}
        
        return regime_distribution
    
    @staticmethod
    def calculate_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly returns from equity curve.
        
        Args:
            equity_curve: DataFrame with 'date' and 'equity' columns
            
        Returns:
            DataFrame with monthly returns
        """
        if len(equity_curve) < 2 or 'equity' not in equity_curve.columns:
            return pd.DataFrame()
        
        # Resample to month-end
        equity_curve['month'] = equity_curve['date'].dt.to_period('M')
        monthly_equity = equity_curve.groupby('month').last().reset_index()
        
        # Calculate returns
        monthly_equity['return'] = monthly_equity['equity'].pct_change() * 100
        
        # Format for output
        result = monthly_equity[['month', 'equity', 'return']].copy()
        result['month'] = result['month'].dt.to_timestamp()
        
        return result.dropna()
    
    @staticmethod
    def calculate_rolling_performance(equity_curve: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            equity_curve: DataFrame with 'date' and 'equity' columns
            window: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(equity_curve) < window or 'equity' not in equity_curve.columns:
            return pd.DataFrame()
        
        # Calculate daily returns
        equity_curve = equity_curve.copy()
        equity_curve['return'] = equity_curve['equity'].pct_change()
        
        # Calculate rolling metrics
        result = pd.DataFrame({
            'date': equity_curve['date'],
            'rolling_return': equity_curve['equity'].pct_change(window) * 100,
            'rolling_volatility': equity_curve['return'].rolling(window).std() * np.sqrt(window) * 100,
            'rolling_sharpe': equity_curve['return'].rolling(window).mean() / 
                              equity_curve['return'].rolling(window).std() * np.sqrt(window) 
                              if window > 0 else 0
        })
        
        # Calculate maximum drawdown within rolling window
        rolling_drawdown = []
        
        for i in range(len(equity_curve)):
            if i < window:
                rolling_drawdown.append(0)
                continue
                
            window_equity = equity_curve['equity'].iloc[i-window:i+1].values
            peak = window_equity[0]
            max_dd = 0
            
            for value in window_equity:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
                    
            rolling_drawdown.append(max_dd * 100)  # Convert to percentage
        
        result['rolling_max_drawdown'] = rolling_drawdown
        
        return result.dropna()
    
    @staticmethod
    def calculate_regime_transition_metrics(trades: List[BacktestTrade], 
                                          regime_transitions: List[Tuple[datetime, str, str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics around regime transitions.
        
        Args:
            trades: List of backtest trades
            regime_transitions: List of tuples (time, old_regime, new_regime, strength)
            
        Returns:
            Dictionary of transition metrics
        """
        if not trades or not regime_transitions:
            return {}
            
        # Sort regime transitions by time
        regime_transitions = sorted(regime_transitions, key=lambda x: x[0])
        
        # Initialize result structure
        transition_metrics = {}
        
        # Group trades by transition
        for i, (transition_time, old_regime, new_regime, strength) in enumerate(regime_transitions):
            # Define window around transition
            window_start = transition_time - timedelta(days=7)
            window_end = transition_time + timedelta(days=7)
            
            # Get trades during this window
            window_trades = [
                t for t in trades
                if t.status != TradeStatus.OPEN
                and window_start <= t.entry_time <= window_end
            ]
            
            # Skip if no trades
            if not window_trades:
                continue
                
            # Calculate metrics for trades during this transition
            basic_metrics = PerformanceMetrics.calculate_basic_metrics(window_trades)
            
            # Split into before and after transition
            before_trades = [t for t in window_trades if t.entry_time < transition_time]
            after_trades = [t for t in window_trades if t.entry_time >= transition_time]
            
            before_metrics = PerformanceMetrics.calculate_basic_metrics(before_trades) if before_trades else {}
            after_metrics = PerformanceMetrics.calculate_basic_metrics(after_trades) if after_trades else {}
            
            # Store combined metrics
            transition_key = f"{old_regime}_to_{new_regime}_{i}"
            transition_metrics[transition_key] = {
                "transition_time": transition_time,
                "old_regime": old_regime,
                "new_regime": new_regime,
                "regime_strength": strength,
                "overall_metrics": basic_metrics,
                "before_transition_metrics": before_metrics,
                "after_transition_metrics": after_metrics,
                "trade_count_before": len(before_trades),
                "trade_count_after": len(after_trades)
            }
        
        return transition_metrics
    
    @staticmethod
    def calculate_win_loss_streaks(trades: List[BacktestTrade]) -> Dict[str, Any]:
        """
        Calculate win and loss streaks from trade history.
        
        Args:
            trades: List of backtest trades
            
        Returns:
            Dictionary with streak metrics
        """
        if not trades:
            return {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "avg_win_streak": 0,
                "avg_loss_streak": 0,
                "current_streak": 0
            }
        
        # Sort trades by exit time
        closed_trades = [t for t in trades if t.status != TradeStatus.OPEN and t.exit_time is not None]
        closed_trades.sort(key=lambda x: x.exit_time)
        
        if not closed_trades:
            return {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "avg_win_streak": 0,
                "avg_loss_streak": 0,
                "current_streak": 0
            }
        
        # Calculate streaks
        streaks = []
        current_streak = 1
        
        for i in range(1, len(closed_trades)):
            current_trade = closed_trades[i]
            prev_trade = closed_trades[i-1]
            
            current_win = current_trade.pnl_absolute > 0
            prev_win = prev_trade.pnl_absolute > 0
            
            if current_win == prev_win:
                # Continuing the streak
                current_streak += 1
            else:
                # Streak ended, record it
                streaks.append((current_streak, prev_win))
                current_streak = 1
        
        # Record the final streak
        streaks.append((current_streak, closed_trades[-1].pnl_absolute > 0))
        
        # Analyze streaks
        win_streaks = [s[0] for s in streaks if s[1]]
        loss_streaks = [s[0] for s in streaks if not s[1]]
        
        max_win_streak = max(win_streaks) if win_streaks else 0
        max_loss_streak = max(loss_streaks) if loss_streaks else 0
        
        avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
        avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
        
        # Determine current streak
        current_streak_count = streaks[-1][0]
        current_streak_type = "win" if streaks[-1][1] else "loss"
        
        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "avg_win_streak": avg_win_streak,
            "avg_loss_streak": avg_loss_streak,
            "current_streak": current_streak_count,
            "current_streak_type": current_streak_type,
            "win_streaks": win_streaks,
            "loss_streaks": loss_streaks
        }
