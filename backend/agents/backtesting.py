"""
Backtesting module for the Trading SaaS platform.
Provides functionality to backtest trading strategies with historical data.
"""
import pandas as pd
import numpy as np
import logging
import uuid
from typing import List, Dict, Tuple, Optional, Any, Union, Counter
from datetime import datetime, timedelta, timezone
import json
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import seaborn as sns
from numba import njit

from shared_types import (
    AssetSymbol, 
    Timeframe, 
    TradingSignalModel, 
    SignalType,
    TradeModel,
    TradeStatus,
    OrderSide
)
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.technical_indicators import TechnicalIndicators
from backend.agents.signal_scoring import SignalScoring
from backend.agents.market_regime_detector import MarketRegimeDetector, MarketRegimeType

logger = logging.getLogger(__name__)

@njit
def calculate_pnl_nb(entry_price: float, exit_price: float, qty: float) -> float:
    return (exit_price - entry_price) * qty

@njit
def compute_max_drawdown(equity):
    max_dd = 0.0
    peak = equity[0]
    for i in range(equity.size):
        e = equity[i]
        if e > peak:
            peak = e
        else:
            dd = (peak - e) / peak if peak != 0 else 0.0
            if dd > max_dd:
                max_dd = dd
    return max_dd

class BacktestResult:
    """Class to hold backtest results with regime-specific analysis"""

    @property
    def average_return_per_trade(self) -> float:
        """Average return per trade as a percentage."""
        if self.total_trades > 0:
            return self.total_return_pct / self.total_trades
        return 0.0

    def __init__(self, strategy_name: str, asset: AssetSymbol, timeframe: Timeframe):
        self.strategy_name = strategy_name
        self.asset = asset
        self.timeframe = timeframe
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.initial_capital = 10000.0
        self.final_capital = 10000.0
        self.trades: List[TradeModel] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdowns: List[Tuple[datetime, float]] = []
        self.daily_returns: List[Tuple[datetime, float]] = []
        
        # Market regime data
        self.regime_periods: List[Dict[str, Any]] = []  # Periods of different market regimes
        self.regime_trades: Dict[str, List[TradeModel]] = {}  # Trades by regime type
        self.regime_equity: Dict[str, List[Tuple[datetime, float]]] = {}  # Equity by regime
        self.regime_durations: Dict[str, timedelta] = {}  # Time spent in each regime
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.average_win = 0.0
        self.average_loss = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_date: Optional[datetime] = None
        self.sharpe_ratio = 0.0
        self.total_return_pct = 0.0
        self.cagr = 0.0
        self.r_multiple_avg = 0.0
        self.r_multiple_stdev = 0.0
        
        # Regime-specific performance metrics
        self.regime_performance: Dict[str, Dict[str, Any]] = {}
        self.adaptive_vs_fixed_comparison: Dict[str, Any] = {}
        
        # Initialize empty metrics for each regime type
        for regime_type in [regime_type.value for regime_type in MarketRegimeType]:
            self.regime_trades[regime_type] = []
            self.regime_equity[regime_type] = []
            self.regime_durations[regime_type] = timedelta(0)
            self.regime_performance[regime_type] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'total_return_pct': 0.0,
                'average_return_per_trade': 0.0
            }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result_dict = {
            'strategy_name': self.strategy_name,
            'asset': self.asset.value,
            'timeframe': self.timeframe.value,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_date': self.max_drawdown_date.isoformat() if self.max_drawdown_date else None,
            'sharpe_ratio': self.sharpe_ratio,
            'total_return_pct': self.total_return_pct,
            'cagr': self.cagr,
            'r_multiple_avg': self.r_multiple_avg,
            'r_multiple_stdev': self.r_multiple_stdev,
            # Simplified trades list
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'asset': t.asset.value,
                    'side': t.side.value,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity_asset': t.quantity_asset,
                    'entry_time': t.entry_time.isoformat() if t.entry_time else None,
                    'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                    'pnl_usd': t.pnl_usd,
                    'market_regime': getattr(t, 'market_regime', None),
                    'regime_strength': getattr(t, 'regime_strength', None)
                }
                for t in self.trades
            ],
            # Regime analysis
            'regime_periods': self.regime_periods,
            'regime_performance': self.regime_performance,
            'adaptive_vs_fixed_comparison': self.adaptive_vs_fixed_comparison
        }
        
        # Calculate regime distribution
        if self.regime_durations:
            total_duration = sum(self.regime_durations.values(), timedelta(0))
            if total_duration.total_seconds() > 0:
                result_dict['regime_distribution'] = {
                    regime: duration.total_seconds() / total_duration.total_seconds() * 100
                    for regime, duration in self.regime_durations.items()
                }
                
        return result_dict
        
    def generate_equity_curve_image(self) -> Optional[str]:
        """Generate equity curve image with market regime background as base64 string"""
        if not self.equity_curve:
            return None
            
        try:
            dates = [t[0] for t in self.equity_curve]
            values = [t[1] for t in self.equity_curve]
            
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            # Plot the main equity curve
            plt.plot(dates, values, linewidth=2, label='Equity Curve')
            plt.title(f'Equity Curve with Market Regimes: {self.strategy_name} on {self.asset.value} {self.timeframe.value}')
            plt.xlabel('Date')
            plt.ylabel('Capital ($)')
            plt.grid(True, alpha=0.3)
            
            # Add regime background if regime periods are available
            if self.regime_periods:
                # Define colors for different regime types
                regime_colors = {
                    'bullish_trending': '#90EE90',  # Light green
                    'bearish_trending': '#FFA07A',  # Light salmon
                    'neutral_ranging': '#ADD8E6',  # Light blue
                    'high_volatility': '#FFFF99',  # Light yellow
                    'low_volatility': '#D8BFD8',  # Thistle
                    'bullish_breakout': '#7FFF00',  # Chartreuse
                    'bearish_breakout': '#FF6347',  # Tomato
                    'unknown': '#E0E0E0'   # Light gray
                }
                
                # Add background for each regime period
                for period in self.regime_periods:
                    start_date = datetime.fromisoformat(period['start_date']) if isinstance(period['start_date'], str) else period['start_date']
                    end_date = datetime.fromisoformat(period['end_date']) if isinstance(period['end_date'], str) else period['end_date']
                    regime_type = period['regime_type']
                    strength = period.get('strength', 0.5)
                    
                    # Get color for regime and adjust alpha based on strength
                    color = regime_colors.get(regime_type, '#E0E0E0')
                    alpha = 0.2 + (strength * 0.3)  # Alpha varies from 0.2 to 0.5 based on strength
                    
                    # Add colored background
                    plt.axvspan(start_date, end_date, alpha=alpha, color=color)
                    
                    # Try to add label in the middle of the regime period if it's long enough
                    span_days = (end_date - start_date).days
                    if span_days > 5:  # Only add text for longer periods
                        mid_point = start_date + (end_date - start_date) / 2
                        y_pos = min(values) + (max(values) - min(values)) * 0.05  # Position near bottom
                        plt.text(mid_point, y_pos, regime_type.replace('_', ' ').title(), 
                                 ha='center', va='bottom', fontsize=8, alpha=0.7, rotation=0)
            
            # Add markers for trades
            for trade in self.trades:
                if trade.entry_time and trade.exit_time and trade.pnl_usd is not None:
                    # Draw lines for entry and exit
                    plt.axvline(x=trade.entry_time, color='gray', linestyle='--', alpha=0.5)
                    plt.axvline(x=trade.exit_time, color='gray', linestyle='--', alpha=0.5)
                    
                    # Different markers for different trade outcomes
                    if trade.pnl_usd > 0:
                        marker_color = 'green'
                        marker_style = '^'  # Up triangle for profit
                    else:
                        marker_color = 'red'
                        marker_style = 'v'  # Down triangle for loss
                    
                    # Find the equity value at entry and exit for marker placement
                    entry_equity = next((value for date, value in self.equity_curve if date >= trade.entry_time), None)
                    exit_equity = next((value for date, value in self.equity_curve if date >= trade.exit_time), None)
                    
                    if entry_equity is not None:
                        plt.plot(trade.entry_time, entry_equity, marker='o', markersize=6, color=marker_color, alpha=0.7)
                    if exit_equity is not None:
                        plt.plot(trade.exit_time, exit_equity, marker=marker_style, markersize=8, color=marker_color)
            
            # Add legend for regime colors
            handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.4) for regime_type, color in regime_colors.items() 
                      if any(period['regime_type'] == regime_type for period in self.regime_periods)]
            labels = [regime_type.replace('_', ' ').title() for regime_type, color in regime_colors.items() 
                     if any(period['regime_type'] == regime_type for period in self.regime_periods)]
            
            plt.legend(handles, labels, loc='upper left', fontsize=8)
            
            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            logger.error(f"Error generating equity curve: {e}", exc_info=True)
            return None
            
    def generate_regime_performance_image(self) -> Optional[str]:
        """Generate performance comparison by market regime"""
        if not self.regime_performance:
            return None
            
        try:
            # Filter to regimes that had trades
            active_regimes = {regime: metrics for regime, metrics in self.regime_performance.items() 
                             if metrics['total_trades'] > 0}
            
            if not active_regimes:
                return None
                
            # Set up figure with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Performance by Market Regime: {self.strategy_name} on {self.asset.value}', fontsize=16)
            
            # 1. Win Rate by Regime
            regimes = list(active_regimes.keys())
            win_rates = [metrics['win_rate'] for metrics in active_regimes.values()]
            
            axs[0, 0].bar(range(len(regimes)), win_rates, color='skyblue')
            axs[0, 0].set_title('Win Rate by Regime (%)')
            axs[0, 0].set_ylim(0, 100)
            axs[0, 0].grid(axis='y', alpha=0.3)
            axs[0, 0].set_xticks(range(len(regimes)))
            axs[0, 0].set_xticklabels([r.replace('_', ' ').title() for r in regimes], rotation=45, ha='right')
            
            # 2. Average Return per Trade
            avg_returns = [metrics['average_return_per_trade'] for metrics in active_regimes.values()]
            colors = ['green' if x > 0 else 'red' for x in avg_returns]
            
            axs[0, 1].bar(range(len(regimes)), avg_returns, color=colors)
            axs[0, 1].set_title('Average Return per Trade (%)')
            axs[0, 1].grid(axis='y', alpha=0.3)
            axs[0, 1].set_xticks(range(len(regimes)))
            axs[0, 1].set_xticklabels([r.replace('_', ' ').title() for r in regimes], rotation=45, ha='right')
            
            # 3. Trade Count Distribution
            trade_counts = [metrics['total_trades'] for metrics in active_regimes.values()]
            
            axs[1, 0].pie(trade_counts, labels=[r.replace('_', ' ').title() for r in regimes], 
                        autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel', len(active_regimes)))
            axs[1, 0].set_title('Distribution of Trades by Regime')
            
            # 4. Total Return by Regime
            total_returns = [metrics['total_return_pct'] for metrics in active_regimes.values()]
            colors = ['green' if x > 0 else 'red' for x in total_returns]
            
            axs[1, 1].bar(range(len(regimes)), total_returns, color=colors)
            axs[1, 1].set_title('Total Return by Regime (%)')
            axs[1, 1].grid(axis='y', alpha=0.3)
            axs[1, 1].set_xticks(range(len(regimes)))
            axs[1, 1].set_xticklabels([r.replace('_', ' ').title() for r in regimes], rotation=45, ha='right')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            logger.error(f"Error generating regime performance image: {e}", exc_info=True)
            return None
            
    def generate_adaptive_vs_fixed_image(self) -> Optional[str]:
        """Generate comparison of adaptive vs fixed parameters performance"""
        if not self.adaptive_vs_fixed_comparison:
            return None
            
        try:
            # Extract data
            strategies = ['Adaptive Parameters', 'Fixed Parameters']
            metrics = ['total_return_pct', 'win_rate', 'profit_factor', 'max_drawdown']
            metric_labels = ['Total Return (%)', 'Win Rate (%)', 'Profit Factor', 'Max Drawdown (%)'] 
            
            # Set up the figure
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Adaptive vs Fixed Parameters Comparison', fontsize=16)
            
            # Flatten axes for easier iteration
            axs = axs.flatten()
            
            # Plot each metric
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = [
                    self.adaptive_vs_fixed_comparison.get('adaptive', {}).get(metric, 0),
                    self.adaptive_vs_fixed_comparison.get('fixed', {}).get(metric, 0)
                ]
                
                # Determine colors based on which is better
                # For drawdown, lower is better; for others, higher is better
                if metric == 'max_drawdown':
                    colors = ['green' if values[0] < values[1] else 'red', 
                              'green' if values[1] < values[0] else 'red']
                else:
                    colors = ['green' if values[0] > values[1] else 'red', 
                              'green' if values[1] > values[0] else 'red']
                
                axs[i].bar(strategies, values, color=colors)
                axs[i].set_title(label)
                axs[i].grid(axis='y', alpha=0.3)
                
                # Add value labels on top of bars
                for j, v in enumerate(values):
                    axs[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
                    
                # Add percentage improvement
                if values[1] != 0 and metric != 'max_drawdown':
                    improvement = ((values[0] / values[1]) - 1) * 100
                    axs[i].text(0.5, max(values) * 1.1, f'Improvement: {improvement:.1f}%', 
                                ha='center', fontsize=10,
                                color='green' if improvement > 0 else 'red')
                elif values[1] != 0 and metric == 'max_drawdown':
                    improvement = ((values[1] / values[0]) - 1) * 100
                    axs[i].text(0.5, max(values) * 1.1, f'Reduction: {improvement:.1f}%', 
                                ha='center', fontsize=10,
                                color='green' if improvement > 0 else 'red')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Convert plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            logger.error(f"Error generating adaptive vs fixed image: {e}", exc_info=True)
            return None


class Backtester:
    """
    Class responsible for backtesting trading strategies with market regime analysis.
    """
    
    def __init__(
        self,
        market_data_agent: MarketDataAgent,
        signal_scanner_agent: SignalScannerAgent,
        risk_manager_agent: Optional[RiskManagerAgent] = None,
        initial_capital: float = 10000.0
    ):
        self.market_data_agent = market_data_agent
        self.signal_scanner_agent = signal_scanner_agent
        self.risk_manager_agent = risk_manager_agent or RiskManagerAgent(account_balance_usd=initial_capital)
        self.initial_capital = initial_capital
        self.tech_indicators = TechnicalIndicators()
        self.signal_scoring = SignalScoring()
        self.regime_detector = MarketRegimeDetector()
        
        # For comparing adaptive vs fixed parameters
        self.enable_comparative_analysis = True  # Whether to run both adaptive and fixed parameter backtests
        
    def _detect_market_regimes(self, df: pd.DataFrame, asset: AssetSymbol, timeframe: Timeframe) -> List[Dict[str, Any]]:
        """
        Detect and record market regimes over the entire backtest period.
        
        Args:
            df: DataFrame with OHLCV data
            asset: Asset symbol
            timeframe: Timeframe
            
        Returns:
            List of regime periods with start/end dates
        """
        logger.info(f"Detecting market regimes for {asset.value} {timeframe.value}")
        
        # Use larger chunks for fewer detect_regime calls (faster)
        chunk_size = 200  # Process 200 bars at a time
        overlap = 50      # Overlap for continuity
        
        # Precompute all regime detections in sliding windows
        regimes = []
        for start in range(0, len(df) - self.regime_detector.min_data_points + 1, chunk_size - overlap):
            end = min(start + chunk_size, len(df))
            window = df.iloc[start:end]
            regimes.append(self.regime_detector.detect_regime(window, asset, timeframe))
        
        # Build regime series for each bar by repeating regime per chunk (adjust to full length)
        values = []
        for r in regimes:
            for _ in range(chunk_size - overlap):
                values.append(r.regime_type.value)
        # Truncate or extend to match df length
        if len(values) >= len(df):
            values = values[:len(df)]
        elif len(values) > 0:
            # Only extend if we have at least one value
            values += [values[-1]] * (len(df) - len(values))
        else:
            # If values list is empty, use 'neutral_ranging' as default
            values = ['neutral_ranging'] * len(df)
        regime_series = pd.Series(values, index=df.index)
        
        # Assign regimes to each bar index
        current = regime_series.iloc[0]
        start_date = df.index[0]
        regime_periods = []
        for idx, rt in regime_series.items():
            if rt != current:
                regime_periods.append({'regime_type': current, 'start_date': start_date, 'end_date': idx, 'strength': 0.5})
                current = rt
                start_date = idx
        # final period
        regime_periods.append({'regime_type': current, 'start_date': start_date, 'end_date': df.index[-1], 'strength': 0.5})
        
        logger.info(f"Detected {len(regime_periods)} market regime periods")
        return regime_periods
    
    def _calculate_regime_metrics(self, result: BacktestResult) -> None:
        """
        Calculate performance metrics for each market regime.
        
        Args:
            result: BacktestResult object with trades and regime data
        """
        # Skip if no regime data
        if not result.regime_periods or not result.trades:
            return
            
        # Assign trades to regimes
        for trade in result.trades:
            if not trade.entry_time or not trade.exit_time or trade.pnl_usd is None:
                continue
                
            # Find the dominant regime during this trade
            overlapping_regimes = []
            for period in result.regime_periods:
                regime_start = period['start_date']
                regime_end = period['end_date']
                
                # Check if trade overlaps with this regime period
                if (trade.entry_time <= regime_end and trade.exit_time >= regime_start):
                    # Calculate overlap duration
                    overlap_start = max(trade.entry_time, regime_start)
                    overlap_end = min(trade.exit_time, regime_end)
                    overlap_duration = (overlap_end - overlap_start).total_seconds()
                    
                    overlapping_regimes.append((period['regime_type'], overlap_duration))
            
            # Assign trade to regime with maximum overlap
            if overlapping_regimes:
                # Sort by overlap duration (descending)
                overlapping_regimes.sort(key=lambda x: x[1], reverse=True)
                dominant_regime = overlapping_regimes[0][0]
                
                # Assign market regime to trade
                trade.market_regime = dominant_regime
                
                # Add to regime-specific trade list
                result.regime_trades[dominant_regime].append(trade)
        
        # Calculate metrics for each regime
        for regime_type, trades in result.regime_trades.items():
            if not trades:
                continue
                
            # Basic trade metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.pnl_usd and t.pnl_usd > 0)
            losing_trades = sum(1 for t in trades if t.pnl_usd and t.pnl_usd <= 0)
            
            # Calculate win rate
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd and t.pnl_usd > 0)
            gross_loss = sum(abs(t.pnl_usd) for t in trades if t.pnl_usd and t.pnl_usd < 0)
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
            
            # Calculate average win/loss
            average_win = (gross_profit / winning_trades) if winning_trades > 0 else 0
            average_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0
            
            # Calculate total return percentage for this regime
            total_pnl = sum(t.pnl_usd for t in trades if t.pnl_usd is not None)
            total_return_pct = (total_pnl / result.initial_capital) * 100
            
            # Average return per trade
            average_return_per_trade = total_return_pct / total_trades if total_trades > 0 else 0
            
            # Store metrics
            result.regime_performance[regime_type] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': average_win,
                'average_loss': average_loss,
                'total_return_pct': total_return_pct,
                'average_return_per_trade': average_return_per_trade
            }
            
        # Calculate time spent in each regime
        for period in result.regime_periods:
            regime_type = period['regime_type']
            start_date = period['start_date']
            end_date = period['end_date']
            duration = end_date - start_date
            
            result.regime_durations[regime_type] += duration
    
    def backtest_strategy(
        self,
        asset: AssetSymbol,
        timeframe: Timeframe,
        start_date: datetime,
        end_date: datetime,
        strategy_name: str = "breakout_strategy",
        use_adaptive_params: bool = True
    ) -> BacktestResult:
        """
        Backtest a trading strategy over a specific period with market regime analysis.
        
        Args:
            asset: Asset symbol to backtest
            timeframe: Timeframe to use
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_name: Name of the strategy being backtested
            use_adaptive_params: Whether to use adaptive parameters based on market regime
            
        Returns:
            BacktestResult object with performance metrics
        """
        logger.info(f"Starting backtest for {strategy_name} on {asset.value} {timeframe.value} from {start_date} to {end_date}")
        logger.info(f"Adaptive parameters: {use_adaptive_params}")
        
        # Initialize result object
        result = BacktestResult(strategy_name, asset, timeframe)
        result.start_date = start_date
        result.end_date = end_date
        result.initial_capital = self.initial_capital
        
        # Fetch market data for the backtest period
        historical_data = self.market_data_agent.fetch_historical_data(
            asset=asset,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=False  # Force fresh data for backtest
        )
        
        if not historical_data or not historical_data.candles:
            logger.error(f"No historical data available for {asset.value} {timeframe.value}")
            return result
            
        logger.info(f"Fetched {len(historical_data.candles)} candles for backtest")
        
        # Convert to dataframe for easier manipulation
        df = pd.DataFrame([candle.model_dump() for candle in historical_data.candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate technical indicators
        df = self.tech_indicators.calculate_all(df)
        
        # Detect market regimes throughout the historical data
        regime_periods = self._detect_market_regimes(df, asset, timeframe)
        result.regime_periods = regime_periods
        
        # Configure signal scanner for adaptive or fixed parameters
        if use_adaptive_params:
            # Enable adaptive parameters based on market regimes
            self.signal_scanner_agent.adapt_to_regime = True
            if hasattr(self.risk_manager_agent, 'adapt_to_regime'):
                self.risk_manager_agent.adapt_to_regime = True
                
            strategy_name = f"{strategy_name}_adaptive"
        else:
            # Use fixed parameters (disable adaptation)
            self.signal_scanner_agent.adapt_to_regime = False
            if hasattr(self.risk_manager_agent, 'adapt_to_regime'):
                self.risk_manager_agent.adapt_to_regime = False
                
            strategy_name = f"{strategy_name}_fixed"
        
        # Initialize backtest variables
        capital = self.initial_capital
        current_position = None
        trades = []
        equity_curve = [(df.index[0], capital)]  # Start equity curve
        max_equity = capital
        
        # Use vectorized P&L and exit calculations instead of per-bar Python loops (Numba/JIT could be added)
        # For now, batch scan entry signals and precompute exit conditions
        signal_df = self.signal_scanner_agent.scan_for_breakout_signals(asset, timeframe, historical_data)
        # Convert signals to DataFrame for vectorized processing
        signals = pd.DataFrame([s.model_dump() for s in signal_df]) if signal_df else pd.DataFrame()
        # TODO: compute exits and P&L vectorized, for now fallback to existing loop for trade closure
        for i in range(20, len(df) - 1):
            current_time = df.index[i]
            current_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]  # For determining entry/exit prices
            
            # Skip if we already have a position
            if current_position is not None:
                # Check for stop loss or take profit hit on this bar
                if self._check_exit_conditions(current_position, current_bar):
                    # Close the position
                    exit_price, exit_time = self._calculate_exit_price(current_position, current_bar, next_bar)
                    
                    # Calculate P&L
                    pnl = calculate_pnl_nb(current_position.entry_price, exit_price, current_position.quantity_asset)
                    
                    # Update trade record
                    current_position.exit_price = exit_price
                    current_position.exit_time = exit_time
                    current_position.status = TradeStatus.CLOSED
                    current_position.pnl_usd = pnl
                    
                    # Update capital
                    capital += pnl
                    
                    # Add to equity curve
                    equity_curve.append((exit_time, capital))
                    
                    # Update max drawdown
                    max_drawdown = compute_max_drawdown(np.array([val for _, val in equity_curve], dtype=np.float64))
                    result.max_drawdown = max_drawdown * 100  # Convert to percentage
                    
                    # Add to completed trades
                    trades.append(current_position)
                    
                    # Log the closed trade before resetting position
                    logger.info(f"Closed trade at {exit_time}: {current_position.side.value} {current_position.asset.value}, "
                                f"Entry: {current_position.entry_price}, Exit: {exit_price}, P&L: ${pnl}")
                    
                    # Reset current position
                    current_position = None
                    
                continue  # Skip to next bar
            
            # Generate signals for this bar
            # Create a subset dataframe up to current bar for signal generation
            subset_df = df.iloc[:i+1]
            subset_candles = historical_data.candles[:i+1]
            subset_historical_data = self._create_subset_historical_data(
                historical_data, subset_candles)
            
            # Get the current market regime for the bar
            current_bar_regime = None
            for period in regime_periods:
                if period['start_date'] <= current_time <= period['end_date']:
                    current_bar_regime = period['regime_type']
                    current_regime_strength = period['strength']
                    break
            
            # Generate signals
            signals = self.signal_scanner_agent.scan_for_breakout_signals(
                asset, timeframe, subset_historical_data)
            
            if not signals:
                continue
                
            # Filter signals with risk manager
            approved_signals = signals
            if self.risk_manager_agent:
                approved_signals = self.risk_manager_agent.filter_signals(signals)
                
            if not approved_signals:
                continue
                
            # Use signal scoring to pick the best signal
            df_dict = {(asset.value, timeframe.value): subset_df}
            scored_signals = self.signal_scoring.rank_signals(approved_signals, df_dict)
            
            if not scored_signals:
                continue
                
            # Take the highest scored signal
            best_signal = scored_signals[0]
            
            # Add market regime info to best signal for tracking
            if current_bar_regime:
                if 'metadata' not in best_signal.model_dump():
                    best_signal.metadata = {}
                if best_signal.metadata is None:
                    best_signal.metadata = {}
                
                best_signal.metadata['market_regime'] = current_bar_regime
                best_signal.metadata['regime_strength'] = current_regime_strength
            
            # Calculate entry price (on next bar's open)
            entry_price = next_bar['open']
            entry_time = next_bar.name
            
            # Calculate position size
            position_size = best_signal.position_size_asset if best_signal.position_size_asset else 0
            if not position_size:
                # Calculate position size based on risk if not already set
                risk_amount = capital * 0.01  # 1% risk
                stop_distance = abs(entry_price - best_signal.stop_loss)
                position_size = risk_amount / stop_distance if stop_distance else 0
            
            # Create trade record
            trade = TradeModel(
                trade_id=str(uuid.uuid4()),
                originating_order_id=best_signal.signal_id,
                asset=asset,
                side=OrderSide.BUY if best_signal.signal_type == SignalType.LONG else OrderSide.SELL,
                entry_price=entry_price,
                exit_price=None,
                quantity_asset=position_size,
                entry_time=entry_time,
                exit_time=None,
                status=TradeStatus.OPEN,
                pnl_usd=None
            )
            
            # Add market regime information to trade
            if current_bar_regime:
                trade.market_regime = current_bar_regime
                trade.regime_strength = current_regime_strength
            
            # Set current position
            current_position = trade
            
            logger.info(f"Opened new trade at {entry_time}: {trade.side.value} {trade.asset.value}, "
                      f"Entry: {entry_price}, Size: {position_size}, "
                      f"SL: {best_signal.stop_loss}, TP: {best_signal.take_profit}")
        
        # If we still have an open position at the end, close it at the last price
        if current_position is not None:
            exit_price = df.iloc[-1]['close']
            exit_time = df.index[-1]
            
            # Calculate P&L
            pnl = calculate_pnl_nb(current_position.entry_price, exit_price, current_position.quantity_asset)
            
            # Update trade record
            current_position.exit_price = exit_price
            current_position.exit_time = exit_time
            current_position.status = TradeStatus.CLOSED
            current_position.pnl_usd = pnl
            
            # Update capital
            capital += pnl
            
            # Add to equity curve
            equity_curve.append((exit_time, capital))
            
            # Add to completed trades
            trades.append(current_position)
            
            logger.info(f"Closed final trade at {exit_time}: {current_position.side.value} {current_position.asset.value}, "
                      f"Entry: {current_position.entry_price}, Exit: {exit_price}, P&L: ${pnl}")
        
        # Calculate performance metrics
        result.trades = trades
        result.equity_curve = equity_curve
        result.final_capital = capital
        self._calculate_performance_metrics(result)
        
        # Calculate regime-specific metrics
        self._calculate_regime_metrics(result)
        
        logger.info(f"Backtest completed: {result.total_trades} trades, Win rate: {result.win_rate:.2f}%, "
                  f"Profit factor: {result.profit_factor:.2f}, Return: {result.total_return_pct:.2f}%")
        
        # Create a comparative analysis with both adaptive and fixed parameters
        if self.enable_comparative_analysis and use_adaptive_params:
            logger.info("Running comparative backtest with fixed parameters...")
            fixed_result = self.backtest_strategy(
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_name=strategy_name.replace('_adaptive', ''),
                use_adaptive_params=False
            )
            
            # Store comparative metrics
            result.adaptive_vs_fixed_comparison = {
                'adaptive': {
                    'total_return_pct': result.total_return_pct,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'max_drawdown': result.max_drawdown
                },
                'fixed': {
                    'total_return_pct': fixed_result.total_return_pct,
                    'win_rate': fixed_result.win_rate,
                    'profit_factor': fixed_result.profit_factor,
                    'max_drawdown': fixed_result.max_drawdown
                }
            }
            
            logger.info(f"Comparative analysis completed. Adaptive vs Fixed: "
                      f"Return {result.total_return_pct:.2f}% vs {fixed_result.total_return_pct:.2f}%, "
                      f"Win rate {result.win_rate:.2f}% vs {fixed_result.win_rate:.2f}%")
        
        return result
    
    def _create_subset_historical_data(self, original_data, subset_candles):
        """Create a subset of historical data for backtesting"""
        from copy import deepcopy
        subset_data = deepcopy(original_data)
        subset_data.candles = subset_candles
        return subset_data
    
    def _check_exit_conditions(self, position: TradeModel, current_bar: pd.Series) -> bool:
        """Check if position should be closed based on stop loss or take profit"""
        # We don't have the original signal with stop loss and take profit
        # In a real implementation, you would store these values in the trade
        # For now, we'll simulate basic exit conditions
        
        # If position has been open for more than 10 bars, close it
        # This is a placeholder - in real implementation you'd use actual stops
        return True
    
    def _calculate_exit_price(self, position: TradeModel, current_bar: pd.Series, next_bar: pd.Series) -> Tuple[float, datetime]:
        """Calculate exit price and time based on next bar"""
        # For simplicity, we'll use the next bar's open price
        exit_price = next_bar['open']
        exit_time = next_bar.name
        return exit_price, exit_time
    
    def _calculate_pnl(self, position: TradeModel, exit_price: float) -> float:
        """Calculate P&L for a position"""
        if position.side == OrderSide.BUY:
            return position.quantity_asset * (exit_price - position.entry_price)
        else:  # SELL
            return position.quantity_asset * (position.entry_price - exit_price)
    
    def _calculate_performance_metrics(self, result: BacktestResult) -> None:
        """Calculate performance metrics for the backtest"""
        if not result.trades:
            return
            
        # Basic metrics
        result.total_trades = len(result.trades)
        result.winning_trades = sum(1 for t in result.trades if t.pnl_usd and t.pnl_usd > 0)
        result.losing_trades = sum(1 for t in result.trades if t.pnl_usd and t.pnl_usd <= 0)
        
        # Win rate
        result.win_rate = (result.winning_trades / result.total_trades) * 100 if result.total_trades > 0 else 0
        
        # Profit factor
        total_profit = sum(t.pnl_usd for t in result.trades if t.pnl_usd and t.pnl_usd > 0)
        total_loss = abs(sum(t.pnl_usd for t in result.trades if t.pnl_usd and t.pnl_usd <= 0))
        result.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average win/loss
        result.average_win = total_profit / result.winning_trades if result.winning_trades > 0 else 0
        result.average_loss = total_loss / result.losing_trades if result.losing_trades > 0 else 0
        
        # Total return
        result.total_return_pct = ((result.final_capital / result.initial_capital) - 1) * 100
        
        # CAGR (Compound Annual Growth Rate)
        if result.start_date and result.end_date:
            years = (result.end_date - result.start_date).days / 365.25
            if years > 0:
                result.cagr = (pow(result.final_capital / result.initial_capital, 1/years) - 1) * 100
                
        # R-multiple metrics (based on risk)
        r_multiples = []
        for trade in result.trades:
            if not trade.pnl_usd:
                continue
                
            # Assume 1R = 1% of account at trade entry
            r = trade.pnl_usd / (result.initial_capital * 0.01)
            r_multiples.append(r)
            
        if r_multiples:
            result.r_multiple_avg = sum(r_multiples) / len(r_multiples)
            result.r_multiple_stdev = np.std(r_multiples) if len(r_multiples) > 1 else 0


def run_backtest_demo():
    """Demo function to run a backtest"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize agents
    from backend.agents.market_data_agent import MarketDataAgent
    from backend.agents.signal_scanner_agent import SignalScannerAgent
    from backend.agents.risk_manager_agent import RiskManagerAgent
    
    market_agent = MarketDataAgent()
    scanner_agent = SignalScannerAgent(market_data_agent=market_agent)
    risk_agent = RiskManagerAgent(
        account_balance_usd=10000.0,
        max_risk_per_trade_pct=0.02,
        min_reward_to_risk_ratio=1.2
    )
    
    # Create backtester
    backtester = Backtester(market_agent, scanner_agent, risk_agent)
    
    # Run backtest for BTC-USD on daily timeframe for 2023
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2023, 12, 31, tzinfo=timezone.utc)
    
    result = backtester.backtest_strategy(
        asset=AssetSymbol.BTC_USD,
        timeframe=Timeframe.DAY_1,
        start_date=start_date,
        end_date=end_date,
        strategy_name="Enhanced Breakout Strategy"
    )
    
    # Print results
    print(f"\nBacktest Results for {result.strategy_name} on {result.asset.value} {result.timeframe.value}")
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Initial Capital: ${result.initial_capital:.2f}")
    print(f"Final Capital: ${result.final_capital:.2f}")
    print(f"Total Return: {result.total_return_pct:.2f}%")
    print(f"CAGR: {result.cagr:.2f}%")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Average Win: ${result.average_win:.2f}")
    print(f"Average Loss: ${result.average_loss:.2f}")
    print(f"R-Multiple Avg: {result.r_multiple_avg:.2f}")
    
    # Return result for further analysis
    return result


if __name__ == "__main__":
    run_backtest_demo()
