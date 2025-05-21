"""
Visualization Module for Backtesting Results

This module provides functions for visualizing backtesting results including
equity curves, trade distributions, regime analysis, and performance metrics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from io import BytesIO
import base64

from backend.backtesting.backtester import BacktestTrade, BacktestResult, TradeStatus
from backend.agents.market_regime_detector import MarketRegimeType

logger = logging.getLogger(__name__)

class BacktestVisualizer:
    """
    Creates visualizations for backtest results.
    """
    
    def __init__(self, 
                figsize: Tuple[int, int] = (12, 8),
                style: str = 'darkgrid',
                palette: str = 'viridis',
                save_path: Optional[str] = None):
        """
        Initialize the BacktestVisualizer.
        
        Args:
            figsize: Default figure size (width, height)
            style: Seaborn style for plots
            palette: Color palette for plots
            save_path: Path to save visualizations
        """
        self.figsize = figsize
        self.style = style
        self.palette = palette
        self.save_path = save_path
        
        # Set default style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        
        # Define color schemes for different regimes
        self.regime_colors = {
            'BULLISH_TRENDING': '#26A69A',  # Teal
            'BEARISH_TRENDING': '#EF5350',  # Red
            'HIGH_VOLATILITY': '#7E57C2',   # Purple
            'LOW_VOLATILITY': '#42A5F5',    # Blue
            'NEUTRAL_RANGING': '#9E9E9E'    # Grey
        }
    
    def create_equity_curve(self, 
                          result: BacktestResult,
                          show_trades: bool = True,
                          show_drawdowns: bool = True,
                          show_regimes: bool = True) -> str:
        """
        Create equity curve visualization with trades and regimes.
        
        Args:
            result: Backtest result to visualize
            show_trades: Whether to show trades on the chart
            show_drawdowns: Whether to highlight drawdowns
            show_regimes: Whether to show market regime background
            
        Returns:
            Base64 encoded PNG image
        """
        if len(result.equity_curve) < 2:
            logger.warning("Insufficient data for equity curve visualization")
            return self._create_error_chart("Insufficient data for equity curve")
        
        # Set up the plot
        plt.close('all')
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot equity curve
        equity_data = result.equity_curve.copy()
        
        if 'date' in equity_data.columns and 'equity' in equity_data.columns:
            ax.plot(equity_data['date'], equity_data['equity'], 
                   linewidth=2, color='#2196F3', label='Equity')
            
            # Format axis
            ax.set_title(f"Equity Curve - {result.strategy_name} on {result.asset.value}", fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Equity', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Format dates on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            
            # Show trades if requested
            if show_trades and result.trades:
                self._add_trades_to_equity_curve(ax, result.trades, equity_data)
            
            # Show drawdowns if requested
            if show_drawdowns and hasattr(result, 'drawdowns') and len(result.drawdowns) > 0:
                self._add_drawdowns_to_equity_curve(ax, result.drawdowns)
            
            # Show market regimes if requested
            if show_regimes and 'market_regime' in equity_data.columns:
                self._add_regimes_to_equity_curve(ax, equity_data)
            
            # Add metrics summary
            self._add_metrics_summary(ax, result.metrics)
            
            # Adjust layout and add legend
            plt.tight_layout()
            plt.legend()
            
            # Convert plot to base64 image
            img_data = self._fig_to_base64(fig)
            plt.close(fig)
            
            return img_data
        else:
            logger.warning("Equity curve data missing required columns")
            return self._create_error_chart("Equity curve data missing required columns")
    
    def create_regime_performance_chart(self, result: BacktestResult) -> str:
        """
        Create a visualization of performance across different market regimes.
        
        Args:
            result: Backtest result to visualize
            
        Returns:
            Base64 encoded PNG image
        """
        if not result.regime_performance:
            logger.warning("No regime performance data available")
            return self._create_error_chart("No regime performance data available")
        
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Extract regime performance data
        regimes = list(result.regime_performance.keys())
        win_rates = [result.regime_performance[r].get('win_rate', 0) * 100 for r in regimes]
        profit_factors = [result.regime_performance[r].get('profit_factor', 0) for r in regimes]
        trade_counts = [result.regime_performance[r].get('total_trades', 0) for r in regimes]
        
        # Create bar charts
        colors = [self.regime_colors.get(r, '#9E9E9E') for r in regimes]
        
        # Win Rate by Regime
        bars1 = ax1.bar(regimes, win_rates, color=colors, alpha=0.8)
        
        # Add trade count labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{trade_counts[i]}",
                    ha='center', va='bottom', fontsize=10,
                    rotation=0)
        
        ax1.set_title('Win Rate by Market Regime', fontsize=14)
        ax1.set_ylabel('Win Rate (%)', fontsize=12)
        ax1.set_ylim(0, max(win_rates) * 1.2 if win_rates else 100)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Profit Factor by Regime
        bars2 = ax2.bar(regimes, profit_factors, color=colors, alpha=0.8)
        
        # Add trade count labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{trade_counts[i]}",
                    ha='center', va='bottom', fontsize=10,
                    rotation=0)
        
        ax2.set_title('Profit Factor by Market Regime', fontsize=14)
        ax2.set_ylabel('Profit Factor', fontsize=12)
        ax2.set_ylim(0, max(profit_factors) * 1.2 if profit_factors else 5)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add overall title
        fig.suptitle(f"Performance by Market Regime - {result.strategy_name}", fontsize=16, y=1.05)
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data
    
    def create_trade_distribution_chart(self, result: BacktestResult) -> str:
        """
        Create a visualization of trade distributions and outcomes.
        
        Args:
            result: Backtest result to visualize
            
        Returns:
            Base64 encoded PNG image
        """
        if not result.trades:
            logger.warning("No trade data available")
            return self._create_error_chart("No trade data available")
        
        # Filter to closed trades only
        closed_trades = [t for t in result.trades if t.status != TradeStatus.OPEN]
        
        if not closed_trades:
            logger.warning("No closed trades available")
            return self._create_error_chart("No closed trades available")
        
        plt.close('all')
        fig = plt.figure(figsize=self.figsize)
        grid = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. P&L Distribution
        ax1 = fig.add_subplot(grid[0, 0])
        pnl_values = [t.pnl_absolute for t in closed_trades]
        sns.histplot(pnl_values, kde=True, ax=ax1, color='#2196F3')
        ax1.set_title('P&L Distribution', fontsize=12)
        ax1.set_xlabel('P&L (absolute)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        
        # 2. Win/Loss by Regime
        ax2 = fig.add_subplot(grid[0, 1])
        
        # Group trades by regime
        regime_trades = {}
        for trade in closed_trades:
            if not trade.market_regime:
                continue
                
            if trade.market_regime not in regime_trades:
                regime_trades[trade.market_regime] = {
                    'win': 0, 'loss': 0
                }
                
            if trade.pnl_absolute > 0:
                regime_trades[trade.market_regime]['win'] += 1
            else:
                regime_trades[trade.market_regime]['loss'] += 1
        
        if regime_trades:
            regimes = list(regime_trades.keys())
            wins = [regime_trades[r]['win'] for r in regimes]
            losses = [regime_trades[r]['loss'] for r in regimes]
            
            x = np.arange(len(regimes))
            width = 0.35
            
            ax2.bar(x - width/2, wins, width, label='Wins', color='#4CAF50')
            ax2.bar(x + width/2, losses, width, label='Losses', color='#F44336')
            
            ax2.set_title('Win/Loss by Regime', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(regimes, rotation=45, ha='right')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No regime data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)
        
        # 3. Trade Durations
        ax3 = fig.add_subplot(grid[1, 0])
        durations = []
        
        for trade in closed_trades:
            if trade.entry_time and trade.exit_time:
                duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                durations.append(duration_hours)
        
        if durations:
            sns.histplot(durations, kde=True, ax=ax3, color='#9C27B0')
            ax3.set_title('Trade Duration Distribution', fontsize=12)
            ax3.set_xlabel('Duration (hours)', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No duration data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes)
        
        # 4. Exit Types
        ax4 = fig.add_subplot(grid[1, 1])
        
        exit_types = {
            'Take Profit': len([t for t in closed_trades if t.status == TradeStatus.CLOSED_TP]),
            'Stop Loss': len([t for t in closed_trades if t.status == TradeStatus.CLOSED_SL]),
            'Exit Signal': len([t for t in closed_trades if t.status == TradeStatus.CLOSED_EXIT]),
            'Manual': len([t for t in closed_trades if t.status == TradeStatus.CLOSED_MANUAL]),
            'Expired': len([t for t in closed_trades if t.status == TradeStatus.EXPIRED])
        }
        
        labels = list(exit_types.keys())
        sizes = list(exit_types.values())
        
        # Filter out zero values
        non_zero_labels = [labels[i] for i in range(len(sizes)) if sizes[i] > 0]
        non_zero_sizes = [sizes[i] for i in range(len(sizes)) if sizes[i] > 0]
        
        if non_zero_sizes:
            ax4.pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%',
                   startangle=90, colors=['#4CAF50', '#F44336', '#2196F3', '#FFC107', '#9E9E9E'])
            ax4.set_title('Exit Types', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'No exit type data available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax4.transAxes)
        
        # Add overall title
        fig.suptitle(f"Trade Analysis - {result.strategy_name}", fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data
    
    def create_monthly_returns_heatmap(self, result: BacktestResult) -> str:
        """
        Create a heatmap of monthly returns.
        
        Args:
            result: Backtest result to visualize
            
        Returns:
            Base64 encoded PNG image
        """
        if len(result.equity_curve) < 30:  # Need at least a month of data
            logger.warning("Insufficient data for monthly returns heatmap")
            return self._create_error_chart("Insufficient data for monthly returns heatmap")
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate monthly returns
        equity_data = result.equity_curve.copy()
        
        if 'date' in equity_data.columns and 'equity' in equity_data.columns:
            equity_data['month'] = equity_data['date'].dt.month
            equity_data['year'] = equity_data['date'].dt.year
            
            # Get month-end equity for each year-month
            monthly_data = equity_data.groupby(['year', 'month']).last().reset_index()
            
            # Calculate returns
            monthly_data['return'] = monthly_data['equity'].pct_change() * 100
            
            # Create pivot table for heatmap
            pivot_data = monthly_data.pivot_table(
                values='return',
                index='year',
                columns='month',
                aggfunc='first'
            ).fillna(0)
            
            # Define color map (red for negative, green for positive)
            cmap = sns.diverging_palette(10, 133, as_cmap=True)
            
            # Create heatmap
            sns.heatmap(pivot_data, cmap=cmap, center=0, annot=True, fmt='.1f',
                       linewidths=0.5, ax=ax, cbar_kws={'label': 'Return (%)'})
            
            # Format axis
            ax.set_title(f"Monthly Returns (%) - {result.strategy_name}", fontsize=16)
            
            # Set month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_names, rotation=0)
            
            plt.tight_layout()
            
            # Convert plot to base64 image
            img_data = self._fig_to_base64(fig)
            plt.close(fig)
            
            return img_data
        else:
            logger.warning("Equity curve data missing required columns")
            return self._create_error_chart("Equity curve data missing required columns")
    
    def create_drawdown_chart(self, result: BacktestResult) -> str:
        """
        Create a visualization of drawdowns over time.
        
        Args:
            result: Backtest result to visualize
            
        Returns:
            Base64 encoded PNG image
        """
        if not hasattr(result, 'drawdowns') or len(result.drawdowns) < 2:
            logger.warning("No drawdown data available")
            return self._create_error_chart("No drawdown data available")
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot drawdowns
        drawdown_data = result.drawdowns.copy()
        
        if 'date' in drawdown_data.columns and 'drawdown' in drawdown_data.columns:
            ax.fill_between(drawdown_data['date'], drawdown_data['drawdown'],
                           color='#F44336', alpha=0.5)
            ax.plot(drawdown_data['date'], drawdown_data['drawdown'],
                   color='#D32F2F', linewidth=1, alpha=0.7)
            
            # Format axis
            ax.set_title(f"Drawdown Analysis - {result.strategy_name}", fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Invert y-axis as drawdowns are negative
            ax.invert_yaxis()
            
            # Format dates on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            
            # Add max drawdown line and annotation
            max_dd = drawdown_data['drawdown'].max()
            max_dd_date = drawdown_data.loc[drawdown_data['drawdown'].idxmax(), 'date']
            
            ax.axhline(y=max_dd, color='r', linestyle='--', alpha=0.7)
            ax.annotate(f'Max Drawdown: {max_dd:.2f}%',
                       xy=(max_dd_date, max_dd),
                       xytext=(30, 30),
                       textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            plt.tight_layout()
            
            # Convert plot to base64 image
            img_data = self._fig_to_base64(fig)
            plt.close(fig)
            
            return img_data
        else:
            logger.warning("Drawdown data missing required columns")
            return self._create_error_chart("Drawdown data missing required columns")
    
    def create_regime_equity_comparison(self, result: BacktestResult) -> str:
        """
        Create a comparison of equity curves across different market regimes.
        
        Args:
            result: Backtest result to visualize
            
        Returns:
            Base64 encoded PNG image
        """
        if 'market_regime' not in result.equity_curve.columns or len(result.equity_curve) < 2:
            logger.warning("No regime equity data available")
            return self._create_error_chart("No regime equity data available")
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data
        equity_data = result.equity_curve.copy()
        
        # Convert MarketRegime objects to string
        if 'market_regime' in equity_data.columns:
            equity_data['regime_type'] = equity_data['market_regime'].apply(
                lambda r: r.regime_type.value if hasattr(r, 'regime_type') else str(r)
            )
        
        # Group by regime type
        regime_groups = equity_data.groupby('regime_type')
        
        # Base starting equity at 100 for each regime
        for regime, group in regime_groups:
            if len(group) < 2:
                continue
                
            # Normalize equity to 100 at start
            start_equity = group['equity'].iloc[0]
            normalized_equity = (group['equity'] / start_equity) * 100
            
            # Plot normalized equity for this regime
            color = self.regime_colors.get(regime, '#9E9E9E')
            ax.plot(group['date'], normalized_equity, label=regime, color=color, linewidth=2)
        
        # Format axis
        ax.set_title(f"Comparative Performance by Regime - {result.strategy_name}", fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Equity (100 = Start)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add legend
        ax.legend(title='Market Regime')
        
        plt.tight_layout()
        
        # Convert plot to base64 image
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data
    
    def _add_trades_to_equity_curve(self, ax: plt.Axes, trades: List[BacktestTrade], equity_data: pd.DataFrame):
        """Add trade markers to equity curve."""
        # Find closest equity value for each trade
        for trade in trades:
            if trade.entry_time and trade.exit_time and trade.status != TradeStatus.OPEN:
                # Find entry and exit points on equity curve
                entry_idx = equity_data['date'].searchsorted(trade.entry_time)
                exit_idx = equity_data['date'].searchsorted(trade.exit_time)
                
                if entry_idx < len(equity_data) and exit_idx < len(equity_data):
                    entry_equity = equity_data['equity'].iloc[entry_idx]
                    exit_equity = equity_data['equity'].iloc[exit_idx]
                    
                    # Plot entry and exit points
                    if trade.signal_type.value == "long":
                        entry_marker = '^'  # Triangle up for long entry
                        exit_marker = 'v' if trade.pnl_absolute <= 0 else 'o'  # Triangle down for loss, circle for win
                    else:  # SHORT
                        entry_marker = 'v'  # Triangle down for short entry
                        exit_marker = '^' if trade.pnl_absolute <= 0 else 'o'  # Triangle up for loss, circle for win
                    
                    # Color based on outcome
                    exit_color = '#F44336' if trade.pnl_absolute <= 0 else '#4CAF50'
                    
                    # Plot markers
                    ax.scatter(trade.entry_time, entry_equity, 
                              marker=entry_marker, s=50, color='#FFC107', zorder=5)
                    ax.scatter(trade.exit_time, exit_equity, 
                              marker=exit_marker, s=50, color=exit_color, zorder=5)
    
    def _add_drawdowns_to_equity_curve(self, ax: plt.Axes, drawdowns: pd.DataFrame):
        """Highlight drawdown periods on equity curve."""
        if 'date' in drawdowns.columns and 'drawdown' in drawdowns.columns:
            # Find significant drawdown periods (e.g., > 5%)
            significant_dd = drawdowns[drawdowns['drawdown'] > 5]
            
            if len(significant_dd) > 0:
                # Group continuous drawdown periods
                period_start = None
                periods = []
                
                for i in range(len(significant_dd)):
                    if i == 0 or significant_dd.iloc[i-1]['drawdown'] <= 5:
                        period_start = significant_dd.iloc[i]['date']
                    
                    if i == len(significant_dd) - 1 or significant_dd.iloc[i+1]['drawdown'] <= 5:
                        period_end = significant_dd.iloc[i]['date']
                        periods.append((period_start, period_end))
                
                # Highlight each period
                for start, end in periods:
                    ax.axvspan(start, end, color='red', alpha=0.2)
    
    def _add_regimes_to_equity_curve(self, ax: plt.Axes, equity_data: pd.DataFrame):
        """Add market regime background colors to equity curve."""
        if 'market_regime' not in equity_data.columns:
            return
        
        # Convert MarketRegime objects to string
        equity_data['regime_type'] = equity_data['market_regime'].apply(
            lambda r: r.regime_type.value if hasattr(r, 'regime_type') else str(r)
        )
        
        # Find regime change points
        changes = []
        current_regime = None
        
        for i, row in equity_data.iterrows():
            if row['regime_type'] != current_regime:
                changes.append((i, row['date'], row['regime_type']))
                current_regime = row['regime_type']
        
        # Add colored background for each regime section
        for i in range(len(changes)):
            start_idx, start_date, regime = changes[i]
            
            if i < len(changes) - 1:
                end_date = changes[i+1][1]
            else:
                end_date = equity_data['date'].iloc[-1]
            
            color = self.regime_colors.get(regime, '#9E9E9E')
            ax.axvspan(start_date, end_date, alpha=0.1, color=color)
            
            # Add regime label in the middle of the span
            mid_date = start_date + (end_date - start_date) / 2
            y_pos = ax.get_ylim()[1] * 0.95  # Near the top
            
            ax.text(mid_date, y_pos, regime, 
                   horizontalalignment='center', verticalalignment='top',
                   fontsize=8, color=color, alpha=0.7, rotation=0)
    
    def _add_metrics_summary(self, ax: plt.Axes, metrics: Dict[str, Any]):
        """Add performance metrics summary to the plot."""
        if not metrics:
            return
        
        # Create summary text
        metrics_text = (
            f"Total Return: {metrics.get('total_return', 0):.2f}%\n"
            f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%\n"
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}"
        )
        
        # Add text box in top left
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str
    
    def _create_error_chart(self, error_message: str) -> str:
        """Create a simple chart displaying an error message."""
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.text(0.5, 0.5, error_message, 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=14)
        
        ax.set_axis_off()
        
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data
    
    def save_visualization(self, img_data: str, filename: str) -> str:
        """
        Save a visualization to a file.
        
        Args:
            img_data: Base64 encoded image
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        if not self.save_path:
            logger.warning("No save_path specified, cannot save visualization")
            return ""
        
        import os
        
        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)
        
        # Convert base64 to image and save
        img_bytes = base64.b64decode(img_data)
        file_path = os.path.join(self.save_path, filename)
        
        with open(file_path, 'wb') as f:
            f.write(img_bytes)
        
        logger.info(f"Saved visualization to {file_path}")
        return file_path
