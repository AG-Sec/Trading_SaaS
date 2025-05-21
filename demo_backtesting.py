#!/usr/bin/env python3
"""
Backtesting Demo Script for Trading SaaS

This script demonstrates the backtesting functionality with the implemented
trading strategies (Mean Reversion and Trend Following).
"""

import logging
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

from shared_types.models import AssetSymbol, Timeframe
from backend.agents.market_data_agent import MarketDataAgent
from backend.strategies.mean_reversion_strategy import MeanReversionStrategy
from backend.strategies.trend_following_strategy import TrendFollowingStrategy
from backend.backtesting.backtester import Backtester, BacktestResult
from backend.backtesting.visualization import BacktestVisualizer
from backend.adaptive_engine.adaptive_engine import AdaptiveEngine
from backend.agents.market_regime_detector import MarketRegimeType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure output directory for visualizations
VISUALIZATION_DIR = os.path.join(os.path.dirname(__file__), 'backtesting_results')
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def run_backtests():
    """Run backtests for different strategies and generate visualizations."""
    logger.info("Starting backtesting demonstration")
    
    # Initialize agents and data
    market_data_agent = MarketDataAgent()
    
    # Import additional required agents
    from backend.agents.signal_scanner_agent import SignalScannerAgent
    from backend.agents.risk_manager_agent import RiskManagerAgent
    from backend.agents.market_regime_detector import MarketRegimeDetector
    
    # Initialize additional agents
    signal_scanner_agent = SignalScannerAgent(market_data_agent=market_data_agent)
    risk_manager_agent = RiskManagerAgent()
    market_regime_detector = MarketRegimeDetector()
    
    # Define test parameters
    assets = [
        AssetSymbol("BTC-USD"),
        AssetSymbol("ETH-USD"),
        AssetSymbol("EURUSD=X"),
        AssetSymbol("SPY")
    ]
    timeframes = [
        Timeframe("1h"),
        Timeframe("1d")
    ]
    
    # Calculate date range (last 6 months)
    end_date = datetime.now().replace(tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=180)
    
    logger.info(f"Testing period: {start_date.date()} to {end_date.date()}")
    
    # Initialize strategies
    mean_reversion = MeanReversionStrategy()
    trend_following = TrendFollowingStrategy()
    adaptive_engine = AdaptiveEngine(
        market_data_agent=market_data_agent,
        signal_scanner_agent=signal_scanner_agent,
        risk_manager_agent=risk_manager_agent
    )
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=10000.0,
        commission_rate=0.001,  # 0.1%
        slippage=0.0005,  # 0.05%
        risk_per_trade=0.02,  # 2% risk per trade
        max_open_trades=3,
        use_regime_detection=True
    )
    
    # Initialize visualizer
    visualizer = BacktestVisualizer(
        figsize=(12, 8),
        style='darkgrid',
        save_path=VISUALIZATION_DIR
    )
    
    # Store results for comparison
    all_results = {}
    
    # Run backtests for each asset/timeframe/strategy combination
    for asset in assets:
        for timeframe in timeframes:
            logger.info(f"Fetching historical data for {asset.value} {timeframe.value}")
            
            try:
                # Fetch historical data
                historical_data = market_data_agent.fetch_historical_data(
                    asset=asset,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True
                )
                
                if not historical_data or not historical_data.candles:
                    logger.warning(f"No data available for {asset.value} {timeframe.value}")
                    continue
                
                logger.info(f"Running backtests for {asset.value} {timeframe.value} with {len(historical_data.candles)} candles")
                
                # 1. Backtest Mean Reversion Strategy
                logger.info("Backtesting Mean Reversion Strategy")
                mr_result = backtester.run_backtest(
                    strategy=mean_reversion,
                    asset=asset,
                    timeframe=timeframe,
                    historical_data=historical_data,
                    start_date=start_date,
                    end_date=end_date
                )
                all_results[f"mean_reversion_{asset.value}_{timeframe.value}"] = mr_result
                
                # 2. Backtest Trend Following Strategy
                logger.info("Backtesting Trend Following Strategy")
                tf_result = backtester.run_backtest(
                    strategy=trend_following,
                    asset=asset,
                    timeframe=timeframe,
                    historical_data=historical_data,
                    start_date=start_date,
                    end_date=end_date
                )
                all_results[f"trend_following_{asset.value}_{timeframe.value}"] = tf_result
                
                # 3. Backtest Adaptive Engine (which selects between strategies)
                logger.info("Backtesting Adaptive Engine")
                ae_result = backtester.run_backtest(
                    strategy=adaptive_engine,
                    asset=asset,
                    timeframe=timeframe,
                    historical_data=historical_data,
                    start_date=start_date,
                    end_date=end_date
                )
                all_results[f"adaptive_engine_{asset.value}_{timeframe.value}"] = ae_result
                
                # Generate visualizations for this asset/timeframe
                generate_visualizations(
                    mr_result=mr_result,
                    tf_result=tf_result,
                    ae_result=ae_result,
                    asset=asset,
                    timeframe=timeframe,
                    visualizer=visualizer
                )
                
            except Exception as e:
                logger.error(f"Error during backtesting for {asset.value} {timeframe.value}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Compare overall performance across strategies
    compare_strategies(all_results)
    
    logger.info("Backtesting demonstration completed")
    return all_results

def generate_visualizations(mr_result: BacktestResult, 
                           tf_result: BacktestResult,
                           ae_result: BacktestResult,
                           asset: AssetSymbol,
                           timeframe: Timeframe,
                           visualizer: BacktestVisualizer):
    """Generate visualizations for backtesting results."""
    logger.info(f"Generating visualizations for {asset.value} {timeframe.value}")
    
    # Generate equity curves
    for result, strategy in [(mr_result, "mean_reversion"), 
                            (tf_result, "trend_following"),
                            (ae_result, "adaptive_engine")]:
        
        # Skip if no trades or insufficient equity data
        if not result.trades or len(result.equity_curve) < 2:
            logger.warning(f"Insufficient data for {strategy} visualizations")
            continue
        
        # 1. Equity curve with trades and regimes
        equity_img = visualizer.create_equity_curve(
            result=result,
            show_trades=True,
            show_drawdowns=True,
            show_regimes=True
        )
        visualizer.save_visualization(
            equity_img, 
            f"{strategy}_{asset.value}_{timeframe.value}_equity.png"
        )
        
        # 2. Regime performance analysis
        if result.regime_performance:
            regime_img = visualizer.create_regime_performance_chart(result)
            visualizer.save_visualization(
                regime_img,
                f"{strategy}_{asset.value}_{timeframe.value}_regimes.png"
            )
        
        # 3. Trade distribution analysis
        trade_img = visualizer.create_trade_distribution_chart(result)
        visualizer.save_visualization(
            trade_img,
            f"{strategy}_{asset.value}_{timeframe.value}_trades.png"
        )
        
        # 4. Monthly returns heatmap
        if len(result.equity_curve) >= 30:  # At least a month of data
            monthly_img = visualizer.create_monthly_returns_heatmap(result)
            visualizer.save_visualization(
                monthly_img,
                f"{strategy}_{asset.value}_{timeframe.value}_monthly.png"
            )
        
        # 5. Drawdown analysis
        if hasattr(result, 'drawdowns') and len(result.drawdowns) > 0:
            drawdown_img = visualizer.create_drawdown_chart(result)
            visualizer.save_visualization(
                drawdown_img,
                f"{strategy}_{asset.value}_{timeframe.value}_drawdowns.png"
            )
    
    # Generate comparison chart between strategies
    compare_strategies_for_asset(mr_result, tf_result, ae_result, asset, timeframe)

def compare_strategies_for_asset(mr_result: BacktestResult,
                               tf_result: BacktestResult,
                               ae_result: BacktestResult,
                               asset: AssetSymbol,
                               timeframe: Timeframe):
    """Compare the performance of different strategies for a specific asset."""
    logger.info(f"Comparing strategies for {asset.value} {timeframe.value}")
    
    # Check if we have sufficient data
    if (len(mr_result.equity_curve) < 2 or
        len(tf_result.equity_curve) < 2 or
        len(ae_result.equity_curve) < 2):
        logger.warning("Insufficient equity data for strategy comparison")
        return
    
    # Create figure for comparison
    plt.figure(figsize=(12, 8))
    
    # Normalize equity curves to 100
    mr_equity = mr_result.equity_curve.copy()
    tf_equity = tf_result.equity_curve.copy()
    ae_equity = ae_result.equity_curve.copy()
    
    mr_equity['normalized'] = mr_equity['equity'] / mr_equity['equity'].iloc[0] * 100
    tf_equity['normalized'] = tf_equity['equity'] / tf_equity['equity'].iloc[0] * 100
    ae_equity['normalized'] = ae_equity['equity'] / ae_equity['equity'].iloc[0] * 100
    
    # Plot equity curves
    plt.plot(mr_equity['date'], mr_equity['normalized'], label='Mean Reversion', linewidth=2)
    plt.plot(tf_equity['date'], tf_equity['normalized'], label='Trend Following', linewidth=2)
    plt.plot(ae_equity['date'], ae_equity['normalized'], label='Adaptive Engine', linewidth=2)
    
    # Add labels and title
    plt.title(f"Strategy Comparison - {asset.value} {timeframe.value}", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Equity (100 = Start)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add metrics text box
    metrics_text = (
        f"Mean Reversion: {mr_result.metrics.get('total_return', 0):.2f}% Return, "
        f"{mr_result.metrics.get('win_rate', 0)*100:.1f}% Win Rate\n"
        f"Trend Following: {tf_result.metrics.get('total_return', 0):.2f}% Return, "
        f"{tf_result.metrics.get('win_rate', 0)*100:.1f}% Win Rate\n"
        f"Adaptive Engine: {ae_result.metrics.get('total_return', 0):.2f}% Return, "
        f"{ae_result.metrics.get('win_rate', 0)*100:.1f}% Win Rate"
    )
    
    plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Save figure
    output_path = os.path.join(VISUALIZATION_DIR, f"strategy_comparison_{asset.value}_{timeframe.value}.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved strategy comparison to {output_path}")

def compare_strategies(results: Dict[str, BacktestResult]):
    """Compare strategy performance across all assets and timeframes."""
    logger.info("Comparing overall strategy performance")
    
    # Group results by strategy
    strategy_results = {
        "mean_reversion": [],
        "trend_following": [],
        "adaptive_engine": []
    }
    
    for key, result in results.items():
        if key.startswith("mean_reversion_"):
            strategy_results["mean_reversion"].append(result)
        elif key.startswith("trend_following_"):
            strategy_results["trend_following"].append(result)
        elif key.startswith("adaptive_engine_"):
            strategy_results["adaptive_engine"].append(result)
    
    # Calculate average metrics for each strategy
    avg_metrics = {}
    
    for strategy, result_list in strategy_results.items():
        if not result_list:
            continue
            
        total_trades = sum(len([t for t in r.trades if t.status != "open"]) for r in result_list)
        
        if total_trades == 0:
            avg_metrics[strategy] = {
                "avg_return": 0,
                "avg_win_rate": 0,
                "avg_profit_factor": 0,
                "total_trades": 0
            }
            continue
        
        # Calculate weighted averages
        weighted_return = sum(r.metrics.get('total_return', 0) * len(r.trades) for r in result_list) / total_trades
        weighted_win_rate = sum(r.metrics.get('win_rate', 0) * len(r.trades) for r in result_list) / total_trades
        weighted_profit_factor = sum(r.metrics.get('profit_factor', 1) * len(r.trades) for r in result_list) / total_trades
        
        avg_metrics[strategy] = {
            "avg_return": weighted_return,
            "avg_win_rate": weighted_win_rate,
            "avg_profit_factor": weighted_profit_factor,
            "total_trades": total_trades
        }
    
    # Compare performance by market regime
    regime_performance = analyze_regime_performance(results)
    
    # Create summary table
    create_summary_table(avg_metrics, regime_performance)

def analyze_regime_performance(results: Dict[str, BacktestResult]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze strategy performance by market regime."""
    logger.info("Analyzing performance by market regime")
    
    # Initialize structure for regime performance
    regime_performance = {
        "mean_reversion": {},
        "trend_following": {},
        "adaptive_engine": {}
    }
    
    for key, result in results.items():
        strategy = None
        if key.startswith("mean_reversion_"):
            strategy = "mean_reversion"
        elif key.startswith("trend_following_"):
            strategy = "trend_following"
        elif key.startswith("adaptive_engine_"):
            strategy = "adaptive_engine"
        else:
            continue
        
        # Process regime-specific performance
        for regime, metrics in result.regime_performance.items():
            if regime not in regime_performance[strategy]:
                regime_performance[strategy][regime] = {
                    "win_rate": [],
                    "profit_factor": [],
                    "trade_count": []
                }
            
            regime_performance[strategy][regime]["win_rate"].append(metrics.get('win_rate', 0))
            regime_performance[strategy][regime]["profit_factor"].append(metrics.get('profit_factor', 1))
            regime_performance[strategy][regime]["trade_count"].append(metrics.get('total_trades', 0))
    
    # Calculate averages for each regime and strategy
    for strategy in regime_performance:
        for regime in regime_performance[strategy]:
            metrics = regime_performance[strategy][regime]
            
            total_trades = sum(metrics["trade_count"])
            
            if total_trades > 0:
                # Calculate weighted averages
                avg_win_rate = sum(metrics["win_rate"][i] * metrics["trade_count"][i] 
                                for i in range(len(metrics["win_rate"]))) / total_trades
                                
                avg_profit_factor = sum(metrics["profit_factor"][i] * metrics["trade_count"][i] 
                                      for i in range(len(metrics["profit_factor"]))) / total_trades
            else:
                avg_win_rate = 0
                avg_profit_factor = 0
            
            # Replace lists with averages
            regime_performance[strategy][regime] = {
                "avg_win_rate": avg_win_rate,
                "avg_profit_factor": avg_profit_factor,
                "total_trades": total_trades
            }
    
    return regime_performance

def create_summary_table(avg_metrics: Dict[str, Dict[str, float]], 
                        regime_performance: Dict[str, Dict[str, Dict[str, float]]]):
    """Create and save a summary table of performance metrics."""
    logger.info("Creating performance summary table")
    
    # Check if we have any metrics
    if not avg_metrics or all(not metrics for metrics in avg_metrics.values()):
        logger.warning("No metrics available for summary table")
        return
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Overall performance table
    strategies = list(avg_metrics.keys())
    metrics = ['avg_return', 'avg_win_rate', 'avg_profit_factor', 'total_trades']
    
    # Prepare data for table
    table_data = []
    for strategy in strategies:
        if strategy not in avg_metrics:
            continue
        row = [
            f"{avg_metrics[strategy].get('avg_return', 0):.2f}%",
            f"{avg_metrics[strategy].get('avg_win_rate', 0)*100:.1f}%",
            f"{avg_metrics[strategy].get('avg_profit_factor', 0):.2f}",
            f"{avg_metrics[strategy].get('total_trades', 0)}"
        ]
        table_data.append(row)
        
    # If no data, return
    if not table_data:
        logger.warning("No table data available for summary")
        return
    
    # Create table
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(
        cellText=table_data,
        rowLabels=[s.replace('_', ' ').title() for s in strategies],
        colLabels=['Return', 'Win Rate', 'Profit Factor', 'Total Trades'],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    
    ax1.set_title('Overall Strategy Performance', fontsize=16)
    
    # 2. Regime-specific performance
    # Prepare data for regime table
    regimes = set()
    for strategy in regime_performance:
        regimes.update(regime_performance[strategy].keys())
    
    regimes = sorted(list(regimes))
    
    # Create data for each regime
    regime_table_data = []
    
    for regime in regimes:
        row = []
        for strategy in strategies:
            if strategy in regime_performance and regime in regime_performance[strategy]:
                metrics = regime_performance[strategy][regime]
                cell_text = (
                    f"WR: {metrics['avg_win_rate']*100:.1f}%\n"
                    f"PF: {metrics['avg_profit_factor']:.2f}\n"
                    f"N: {metrics['total_trades']}"
                )
                row.append(cell_text)
            else:
                row.append("N/A")
        
        regime_table_data.append(row)
    
    # Create table
    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(
        cellText=regime_table_data,
        rowLabels=[r.replace('_', ' ').title() for r in regimes],
        colLabels=[s.replace('_', ' ').title() for s in strategies],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1, 1.8)
    
    ax2.set_title('Performance by Market Regime', fontsize=16)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(VISUALIZATION_DIR, "performance_summary.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved performance summary to {output_path}")

if __name__ == "__main__":
    results = run_backtests()
    print(f"Backtesting completed. Results saved to {VISUALIZATION_DIR}")
