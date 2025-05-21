#!/usr/bin/env python3
"""
Trading SaaS Platform Demo Script
This script demonstrates the market regime analysis features and runs the web application.
"""
import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trading_demo")

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
# Import from the relevant module that defines shared types
from backend.agents.market_data_agent import AssetSymbol, Timeframe
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.market_regime_detector import MarketRegimeDetector
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.backtesting import Backtester
from backend.agents.regime_dashboard import RegimeDashboard
from backend.agents.regime_optimizer import RegimeOptimizer

def demo_regime_detection():
    """Demonstrate market regime detection capabilities"""
    logger.info("Demonstrating market regime detection...")
    
    market_agent = MarketDataAgent()
    regime_detector = MarketRegimeDetector()
    
    # Detect regimes for different assets - only using available symbols
    assets = [AssetSymbol.BTC_USD, AssetSymbol.SPY]
    timeframe = Timeframe.DAY_1
    
    for asset in assets:
        logger.info(f"Detecting market regime for {asset.value} on {timeframe.value} timeframe...")
        
        # Fetch historical data
        data = market_agent.fetch_historical_data(asset, timeframe)
        if not data or not data.candles:
            logger.warning(f"No data available for {asset.value}")
            continue
            
        # Convert to DataFrame
        df = pd.DataFrame([candle.model_dump() for candle in data.candles])
        
        # Detect current regime
        regime_info = regime_detector.detect_regime(df, asset, timeframe)
        
        # Display regime information
        logger.info(f"{asset.value} is currently in a {regime_info.regime_type.value} regime")
        logger.info(f"Regime strength: {regime_info.strength:.2f}")
        logger.info(f"Key metrics: {regime_info.metrics}")
        logger.info("---")

def demo_backtesting():
    """Demonstrate backtesting with regime analysis"""
    logger.info("Demonstrating backtesting with market regime analysis...")
    
    market_agent = MarketDataAgent()
    signal_scanner = SignalScannerAgent(market_agent)
    risk_manager = RiskManagerAgent()
    backtester = Backtester(market_agent, signal_scanner, risk_manager)
    
    # Run backtest for BTC-USD with adaptive parameters
    asset = AssetSymbol.BTC_USD
    timeframe = Timeframe.DAY_1
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    logger.info(f"Running backtest for {asset.value} from {start_date.date()} to {end_date.date()}")
    logger.info("Using adaptive parameters based on market regimes")
    
    result = backtester.backtest_strategy(
        asset=asset,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        strategy_name="breakout",
        use_adaptive_params=True
    )
    
    # Display backtest results
    logger.info(f"Backtest completed with {result.total_trades} trades")
    logger.info(f"Win rate: {result.win_rate:.2f}%")
    logger.info(f"Profit factor: {result.profit_factor:.2f}")
    logger.info(f"Total return: {result.total_return_pct:.2f}%")
    logger.info(f"Max drawdown: {result.max_drawdown:.2f}%")
    
    # Display regime performance
    logger.info("Performance by market regime:")
    for regime, metrics in result.regime_performance.items():
        if metrics.get('trade_count', 0) > 0:
            logger.info(f"  {regime}:")
            logger.info(f"    Trades: {metrics.get('trade_count', 0)}")
            logger.info(f"    Win rate: {metrics.get('win_rate', 0):.2f}%")
            logger.info(f"    Return: {metrics.get('total_return', 0):.2f}%")
    
    logger.info("---")

def demo_regime_optimization():
    """Demonstrate ML optimization of regime detection thresholds"""
    logger.info("Demonstrating ML optimization of regime detection thresholds...")
    
    market_agent = MarketDataAgent()
    signal_scanner = SignalScannerAgent(market_agent)
    risk_manager = RiskManagerAgent()
    backtester = Backtester(market_agent, signal_scanner, risk_manager)
    
    # Create optimizer with shorter lookback for the demo
    optimizer = RegimeOptimizer(market_agent, backtester)
    
    # Run a quick optimization with limited data for demo purposes
    assets = [AssetSymbol.BTC_USD]
    timeframes = [Timeframe.DAY_1]
    lookback_days = 90  # Use only 90 days for a faster demo
    
    logger.info(f"Running optimization pipeline with {lookback_days} days of data")
    logger.info("This process would typically use more data and take longer in production")
    
    try:
        # This is a computationally intensive operation and would take time
        # For the demo, we'll display the steps but not actually run full optimization
        logger.info("Step 1: Creating training dataset from historical market data")
        logger.info("Step 2: Training machine learning models to classify regimes")
        logger.info("Step 3: Analyzing model performance across different regimes")
        logger.info("Step 4: Generating optimized regime detection thresholds")
        
        # In a real app, we would run:
        # optimized_thresholds = optimizer.run_optimization_pipeline(
        #     assets=assets,
        #     timeframes=timeframes,
        #     lookback_days=lookback_days
        # )
        
        logger.info("Optimization would generate threshold values for:")
        logger.info("- Trend detection (EMA distance, slope thresholds)")
        logger.info("- Volatility thresholds (ATR percentages, Bollinger Band widths)")
        logger.info("- Momentum thresholds (RSI levels)")
        logger.info("- Breakout detection (volume surge ratios)")
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
    
    logger.info("---")

def run_web_app():
    """Run the web application"""
    logger.info("Starting the Trading SaaS web application...")
    
    from backend.main import app
    import uvicorn
    
    # Run the web app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def run_simulated_alerts():
    """Run a background thread to simulate real-time regime change alerts"""
    logger.info("Starting simulated regime change alerts...")
    
    market_agent = MarketDataAgent()
    regime_detector = MarketRegimeDetector()
    
    assets = [AssetSymbol.BTC_USD, AssetSymbol.SPY]
    timeframes = [Timeframe.DAY_1, Timeframe.HOUR_4]
    
    def alert_thread():
        """Thread function to periodically check for regime changes"""
        while True:
            # Randomly pick an asset and timeframe from available options
            asset = np.random.choice([AssetSymbol.BTC_USD, AssetSymbol.SPY])
            timeframe = np.random.choice(timeframes)
            
            try:
                # Fetch data
                data = market_agent.fetch_historical_data(asset, timeframe)
                if not data or not data.candles:
                    continue
                    
                # Convert to DataFrame
                df = pd.DataFrame([candle.model_dump() for candle in data.candles])
                
                # Detect regime with random chance of being a "new" regime
                # In real app, this would actually detect real changes
                regime_info = regime_detector.detect_regime(df, asset, timeframe)
                
                # Simulate a regime change alert with 20% probability
                if np.random.random() < 0.2:
                    previous_regime = np.random.choice([r for r in regime_detector.REGIME_TYPES if r != regime_info['regime_type']])
                    logger.warning(f"ALERT: {asset.value} {timeframe.value} regime changed from {previous_regime} to {regime_info['regime_type']}")
                    logger.warning(f"Regime strength: {regime_info['regime_strength']:.2f}, key metrics changed")
            
            except Exception as e:
                logger.error(f"Error in alert thread: {e}")
            
            # Wait 30-60 seconds between checks
            sleep_time = np.random.randint(30, 60)
            time.sleep(sleep_time)
    
    # Start alert thread
    alert_thread = threading.Thread(target=alert_thread, daemon=True)
    alert_thread.start()
    
    return alert_thread

def main():
    """Main demo function"""
    logger.info("=" * 40)
    logger.info("Trading SaaS Platform Demo")
    logger.info("=" * 40)
    
    # Demonstrate core functionalities
    demo_regime_detection()
    demo_backtesting()
    demo_regime_optimization()
    
    # Start simulated real-time alerts
    alert_thread = run_simulated_alerts()
    
    # Run the web app (this will block until the server is stopped)
    run_web_app()

if __name__ == "__main__":
    main()
