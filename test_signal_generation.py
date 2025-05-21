#!/usr/bin/env python
"""
Test script to verify signal generation functionality
"""
import logging
import sys
from datetime import datetime, timedelta, timezone

from shared_types import AssetSymbol, Timeframe
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_signal_generation():
    """Test the complete signal generation pipeline for all assets and timeframes"""
    # Initialize agents with optimized parameters
    market_agent = MarketDataAgent()
    scanner_agent = SignalScannerAgent(market_data_agent=market_agent)
    risk_agent = RiskManagerAgent(
        account_balance_usd=1000.0,
        max_risk_per_trade_pct=0.02,     # More aggressive risk
        min_reward_to_risk_ratio=1.2,    # More permissive R:R
        max_signals_per_asset=3,         # Allow multiple signals per asset
        journal_agent=None                # Skip journaling for this test
    )
    
    # Assets and timeframes to test (from PRD)
    assets = [
        AssetSymbol.BTC_USD,
        AssetSymbol.ETH_USD,
        AssetSymbol.EUR_USD_FX,  # Correct enum for EURUSD=X
        AssetSymbol.SPY
    ]
    
    # Focus more on lower timeframes for testing as they provide more data points
    timeframes = [
        Timeframe.MIN_15,
        Timeframe.HOUR_1,
        Timeframe.HOUR_4,
        Timeframe.DAY_1
    ]
    
    # For 15m data, we need a shorter lookback due to API limitations
    # Define custom lookback periods per timeframe
    lookback_periods = {
        Timeframe.MIN_15: 7,  # 7 days for 15min data (API limitation)
        Timeframe.HOUR_1: 60,  # 60 days for hourly
        Timeframe.HOUR_4: 120,  # 120 days for 4h
        Timeframe.DAY_1: 180,  # 180 days for daily
    }
    
    total_signals = 0
    approved_signals = 0
    
    # Set longer lookback period to capture more patterns
    # For 15-period strategy, we need enough data to find patterns
    lookback_days = 180  # Increased from 60 to 180 days for more historical patterns
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)
    
    logger.info("=" * 80)
    logger.info(f"Starting signal generation test (lookback: {lookback_days} days)")
    logger.info("=" * 80)
    
    # Helper to process one asset-timeframe combo
    def process_combo(asset, timeframe):
        tf_lookback = lookback_periods[timeframe]
        tf_start = end_date - timedelta(days=tf_lookback)
        data = market_agent.fetch_historical_data(
            asset=asset, timeframe=timeframe,
            start_date=tf_start, end_date=end_date,
            use_cache=False
        )
        if not data or not data.candles:
            logger.warning(f"No data for {asset.value} {timeframe.value}")
            return 0, 0
        raw = scanner_agent.scan_for_breakout_signals(
            asset=asset, timeframe=timeframe, historical_data=data
        ) or []
        approved = risk_agent.filter_signals(raw) if raw else []
        return len(raw), len(approved)

    # Parallel execution
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for asset in assets:
            for timeframe in timeframes:
                tasks.append(executor.submit(process_combo, asset, timeframe))
        for future in as_completed(tasks):
            raw_count, appr_count = future.result()
            total_signals += raw_count
            approved_signals += appr_count

    logger.info("=" * 80)
    logger.info(f"Test complete: {total_signals} raw signals across combos, {approved_signals} approved")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_signal_generation()
