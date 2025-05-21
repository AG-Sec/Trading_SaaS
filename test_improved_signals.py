#!/usr/bin/env python3
"""
Improved Signal Generation Test Script
This script tests the enhanced signal generation with the new parameters.
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
import json

from shared_types import AssetSymbol, Timeframe, SignalType
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.journal_agent import JournalAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("signal_test")

def test_improved_signals():
    """Test the improved signal generation implementation"""
    logger.info("=" * 80)
    logger.info("Testing improved signal generation")
    logger.info("=" * 80)
    
    # Initialize agents
    market_agent = MarketDataAgent()
    scanner_agent = SignalScannerAgent(market_data_agent=market_agent)
    journal_agent = JournalAgent()
    risk_agent = RiskManagerAgent(
        account_balance_usd=1000.0,
        journal_agent=journal_agent
    )
    
    # Test all assets
    assets = [
        AssetSymbol.BTC_USD,
        AssetSymbol.ETH_USD,
        AssetSymbol.SPY,
        AssetSymbol.EUR_USD_FX
    ]
    
    # Test all timeframes
    timeframes = [
        Timeframe.MIN_15,
        Timeframe.HOUR_1,
        Timeframe.HOUR_4,
        Timeframe.DAY_1
    ]
    
    # Track results
    all_signals = []
    total_raw_signals = 0
    total_approved_signals = 0
    
    # Test each asset and timeframe combination
    for asset in assets:
        asset_signals = 0
        asset_approved = 0
        
        logger.info(f"Testing asset: {asset.value}")
        logger.info("-" * 40)
        
        for timeframe in timeframes:
            logger.info(f"  Timeframe: {timeframe.value}")
            
            # Fetch historical data
            data = market_agent.fetch_historical_data(asset, timeframe)
            if not data or len(data.candles) < 30:
                logger.warning(f"    Not enough data for {asset.value} on {timeframe.value}")
                continue
                
            # Generate signals
            raw_signals = scanner_agent.scan_for_breakout_signals(asset, timeframe, data)
            raw_signal_count = len(raw_signals)
            total_raw_signals += raw_signal_count
            asset_signals += raw_signal_count
            
            # Evaluate signals with risk manager
            approved_signals = []
            for signal in raw_signals:
                approved_signal = risk_agent.evaluate_signal(signal)
                if approved_signal:
                    approved_signals.append(approved_signal)
                    
            approved_count = len(approved_signals)
            total_approved_signals += approved_count
            asset_approved += approved_count
            
            # Log results
            logger.info(f"    Raw signals: {raw_signal_count}, Approved: {approved_count}")
            
            # Store signals for review
            all_signals.extend([s.model_dump() for s in approved_signals])
        
        # Log asset summary
        if asset_signals > 0:
            approval_rate = (asset_approved / asset_signals) * 100
            logger.info(f"  {asset.value} summary: {asset_signals} signals, {asset_approved} approved ({approval_rate:.1f}%)")
        else:
            logger.info(f"  {asset.value} summary: No signals generated")
        
        logger.info("")
    
    # Log overall summary
    logger.info("=" * 80)
    if total_raw_signals > 0:
        overall_rate = (total_approved_signals / total_raw_signals) * 100
        logger.info(f"Overall results: {total_raw_signals} signals, {total_approved_signals} approved ({overall_rate:.1f}%)")
    else:
        logger.info("Overall results: No signals generated")
    
    logger.info("=" * 80)
    
    # Save approved signals to file for review
    if all_signals:
        # Custom JSON encoder for datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        with open('improved_signals.json', 'w') as f:
            json.dump(all_signals, f, indent=2, cls=DateTimeEncoder)
        logger.info(f"Saved {len(all_signals)} approved signals to improved_signals.json")

if __name__ == "__main__":
    test_improved_signals()
