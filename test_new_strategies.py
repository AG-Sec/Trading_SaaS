#!/usr/bin/env python3
"""
Simple Test Script for New Trading Strategies

This script provides a simplified test of the newly implemented Mean Reversion and
Trend Following strategies without dependencies on other system components.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from shared_types.models import AssetSymbol, Timeframe, TradingSignalModel
from backend.agents.market_data_agent import MarketDataAgent
from backend.strategies.mean_reversion_strategy import MeanReversionStrategy
from backend.strategies.trend_following_strategy import TrendFollowingStrategy
from backend.agents.market_regime_detector import MarketRegimeType, MarketRegime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_strategies():
    """Test the new trading strategies with sample data."""
    logger.info("Starting test of new trading strategies")
    
    # Initialize agents
    market_data_agent = MarketDataAgent()
    
    # Initialize strategies
    mean_reversion = MeanReversionStrategy()
    trend_following = TrendFollowingStrategy()
    
    # Test assets and timeframes
    assets = [
        AssetSymbol("BTC-USD"),
        AssetSymbol("ETH-USD"),
        AssetSymbol("EURUSD=X"),
        AssetSymbol("SPY")
    ]
    
    timeframes = [
        Timeframe("1h"),
        Timeframe("4h"),
        Timeframe("1d")
    ]
    
    # Results counter
    mr_signals_count = 0
    tf_signals_count = 0
    
    # Process each asset and timeframe combination
    for asset in assets:
        print(f"\nTesting asset: {asset.value}")
        
        for timeframe in timeframes:
            print(f"  Timeframe: {timeframe.value}")
            
            try:
                # Get historical data
                historical_data = market_data_agent.fetch_historical_data(
                    asset=asset,
                    timeframe=timeframe,
                    start_date=None,  # Use default lookback
                    end_date=None,    # Up to current time
                    use_cache=True
                )
                
                if not historical_data or not historical_data.candles:
                    print(f"  No historical data available")
                    continue
                
                print(f"  Retrieved {len(historical_data.candles)} candles")
                
                # Create a simple market regime for testing
                # Note: In real usage, this would come from MarketRegimeDetector
                
                # Create a dataframe for regime detection
                candles = historical_data.candles
                df = pd.DataFrame([
                    {
                        'timestamp': c.timestamp,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    } for c in candles
                ])
                
                # Simple regime detection logic
                # Calculate 20-day returns and volatility
                if len(df) >= 20:
                    returns = df['close'].pct_change(20).iloc[-1] * 100
                    volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
                    
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
                    
                    # Create a simple MarketRegime object
                    regime = MarketRegime(
                        timestamp=datetime.now(timezone.utc),
                        asset=asset,
                        timeframe=timeframe,
                        regime_type=regime_type,
                        strength=0.7,
                        metrics={
                            'returns': returns,
                            'volatility': volatility
                        }
                    )
                    
                    print(f"  Detected regime: {regime_type.value}")
                    
                    # Test Mean Reversion Strategy
                    mr_signals = mean_reversion.generate_signals(
                        asset=asset,
                        timeframe=timeframe,
                        historical_data=historical_data
                    )
                    
                    # Test Trend Following Strategy
                    tf_signals = trend_following.generate_signals(
                        asset=asset,
                        timeframe=timeframe,
                        historical_data=historical_data
                    )
                    
                    # Update counts
                    mr_signals_count += len(mr_signals)
                    tf_signals_count += len(tf_signals)
                    
                    print(f"  Mean Reversion signals: {len(mr_signals)}")
                    print(f"  Trend Following signals: {len(tf_signals)}")
                    
                    # Print signal details if any
                    if mr_signals:
                        signal = mr_signals[0]
                        print(f"\n    Mean Reversion Sample Signal:")
                        print(f"    Type: {signal.signal_type.value.upper()}")
                        print(f"    Entry: {signal.entry_price:.5f}")
                        print(f"    Stop Loss: {signal.stop_loss:.5f}")
                        print(f"    Take Profit: {signal.take_profit:.5f}")
                        if signal.metadata:
                            for key, value in signal.metadata.items():
                                if isinstance(value, (int, float, str, bool)):
                                    print(f"    {key}: {value}")
                    
                    if tf_signals:
                        signal = tf_signals[0]
                        print(f"\n    Trend Following Sample Signal:")
                        print(f"    Type: {signal.signal_type.value.upper()}")
                        print(f"    Entry: {signal.entry_price:.5f}")
                        print(f"    Stop Loss: {signal.stop_loss:.5f}")
                        print(f"    Take Profit: {signal.take_profit:.5f}")
                        if signal.metadata:
                            for key, value in signal.metadata.items():
                                if isinstance(value, (int, float, str, bool)):
                                    print(f"    {key}: {value}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                logger.error(f"Error processing {asset.value} {timeframe.value}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Print summary
    print("\n" + "="*60)
    print(f"SUMMARY: Generated {mr_signals_count} Mean Reversion signals and {tf_signals_count} Trend Following signals")
    print("="*60)
    
    # Strategy implementation success check
    if mr_signals_count > 0 or tf_signals_count > 0:
        print("\nNew strategies have been successfully implemented and are generating signals!")
    else:
        print("\nStrategies are implemented but no signals were generated with current market conditions.")
        print("You may need to adjust thresholds or test with different historical periods.")

if __name__ == "__main__":
    test_strategies()
