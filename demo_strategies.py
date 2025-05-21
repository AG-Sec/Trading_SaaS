#!/usr/bin/env python3
"""
Trading Strategies Demo Script for Trading SaaS

This script demonstrates the newly implemented trading strategies:
1. Mean Reversion Strategy
2. Trend Following Strategy
"""

import logging
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from typing import List, Dict, Any

from shared_types.models import AssetSymbol, Timeframe, TradingSignalModel
from backend.agents.market_data_agent import MarketDataAgent
from backend.strategies.mean_reversion_strategy import MeanReversionStrategy
from backend.strategies.trend_following_strategy import TrendFollowingStrategy
from backend.agents.market_regime_detector import MarketRegimeDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_strategies_demo():
    """
    Demonstrate the new trading strategies.
    """
    logger.info("Starting Trading Strategies demonstration")
    
    # Initialize agents
    market_data_agent = MarketDataAgent()
    market_regime_detector = MarketRegimeDetector()
    
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
    
    # Dictionary to hold all results
    all_signals = {}
    regime_detections = {}
    
    # Process each asset and timeframe combination
    for asset in assets:
        asset_signals = {}
        asset_regimes = {}
        
        for timeframe in timeframes:
            logger.info(f"Processing {asset.value} on {timeframe.value} timeframe")
            
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
                    logger.warning(f"No historical data available for {asset.value} {timeframe.value}")
                    continue
                
                logger.info(f"Retrieved {len(historical_data.candles)} candles for {asset.value} {timeframe.value}")
                
                # Get market regime information
                # Convert historical_data to DataFrame for regime detection
                df = pd.DataFrame([
                    {
                        'timestamp': c.timestamp,
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume
                    } for c in historical_data.candles
                ])
                df.set_index('timestamp', inplace=True)
                
                regime = market_regime_detector.detect_regime(
                    df=df,
                    asset=asset,
                    timeframe=timeframe
                )
                
                if regime:
                    logger.info(f"Detected {regime.regime_type.value} regime with strength {regime.strength:.2f}")
                    
                    # Generate signals with Mean Reversion strategy
                    mr_signals = mean_reversion.generate_signals(asset, timeframe, historical_data)
                    logger.info(f"Mean Reversion generated {len(mr_signals)} signals")
                    
                    # Generate signals with Trend Following strategy
                    tf_signals = trend_following.generate_signals(asset, timeframe, historical_data)
                    logger.info(f"Trend Following generated {len(tf_signals)} signals")
                    
                    # Store results
                    asset_signals[timeframe.value] = {
                        "mean_reversion": {
                            "count": len(mr_signals),
                            "signals": mr_signals
                        },
                        "trend_following": {
                            "count": len(tf_signals),
                            "signals": tf_signals
                        }
                    }
                    
                    asset_regimes[timeframe.value] = {
                        "regime_type": regime.regime_type.value,
                        "strength": regime.strength,
                        "timestamp": regime.timestamp.isoformat() if hasattr(regime, 'timestamp') else None
                    }
                
            except Exception as e:
                logger.error(f"Error processing {asset.value} {timeframe.value}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        all_signals[asset.value] = asset_signals
        regime_detections[asset.value] = asset_regimes
    
    # Print summary
    print_results_summary(all_signals, regime_detections)
    
    return {
        "signals": all_signals,
        "regimes": regime_detections
    }

def print_results_summary(all_signals, regime_detections):
    """
    Print a summary of results from the demonstration.
    
    Args:
        all_signals: Dictionary of signals by asset and timeframe
        regime_detections: Dictionary of regime detections
    """
    print("\n" + "="*80)
    print(" TRADING STRATEGIES DEMONSTRATION RESULTS ")
    print("="*80)
    
    total_mr_signals = 0
    total_tf_signals = 0
    
    for asset in all_signals:
        print(f"\nASSET: {asset}")
        print("-" * 40)
        
        for timeframe in all_signals[asset]:
            signals = all_signals[asset][timeframe]
            regime = regime_detections[asset].get(timeframe, {})
            
            mr_count = signals.get("mean_reversion", {}).get("count", 0)
            tf_count = signals.get("trend_following", {}).get("count", 0)
            
            total_mr_signals += mr_count
            total_tf_signals += tf_count
            
            print(f"\n  Timeframe: {timeframe}")
            print(f"  Market Regime: {regime.get('regime_type', 'Unknown')} (Strength: {regime.get('strength', 0):.2f})")
            print(f"  Signal Count: Mean Reversion={mr_count}, Trend Following={tf_count}")
            
            # Print Mean Reversion signal details if any
            mr_signals = signals.get("mean_reversion", {}).get("signals", [])
            if mr_signals:
                print("\n    Mean Reversion Signals:")
                for i, signal in enumerate(mr_signals[:2]):  # Show only first 2 signals
                    print(f"    Signal {i+1}: {signal.signal_type.value.upper()} Entry: {signal.entry_price:.5f}, "
                          f"SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
                    
                    if signal.metadata:
                        strategy_type = signal.metadata.get("strategy_type", "Unknown")
                        print(f"      Strategy Type: {strategy_type}")
                        
                        # Print additional metadata
                        for key, value in signal.metadata.items():
                            if key in ["rsi", "ma_deviation", "bb_position", "atr", "volume_percentile"]:
                                print(f"      {key}: {value}")
                
                if len(mr_signals) > 2:
                    print(f"      ... and {len(mr_signals) - 2} more signals")
            
            # Print Trend Following signal details if any
            tf_signals = signals.get("trend_following", {}).get("signals", [])
            if tf_signals:
                print("\n    Trend Following Signals:")
                for i, signal in enumerate(tf_signals[:2]):  # Show only first 2 signals
                    print(f"    Signal {i+1}: {signal.signal_type.value.upper()} Entry: {signal.entry_price:.5f}, "
                          f"SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
                    
                    if signal.metadata:
                        strategy_type = signal.metadata.get("strategy_type", "Unknown")
                        print(f"      Strategy Type: {strategy_type}")
                        
                        # Print additional metadata
                        for key, value in signal.metadata.items():
                            if key in ["adx", "ema_fast", "ema_slow", "macd", "signal_type"]:
                                print(f"      {key}: {value}")
                
                if len(tf_signals) > 2:
                    print(f"      ... and {len(tf_signals) - 2} more signals")
    
    print("\n" + "="*80)
    print(f" SUMMARY: Generated {total_mr_signals} Mean Reversion signals and {total_tf_signals} Trend Following signals")
    print("="*80 + "\n")
    
    # Print regime-strategy fit analysis
    print("\nMarket Regime Suitability Analysis:")
    print("-" * 40)
    
    regime_strategy_fit = analyze_regime_strategy_fit(all_signals, regime_detections)
    
    for regime_type, strategies in regime_strategy_fit.items():
        print(f"\n{regime_type.replace('_', ' ').title()} Regime:")
        for strategy, count in strategies.items():
            strategy_name = strategy.replace('_', ' ').title()
            print(f"  {strategy_name}: {count} signals")

def analyze_regime_strategy_fit(all_signals, regime_detections):
    """
    Analyze how well each strategy performs in different regimes.
    
    Args:
        all_signals: Dictionary of signals by asset and timeframe
        regime_detections: Dictionary of regime detections
        
    Returns:
        Dictionary of regime types with strategy counts
    """
    regime_strategy_counts = {}
    
    for asset in all_signals:
        for timeframe in all_signals[asset]:
            regime_info = regime_detections[asset].get(timeframe, {})
            regime_type = regime_info.get("regime_type", "unknown")
            
            if regime_type not in regime_strategy_counts:
                regime_strategy_counts[regime_type] = {
                    "mean_reversion": 0,
                    "trend_following": 0
                }
            
            signals = all_signals[asset][timeframe]
            mr_count = signals.get("mean_reversion", {}).get("count", 0)
            tf_count = signals.get("trend_following", {}).get("count", 0)
            
            regime_strategy_counts[regime_type]["mean_reversion"] += mr_count
            regime_strategy_counts[regime_type]["trend_following"] += tf_count
    
    return regime_strategy_counts

if __name__ == "__main__":
    results = run_strategies_demo()
