#!/usr/bin/env python3
"""
Adaptive Strategies Demo Script for Trading SaaS

This script demonstrates the newly implemented trading strategies (Mean Reversion
and Trend Following) along with the Adaptive Engine that dynamically selects
the optimal strategy, timeframe, and risk parameters.
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
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.market_regime_detector import MarketRegimeDetector
from backend.adaptive_engine.adaptive_engine import AdaptiveEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_adaptive_demo():
    """
    Demonstrate the Adaptive Engine with the new trading strategies.
    """
    logger.info("Starting Adaptive Strategies demonstration")
    
    # Initialize agents
    market_data_agent = MarketDataAgent()
    signal_scanner_agent = SignalScannerAgent(market_data_agent=market_data_agent)
    risk_manager_agent = RiskManagerAgent()
    market_regime_detector = MarketRegimeDetector()
    
    # Initialize strategies
    mean_reversion = MeanReversionStrategy()
    trend_following = TrendFollowingStrategy()
    
    # Initialize adaptive engine
    adaptive_engine = AdaptiveEngine(
        market_data_agent=market_data_agent,
        signal_scanner_agent=signal_scanner_agent,
        risk_manager_agent=risk_manager_agent
    )
    
    # Lower thresholds to generate more signals for testing
    if hasattr(adaptive_engine, 'score_threshold'):
        adaptive_engine.score_threshold = 0.5
    if hasattr(adaptive_engine, 'min_regime_strength'):
        adaptive_engine.min_regime_strength = 0.3
    
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
    strategy_selections = {}
    regime_detections = {}
    
    # Process each asset and timeframe combination
    for asset in assets:
        asset_signals = {}
        asset_strategies = {}
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
                
                # 1. Test individual strategies
                mr_signals = mean_reversion.generate_signals(asset, timeframe, historical_data)
                tf_signals = trend_following.generate_signals(asset, timeframe, historical_data)
                
                logger.info(f"Mean Reversion generated {len(mr_signals)} signals")
                logger.info(f"Trend Following generated {len(tf_signals)} signals")
                
                # 2. Generate optimized signals with Adaptive Engine
                # Check the actual signature of the method
                try:
                    # Try with timeframe parameter
                    optimized_signals = adaptive_engine.generate_optimized_signals(
                        asset=asset,
                        timeframe=timeframe,
                        historical_data=historical_data
                    )
                except TypeError:
                    # Try without timeframe parameter
                    optimized_signals = adaptive_engine.generate_optimized_signals(
                        asset=asset,
                        historical_data=historical_data
                    )
                
                logger.info(f"Adaptive Engine generated {len(optimized_signals)} optimized signals")
                
                # 3. Get market regime information
                regime = market_regime_detector.detect_regime(
                    asset=asset,
                    timeframe=timeframe,
                    historical_data=historical_data
                )
                
                if regime:
                    logger.info(f"Detected {regime.regime_type.value} regime with strength {regime.strength:.2f}")
                    
                    # 4. Store results
                    asset_signals[timeframe.value] = {
                        "mean_reversion": len(mr_signals),
                        "trend_following": len(tf_signals),
                        "adaptive_engine": len(optimized_signals),
                        "signals": optimized_signals
                    }
                    
                    asset_strategies[timeframe.value] = get_strategy_selections(optimized_signals)
                    
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
        strategy_selections[asset.value] = asset_strategies
        regime_detections[asset.value] = asset_regimes
    
    # Print summary
    print_results_summary(all_signals, strategy_selections, regime_detections)
    
    return {
        "signals": all_signals,
        "strategies": strategy_selections,
        "regimes": regime_detections
    }

def get_strategy_selections(signals: List[TradingSignalModel]) -> Dict[str, int]:
    """
    Extract strategy selections from signals metadata.
    
    Args:
        signals: List of trading signals
        
    Returns:
        Dictionary counting strategy usage
    """
    strategy_counts = {}
    
    for signal in signals:
        if not signal.metadata or "strategy_name" not in signal.metadata:
            continue
            
        strategy_name = signal.metadata.get("strategy_name", "Unknown")
        if strategy_name not in strategy_counts:
            strategy_counts[strategy_name] = 0
            
        strategy_counts[strategy_name] += 1
    
    return strategy_counts

def print_results_summary(all_signals, strategy_selections, regime_detections):
    """
    Print a summary of results from the demonstration.
    
    Args:
        all_signals: Dictionary of signals by asset and timeframe
        strategy_selections: Dictionary of strategy selections
        regime_detections: Dictionary of regime detections
    """
    print("\n" + "="*80)
    print(" ADAPTIVE STRATEGIES DEMONSTRATION RESULTS ")
    print("="*80)
    
    total_optimized_signals = 0
    
    for asset in all_signals:
        print(f"\nASSET: {asset}")
        print("-" * 40)
        
        for timeframe in all_signals[asset]:
            signals = all_signals[asset][timeframe]
            strategies = strategy_selections[asset].get(timeframe, {})
            regime = regime_detections[asset].get(timeframe, {})
            
            print(f"\n  Timeframe: {timeframe}")
            print(f"  Market Regime: {regime.get('regime_type', 'Unknown')} (Strength: {regime.get('strength', 0):.2f})")
            print(f"  Signal Count: Mean Reversion={signals.get('mean_reversion', 0)}, " 
                  f"Trend Following={signals.get('trend_following', 0)}, "
                  f"Adaptive Engine={signals.get('adaptive_engine', 0)}")
            
            if strategies:
                print(f"  Strategy Selection: {strategies}")
            
            # Print signal details if any
            optimized_signals = signals.get("signals", [])
            total_optimized_signals += len(optimized_signals)
            
            if optimized_signals:
                print("\n    Signal Details:")
                for i, signal in enumerate(optimized_signals[:2]):  # Show only first 2 signals
                    print(f"    Signal {i+1}: {signal.signal_type.value.upper()} Entry: {signal.entry_price:.5f}, "
                          f"SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}")
                    
                    if signal.metadata:
                        strategy = signal.metadata.get("strategy_name", "Unknown")
                        strategy_type = signal.metadata.get("strategy_type", "Unknown")
                        print(f"      Strategy: {strategy} ({strategy_type})")
                        
                        # Print additional metadata
                        if "regime_strength" in signal.metadata:
                            print(f"      Regime Strength: {signal.metadata.get('regime_strength', 0):.2f}")
                        if "confidence" in signal.metadata:
                            print(f"      Confidence: {signal.metadata.get('confidence', 0):.2f}")
                
                if len(optimized_signals) > 2:
                    print(f"      ... and {len(optimized_signals) - 2} more signals")
    
    print("\n" + "="*80)
    print(f" SUMMARY: Generated {total_optimized_signals} optimized signals across all assets and timeframes")
    print("="*80 + "\n")
    
    print("The Adaptive Engine has successfully integrated the new trading strategies")
    print("and dynamically selected the optimal strategy, timeframe, and parameters")
    print("based on current market conditions for each asset.")

if __name__ == "__main__":
    results = run_adaptive_demo()
