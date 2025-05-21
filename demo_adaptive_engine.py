#!/usr/bin/env python
"""
Adaptive Engine Demo for Trading SaaS

This script demonstrates how to use the Adaptive Engine to generate optimized trading signals
that dynamically adjust to market conditions, using the optimal timeframe, strategy, and
risk parameters for each asset.
"""

import logging
import sys
from datetime import datetime, timedelta

from shared_types.models import AssetSymbol
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.market_regime_detector import MarketRegimeDetector
from backend.agents.regime_optimizer import RegimeOptimizer
from backend.adaptive_engine.adaptive_engine import AdaptiveEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('adaptive_engine_demo.log')
    ]
)
logger = logging.getLogger('adaptive_engine_demo')

def main():
    """Run the Adaptive Engine demo."""
    logger.info("Starting Adaptive Engine Demo")
    
    # Initialize core agents
    market_agent = MarketDataAgent()
    scanner_agent = SignalScannerAgent(market_data_agent=market_agent)
    risk_agent = RiskManagerAgent()
    regime_detector = MarketRegimeDetector()
    
    # Lower signal scanner thresholds to generate more signals for the demo
    scanner_agent.breakout_period = 5  # Shorter lookback period (default is 10)
    scanner_agent.atr_multiplier_sl = 1.0  # Tighter stops
    scanner_agent.volume_threshold = 1.0  # No volume confirmation requirement
    scanner_agent.rsi_oversold = 40  # Less strict RSI requirement for oversold
    scanner_agent.rsi_overbought = 60  # Less strict RSI requirement for overbought
    
    # Initialize Adaptive Engine with lower thresholds
    adaptive_engine = AdaptiveEngine(
        market_data_agent=market_agent,
        signal_scanner_agent=scanner_agent,
        risk_manager_agent=risk_agent,
        regime_detector=regime_detector,
        min_strategy_confidence=0.55,  # Lower confidence threshold
        multi_timeframe_confirmation=False,  # Disable to allow more signals
        minimum_alert_score=0.5  # Lower the score threshold
    )
    
    # Also lower risk manager thresholds
    risk_agent.max_volatility_factor = 5.0  # Increase max volatility tolerance
    risk_agent.min_reward_to_risk_ratio = 1.1  # Lower R:R requirement
    
    # Define assets to analyze
    assets = [
        AssetSymbol.BTC_USD,
        AssetSymbol.ETH_USD,
        AssetSymbol.EUR_USD_FX,
        AssetSymbol.SPY
    ]
    
    # Generate and display optimized signals for each asset
    for asset in assets:
        logger.info(f"\n{'='*60}\nGenerating optimized signals for {asset.value}\n{'='*60}")
        
        # Generate signals with the Adaptive Engine
        signals = adaptive_engine.generate_optimized_signals(asset)
        
        # Display results
        logger.info(f"Generated {len(signals)} optimized signals for {asset.value}")
        
        # Process signals through risk manager for final filtering and position sizing
        if signals:
            approved_signals = []
            for signal in signals:
                approved = risk_agent.evaluate_signal(signal)
                if approved:
                    approved_signals.append(approved)
            
            logger.info(f"Risk manager approved {len(approved_signals)} out of {len(signals)} signals")
            
            # Display details of approved signals
            for i, signal in enumerate(approved_signals, 1):
                logger.info(f"\nSignal {i}:")
                logger.info(f"  Type: {signal.signal_type.value.upper()}")
                logger.info(f"  Asset: {signal.asset.value}")
                logger.info(f"  Timeframe: {signal.timeframe.value}")
                logger.info(f"  Entry: {signal.entry_price:.5f}")
                logger.info(f"  Stop Loss: {signal.stop_loss:.5f}")
                logger.info(f"  Take Profit: {signal.take_profit:.5f}")
                logger.info(f"  R:R Ratio: {signal.risk_reward_ratio:.2f}")
                
                # Display key adaptive engine metadata
                if "adaptive_engine" in signal.metadata:
                    ae_meta = signal.metadata["adaptive_engine"]
                    logger.info(f"  Signal Score: {ae_meta.get('signal_score', 'N/A')}")
                    logger.info(f"  Market Regime: {ae_meta.get('market_regime', 'N/A')}")
                    logger.info(f"  Regime Strength: {ae_meta.get('regime_strength', 'N/A'):.2f}")
                
                # Display position sizing if available
                if signal.position_size_usd:
                    logger.info(f"  Position Size: ${signal.position_size_usd:.2f}")
                    logger.info(f"  Risk Amount: ${signal.risk_per_trade_usd:.2f}")
        else:
            logger.info(f"No signals generated for {asset.value}")
    
    logger.info("\nAdaptive Engine Demo completed")

if __name__ == "__main__":
    main()
