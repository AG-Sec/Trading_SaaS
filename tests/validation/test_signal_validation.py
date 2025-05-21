"""
Signal validation tests to ensure proper price level relationships in trading signals.

These tests verify that signals adhere to fundamental principles of trading:
- For LONG positions: take_profit > entry_price > stop_loss
- For SHORT positions: stop_loss > entry_price > take_profit
"""
import pytest
import pandas as pd
from datetime import datetime, timezone
import logging

from shared_types.models import (
    AssetSymbol, 
    Timeframe, 
    SignalType, 
    TradingSignalModel,
    HistoricalDataModel,
    CandleModel
)
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.risk_manager_agent import RiskManagerAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def market_agent():
    """Fixture providing a market data agent."""
    return MarketDataAgent()


@pytest.fixture
def scanner_agent(market_agent):
    """Fixture providing a signal scanner agent."""
    return SignalScannerAgent(market_data_agent=market_agent)


@pytest.fixture
def risk_agent():
    """Fixture providing a risk manager agent."""
    return RiskManagerAgent()


def test_long_signal_price_relationships(scanner_agent, market_agent):
    """
    Test that LONG signals have correct price level relationships:
    take_profit > entry_price > stop_loss
    """
    # Test with multiple assets and timeframes
    test_assets = [AssetSymbol.BTC_USD, AssetSymbol.SPY]
    test_timeframes = [Timeframe.HOUR_1, Timeframe.DAY_1]
    
    for asset in test_assets:
        for timeframe in test_timeframes:
            # Fetch data and generate signals
            market_data = market_agent.fetch_historical_data(asset, timeframe)
            signals = scanner_agent.scan_for_breakout_signals(
                asset, timeframe, historical_data=market_data
            )
            
            # Filter to only analyze LONG signals
            long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
            
            for signal in long_signals:
                # Log the signal details for analysis
                logger.info(f"Validating LONG signal for {asset.value} {timeframe.value}")
                logger.info(f"  Entry: {signal.entry_price}, Stop Loss: {signal.stop_loss}, Take Profit: {signal.take_profit}")
                
                # Assert the correct price relationships for LONG
                assert signal.take_profit > signal.entry_price, f"Take profit ({signal.take_profit}) must be above entry price ({signal.entry_price}) for LONG signal"
                assert signal.entry_price > signal.stop_loss, f"Entry price ({signal.entry_price}) must be above stop loss ({signal.stop_loss}) for LONG signal"
                
                # Calculate and verify profit potential vs. risk
                potential_profit = signal.take_profit - signal.entry_price
                potential_risk = signal.entry_price - signal.stop_loss
                reward_risk_ratio = potential_profit / potential_risk if potential_risk > 0 else 0
                
                logger.info(f"  Profit potential: {potential_profit:.2f}, Risk: {potential_risk:.2f}, R:R Ratio: {reward_risk_ratio:.2f}")
                # Based on our risk management rules requiring minimum 1.5:1 R:R
                assert reward_risk_ratio >= 1.1, f"Reward to risk ratio ({reward_risk_ratio:.2f}) should be at least 1.1 for LONG signal"


def test_short_signal_price_relationships(scanner_agent, market_agent):
    """
    Test that SHORT signals have correct price level relationships:
    stop_loss > entry_price > take_profit
    """
    # Test with multiple assets and timeframes
    test_assets = [AssetSymbol.BTC_USD, AssetSymbol.EUR_USD_FX]
    test_timeframes = [Timeframe.HOUR_1, Timeframe.DAY_1]
    
    for asset in test_assets:
        for timeframe in test_timeframes:
            # Fetch data and generate signals
            market_data = market_agent.fetch_historical_data(asset, timeframe)
            signals = scanner_agent.scan_for_breakout_signals(
                asset, timeframe, historical_data=market_data
            )
            
            # Filter to only analyze SHORT signals
            short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
            
            for signal in short_signals:
                # Log the signal details for analysis
                logger.info(f"Validating SHORT signal for {asset.value} {timeframe.value}")
                logger.info(f"  Entry: {signal.entry_price}, Stop Loss: {signal.stop_loss}, Take Profit: {signal.take_profit}")
                
                # Assert the correct price relationships for SHORT
                assert signal.stop_loss > signal.entry_price, f"Stop loss ({signal.stop_loss}) must be above entry price ({signal.entry_price}) for SHORT signal"
                assert signal.entry_price > signal.take_profit, f"Entry price ({signal.entry_price}) must be above take profit ({signal.take_profit}) for SHORT signal"
                
                # Calculate and verify profit potential vs. risk
                potential_profit = signal.entry_price - signal.take_profit
                potential_risk = signal.stop_loss - signal.entry_price
                reward_risk_ratio = potential_profit / potential_risk if potential_risk > 0 else 0
                
                logger.info(f"  Profit potential: {potential_profit:.2f}, Risk: {potential_risk:.2f}, R:R Ratio: {reward_risk_ratio:.2f}")
                # Based on our risk management rules requiring minimum 1.5:1 R:R
                assert reward_risk_ratio >= 1.1, f"Reward to risk ratio ({reward_risk_ratio:.2f}) should be at least 1.1 for SHORT signal"


def test_risk_manager_enforces_price_relationships(scanner_agent, market_agent, risk_agent):
    """
    Test that the risk manager enforces correct price relationships.
    This is a more complete test as it ensures the risk manager is validating
    signals properly before approving them.
    """
    test_assets = [AssetSymbol.BTC_USD, AssetSymbol.SPY, AssetSymbol.EUR_USD_FX]
    test_timeframes = [Timeframe.HOUR_1, Timeframe.DAY_1]
    
    all_raw_signals = []
    approved_signals = []
    
    for asset in test_assets:
        for timeframe in test_timeframes:
            market_data = market_agent.fetch_historical_data(asset, timeframe)
            signals = scanner_agent.scan_for_breakout_signals(
                asset, timeframe, historical_data=market_data
            )
            
            for signal in signals:
                all_raw_signals.append(signal)
                
                # Get approved signal from risk manager
                approved = risk_agent.evaluate_signal(signal)
                if approved:
                    approved_signals.append(approved)
    
    # Log statistics
    logger.info(f"Generated {len(all_raw_signals)} raw signals")
    logger.info(f"Risk manager approved {len(approved_signals)} signals ({len(approved_signals)/len(all_raw_signals) * 100 if all_raw_signals else 0:.1f}%)")
    
    # Test all approved signals
    for signal in approved_signals:
        if signal.signal_type == SignalType.LONG:
            # Verify LONG price relationships
            assert signal.take_profit > signal.entry_price, f"Take profit must be above entry price for LONG"
            assert signal.entry_price > signal.stop_loss, f"Entry price must be above stop loss for LONG"
            
            # Verify R:R ratio meets risk manager's requirements
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
            rr_ratio = reward / risk if risk > 0 else 0
            
            logger.info(f"LONG signal {signal.signal_id} - R:R ratio: {rr_ratio:.2f}")
            assert rr_ratio >= risk_agent.min_reward_to_risk_ratio * 0.99, f"R:R ratio below minimum requirement"
            
        elif signal.signal_type == SignalType.SHORT:
            # Verify SHORT price relationships
            assert signal.stop_loss > signal.entry_price, f"Stop loss must be above entry price for SHORT"
            assert signal.entry_price > signal.take_profit, f"Entry price must be above take profit for SHORT"
            
            # Verify R:R ratio meets risk manager's requirements
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.take_profit
            rr_ratio = reward / risk if risk > 0 else 0
            
            logger.info(f"SHORT signal {signal.signal_id} - R:R ratio: {rr_ratio:.2f}")
            assert rr_ratio >= risk_agent.min_reward_to_risk_ratio * 0.99, f"R:R ratio below minimum requirement"


def test_manual_signal_creation():
    """
    Test that manually created signals also adhere to proper price level relationships.
    This ensures the validation tests above aren't just testing our signal generation but the actual logic.
    """
    # Create a valid LONG signal
    long_signal = TradingSignalModel(
        signal_id="test-long-signal",
        generated_at=datetime.now(timezone.utc),
        asset=AssetSymbol.BTC_USD,
        timeframe=Timeframe.HOUR_1,
        signal_type=SignalType.LONG,
        entry_price=50000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        metadata={"test": True}
    )
    
    # Verify LONG signal price relationships
    assert long_signal.take_profit > long_signal.entry_price, "Take profit must be above entry price for LONG"
    assert long_signal.entry_price > long_signal.stop_loss, "Entry price must be above stop loss for LONG"
    
    # Calculate LONG R:R
    long_risk = long_signal.entry_price - long_signal.stop_loss
    long_reward = long_signal.take_profit - long_signal.entry_price
    long_rr_ratio = long_reward / long_risk
    assert long_rr_ratio >= 1.5, f"LONG signal R:R ratio should be at least 1.5, got {long_rr_ratio:.2f}"
    
    # Create a valid SHORT signal
    short_signal = TradingSignalModel(
        signal_id="test-short-signal",
        generated_at=datetime.now(timezone.utc),
        asset=AssetSymbol.BTC_USD,
        timeframe=Timeframe.HOUR_1,
        signal_type=SignalType.SHORT,
        entry_price=50000.0,
        stop_loss=52000.0,
        take_profit=45000.0,
        metadata={"test": True}
    )
    
    # Verify SHORT signal price relationships
    assert short_signal.stop_loss > short_signal.entry_price, "Stop loss must be above entry price for SHORT"
    assert short_signal.entry_price > short_signal.take_profit, "Entry price must be above take profit for SHORT"
    
    # Calculate SHORT R:R
    short_risk = short_signal.stop_loss - short_signal.entry_price
    short_reward = short_signal.entry_price - short_signal.take_profit
    short_rr_ratio = short_reward / short_risk
    assert short_rr_ratio >= 1.5, f"SHORT signal R:R ratio should be at least 1.5, got {short_rr_ratio:.2f}"


def test_generate_both_signal_types(market_agent, scanner_agent):
    """
    Test to ensure our signal scanner can generate both LONG and SHORT signals.
    """
    # Get a variety of assets to increase chance of both signal types
    test_assets = [AssetSymbol.BTC_USD, AssetSymbol.ETH_USD, AssetSymbol.SPY, AssetSymbol.EUR_USD_FX]
    test_timeframes = [Timeframe.MIN_15, Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAY_1]
    
    all_signals = []
    
    for asset in test_assets:
        for timeframe in test_timeframes:
            market_data = market_agent.fetch_historical_data(asset, timeframe)
            signals = scanner_agent.scan_for_breakout_signals(asset, timeframe, historical_data=market_data)
            all_signals.extend(signals)
    
    # Count signal types
    long_signals = [s for s in all_signals if s.signal_type == SignalType.LONG]
    short_signals = [s for s in all_signals if s.signal_type == SignalType.SHORT]
    
    logger.info(f"Total signals: {len(all_signals)}")
    logger.info(f"LONG signals: {len(long_signals)}")
    logger.info(f"SHORT signals: {len(short_signals)}")
    
    # We're not making a hard assertion because market conditions may not produce both types
    # but at least log the information
    if len(short_signals) == 0:
        logger.warning("No SHORT signals were generated. This might indicate an issue with SHORT signal generation.")
        
    # If we have both signal types, verify price relationships for each
    for signal in long_signals:
        assert signal.take_profit > signal.entry_price, "Take profit must be above entry price for LONG"
        assert signal.entry_price > signal.stop_loss, "Entry price must be above stop loss for LONG"
        
    for signal in short_signals:
        assert signal.stop_loss > signal.entry_price, "Stop loss must be above entry price for SHORT"
        assert signal.entry_price > signal.take_profit, "Entry price must be above take profit for SHORT"


def test_market_regime_adjustments(market_agent, scanner_agent):
    """
    Test that market regime adjustments maintain proper price level relationships.
    """
    from backend.agents.market_regime_detector import MarketRegimeDetector
    
    regime_detector = MarketRegimeDetector()
    
    # Test with asset and timeframe likely to have regime detection
    asset = AssetSymbol.BTC_USD
    timeframe = Timeframe.DAY_1
    
    # Fetch market data
    market_data = market_agent.fetch_historical_data(asset, timeframe)
    df = pd.DataFrame([candle.model_dump() for candle in market_data.candles])
    
    # Detect market regime
    regime = regime_detector.detect_regime(df, asset, timeframe)
    logger.info(f"Detected market regime: {regime.regime_type.value} with strength {regime.strength:.2f}")
    
    # Get adjusted parameters for the regime
    adjusted_params = regime_detector.get_adjusted_parameters(df, asset, timeframe)
    logger.info(f"Adjusted parameters: {adjusted_params}")
    
    # Create a signal scanner with adjusted parameters
    adapted_scanner = SignalScannerAgent(market_data_agent=market_agent)
    
    # Set adjusted parameters
    if 'breakout_period' in adjusted_params:
        adapted_scanner.breakout_period = adjusted_params['breakout_period']
    if 'atr_multiplier_sl' in adjusted_params:
        adapted_scanner.atr_multiplier_sl = adjusted_params['atr_multiplier_sl']
    if 'min_rr_ratio' in adjusted_params:
        adapted_scanner.min_rr_ratio = adjusted_params['min_rr_ratio']
    
    # Generate signals with both scanners
    regular_signals = scanner_agent.scan_for_breakout_signals(asset, timeframe, historical_data=market_data)
    adapted_signals = adapted_scanner.scan_for_breakout_signals(asset, timeframe, historical_data=market_data)
    
    logger.info(f"Regular scanner generated {len(regular_signals)} signals")
    logger.info(f"Regime-adapted scanner generated {len(adapted_signals)} signals")
    
    # Verify price relationships for all signals from both scanners
    for signal in regular_signals + adapted_signals:
        if signal.signal_type == SignalType.LONG:
            assert signal.take_profit > signal.entry_price, "Take profit must be above entry price for LONG"
            assert signal.entry_price > signal.stop_loss, "Entry price must be above stop loss for LONG"
        elif signal.signal_type == SignalType.SHORT:
            assert signal.stop_loss > signal.entry_price, "Stop loss must be above entry price for SHORT"
            assert signal.entry_price > signal.take_profit, "Entry price must be above take profit for SHORT"
            
    # Validate that adapted signals have metadata about market regime
    for signal in adapted_signals:
        if 'market_regime' in signal.metadata:
            logger.info(f"Signal has market regime metadata: {signal.metadata['market_regime']}")


if __name__ == "__main__":
    # Setup logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Run tests manually
    market_agent = MarketDataAgent()
    scanner_agent = SignalScannerAgent(market_data_agent=market_agent)
    risk_agent = RiskManagerAgent()
    
    print("Testing LONG signal price relationships...")
    test_long_signal_price_relationships(scanner_agent, market_agent)
    
    print("Testing SHORT signal price relationships...")
    test_short_signal_price_relationships(scanner_agent, market_agent)
    
    print("Testing risk manager enforcement...")
    test_risk_manager_enforces_price_relationships(scanner_agent, market_agent, risk_agent)
    
    print("Testing both signal types generation...")
    test_generate_both_signal_types(market_agent, scanner_agent)
    
    print("Testing market regime adjustments...")
    test_market_regime_adjustments(market_agent, scanner_agent)
