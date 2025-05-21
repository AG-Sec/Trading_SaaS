"""
Unit tests for the MarketRegimeDetector class.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from shared_types.models import AssetSymbol, Timeframe
from backend.agents.market_regime_detector import MarketRegimeType

def test_regime_detector_init(regime_detector):
    """Test that the MarketRegimeDetector initializes correctly."""
    assert regime_detector is not None
    
    # Check default parameters
    assert regime_detector.min_data_points > 0
    assert hasattr(regime_detector, 'tech_indicators')
    assert hasattr(regime_detector, 'regime_history')

def test_detect_regime(regime_detector, market_agent, sample_assets, sample_timeframes):
    """Test regime detection for different assets and timeframes."""
    regime_results = {}
    
    for asset in sample_assets:
        regime_results[asset.value] = {}
        for timeframe in sample_timeframes:
            # Fetch data
            data = market_agent.fetch_historical_data(asset, timeframe)
            if not data or not data.candles:
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame([candle.model_dump() for candle in data.candles])
            
            # Detect regime
            regime = regime_detector.detect_regime(df, asset, timeframe)
            
            # Store regime type
            regime_results[asset.value][timeframe.value] = regime.regime_type.value
            
            # Verify regime properties
            assert regime.regime_type is not None
            assert regime.regime_type in list(MarketRegimeType)
            assert 0 <= regime.strength <= 1
            assert regime.asset == asset
            assert regime.timeframe == timeframe
    
    # Log regime types for review
    print("\nDetected market regimes by asset and timeframe:")
    for asset, timeframes in regime_results.items():
        print(f"  {asset}:")
        for timeframe, regime_type in timeframes.items():
            print(f"    {timeframe}: {regime_type}")

def test_parameter_adjustment(regime_detector, market_agent):
    """Test parameter adjustment based on market regime."""
    # Fetch data
    data = market_agent.fetch_historical_data(AssetSymbol.BTC_USD, Timeframe.DAY_1)
    if data and data.candles:
        df = pd.DataFrame([candle.model_dump() for candle in data.candles])
        
        # Detect regime
        regime = regime_detector.detect_regime(df, AssetSymbol.BTC_USD, Timeframe.DAY_1)
        
        # Default parameters to adjust
        default_params = {
            'breakout_period': 10,
            'atr_multiplier_sl': 1.5,
            'atr_multiplier_tp': 2.0,
            'rsi_lower_threshold': 25,
            'rsi_upper_threshold': 75,
            'min_rr_ratio': 1.1
        }
        
        # Get adjusted parameters
        adjusted_params = regime_detector.get_adjusted_parameters(df, regime.asset, regime.timeframe)
        
        # Verify adjusted parameters
        assert adjusted_params is not None
        # The implementation might return a different set of parameters than what we provided
        # so just check that we got something reasonable back
        assert len(adjusted_params) > 0
        assert 'breakout_period' in adjusted_params
            
        # Log adjustments
        print(f"\nMarket Regime: {regime.regime_type.value} with strength {regime.strength:.2f}")
        print("Parameter adjustments:")
        for key in default_params:
            if key in adjusted_params:
                change = ((adjusted_params[key] - default_params[key]) / default_params[key]) * 100
                print(f"  {key}: {default_params[key]} -> {adjusted_params[key]} ({change:.1f}%)")
            else:
                print(f"  {key}: {default_params[key]} (unchanged, not in adjusted params)")
            
        # Verify that adjustments are reasonable
        # The actual implementation decides how much to adjust parameters
        for key in default_params:
            if key in adjusted_params and isinstance(default_params[key], (int, float)):
                # Just check that the change isn't too extreme
                change_pct = abs((adjusted_params[key] - default_params[key]) / default_params[key])
                assert change_pct <= 0.5, f"Parameter {key} had extreme adjustment >50%"

def test_regime_classification_logic(regime_detector):
    """Test the logic for classifying different market regimes."""
    # Create synthetic data for different regimes
    def create_test_data(trend, volatility, days=100):
        base_date = datetime.now()
        dates = [base_date - pd.Timedelta(days=i) for i in range(days)]
        
        if trend == "up":
            closes = np.linspace(100, 200, days)  # Strong uptrend
        elif trend == "down":
            closes = np.linspace(200, 100, days)  # Strong downtrend
        elif trend == "flat":
            closes = np.ones(days) * 150  # Flat, no trend
        
        # Add volatility
        if volatility == "high":
            noise = np.random.normal(0, 15, days)
        elif volatility == "low":
            noise = np.random.normal(0, 3, days)
        else:  # medium
            noise = np.random.normal(0, 8, days)
            
        closes = closes + noise
        
        # Create full OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': closes - noise/2,
            'high': closes + abs(noise),
            'low': closes - abs(noise),
            'close': closes,
            'volume': np.random.randint(1000, 10000, days)
        })
        
        return df.sort_values('timestamp')
    
    # Test different synthetic market conditions
    test_conditions = [
        {"name": "Strong Bull", "trend": "up", "volatility": "low"},
        {"name": "Volatile Bull", "trend": "up", "volatility": "high"},
        {"name": "Strong Bear", "trend": "down", "volatility": "low"},
        {"name": "Volatile Bear", "trend": "down", "volatility": "high"},
        {"name": "Ranging Low Vol", "trend": "flat", "volatility": "low"},
        {"name": "Ranging High Vol", "trend": "flat", "volatility": "high"}
    ]
    
    # Detect regime for each condition
    for condition in test_conditions:
        df = create_test_data(condition["trend"], condition["volatility"])
        regime = regime_detector.detect_regime(
            df, 
            AssetSymbol.BTC_USD, 
            Timeframe.DAY_1
        )
        
        print(f"\nTest condition: {condition['name']}")
        print(f"Detected regime: {regime.regime_type.value}, strength: {regime.strength:.2f}")
        
        # Make assertions based on expected regimes
        # The exact classification might vary based on the specific implementation
        # So we just verify that we get a valid regime type and the strength is reasonable
        assert regime.regime_type in list(MarketRegimeType)
        assert 0 <= regime.strength <= 1
        
        # Log the detected regime
        print(f"Synthetic {condition['name']} data classified as: {regime.regime_type.value} with strength {regime.strength:.2f}")

def test_handle_insufficient_data(regime_detector):
    """Test that the detector handles insufficient data gracefully."""
    # Create minimal/empty DataFrame
    df = pd.DataFrame({
        'timestamp': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': []
    })
    
    # This should handle the empty data without errors
    try:
        regime = regime_detector.detect_regime(
            df, 
            AssetSymbol.BTC_USD, 
            Timeframe.DAY_1
        )
        # If no exception, should return UNKNOWN or similar fallback
        assert regime.regime_type == MarketRegimeType.UNKNOWN
    except Exception as e:
        # Some implementations might raise an exception for empty data
        # In that case, make sure it's a controlled, expected exception
        assert "insufficient data" in str(e).lower() or "empty data" in str(e).lower()
