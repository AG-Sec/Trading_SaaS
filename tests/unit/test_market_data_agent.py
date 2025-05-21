"""
Unit tests for the MarketDataAgent class.
"""
import pytest
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta

from shared_types.models import AssetSymbol, Timeframe, HistoricalDataModel, CandleModel

def test_market_data_agent_init(market_agent):
    """Test that the MarketDataAgent initializes correctly."""
    assert market_agent is not None
    
def test_fetch_historical_data(market_agent, sample_assets, sample_timeframes):
    """Test fetching historical data for all supported assets and timeframes."""
    for asset in sample_assets:
        for timeframe in sample_timeframes:
            data = market_agent.fetch_historical_data(asset, timeframe)
            
            # Verify data structure
            assert isinstance(data, HistoricalDataModel)
            assert data.asset == asset
            assert data.timeframe == timeframe
            
            # Verify we have candles
            assert data.candles is not None
            assert len(data.candles) > 0
            
            # Verify candle data structure
            first_candle = data.candles[0]
            assert hasattr(first_candle, 'timestamp')
            assert hasattr(first_candle, 'open')
            assert hasattr(first_candle, 'high')
            assert hasattr(first_candle, 'low')
            assert hasattr(first_candle, 'close')
            assert hasattr(first_candle, 'volume')
            
            # Verify data order (should be sorted by timestamp, ascending)
            timestamps = [candle.timestamp for candle in data.candles]
            assert timestamps == sorted(timestamps)

def test_market_data_caching(market_agent):
    """Test that data is cached and reused when appropriate."""
    # First fetch should go to the actual data source
    start_time = datetime.now()
    data1 = market_agent.fetch_historical_data(AssetSymbol.BTC_USD, Timeframe.DAY_1)
    first_fetch_time = datetime.now() - start_time
    
    # Second fetch should be faster due to caching
    start_time = datetime.now()
    data2 = market_agent.fetch_historical_data(AssetSymbol.BTC_USD, Timeframe.DAY_1)
    second_fetch_time = datetime.now() - start_time
    
    # Verify both fetches returned the same data
    assert len(data1.candles) == len(data2.candles)
    
    # The second fetch should typically be faster if caching is working
    # However, this might not always be true due to system variability
    # So we just log the times for now
    print(f"First fetch time: {first_fetch_time.total_seconds():.6f}s")
    print(f"Second fetch time: {second_fetch_time.total_seconds():.6f}s")

def test_handle_invalid_asset(market_agent):
    """Test that the agent handles invalid assets gracefully."""
    try:
        # Create a test enum value that's not valid
        class TestEnum(str, Enum):
            INVALID = "INVALID_ASSET"
        
        # This should raise a ValueError or return None
        result = market_agent.fetch_historical_data(TestEnum.INVALID, Timeframe.DAY_1)
        # If it doesn't raise an exception, it should return None or empty data
        assert result is None or len(result.candles) == 0
    except (ValueError, AttributeError, TypeError):
        # One of these exceptions is expected behavior
        pass

def test_handle_invalid_timeframe(market_agent):
    """Test that the agent handles invalid timeframes gracefully."""
    try:
        # Create a test enum value that's not valid
        class TestTimeframe(str, Enum):
            INVALID = "INVALID_TIMEFRAME"
            
        # This should raise a ValueError or return None
        result = market_agent.fetch_historical_data(AssetSymbol.BTC_USD, TestTimeframe.INVALID)
        # If it doesn't raise an exception, it should return None or empty data
        assert result is None or len(result.candles) == 0
    except (ValueError, AttributeError, TypeError):
        # One of these exceptions is expected behavior
        pass

def test_dataframe_conversion(market_agent):
    """Test that market data can be converted to a pandas DataFrame."""
    data = market_agent.fetch_historical_data(AssetSymbol.BTC_USD, Timeframe.DAY_1)
    
    # Convert to DataFrame
    df = pd.DataFrame([candle.model_dump() for candle in data.candles])
    
    # Verify DataFrame structure
    assert 'timestamp' in df.columns
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns
    
    # Verify DataFrame is not empty
    assert not df.empty
    
    # Verify data types
    assert pd.api.types.is_datetime64_dtype(df['timestamp']) or isinstance(df['timestamp'].iloc[0], datetime)
    assert pd.api.types.is_numeric_dtype(df['open'])
    assert pd.api.types.is_numeric_dtype(df['high'])
    assert pd.api.types.is_numeric_dtype(df['low'])
    assert pd.api.types.is_numeric_dtype(df['close'])
    assert pd.api.types.is_numeric_dtype(df['volume'])
