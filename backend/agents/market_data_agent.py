import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
import logging
import os
import redis
import pickle

from shared_types import AssetSymbol, Timeframe, CandleModel, HistoricalDataModel

# Configure logging
logger = logging.getLogger(__name__)

# Mapping from our Timeframe enum to yfinance interval strings
TIMEFRAME_TO_YFINANCE_INTERVAL: Dict[Timeframe, str] = {
    Timeframe.MIN_5: "5m",
    Timeframe.MIN_15: "15m",
    Timeframe.HOUR_1: "1h", # yfinance uses '60m' or '1h'
    Timeframe.HOUR_4: "4h", # yfinance does not directly support 4h, needs to be handled or approximated
    Timeframe.DAY_1: "1d",
}

# Mapping for periods based on timeframe for yfinance history call
# yfinance 'period' parameter: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
# These are just defaults, can be overridden by start_date/end_date
DEFAULT_PERIOD_FOR_TIMEFRAME: Dict[Timeframe, str] = {
    Timeframe.MIN_5: "7d",    # Fetch last 7 days of 5-min data
    Timeframe.MIN_15: "60d",   # Fetch last 60 days of 15-min data
    Timeframe.HOUR_1: "730d",  # Fetch last 2 years of 1-hour data (max yfinance allows for <1d intervals)
    Timeframe.HOUR_4: "730d",  # Fetch last 2 years of 1-hour data (to be resampled to 4h)
    Timeframe.DAY_1: "5y",    # Fetch last 5 years of daily data
}

class MarketDataAgent:
    """Agent responsible for fetching and providing market data."""

    def __init__(self, cache_size: int = 100):
        """
        Initializes the MarketDataAgent.
        Args:
            cache_size: Maximum number of historical data results to cache.
        """
        self.cache: Dict[tuple[AssetSymbol, Timeframe], HistoricalDataModel] = {}
        self.cache_size = cache_size
        self.cache_keys_order: List[tuple[AssetSymbol, Timeframe]] = [] # For LRU eviction
        # Initialize Redis client if URL provided
        redis_url = os.getenv("REDIS_URL")
        self.redis_client = redis.Redis.from_url(redis_url) if redis_url else None

    def _add_to_cache(self, key: tuple[AssetSymbol, Timeframe], data: HistoricalDataModel):
        if len(self.cache) >= self.cache_size:
            oldest_key = self.cache_keys_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
        
        self.cache[key] = data
        self.cache_keys_order.append(key)

    def _get_from_cache(self, key: tuple[AssetSymbol, Timeframe]) -> Optional[HistoricalDataModel]:
        if key in self.cache:
            # Move key to end to signify it was recently used (for LRU)
            self.cache_keys_order.remove(key)
            self.cache_keys_order.append(key)
            return self.cache[key]
        return None

    def fetch_historical_data(
        self,
        asset: AssetSymbol,
        timeframe: Timeframe,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> Optional[HistoricalDataModel]:
        """
        Fetches historical candle data for a given asset and timeframe.

        Args:
            asset: The asset symbol (e.g., AssetSymbol.BTC_USD).
            timeframe: The timeframe for the candles (e.g., Timeframe.HOUR_1).
            start_date: Optional start datetime (UTC) for the data range. 
                        If None, a default period based on timeframe is used.
            end_date: Optional end datetime (UTC) for the data range. 
                      If None, defaults to now.
            use_cache: Whether to use the in-memory cache.

        Returns:
            A HistoricalDataModel containing the candles, or None if an error occurs.
        """
        cache_key = (asset, timeframe)
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            # Basic cache check: if start/end dates are not provided, cached data might be fine.
            # For more robust caching with date ranges, a more complex key or validation is needed.
            if cached_data and not start_date and not end_date:
                logger.info(f"Cache hit for {asset.value} {timeframe.value}")
                return cached_data

        # Try Redis cache
        if self.redis_client and use_cache and start_date and end_date:
            redis_key = f"historical:{asset.value}:{timeframe.value}:{start_date.isoformat()}:{end_date.isoformat()}"
            cached = self.redis_client.get(redis_key)
            if cached:
                logger.info(f"Redis cache hit for {asset.value} {timeframe.value}")
                return pickle.loads(cached)

        yf_ticker_symbol = asset.value
        yf_interval = TIMEFRAME_TO_YFINANCE_INTERVAL.get(timeframe)

        if not yf_interval:
            logger.error(f"Unsupported timeframe for yfinance: {timeframe.value}")
            return None

        # Handle 4h timeframe by fetching 1h and resampling
        # This is a simplification; robust resampling might need more logic
        # e.g. aligning to specific 4h candle open times (00, 04, 08, 12, 16, 20 UTC)
        resample_to_4h = False
        if timeframe == Timeframe.HOUR_4:
            yf_interval = TIMEFRAME_TO_YFINANCE_INTERVAL[Timeframe.HOUR_1] # Fetch 1h data
            resample_to_4h = True

        try:
            ticker = yf.Ticker(yf_ticker_symbol)
            df_history: pd.DataFrame
            if start_date and end_date:
                df_history = ticker.history(interval=yf_interval, start=start_date, end=end_date)
            elif start_date:
                df_history = ticker.history(interval=yf_interval, start=start_date)
            else:
                # Use default period if no start/end date
                period = DEFAULT_PERIOD_FOR_TIMEFRAME.get(timeframe, "1mo")
                df_history = ticker.history(interval=yf_interval, period=period)

            if df_history.empty:
                logger.warning(f"No data returned for {asset.value} with interval {yf_interval}")
                return HistoricalDataModel(asset=asset, timeframe=timeframe, candles=[])

            # Ensure timezone is UTC for consistency, yfinance usually returns tz-aware UTC for recent data
            if df_history.index.tz is None:
                df_history.index = df_history.index.tz_localize('UTC')
            else:
                df_history.index = df_history.index.tz_convert('UTC')

            # Handle yfinance MultiIndex for Forex (e.g., ('EURUSD=X', 'Open'))
            if isinstance(df_history.columns, pd.MultiIndex):
                df_history.columns = df_history.columns.droplevel(0) # Drop the asset symbol level

            # Rename columns to match our CandleModel, yfinance uses capitalized names
            df_history.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)

            # Resample to 4H if needed
            if resample_to_4h:
                # Resampling logic: 'first' for open, 'max' for high, 'min' for low, 'last' for close, 'sum' for volume
                # Ensure base=0 to align with 00:00, 04:00, etc. for 4H candles.
                # Pandas resample 'base' or 'offset' might be deprecated for 'origin'
                # For simplicity, we'll use standard resampling logic. Alignment may vary.
                resample_config = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                # df_history = df_history.resample('4H', origin='start_day').apply(resample_config).dropna()
                # 'origin' can be tricky with timezones. Using loffset or a similar approach might be needed for exact alignment.
                # A simpler resample for now, will produce candles based on available 1H data boundaries.
                df_history = df_history.resample('4H').apply(resample_config).dropna()
                if df_history.empty:
                    logger.warning(f"Resampling to 4H for {asset.value} resulted in empty data.")
                    return HistoricalDataModel(asset=asset, timeframe=timeframe, candles=[])

            candles: List[CandleModel] = []
            for timestamp, row in df_history.iterrows():
                # Ensure timestamp is a datetime object (it should be from DataFrame index)
                candle_ts = pd.to_datetime(timestamp).to_pydatetime()
                # Ensure it's offset-aware UTC
                if candle_ts.tzinfo is None or candle_ts.tzinfo.utcoffset(candle_ts) is None:
                    candle_ts = candle_ts.replace(tzinfo=timezone.utc)
                else:
                    candle_ts = candle_ts.astimezone(timezone.utc)

                candles.append(CandleModel(
                    timestamp=candle_ts,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"]
                ))
            
            historical_data = HistoricalDataModel(asset=asset, timeframe=timeframe, candles=candles)
            if use_cache:
                self._add_to_cache(cache_key, historical_data)
            
            # Store in Redis
            if self.redis_client and use_cache and start_date and end_date:
                redis_key = f"historical:{asset.value}:{timeframe.value}:{start_date.isoformat()}:{end_date.isoformat()}"
                self.redis_client.set(redis_key, pickle.dumps(historical_data), ex=3600)
            
            return historical_data

        except Exception as e:
            logger.error(f"Error fetching/processing data for {asset.value} ({timeframe.value}): {e}", exc_info=True)
            return None

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)
    agent = MarketDataAgent()

    # Test 1: BTC-USD 1h data for a specific period
    logger.info("\n--- Test 1: BTC-USD 1h data (specific period) ---")
    btc_start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    btc_end = datetime(2023, 1, 2, tzinfo=timezone.utc)
    btc_data_1h = agent.fetch_historical_data(AssetSymbol.BTC_USD, Timeframe.HOUR_1, start_date=btc_start, end_date=btc_end)
    if btc_data_1h and btc_data_1h.candles:
        logger.info(f"Fetched {len(btc_data_1h.candles)} candles for {btc_data_1h.asset.value} {btc_data_1h.timeframe.value}")
        logger.info(f"First candle: {btc_data_1h.candles[0]}")
        logger.info(f"Last candle: {btc_data_1h.candles[-1]}")
    else:
        logger.error("Failed to fetch BTC-USD 1h data or no candles found.")

    # Test 2: SPY 1d data using default period
    logger.info("\n--- Test 2: SPY 1d data (default period) ---")
    spy_data_1d = agent.fetch_historical_data(AssetSymbol.SPY, Timeframe.DAY_1)
    if spy_data_1d and spy_data_1d.candles:
        logger.info(f"Fetched {len(spy_data_1d.candles)} candles for {spy_data_1d.asset.value} {spy_data_1d.timeframe.value}")
        logger.info(f"Most recent candle: {spy_data_1d.candles[-1]}")
    else:
        logger.error("Failed to fetch SPY 1d data or no candles found.")

    # Test 3: EUR/USD 15m data
    logger.info("\n--- Test 3: EUR/USD 15m data (default period) ---")
    eurusd_data_15m = agent.fetch_historical_data(AssetSymbol.EUR_USD_FX, Timeframe.MIN_15)
    if eurusd_data_15m and eurusd_data_15m.candles:
        logger.info(f"Fetched {len(eurusd_data_15m.candles)} candles for {eurusd_data_15m.asset.value} {eurusd_data_15m.timeframe.value}")
        logger.info(f"Most recent candle: {eurusd_data_15m.candles[-1]}")
    else:
        logger.error("Failed to fetch EURUSD=X 15m data or no candles found.")
        
    # Test 4: BTC-USD 4h data (resampled from 1h)
    logger.info("\n--- Test 4: BTC-USD 4h data (resampled from 1h) ---")
    btc_data_4h = agent.fetch_historical_data(AssetSymbol.BTC_USD, Timeframe.HOUR_4, start_date=btc_start, end_date=btc_end)
    if btc_data_4h and btc_data_4h.candles:
        logger.info(f"Fetched {len(btc_data_4h.candles)} candles for {btc_data_4h.asset.value} {btc_data_4h.timeframe.value}")
        logger.info(f"First 4h candle: {btc_data_4h.candles[0]}")
        logger.info(f"Last 4h candle: {btc_data_4h.candles[-1]}")
    else:
        logger.error("Failed to fetch BTC-USD 4h data or no candles found.")

    # Test 5: Cache test
    logger.info("\n--- Test 5: Cache test for SPY 1d data ---")
    spy_data_1d_cached = agent.fetch_historical_data(AssetSymbol.SPY, Timeframe.DAY_1)
    if spy_data_1d_cached and spy_data_1d_cached.candles and spy_data_1d and spy_data_1d.candles and spy_data_1d_cached.candles[-1] == spy_data_1d.candles[-1]:
        logger.info("Successfully fetched SPY 1d data from cache.")
    else:
        logger.error("Cache test failed for SPY 1d data.")
