import logging
import random
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from backend.models.portfolio import AssetClass

# Configure logging
logger = logging.getLogger(__name__)

class MarketDataService:
    """Service for fetching market data like prices, OHLCV data, etc."""
    
    _price_cache = {}
    _cache_expiry = {}
    CACHE_DURATION = 300  # 5 minutes in seconds
    
    @staticmethod
    async def get_current_prices(symbols: List[str], asset_class: AssetClass) -> Dict[str, float]:
        """
        Get current prices for a list of symbols.
        
        Args:
            symbols: List of asset symbols
            asset_class: Asset class
            
        Returns:
            Dictionary mapping symbol to price
        """
        result = {}
        uncached_symbols = []
        
        # Check cache first
        current_time = datetime.now().timestamp()
        for symbol in symbols:
            cache_key = f"{symbol}_{asset_class}"
            if (cache_key in MarketDataService._price_cache and 
                current_time - MarketDataService._cache_expiry.get(cache_key, 0) < MarketDataService.CACHE_DURATION):
                result[symbol] = MarketDataService._price_cache[cache_key]
            else:
                uncached_symbols.append(symbol)
        
        if not uncached_symbols:
            return result
        
        try:
            if asset_class == AssetClass.CRYPTO or asset_class == AssetClass.STOCK or asset_class == AssetClass.ETF:
                # Try to fetch data from yfinance
                try:
                    data = yf.download(uncached_symbols, period="1d", progress=False)
                    if 'Close' in data.columns:
                        closes = data['Close']
                        if len(uncached_symbols) == 1:
                            # If only one symbol, the result is a Series
                            symbol = uncached_symbols[0]
                            price = closes.iloc[-1]
                            result[symbol] = float(price)
                            
                            # Update cache
                            cache_key = f"{symbol}_{asset_class}"
                            MarketDataService._price_cache[cache_key] = float(price)
                            MarketDataService._cache_expiry[cache_key] = current_time
                        else:
                            # For multiple symbols, result is a DataFrame
                            for symbol in uncached_symbols:
                                if symbol in closes.columns:
                                    price = closes[symbol].iloc[-1]
                                    if not pd.isna(price):
                                        result[symbol] = float(price)
                                        
                                        # Update cache
                                        cache_key = f"{symbol}_{asset_class}"
                                        MarketDataService._price_cache[cache_key] = float(price)
                                        MarketDataService._cache_expiry[cache_key] = current_time
                except Exception as e:
                    logger.error(f"Error fetching data from yfinance: {str(e)}")
                    # Fall back to mock prices for failed symbols
                    for symbol in uncached_symbols:
                        if symbol not in result:
                            result[symbol] = MarketDataService._get_mock_price(symbol, asset_class)
            else:
                # For other asset classes, use mock prices
                for symbol in uncached_symbols:
                    result[symbol] = MarketDataService._get_mock_price(symbol, asset_class)
                
        except Exception as e:
            logger.error(f"Error fetching prices: {str(e)}")
            # Return mock prices in case of error
            for symbol in uncached_symbols:
                if symbol not in result:
                    result[symbol] = MarketDataService._get_mock_price(symbol, asset_class)
        
        return result
    
    @staticmethod
    def _get_mock_price(symbol: str, asset_class: AssetClass) -> float:
        """Generate a realistic mock price for testing"""
        # Use symbol string hash to generate a stable price
        symbol_hash = sum(ord(char) for char in symbol)
        base_price = (symbol_hash % 900) + 100  # Range from $100 to $1000
        
        # Asset class adjustments
        if asset_class == AssetClass.CRYPTO:
            if symbol.startswith(("BTC", "XBT")):
                base_price = 30000 + (symbol_hash % 5000)  # Bitcoin range
            elif symbol.startswith("ETH"):
                base_price = 2000 + (symbol_hash % 300)  # Ethereum range
            else:
                base_price = (symbol_hash % 90) + 10  # Other crypto range
        elif asset_class == AssetClass.FOREX:
            base_price = 1.0 + ((symbol_hash % 200) / 100)  # Forex around 1.0-3.0
        elif asset_class == AssetClass.OPTIONS:
            base_price = (symbol_hash % 30) + 1  # Options in lower ranges
        
        # Add some small random variation
        variation = (random.random() - 0.5) * 0.02 * base_price  # ±1% random fluctuation
        
        price = base_price + variation
        
        # Update cache
        cache_key = f"{symbol}_{asset_class}"
        MarketDataService._price_cache[cache_key] = price
        MarketDataService._cache_expiry[cache_key] = datetime.now().timestamp()
        
        return price
    
    @staticmethod
    async def get_historical_data(
        symbol: str, 
        asset_class: AssetClass, 
        timeframe: str = "1d", 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        periods: int = 100
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Asset symbol
            asset_class: Asset class
            timeframe: Data timeframe (e.g., "1d", "1h")
            start_date: Start date for data
            end_date: End date for data
            periods: Number of periods if dates not specified
            
        Returns:
            DataFrame with historical data
        """
        if end_date is None:
            end_date = datetime.now()
            
        if start_date is None:
            if timeframe.endswith("d"):
                days = int(timeframe[:-1]) * periods
                start_date = end_date - timedelta(days=days)
            elif timeframe.endswith("h"):
                hours = int(timeframe[:-1]) * periods
                start_date = end_date - timedelta(hours=hours)
            else:
                # Default to 100 days
                start_date = end_date - timedelta(days=100)
        
        try:
            if asset_class in [AssetClass.STOCK, AssetClass.ETF, AssetClass.CRYPTO]:
                # Convert timeframe to yfinance interval
                if timeframe == "1d":
                    interval = "1d"
                elif timeframe == "1h":
                    interval = "1h"
                elif timeframe == "15m":
                    interval = "15m"
                else:
                    interval = "1d"  # Default
                
                try:
                    # Try to fetch from yfinance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval=interval
                    )
                    
                    if not data.empty:
                        return data
                except Exception as e:
                    logger.error(f"Error fetching historical data from yfinance: {str(e)}")
                    # Fall back to mock data
            
            # Generate mock data
            return MarketDataService._get_mock_historical_data(
                symbol, asset_class, timeframe, start_date, end_date
            )
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            # Return mock data in case of error
            return MarketDataService._get_mock_historical_data(
                symbol, asset_class, timeframe, start_date, end_date
            )
    
    @staticmethod
    def _get_mock_historical_data(
        symbol: str, 
        asset_class: AssetClass, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate realistic mock historical data for testing"""
        # Use symbol to generate stable price trends
        symbol_hash = sum(ord(char) for char in symbol)
        
        # Determine price range based on asset class and symbol
        if asset_class == AssetClass.CRYPTO:
            if symbol.startswith(("BTC", "XBT")):
                base_price = 30000 + (symbol_hash % 5000)
                volatility = 0.03  # 3% daily volatility
            elif symbol.startswith("ETH"):
                base_price = 2000 + (symbol_hash % 300)
                volatility = 0.04
            else:
                base_price = (symbol_hash % 90) + 10
                volatility = 0.05
        elif asset_class == AssetClass.FOREX:
            base_price = 1.0 + ((symbol_hash % 200) / 100)
            volatility = 0.005
        elif asset_class == AssetClass.OPTIONS:
            base_price = (symbol_hash % 30) + 1
            volatility = 0.08
        else:  # Stocks, ETFs, and others
            base_price = (symbol_hash % 900) + 100
            volatility = 0.015
        
        # Determine time delta based on timeframe
        if timeframe.endswith("d"):
            delta = timedelta(days=1)
            periods = int((end_date - start_date).days)
        elif timeframe.endswith("h"):
            delta = timedelta(hours=1)
            periods = int((end_date - start_date).total_seconds() / 3600)
        elif timeframe.endswith("m"):
            minutes = int(timeframe[:-1])
            delta = timedelta(minutes=minutes)
            periods = int((end_date - start_date).total_seconds() / (60 * minutes))
        else:
            delta = timedelta(days=1)
            periods = int((end_date - start_date).days)
        
        periods = max(periods, 2)  # Ensure at least 2 periods
        
        # Generate date range
        dates = [start_date + delta * i for i in range(periods)]
        
        # Generate price data with random walk and trend
        trend = (symbol_hash % 5 - 2) / 100  # Random trend between -2% and +2%
        prices = [base_price]
        for i in range(1, periods):
            # Random walk with trend
            change = (random.random() - 0.5) * 2 * volatility + trend
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Generate OHLC from close prices
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            high = close * (1 + random.random() * volatility / 2)
            low = close * (1 - random.random() * volatility / 2)
            open_price = low + random.random() * (high - low)
            volume = int(base_price * 1000 * (0.5 + random.random()))
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        return df
