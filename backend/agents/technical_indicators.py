"""
Technical indicators module for the Trading SaaS platform.
Provides common technical indicators using TA-Lib and additional functions.
"""
import numpy as np
import pandas as pd
import talib
import logging
from typing import Optional, Dict, Tuple, List, Union

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Class that provides technical indicators and pattern recognition.
    """
    def __init__(self):
        # Indicator default parameters
        self.ema_fast_period = 9
        self.ema_slow_period = 21
        self.sma_period = 200
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2.0
        self.atr_period = 14
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        self.stoch_slowing = 3
        self.vwap_periods = [14, 30]  # Multiple VWAP periods

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the dataframe.
        
        Args:
            df: DataFrame with at least OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        if len(df) < 50:  # Need sufficient data
            logger.warning(f"Not enough data for indicator calculations. Need at least 50 bars, got {len(df)}")
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate all indicators
        try:
            # Moving Averages
            result_df['ema_fast'] = talib.EMA(result_df['close'], timeperiod=self.ema_fast_period)
            result_df['ema_slow'] = talib.EMA(result_df['close'], timeperiod=self.ema_slow_period)
            result_df['sma_200'] = talib.SMA(result_df['close'], timeperiod=self.sma_period)
            
            # RSI
            result_df['rsi'] = talib.RSI(result_df['close'], timeperiod=self.rsi_period)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                result_df['close'], 
                fastperiod=self.macd_fast, 
                slowperiod=self.macd_slow, 
                signalperiod=self.macd_signal
            )
            result_df['macd'] = macd
            result_df['macd_signal'] = macd_signal
            result_df['macd_hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                result_df['close'],
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std
            )
            result_df['bb_upper'] = bb_upper
            result_df['bb_middle'] = bb_middle
            result_df['bb_lower'] = bb_lower
            
            # ATR
            result_df['atr'] = talib.ATR(
                result_df['high'],
                result_df['low'],
                result_df['close'],
                timeperiod=self.atr_period
            )
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                result_df['high'],
                result_df['low'],
                result_df['close'],
                fastk_period=self.stoch_k_period,
                slowk_period=self.stoch_slowing,
                slowk_matype=0,
                slowd_period=self.stoch_d_period,
                slowd_matype=0
            )
            result_df['stoch_k'] = stoch_k
            result_df['stoch_d'] = stoch_d
            
            # Calculate VWAP (custom implementation since TA-Lib doesn't provide it)
            for period in self.vwap_periods:
                result_df[f'vwap_{period}'] = self._calculate_vwap(result_df, period)
            
            # Add derived indicators
            result_df['ema_cross'] = self._calculate_ema_cross(result_df)
            result_df['price_vs_sma200'] = (result_df['close'] > result_df['sma_200']).astype(int)
            result_df['macd_cross'] = self._calculate_macd_cross(result_df)
            result_df['bb_position'] = self._calculate_bb_position(result_df)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            
        return result_df

    def _calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP for the specified period"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        typical_price_volume = typical_price * df['volume']
        
        # Use rolling window
        sum_typical_price_volume = typical_price_volume.rolling(window=period).sum()
        sum_volume = df['volume'].rolling(window=period).sum()
        
        # Avoid division by zero
        vwap = np.where(sum_volume > 0, sum_typical_price_volume / sum_volume, np.nan)
        return pd.Series(vwap, index=df.index)
    
    def _calculate_ema_cross(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate EMA cross signal:
        1 = Fast EMA crosses above Slow EMA (bullish)
        -1 = Fast EMA crosses below Slow EMA (bearish)
        0 = No cross
        """
        # Current state
        current_cross = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        
        # Previous state (shifted by 1)
        previous_cross = np.roll(current_cross, 1)
        previous_cross[0] = 0  # First value can't have a cross
        
        # Signal only when cross occurs
        cross_signal = np.where(current_cross != previous_cross, current_cross, 0)
        
        return pd.Series(cross_signal, index=df.index)
    
    def _calculate_macd_cross(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate MACD cross signal:
        1 = MACD line crosses above signal line (bullish)
        -1 = MACD line crosses below signal line (bearish)
        0 = No cross
        """
        # Current state
        current_cross = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Previous state (shifted by 1)
        previous_cross = np.roll(current_cross, 1)
        previous_cross[0] = 0  # First value can't have a cross
        
        # Signal only when cross occurs
        cross_signal = np.where(current_cross != previous_cross, current_cross, 0)
        
        return pd.Series(cross_signal, index=df.index)
    
    def _calculate_bb_position(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate position within Bollinger Bands:
        Value from -1 to 1 where:
        -1 = At or below lower band
        0 = At middle band
        1 = At or above upper band
        """
        # Calculate position as percentage of the way from lower to upper band
        bb_range = df['bb_upper'] - df['bb_lower']
        position = (df['close'] - df['bb_lower']) / bb_range
        
        # Clip values to range [-1, 1]
        position = position.clip(0, 1) * 2 - 1
        
        return position
        
    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average for the given series and period.
        
        Args:
            series: Series of price data
            period: SMA period
            
        Returns:
            Series with SMA values
        """
        try:
            return talib.SMA(series, timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.Series(np.nan, index=series.index)
            
    def calculate_bollinger_bands(self, series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict:
        """
        Calculate Bollinger Bands for the given series.
        
        Args:
            series: Series of price data
            period: Bollinger Band period
            num_std: Number of standard deviations
            
        Returns:
            Dictionary with upper, middle, and lower bands as Series
        """
        try:
            upper, middle, lower = talib.BBANDS(
                series,
                timeperiod=period,
                nbdevup=num_std,
                nbdevdn=num_std
            )
            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            empty = pd.Series(np.nan, index=series.index)
            return {'upper': empty, 'middle': empty, 'lower': empty}

    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detect candlestick patterns using TA-Lib.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dictionary with pattern names and their signals
        """
        patterns = {}
        
        try:
            # Bullish patterns
            patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            patterns['inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
            patterns['engulfing_bullish'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            patterns['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            patterns['piercing'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
            
            # Bearish patterns
            patterns['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            patterns['engulfing_bearish'] = -talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            patterns['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}", exc_info=True)
            
        return patterns

    def market_regime(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        Detect market regime (trend, range, volatile) and its strength.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Tuple of (regime_type, regime_strength, additional_info)
            where regime_type is one of: 'bullish', 'bearish', 'neutral', 'volatile'
        """
        if len(df) < 50:
            return 'unknown', 0.0, {}
            
        # Get latest values for analysis
        latest = df.iloc[-1]
        recent = df.iloc[-20:]  # Last 20 bars
        
        # Calculate volatility metrics
        atr_pct = latest['atr'] / latest['close'] * 100  # ATR as percentage of price
        
        # BB Width as volatility indicator
        bb_width = (latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']
        
        # ADX for trend strength - we'll approximate with the distance between EMAs
        ema_distance = abs(latest['ema_fast'] - latest['ema_slow']) / latest['close'] * 100
        
        # Analyze price in relation to key MAs
        above_sma200 = latest['close'] > latest['sma_200']
        fast_above_slow = latest['ema_fast'] > latest['ema_slow']
        
        # RSI for overbought/oversold
        rsi_overbought = latest['rsi'] > 70
        rsi_oversold = latest['rsi'] < 30
        
        # MACD for momentum
        macd_positive = latest['macd'] > 0
        macd_above_signal = latest['macd'] > latest['macd_signal']
        
        # Look for recent crosses
        recent_ema_cross_up = any(recent['ema_cross'] == 1)
        recent_ema_cross_down = any(recent['ema_cross'] == -1)
        recent_macd_cross_up = any(recent['macd_cross'] == 1)
        recent_macd_cross_down = any(recent['macd_cross'] == -1)
        
        # Identify market regime
        regime_type = 'neutral'
        regime_strength = 0.5  # Default middle value
        
        # Bullish signals
        bullish_points = 0
        if above_sma200: bullish_points += 1
        if fast_above_slow: bullish_points += 1
        if macd_positive: bullish_points += 1
        if macd_above_signal: bullish_points += 1
        if recent_ema_cross_up: bullish_points += 1
        if recent_macd_cross_up: bullish_points += 1
        if rsi_oversold: bullish_points += 0.5  # Potential reversal
        
        # Bearish signals
        bearish_points = 0
        if not above_sma200: bearish_points += 1
        if not fast_above_slow: bearish_points += 1
        if not macd_positive: bearish_points += 1
        if not macd_above_signal: bearish_points += 1
        if recent_ema_cross_down: bearish_points += 1
        if recent_macd_cross_down: bearish_points += 1
        if rsi_overbought: bearish_points += 0.5  # Potential reversal
        
        # Determine regime type
        if bullish_points >= 4 and bullish_points > bearish_points:
            regime_type = 'bullish'
            regime_strength = min(1.0, bullish_points / 6.0)
        elif bearish_points >= 4 and bearish_points > bullish_points:
            regime_type = 'bearish'
            regime_strength = min(1.0, bearish_points / 6.0)
        
        # Check for volatility - overrides trend determination if very high
        if atr_pct > 3.0 or bb_width > 0.1:  # High volatility thresholds
            regime_type = 'volatile'
            regime_strength = min(1.0, max(atr_pct / 5.0, bb_width / 0.2))
            
        # Build additional info dict
        additional_info = {
            'atr_pct': atr_pct,
            'bb_width': bb_width,
            'ema_distance_pct': ema_distance,
            'rsi': latest['rsi'],
            'bullish_signals': bullish_points,
            'bearish_signals': bearish_points,
            'above_sma200': above_sma200,
            'macd_histogram': latest['macd_hist']
        }
            
        return regime_type, regime_strength, additional_info
