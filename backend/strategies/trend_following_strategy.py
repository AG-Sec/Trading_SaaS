"""
Trend Following Strategy for Trading SaaS

This module implements a trend following strategy that identifies and trades
in the direction of established market trends using moving averages,
momentum indicators, and price action.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from shared_types.models import (
    AssetSymbol, Timeframe, TradingSignalModel, 
    HistoricalDataModel, SignalType
)
from backend.agents.market_regime_detector import MarketRegime, MarketRegimeType
from backend.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend Following Strategy Implementation.
    
    This strategy looks for:
    1. Directional price movement using EMA crossovers
    2. Strong momentum confirmed by ADX
    3. Pullbacks to moving averages as entry opportunities
    """
    
    def __init__(self):
        """Initialize the Trend Following Strategy."""
        super().__init__(
            name="Trend Following",
            description="Identifies and follows established market trends using moving averages and momentum indicators."
        )
        # Set default parameters
        self.set_parameters(self.get_default_parameters())
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for this strategy.
        
        Returns:
            Dictionary of default parameters
        """
        return {
            # Moving average parameters
            "ema_fast_period": 9,
            "ema_slow_period": 21,
            "ema_signal_period": 50,  # Trend direction confirmation
            
            # ADX parameters (Average Directional Index for trend strength)
            "adx_period": 14,
            "adx_threshold": 25,  # Minimum ADX for strong trend
            
            # MACD parameters
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            
            # Pullback parameters
            "allow_pullback_entry": True,
            "pullback_threshold": 0.3,  # 30% retracement to EMAs
            
            # Risk management
            "atr_period": 14,
            "atr_multiplier_sl": 1.5,
            "min_rr_ratio": 1.8,
            
            # Filters
            "min_volume_percentile": 40,  # Min volume must be in the top 40th percentile
        }
    
    def get_regime_adapted_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Adapt parameters based on the current market regime.
        
        Args:
            regime: The current market regime
            
        Returns:
            Dictionary of adapted parameters
        """
        params = self.get_default_parameters()
        
        # Adjust based on regime type
        if regime.regime_type == MarketRegimeType.BULLISH_TRENDING:
            # Best scenario for trend following - optimize for performance
            params["ema_fast_period"] = 8
            params["adx_threshold"] = 20
            params["atr_multiplier_sl"] = 1.3
            params["min_rr_ratio"] = 2.0
            params["pullback_threshold"] = 0.4  # Allow deeper pullbacks
            
        elif regime.regime_type == MarketRegimeType.BEARISH_TRENDING:
            # Bearish trends - be slightly more conservative
            params["ema_fast_period"] = 10
            params["adx_threshold"] = 22
            params["atr_multiplier_sl"] = 1.4
            params["min_rr_ratio"] = 1.8
            params["pullback_threshold"] = 0.35
            
        elif regime.regime_type == MarketRegimeType.HIGH_VOLATILITY:
            # More volatile - wider stops, higher returns
            params["ema_fast_period"] = 7
            params["adx_threshold"] = 30  # Require stronger trends
            params["atr_multiplier_sl"] = 2.0
            params["min_rr_ratio"] = 2.5
            params["pullback_threshold"] = 0.5  # Allow deeper pullbacks
            
        elif regime.regime_type == MarketRegimeType.LOW_VOLATILITY:
            # Less volatile - tighter parameters
            params["ema_fast_period"] = 12
            params["adx_threshold"] = 20
            params["atr_multiplier_sl"] = 1.2
            params["min_rr_ratio"] = 1.5
            params["pullback_threshold"] = 0.25  # Smaller pullbacks
            
        elif regime.regime_type == MarketRegimeType.NEUTRAL_RANGING:
            # Ranging markets - not ideal for trend following
            params["ema_fast_period"] = 6
            params["adx_threshold"] = 35  # Only strongest trends
            params["atr_multiplier_sl"] = 1.3
            params["min_rr_ratio"] = 2.0
            params["pullback_threshold"] = 0.2  # More conservative
        
        # Adjust for regime strength
        strength_factor = regime.strength
        
        # For favorable regimes, higher strength = more aggressive
        if regime.regime_type in [MarketRegimeType.BULLISH_TRENDING, MarketRegimeType.BEARISH_TRENDING]:
            params["min_rr_ratio"] += strength_factor * 0.5
            params["adx_threshold"] = max(15, params["adx_threshold"] - (strength_factor * 5))
        else:
            # For less favorable regimes, higher strength = more conservative
            params["adx_threshold"] += strength_factor * 5
            
        return params
    
    def is_suitable_for_regime(self, regime_type: MarketRegimeType) -> bool:
        """
        Determine if this strategy is suitable for the given market regime.
        
        Args:
            regime_type: The market regime type
            
        Returns:
            True if suitable, False otherwise
        """
        return regime_type in [
            MarketRegimeType.BULLISH_TRENDING,
            MarketRegimeType.BEARISH_TRENDING,
            MarketRegimeType.HIGH_VOLATILITY
        ]
    
    def get_confidence_for_regime(self, regime: MarketRegime) -> float:
        """
        Get the confidence score for this strategy in the given market regime.
        
        Args:
            regime: The market regime
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        if regime.regime_type == MarketRegimeType.BULLISH_TRENDING:
            return 0.8 + (regime.strength * 0.2)  # 0.8 to 1.0
        elif regime.regime_type == MarketRegimeType.BEARISH_TRENDING:
            return 0.75 + (regime.strength * 0.15)  # 0.75 to 0.9
        elif regime.regime_type == MarketRegimeType.HIGH_VOLATILITY:
            return 0.6 + (regime.strength * 0.1)  # 0.6 to 0.7
        elif regime.regime_type == MarketRegimeType.LOW_VOLATILITY:
            return 0.5 - (regime.strength * 0.1)  # 0.4 to 0.5
        elif regime.regime_type == MarketRegimeType.NEUTRAL_RANGING:
            return 0.3 - (regime.strength * 0.1)  # 0.2 to 0.3
        else:
            return 0.5  # Neutral
    
    def generate_signals(self, 
                        asset: AssetSymbol, 
                        timeframe: Timeframe, 
                        historical_data: HistoricalDataModel) -> List[TradingSignalModel]:
        """
        Generate trend following trading signals.
        
        Args:
            asset: The asset to generate signals for
            timeframe: The timeframe to use
            historical_data: Historical price data
            
        Returns:
            List of generated trading signals
        """
        # Preprocess data
        df = self.preprocess_data(historical_data)
        
        # Check if we have enough data
        min_required_bars = max(
            self.parameters["ema_slow_period"],
            self.parameters["ema_signal_period"],
            self.parameters["adx_period"],
            self.parameters["macd_slow_period"],
            self.parameters["atr_period"]
        ) + 10  # Add some buffer
        
        if len(df) < min_required_bars:
            logger.warning(f"Insufficient data for {asset.value} {timeframe.value} to generate trend following signals. Need at least {min_required_bars} bars.")
            return []
        
        # Get current market regime
        regime = self.detect_regime(df, asset, timeframe)
        logger.info(f"Detected {regime.regime_type.value} regime for {asset.value} {timeframe.value} with strength {regime.strength:.2f}")
        
        # Adapt parameters to the current regime
        params = self.get_regime_adapted_parameters(regime)
        logger.info(f"Using regime-adapted parameters for {asset.value} {timeframe.value}: {params}")
        
        # Calculate indicators
        signals = []
        
        # 1. Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=params["ema_fast_period"], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=params["ema_slow_period"], adjust=False).mean()
        df['ema_signal'] = df['close'].ewm(span=params["ema_signal_period"], adjust=False).mean()
        
        # 2. Calculate ADX (Average Directional Index)
        df = self._calculate_adx(df, params["adx_period"])
        
        # 3. Calculate MACD
        df = self._calculate_macd(df, params["macd_fast_period"], params["macd_slow_period"], params["macd_signal_period"])
        
        # 4. Calculate ATR for stop loss
        df['atr'] = self._calculate_atr(df, params["atr_period"])
        
        # 5. Calculate volume percentile (20-day rolling)
        df['volume_pct'] = df['volume'].rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
        
        # Generate signals for the most recent candle
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Check for volume threshold
        volume_confirmed = current['volume_pct'] >= params["min_volume_percentile"]
        
        # Trend direction based on EMAs and signal line
        bullish_trend = (
            current['ema_fast'] > current['ema_slow'] and 
            current['ema_slow'] > current['ema_signal'] and
            current['close'] > current['ema_signal']
        )
        
        bearish_trend = (
            current['ema_fast'] < current['ema_slow'] and 
            current['ema_slow'] < current['ema_signal'] and
            current['close'] < current['ema_signal']
        )
        
        # Check for strong trend using ADX
        strong_trend = current['adx'] > params["adx_threshold"]
        
        # Check for EMA crossover signals
        ema_crossover_bullish = (
            prev['ema_fast'] <= prev['ema_slow'] and 
            current['ema_fast'] > current['ema_slow']
        )
        
        ema_crossover_bearish = (
            prev['ema_fast'] >= prev['ema_slow'] and 
            current['ema_fast'] < current['ema_slow']
        )
        
        # Check for MACD signals
        macd_bullish = (
            prev['macd_histogram'] < 0 and
            current['macd_histogram'] > 0
        )
        
        macd_bearish = (
            prev['macd_histogram'] > 0 and
            current['macd_histogram'] < 0
        )
        
        # Check for pullback entries
        pullback_to_ema_bullish = (
            params["allow_pullback_entry"] and
            bullish_trend and
            prev['close'] <= prev['ema_slow'] and
            current['close'] > current['ema_slow']
        )
        
        pullback_to_ema_bearish = (
            params["allow_pullback_entry"] and
            bearish_trend and
            prev['close'] >= prev['ema_slow'] and
            current['close'] < current['ema_slow']
        )
        
        # 1. Long signal conditions
        long_signal_condition = (
            (ema_crossover_bullish or macd_bullish or pullback_to_ema_bullish) and
            bullish_trend and
            strong_trend
        )
        
        if long_signal_condition and volume_confirmed:
            # Generate long signal
            entry_price = current['close']
            stop_loss = entry_price - (current['atr'] * params["atr_multiplier_sl"])
            
            # Calculate take profit based on minimum risk-reward ratio
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * params["min_rr_ratio"])
            
            # Create signal object
            signal_id = str(uuid.uuid4())
            signal = TradingSignalModel(
                signal_id=signal_id,
                generated_at=datetime.now(timezone.utc),
                asset=asset,
                timeframe=timeframe,
                signal_type=SignalType.LONG,
                entry_price=round(entry_price, 5),
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                metadata={
                    "strategy_name": self.name,
                    "strategy_type": "trend_following",
                    "ema_fast": round(current['ema_fast'], 5),
                    "ema_slow": round(current['ema_slow'], 5),
                    "adx": round(current['adx'], 2),
                    "macd": round(current['macd'], 5),
                    "macd_signal": round(current['macd_signal'], 5),
                    "macd_histogram": round(current['macd_histogram'], 5),
                    "atr": round(current['atr'], 5),
                    "market_regime": regime.regime_type.value,
                    "regime_strength": round(regime.strength, 2),
                    "volume_percentile": round(current['volume_pct'], 1),
                    "signal_type": "ema_crossover" if ema_crossover_bullish else 
                                  "macd_signal" if macd_bullish else
                                  "pullback_entry"
                }
            )
            
            logger.info(f"Generated LONG trend following signal for {asset.value} {timeframe.value}")
            logger.info(f"  Entry: {entry_price:.5f}, Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
            logger.info(f"  ADX: {current['adx']:.2f}, Fast EMA: {current['ema_fast']:.5f}, Slow EMA: {current['ema_slow']:.5f}")
            
            signals.append(signal)
            
        # 2. Short signal conditions
        short_signal_condition = (
            (ema_crossover_bearish or macd_bearish or pullback_to_ema_bearish) and
            bearish_trend and
            strong_trend
        )
        
        if short_signal_condition and volume_confirmed:
            # Generate short signal
            entry_price = current['close']
            stop_loss = entry_price + (current['atr'] * params["atr_multiplier_sl"])
            
            # Calculate take profit based on minimum risk-reward ratio
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * params["min_rr_ratio"])
            
            # Create signal object
            signal_id = str(uuid.uuid4())
            signal = TradingSignalModel(
                signal_id=signal_id,
                generated_at=datetime.now(timezone.utc),
                asset=asset,
                timeframe=timeframe,
                signal_type=SignalType.SHORT,
                entry_price=round(entry_price, 5),
                stop_loss=round(stop_loss, 5),
                take_profit=round(take_profit, 5),
                metadata={
                    "strategy_name": self.name,
                    "strategy_type": "trend_following",
                    "ema_fast": round(current['ema_fast'], 5),
                    "ema_slow": round(current['ema_slow'], 5),
                    "adx": round(current['adx'], 2),
                    "macd": round(current['macd'], 5),
                    "macd_signal": round(current['macd_signal'], 5),
                    "macd_histogram": round(current['macd_histogram'], 5),
                    "atr": round(current['atr'], 5),
                    "market_regime": regime.regime_type.value,
                    "regime_strength": round(regime.strength, 2),
                    "volume_percentile": round(current['volume_pct'], 1),
                    "signal_type": "ema_crossover" if ema_crossover_bearish else 
                                  "macd_signal" if macd_bearish else
                                  "pullback_entry"
                }
            )
            
            logger.info(f"Generated SHORT trend following signal for {asset.value} {timeframe.value}")
            logger.info(f"  Entry: {entry_price:.5f}, Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
            logger.info(f"  ADX: {current['adx']:.2f}, Fast EMA: {current['ema_fast']:.5f}, Slow EMA: {current['ema_slow']:.5f}")
            
            signals.append(signal)
            
        return signals
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            df: DataFrame with high, low, close columns
            period: ADX period
            
        Returns:
            DataFrame with ADX, +DI, -DI columns added
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        df['tr1'] = abs(high - low)
        df['tr2'] = abs(high - close.shift(1))
        df['tr3'] = abs(low - close.shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Directional Movement
        df['up_move'] = high - high.shift(1)
        df['down_move'] = low.shift(1) - low
        
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Calculate smoothed values
        df['tr_' + str(period)] = df['tr'].rolling(window=period).sum()
        df['plus_dm_' + str(period)] = df['plus_dm'].rolling(window=period).sum()
        df['minus_dm_' + str(period)] = df['minus_dm'].rolling(window=period).sum()
        
        # Calculate +DI and -DI
        df['plus_di'] = 100 * (df['plus_dm_' + str(period)] / df['tr_' + str(period)])
        df['minus_di'] = 100 * (df['minus_dm_' + str(period)] / df['tr_' + str(period)])
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame with close column
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with MACD, signal line, and histogram columns added
        """
        df['ema_fast_macd'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow_macd'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        df['macd'] = df['ema_fast_macd'] - df['ema_slow_macd']
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with high, low, close columns
            period: ATR period
            
        Returns:
            Series of ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
