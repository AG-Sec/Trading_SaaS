"""
Mean Reversion Strategy for Trading SaaS

This module implements a mean reversion trading strategy that looks for price movements
that deviate significantly from their historical mean and are likely to revert back.
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

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy Implementation.
    
    This strategy looks for:
    1. Overbought/oversold conditions using RSI
    2. Price deviations from a moving average
    3. Bollinger Band extremes
    """
    
    def __init__(self):
        """Initialize the Mean Reversion Strategy."""
        super().__init__(
            name="Mean Reversion",
            description="Identifies assets that have moved significantly away from their average price and are likely to revert back to the mean."
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
            # RSI parameters
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            
            # Moving average parameters
            "ma_period": 20,
            "deviation_threshold": 0.03,  # 3% deviation from MA
            
            # Bollinger Band parameters
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "bb_threshold": 0.05,  # 5% from band to trigger
            
            # Risk management
            "atr_period": 14,
            "atr_multiplier_sl": 1.5,
            "min_rr_ratio": 1.5,
            
            # Filters
            "min_volume_percentile": 30,  # Min volume must be in the top 30th percentile
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
        if regime.regime_type == MarketRegimeType.NEUTRAL_RANGING:
            # Best scenario for mean reversion - tighten parameters
            params["rsi_overbought"] = 65
            params["rsi_oversold"] = 35
            params["deviation_threshold"] = 0.02
            params["bb_threshold"] = 0.03
            params["atr_multiplier_sl"] = 1.2
            params["min_rr_ratio"] = 1.7
            
        elif regime.regime_type == MarketRegimeType.HIGH_VOLATILITY:
            # More volatile - widen parameters
            params["rsi_overbought"] = 80
            params["rsi_oversold"] = 20
            params["deviation_threshold"] = 0.05
            params["bb_threshold"] = 0.08
            params["atr_multiplier_sl"] = 2.0
            params["min_rr_ratio"] = 2.0
            
        elif regime.regime_type == MarketRegimeType.LOW_VOLATILITY:
            # Less volatile - use tighter parameters
            params["rsi_overbought"] = 70
            params["rsi_oversold"] = 30
            params["deviation_threshold"] = 0.02
            params["bb_threshold"] = 0.04
            params["atr_multiplier_sl"] = 1.0
            params["min_rr_ratio"] = 1.2
            
        elif regime.regime_type in [MarketRegimeType.BULLISH_TRENDING, MarketRegimeType.BEARISH_TRENDING]:
            # Trending markets are less ideal for mean reversion - be more selective
            params["rsi_overbought"] = 80
            params["rsi_oversold"] = 20
            params["deviation_threshold"] = 0.04
            params["bb_threshold"] = 0.06
            params["atr_multiplier_sl"] = 1.8
            params["min_rr_ratio"] = 1.8
        
        # Adjust for regime strength
        strength_factor = regime.strength
        
        # For certain parameters, adjust based on strength
        if regime.regime_type in [MarketRegimeType.NEUTRAL_RANGING, MarketRegimeType.LOW_VOLATILITY]:
            # For favorable regimes, higher strength = more aggressive
            params["min_rr_ratio"] *= max(0.8, 1.0 - (strength_factor * 0.2))
        else:
            # For less favorable regimes, higher strength = more conservative
            params["min_rr_ratio"] *= min(1.5, 1.0 + (strength_factor * 0.5))
            
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
            MarketRegimeType.NEUTRAL_RANGING,
            MarketRegimeType.LOW_VOLATILITY
        ]
    
    def get_confidence_for_regime(self, regime: MarketRegime) -> float:
        """
        Get the confidence score for this strategy in the given market regime.
        
        Args:
            regime: The market regime
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        if regime.regime_type == MarketRegimeType.NEUTRAL_RANGING:
            return 0.8 + (regime.strength * 0.2)  # 0.8 to 1.0
        elif regime.regime_type == MarketRegimeType.LOW_VOLATILITY:
            return 0.7 + (regime.strength * 0.1)  # 0.7 to 0.8
        elif regime.regime_type == MarketRegimeType.HIGH_VOLATILITY:
            return 0.4 - (regime.strength * 0.1)  # 0.3 to 0.4
        elif regime.regime_type in [MarketRegimeType.BULLISH_TRENDING, MarketRegimeType.BEARISH_TRENDING]:
            return 0.3 - (regime.strength * 0.1)  # 0.2 to 0.3
        else:
            return 0.5  # Neutral
    
    def generate_signals(self, 
                        asset: AssetSymbol, 
                        timeframe: Timeframe, 
                        historical_data: HistoricalDataModel) -> List[TradingSignalModel]:
        """
        Generate mean reversion trading signals.
        
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
            self.parameters["rsi_period"],
            self.parameters["ma_period"],
            self.parameters["bb_period"],
            self.parameters["atr_period"]
        ) + 10  # Add some buffer
        
        if len(df) < min_required_bars:
            logger.warning(f"Insufficient data for {asset.value} {timeframe.value} to generate mean reversion signals. Need at least {min_required_bars} bars.")
            return []
        
        # Get current market regime
        regime = self.detect_regime(df, asset, timeframe)
        logger.info(f"Detected {regime.regime_type.value} regime for {asset.value} {timeframe.value} with strength {regime.strength:.2f}")
        
        # Adapt parameters to the current regime
        params = self.get_regime_adapted_parameters(regime)
        logger.info(f"Using regime-adapted parameters for {asset.value} {timeframe.value}: {params}")
        
        # Calculate indicators
        signals = []
        
        # 1. Calculate RSI
        df['rsi'] = self._calculate_rsi(df['close'], params["rsi_period"])
        
        # 2. Calculate Simple Moving Average
        df['ma'] = df['close'].rolling(window=params["ma_period"]).mean()
        df['ma_deviation'] = (df['close'] - df['ma']) / df['ma']
        
        # 3. Calculate Bollinger Bands
        df['ma'] = df['close'].rolling(window=params["bb_period"]).mean()
        df['std'] = df['close'].rolling(window=params["bb_period"]).std()
        df['upper_band'] = df['ma'] + (params["bb_std_dev"] * df['std'])
        df['lower_band'] = df['ma'] - (params["bb_std_dev"] * df['std'])
        df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # 4. Calculate ATR for stop loss
        df['atr'] = self._calculate_atr(df, params["atr_period"])
        
        # 5. Calculate volume percentile (20-day rolling)
        df['volume_pct'] = df['volume'].rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )
        
        # Generate signals for the most recent candle
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for volume threshold
        volume_confirmed = current['volume_pct'] >= params["min_volume_percentile"]
        
        # 1. Long signal (oversold conditions)
        long_signal_condition = (
            # RSI oversold and rising
            (current['rsi'] < params["rsi_oversold"] and current['rsi'] > prev['rsi']) or 
            # Price significantly below MA and mean reverting
            (current['ma_deviation'] < -params["deviation_threshold"] and current['close'] > prev['close']) or
            # Price at lower Bollinger Band and bouncing
            (current['bb_position'] < params["bb_threshold"] and current['close'] > prev['close'])
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
                    "strategy_type": "mean_reversion",
                    "rsi": round(current['rsi'], 2),
                    "ma_deviation": round(current['ma_deviation'] * 100, 2),
                    "bb_position": round(current['bb_position'], 2),
                    "atr": round(current['atr'], 5),
                    "market_regime": regime.regime_type.value,
                    "regime_strength": round(regime.strength, 2),
                    "volume_percentile": round(current['volume_pct'], 1)
                }
            )
            
            logger.info(f"Generated LONG mean reversion signal for {asset.value} {timeframe.value}")
            logger.info(f"  Entry: {entry_price:.5f}, Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
            logger.info(f"  RSI: {current['rsi']:.2f}, MA Deviation: {current['ma_deviation']*100:.2f}%, BB Position: {current['bb_position']:.2f}")
            
            signals.append(signal)
            
        # 2. Short signal (overbought conditions)
        short_signal_condition = (
            # RSI overbought and falling
            (current['rsi'] > params["rsi_overbought"] and current['rsi'] < prev['rsi']) or 
            # Price significantly above MA and mean reverting
            (current['ma_deviation'] > params["deviation_threshold"] and current['close'] < prev['close']) or
            # Price at upper Bollinger Band and dropping
            (current['bb_position'] > (1 - params["bb_threshold"]) and current['close'] < prev['close'])
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
                    "strategy_type": "mean_reversion",
                    "rsi": round(current['rsi'], 2),
                    "ma_deviation": round(current['ma_deviation'] * 100, 2),
                    "bb_position": round(current['bb_position'], 2),
                    "atr": round(current['atr'], 5),
                    "market_regime": regime.regime_type.value,
                    "regime_strength": round(regime.strength, 2),
                    "volume_percentile": round(current['volume_pct'], 1)
                }
            )
            
            logger.info(f"Generated SHORT mean reversion signal for {asset.value} {timeframe.value}")
            logger.info(f"  Entry: {entry_price:.5f}, Stop Loss: {stop_loss:.5f}, Take Profit: {take_profit:.5f}")
            logger.info(f"  RSI: {current['rsi']:.2f}, MA Deviation: {current['ma_deviation']*100:.2f}%, BB Position: {current['bb_position']:.2f}")
            
            signals.append(signal)
            
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series of RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
