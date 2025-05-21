"""
Market Regime Detector module for the Trading SaaS platform.
Analyzes market conditions and adjusts strategy parameters accordingly.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
from datetime import datetime, timezone
import json

from shared_types import AssetSymbol, Timeframe
from backend.agents.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class MarketRegimeType(str, Enum):
    """Enum for market regime types."""
    BULLISH_TRENDING = "bullish_trending"
    BEARISH_TRENDING = "bearish_trending"
    NEUTRAL_RANGING = "neutral_ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKOUT = "bearish_breakout"
    UNKNOWN = "unknown"

class MarketRegime:
    """Class to hold market regime information"""
    
    def __init__(
        self,
        regime_type: MarketRegimeType,
        strength: float,
        timestamp: datetime,
        asset: AssetSymbol,
        timeframe: Timeframe,
        metrics: Dict[str, Any] = None
    ):
        self.regime_type = regime_type
        self.strength = strength  # 0.0 to 1.0
        self.timestamp = timestamp
        self.asset = asset
        self.timeframe = timeframe
        self.metrics = metrics or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'regime_type': self.regime_type.value,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset.value,
            'timeframe': self.timeframe.value,
            'metrics': self.metrics
        }
        
    def get_adjusted_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters adjusted for this market regime.
        
        Returns:
            Dictionary of parameter adjustments for the current regime
        """
        params = {}
        
        # Default parameters (updated to match SignalScannerAgent changes)
        params['breakout_period'] = 10  # Base parameter from SignalScannerAgent
        params['atr_multiplier_sl'] = 1.5  # Base parameter from SignalScannerAgent
        params['min_rr_ratio'] = 1.2  # Base parameter from RiskManagerAgent
        params['volume_threshold'] = 1.1  # Base parameter from SignalScannerAgent
        params['allow_pullback_entry'] = True  # Base parameter from SignalScannerAgent
        
        # Adjust based on regime type (less aggressively)
        if self.regime_type == MarketRegimeType.BULLISH_TRENDING:
            # In bullish trends, we can be more aggressive
            params['breakout_period'] = max(8, int(params['breakout_period'] * 0.9))  # Shorter breakout period
            params['atr_multiplier_sl'] = max(1.2, params['atr_multiplier_sl'] * 0.95)  # Slightly tighter stops
            params['min_rr_ratio'] = max(1.0, params['min_rr_ratio'] * 0.95)  # Accept slightly lower R:R
            
        elif self.regime_type == MarketRegimeType.BEARISH_TRENDING:
            # In bearish trends, less cautious adjustment
            params['breakout_period'] = min(12, int(params['breakout_period'] * 1.1))  # Slightly longer for confirmation
            params['atr_multiplier_sl'] = min(1.8, params['atr_multiplier_sl'] * 1.05)  # Slightly wider stops
            params['volume_threshold'] = min(1.2, params['volume_threshold'] * 1.1)  # Slightly stronger volume
            
        elif self.regime_type == MarketRegimeType.NEUTRAL_RANGING:
            # In ranging markets, focus more on pullbacks and range boundaries
            params['allow_pullback_entry'] = True
            # Keep breakout period nearly the same
            params['breakout_period'] = min(11, int(params['breakout_period'] * 1.05))
            
        elif self.regime_type == MarketRegimeType.HIGH_VOLATILITY:
            # In high volatility, use wider stops but with milder adjustments
            params['atr_multiplier_sl'] = min(1.8, params['atr_multiplier_sl'] * 1.1)  # Wider stops but not as extreme
            params['min_rr_ratio'] = min(1.5, params['min_rr_ratio'] * 1.1)  # Better R:R but more reasonable
            params['volume_threshold'] = min(1.3, params['volume_threshold'] * 1.2)  # Stronger volume but not as strict
            
        elif self.regime_type == MarketRegimeType.LOW_VOLATILITY:
            # In low volatility, slightly tighter stops and more sensitivity
            params['atr_multiplier_sl'] = max(1.2, params['atr_multiplier_sl'] * 0.9)  # Slightly tighter stops
            params['min_rr_ratio'] = max(1.0, params['min_rr_ratio'] * 0.95)  # Accept slightly lower R:R
            params['breakout_period'] = max(8, int(params['breakout_period'] * 0.8))  # More sensitive
            
        elif self.regime_type == MarketRegimeType.BULLISH_BREAKOUT:
            # During breakouts, be aggressive with milder volume requirements
            params['breakout_period'] = max(8, int(params['breakout_period'] * 0.7))  # More sensitive
            params['volume_threshold'] = min(1.2, params['volume_threshold'] * 1.1)  # Reduced volume requirement
            
        elif self.regime_type == MarketRegimeType.BEARISH_BREAKOUT:
            # Similar to bullish breakout but for downside
            params['breakout_period'] = max(8, int(params['breakout_period'] * 0.7))  # More sensitive
            params['volume_threshold'] = min(1.2, params['volume_threshold'] * 1.1)  # Reduced volume requirement
            
        # Scale parameters based on regime strength (0.0-1.0)
        # The stronger the regime, the more we adjust, but with reduced impact
        for key in params:
            # Only scale numeric parameters
            if isinstance(params[key], (int, float)) and key != 'breakout_period':
                # Calculate adjustment factor based on strength, but with reduced impact
                base_value = params[key]
                
                # Midpoint is 0.5 strength (no adjustment)
                if self.strength > 0.5:
                    # Scale up adjustments for strong regimes (reduced from 0.5 to 0.3)
                    scale_factor = 1.0 + (self.strength - 0.5) * 0.3  # Max +15% at strength=1.0
                    params[key] = base_value * scale_factor
                elif self.strength < 0.5:
                    # Scale down adjustments for weak regimes (reduced from 0.4 to 0.2)
                    scale_factor = 1.0 - (0.5 - self.strength) * 0.2  # Max -10% at strength=0.0
                    params[key] = base_value * scale_factor
                    
            # Handle integer parameters separately
            elif key == 'breakout_period':
                # Ensure breakout_period stays an integer
                params[key] = int(params[key])
                
        return params


class MarketRegimeDetector:
    """
    Class responsible for detecting market regimes and adjusting strategy parameters.
    """
    
    def __init__(self):
        self.tech_indicators = TechnicalIndicators()
        
        # Required lookback periods for reliable regime detection
        self.min_data_points = 50
        
        # Tracking of regime history
        self.regime_history: Dict[Tuple[str, str], List[MarketRegime]] = {}
        
    def detect_regime(
        self, 
        df: pd.DataFrame,
        asset: AssetSymbol,
        timeframe: Timeframe
    ) -> MarketRegime:
        """
        Detect current market regime based on technical indicators.
        
        Args:
            df: DataFrame with price data
            asset: Asset symbol
            timeframe: Timeframe
            
        Returns:
            MarketRegime object with detected regime information
        """
        if len(df) < self.min_data_points:
            logger.warning(f"Not enough data points for regime detection. Need {self.min_data_points}, got {len(df)}")
            return MarketRegime(
                regime_type=MarketRegimeType.UNKNOWN,
                strength=0.5,
                timestamp=df.index[-1] if not df.empty else datetime.now(timezone.utc),
                asset=asset,
                timeframe=timeframe
            )
            
        # Ensure all indicators are calculated
        if 'ema_fast' not in df.columns:
            df = self.tech_indicators.calculate_all(df)
            
        # Get latest data
        current = df.iloc[-1]
        
        # Key metrics for regime detection
        metrics = {}
        
        # 1. Trend metrics
        metrics['close'] = current['close']
        metrics['sma200'] = current['sma_200']
        metrics['above_sma200'] = current['close'] > current['sma_200']
        metrics['ema_fast'] = current['ema_fast']
        metrics['ema_slow'] = current['ema_slow']
        metrics['ema_trend'] = current['ema_fast'] > current['ema_slow']
        
        # Recent crosses (last 10 bars)
        recent = df.iloc[-10:]
        metrics['recent_ema_cross_up'] = any(recent['ema_cross'] == 1)
        metrics['recent_ema_cross_down'] = any(recent['ema_cross'] == -1)
        metrics['recent_macd_cross_up'] = any(recent['macd_cross'] == 1)
        metrics['recent_macd_cross_down'] = any(recent['macd_cross'] == -1)
        
        # 2. Volatility metrics
        metrics['atr'] = current['atr']
        metrics['atr_pct'] = current['atr'] / current['close'] * 100 if current['close'] > 0 else 0
        metrics['bb_width'] = (current['bb_upper'] - current['bb_lower']) / current['bb_middle'] if current['bb_middle'] > 0 else 0
        
        # Average volatility (last 20 bars)
        volatility_window = df.iloc[-20:]
        metrics['avg_atr_pct'] = (volatility_window['atr'] / volatility_window['close'] * 100).mean()
        metrics['volatility_change'] = metrics['atr_pct'] / metrics['avg_atr_pct'] if metrics['avg_atr_pct'] > 0 else 1.0
        
        # 3. Momentum metrics
        metrics['rsi'] = current['rsi']
        metrics['macd'] = current['macd']
        metrics['macd_signal'] = current['macd_signal']
        metrics['macd_hist'] = current['macd_hist']
        metrics['stoch_k'] = current['stoch_k']
        metrics['stoch_d'] = current['stoch_d']
        
        # 4. Volume metrics
        avg_volume = df['volume'].iloc[-20:-1].mean()  # Exclude current volume
        metrics['volume'] = current['volume']
        metrics['volume_ratio'] = current['volume'] / avg_volume if avg_volume > 0 else 1.0
        
        # Regime scoring
        regime_scores = {
            MarketRegimeType.BULLISH_TRENDING: 0,
            MarketRegimeType.BEARISH_TRENDING: 0,
            MarketRegimeType.NEUTRAL_RANGING: 0,
            MarketRegimeType.HIGH_VOLATILITY: 0,
            MarketRegimeType.LOW_VOLATILITY: 0,
            MarketRegimeType.BULLISH_BREAKOUT: 0,
            MarketRegimeType.BEARISH_BREAKOUT: 0
        }
        
        # Bullish trending signals
        if metrics.get('ema_trend', False) and metrics.get('above_sma200', False):
            regime_scores[MarketRegimeType.BULLISH_TRENDING] += 1
        if metrics.get('rsi', 50) > 50:
            regime_scores[MarketRegimeType.BULLISH_TRENDING] += 0.51
        if metrics.get('macd', 0) > 0:
            regime_scores[MarketRegimeType.BULLISH_TRENDING] += 1
        if metrics.get('macd', 0) > metrics.get('macd_signal', 0):
            regime_scores[MarketRegimeType.BULLISH_TRENDING] += 1
        if 40 < metrics.get('rsi', 50) < 70:
            regime_scores[MarketRegimeType.BULLISH_TRENDING] += 1
        if metrics.get('stoch_k', 0) > metrics.get('stoch_d', 0) and metrics.get('stoch_k', 0) < 80:
            regime_scores[MarketRegimeType.BULLISH_TRENDING] += 1
            
        # Bearish trending signals
        if not metrics.get('ema_trend', True) and not metrics.get('above_sma200', True):
            regime_scores[MarketRegimeType.BEARISH_TRENDING] += 1
        if metrics.get('rsi', 50) < 50:
            regime_scores[MarketRegimeType.BEARISH_TRENDING] += 0.51
        if metrics.get('macd', 0) < 0:
            regime_scores[MarketRegimeType.BEARISH_TRENDING] += 1
        if metrics.get('macd', 0) < metrics.get('macd_signal', 0):
            regime_scores[MarketRegimeType.BEARISH_TRENDING] += 1
        if 30 < metrics.get('rsi', 50) < 60:
            regime_scores[MarketRegimeType.BEARISH_TRENDING] += 1
        if metrics.get('stoch_k', 0) < metrics.get('stoch_d', 0) and metrics.get('stoch_k', 0) > 20:
            regime_scores[MarketRegimeType.BEARISH_TRENDING] += 1
            
        # Neutral ranging signals
        if 0.8 < (metrics.get('ema_fast', 0) / metrics.get('ema_slow', 0)) < 1.2:  # EMAs close together
            regime_scores[MarketRegimeType.NEUTRAL_RANGING] += 2
        rsi_value = metrics.get('rsi', 50)
        if 40 < rsi_value < 60:  # RSI in neutral zone
            regime_scores[MarketRegimeType.NEUTRAL_RANGING] += 1
        if abs(metrics.get('bb_position', 0)) < 0.5:  # Price near BB middle
            regime_scores[MarketRegimeType.NEUTRAL_RANGING] += 1
        if metrics.get('bb_width', 0) < 0.03:  # Narrow Bollinger Bands
            regime_scores[MarketRegimeType.NEUTRAL_RANGING] += 2
            
        # High volatility signals
        if metrics.get('atr_pct', 0) > 0.03:  # ATR > 3% of price
            regime_scores[MarketRegimeType.HIGH_VOLATILITY] += 1
        if metrics.get('bb_width', 0) > 0.06:  # Wide BBs
            regime_scores[MarketRegimeType.HIGH_VOLATILITY] += 1
        if metrics.get('volatility_change', 0) > 1.5:  # Increasing volatility
            regime_scores[MarketRegimeType.HIGH_VOLATILITY] += 1
        if abs(metrics.get('rsi', 50) - 50) > 20:  # Extreme RSI
            regime_scores[MarketRegimeType.HIGH_VOLATILITY] += 1
            
        # Low volatility signals
        if metrics.get('atr_pct', 0.02) < 0.015:  # ATR < 1.5% of price
            regime_scores[MarketRegimeType.LOW_VOLATILITY] += 1
        if metrics.get('bb_width', 0.04) < 0.025:  # Narrow BBs
            regime_scores[MarketRegimeType.LOW_VOLATILITY] += 1
        if metrics.get('volatility_change', 0) < 0.7:  # Decreasing volatility
            regime_scores[MarketRegimeType.LOW_VOLATILITY] += 1
        if 40 < metrics.get('rsi', 50) < 60:  # Neutral RSI
            regime_scores[MarketRegimeType.LOW_VOLATILITY] += 1
            
        # Breakout signals
        
        # Bullish breakout
        if metrics.get('recent_ema_cross_up', False) or metrics.get('recent_macd_cross_up', False):
            regime_scores[MarketRegimeType.BULLISH_BREAKOUT] += 1
        if metrics.get('volume_ratio', 0) > 1.5:  # High volume
            regime_scores[MarketRegimeType.BULLISH_BREAKOUT] += 1
        if metrics.get('bb_position', 0) > 0.8:  # Near upper BB
            regime_scores[MarketRegimeType.BULLISH_BREAKOUT] += 1
        if metrics.get('ema_trend', False) and metrics.get('above_sma200', False) and metrics.get('macd_hist', 0) > 0:
            regime_scores[MarketRegimeType.BULLISH_BREAKOUT] += 1
            
        # Bearish breakout
        if metrics.get('recent_ema_cross_down', False) or metrics.get('recent_macd_cross_down', False):
            regime_scores[MarketRegimeType.BEARISH_BREAKOUT] += 1
        if metrics.get('volume_ratio', 0) > 1.5:  # High volume
            regime_scores[MarketRegimeType.BEARISH_BREAKOUT] += 1
        if metrics.get('bb_position', 0) < -0.8:  # Near lower BB
            regime_scores[MarketRegimeType.BEARISH_BREAKOUT] += 1
        if not metrics.get('ema_trend', False) and not metrics.get('above_sma200', False) and metrics.get('macd_hist', 0) < 0:
            regime_scores[MarketRegimeType.BEARISH_BREAKOUT] += 1
            
        # Determine primary regime type
        max_score = max(regime_scores.values())
        primary_regimes = [regime for regime, score in regime_scores.items() if score == max_score]
        
        # If multiple regimes have the same score, prioritize based on a hierarchy
        if len(primary_regimes) > 1:
            # Prioritize: High Vol > Breakouts > Trending > Low Vol > Neutral
            for regime_type in [
                MarketRegimeType.HIGH_VOLATILITY,
                MarketRegimeType.BULLISH_BREAKOUT,
                MarketRegimeType.BEARISH_BREAKOUT,
                MarketRegimeType.BULLISH_TRENDING,
                MarketRegimeType.BEARISH_TRENDING,
                MarketRegimeType.LOW_VOLATILITY,
                MarketRegimeType.NEUTRAL_RANGING
            ]:
                if regime_type in primary_regimes:
                    primary_regime = regime_type
                    break
        else:
            primary_regime = primary_regimes[0]
            
        # Calculate regime strength (0.0-1.0)
        # Max possible score depends on regime type
        max_possible = {
            MarketRegimeType.BULLISH_TRENDING: 6,
            MarketRegimeType.BEARISH_TRENDING: 6,
            MarketRegimeType.NEUTRAL_RANGING: 6,
            MarketRegimeType.HIGH_VOLATILITY: 6,
            MarketRegimeType.LOW_VOLATILITY: 6,
            MarketRegimeType.BULLISH_BREAKOUT: 4,
            MarketRegimeType.BEARISH_BREAKOUT: 4
        }
        
        regime_strength = regime_scores[primary_regime] / max_possible[primary_regime]
        
        # Create market regime object
        regime = MarketRegime(
            regime_type=primary_regime,
            strength=regime_strength,
            timestamp=df.index[-1],
            asset=asset,
            timeframe=timeframe,
            metrics=metrics
        )
        
        # Store in history
        key = (asset.value, timeframe.value)
        if key not in self.regime_history:
            self.regime_history[key] = []
        self.regime_history[key].append(regime)
        
        # Keep history at a reasonable size
        if len(self.regime_history[key]) > 100:
            self.regime_history[key] = self.regime_history[key][-100:]
            
        logger.info(f"Detected {primary_regime.value} regime for {asset.value} {timeframe.value} "
                   f"with strength {regime_strength:.2f}")
        
        return regime
        
    def get_regime_history(self, asset: AssetSymbol, timeframe: Timeframe) -> List[MarketRegime]:
        """Get historical regimes for an asset and timeframe"""
        key = (asset.value, timeframe.value)
        return self.regime_history.get(key, [])
        
    def get_adjusted_parameters(self, df: pd.DataFrame, asset: AssetSymbol, timeframe: Timeframe) -> Dict[str, Any]:
        """
        Get strategy parameters adjusted for current market regime.
        
        Args:
            df: DataFrame with price data
            asset: Asset symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary of parameter adjustments for the current regime
        """
        regime = self.detect_regime(df, asset, timeframe)
        return regime.get_adjusted_parameters()
        
    def get_summary(self, asset: AssetSymbol = None, timeframe: Timeframe = None) -> Dict[str, Any]:
        """
        Get a summary of current market regimes across assets and timeframes.
        
        Args:
            asset: Optional filter for specific asset
            timeframe: Optional filter for specific timeframe
            
        Returns:
            Dictionary with regime summaries
        """
        summary = {}
        
        for key, regimes in self.regime_history.items():
            if not regimes:
                continue
                
            asset_val, tf_val = key
            
            # Apply filters if specified
            if asset and asset.value != asset_val:
                continue
            if timeframe and timeframe.value != tf_val:
                continue
                
            # Get most recent regime
            latest_regime = regimes[-1]
            
            # Add to summary
            if asset_val not in summary:
                summary[asset_val] = {}
                
            summary[asset_val][tf_val] = {
                'regime': latest_regime.regime_type.value,
                'strength': latest_regime.strength,
                'timestamp': latest_regime.timestamp.isoformat(),
                'parameters': latest_regime.get_adjusted_parameters()
            }
            
        return summary
