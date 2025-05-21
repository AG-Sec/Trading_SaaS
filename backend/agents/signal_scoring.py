"""
Signal scoring module for the Trading SaaS platform.
Provides a comprehensive scoring system to rank trading signals.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from shared_types import TradingSignalModel, SignalType
from backend.agents.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class SignalScoring:
    """
    Class responsible for scoring and ranking trading signals based on
    multiple technical and market factors.
    """
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        
        # Scoring weights for different factors (must sum to 1.0)
        self.weights = {
            'trend_alignment': 0.20,      # Signal aligned with overall trend
            'indicator_confirmation': 0.25,  # Multiple indicators agree
            'pattern_strength': 0.15,      # Strength of chart pattern
            'risk_reward': 0.25,          # Quality of R:R ratio
            'volatility_fit': 0.15        # Appropriate for current volatility
        }
        
    def score_signal(self, signal: TradingSignalModel, df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Score a trading signal based on multiple factors.
        
        Args:
            signal: The trading signal to score
            df: DataFrame with price data and calculated indicators
            
        Returns:
            Tuple of (total_score, factor_scores)
            where total_score is 0.0-1.0 (higher is better)
            and factor_scores is a dict of individual category scores
        """
        if df.empty or len(df) < 50:
            logger.warning("Not enough data to score signal properly")
            return 0.0, {}
            
        # Ensure indicators are calculated
        if 'ema_fast' not in df.columns:
            df = self.technical_indicators.calculate_all(df)
            
        # Get latest data for scoring
        latest = df.iloc[-1]
        
        # Calculate market regime
        regime_type, regime_strength, regime_info = self.technical_indicators.market_regime(df)
        
        # Calculate individual factor scores
        factor_scores = {}
        
        # 1. Trend alignment score
        factor_scores['trend_alignment'] = self._score_trend_alignment(
            signal.signal_type, latest, regime_type, regime_info)
            
        # 2. Indicator confirmation score
        factor_scores['indicator_confirmation'] = self._score_indicator_confirmation(
            signal.signal_type, latest)
            
        # 3. Pattern strength score
        factor_scores['pattern_strength'] = self._score_pattern_strength(
            signal.signal_type, df)
            
        # 4. Risk-reward quality score
        factor_scores['risk_reward'] = self._score_risk_reward(signal)
        
        # 5. Volatility appropriateness score
        factor_scores['volatility_fit'] = self._score_volatility_fit(
            signal, latest, regime_type)
        
        # Calculate weighted average for total score
        total_score = sum(score * self.weights[factor] 
                          for factor, score in factor_scores.items())
        
        # Add score to signal metadata
        if not signal.metadata:
            signal.metadata = {}
        signal.metadata['signal_score'] = round(total_score, 2)
        signal.metadata['factor_scores'] = {k: round(v, 2) for k, v in factor_scores.items()}
        signal.metadata['market_regime'] = {
            'type': regime_type,
            'strength': round(regime_strength, 2)
        }
        
        return total_score, factor_scores
        
    def _score_trend_alignment(self, signal_type: SignalType, latest: pd.Series, 
                              regime_type: str, regime_info: Dict) -> float:
        """Score how well signal aligns with market trend"""
        score = 0.5  # Neutral starting point
        
        # Check if signal type aligns with overall trend
        trend_alignment = ((signal_type == SignalType.LONG and regime_type == 'bullish') or
                          (signal_type == SignalType.SHORT and regime_type == 'bearish'))
        
        # Boost score for trend alignment
        if trend_alignment:
            score += 0.3
        # Slightly reduce score for counter-trend
        elif ((signal_type == SignalType.LONG and regime_type == 'bearish') or
              (signal_type == SignalType.SHORT and regime_type == 'bullish')):
            score -= 0.2
            
        # Check price vs SMA200 (major trend)
        if signal_type == SignalType.LONG and regime_info['above_sma200']:
            score += 0.1
        elif signal_type == SignalType.SHORT and not regime_info['above_sma200']:
            score += 0.1
        # Going against SMA200 is risky
        elif signal_type == SignalType.LONG and not regime_info['above_sma200']:
            score -= 0.1
        elif signal_type == SignalType.SHORT and regime_info['above_sma200']:
            score -= 0.1
            
        # Check MACD histogram for momentum
        if signal_type == SignalType.LONG and regime_info['macd_histogram'] > 0:
            score += 0.1
        elif signal_type == SignalType.SHORT and regime_info['macd_histogram'] < 0:
            score += 0.1
            
        # Neutral regime is fine for both directions
        if regime_type == 'neutral':
            score += 0.1
            
        # Volatile regime is challenging for any signal
        if regime_type == 'volatile':
            score -= 0.2
            
        return max(0.0, min(1.0, score))  # Clamp to range [0, 1]
    
    def _score_indicator_confirmation(self, signal_type: SignalType, latest: pd.Series) -> float:
        """Score based on how many indicators confirm the signal direction"""
        score = 0.0
        confirmations = 0
        
        # For long signals, check bullish indicators
        if signal_type == SignalType.LONG:
            # EMA alignment
            if latest['ema_fast'] > latest['ema_slow']:
                confirmations += 1
            
            # RSI momentum
            if 40 <= latest['rsi'] <= 70:  # Not overbought, with momentum
                confirmations += 1
            
            # MACD direction
            if latest['macd'] > latest['macd_signal']:
                confirmations += 1
                
            # Stochastic momentum
            if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 80:
                confirmations += 1
                
            # Price vs VWAP
            if latest['close'] > latest.get('vwap_14', 0):
                confirmations += 1
                
            # BB position
            if latest['bb_position'] > 0:  # Upper half of bands
                confirmations += 1
        
        # For short signals, check bearish indicators
        elif signal_type == SignalType.SHORT:
            # EMA alignment
            if latest['ema_fast'] < latest['ema_slow']:
                confirmations += 1
            
            # RSI momentum
            if 30 <= latest['rsi'] <= 60:  # Not oversold, with downward room
                confirmations += 1
            
            # MACD direction
            if latest['macd'] < latest['macd_signal']:
                confirmations += 1
                
            # Stochastic momentum
            if latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] > 20:
                confirmations += 1
                
            # Price vs VWAP
            if latest['close'] < latest.get('vwap_14', float('inf')):
                confirmations += 1
                
            # BB position
            if latest['bb_position'] < 0:  # Lower half of bands
                confirmations += 1
        
        # Score based on number of confirmations (0-6)
        score = confirmations / 6.0
        return score
    
    def _score_pattern_strength(self, signal_type: SignalType, df: pd.DataFrame) -> float:
        """Score based on strength of chart patterns"""
        score = 0.5  # Neutral starting point
        
        # Get recent bars
        recent = df.iloc[-3:]
        
        # Detect patterns
        patterns = self.technical_indicators.detect_patterns(df)
        
        # For long signals, check bullish patterns in last 3 bars
        if signal_type == SignalType.LONG:
            # Check for bullish reversal patterns
            if any(patterns['hammer'].iloc[-3:] > 0):
                score += 0.15
            if any(patterns['inverted_hammer'].iloc[-3:] > 0):
                score += 0.1
            if any(patterns['engulfing_bullish'].iloc[-3:] > 0):
                score += 0.2
            if any(patterns['morning_star'].iloc[-3:] > 0):
                score += 0.25
            if any(patterns['piercing'].iloc[-3:] > 0):
                score += 0.15
                
            # Check for bearish patterns - reduce score
            if any(patterns['hanging_man'].iloc[-3:] > 0):
                score -= 0.1
            if any(patterns['shooting_star'].iloc[-3:] > 0):
                score -= 0.15
            if any(patterns['engulfing_bearish'].iloc[-3:] > 0):
                score -= 0.2
                
        # For short signals, check bearish patterns in last 3 bars 
        elif signal_type == SignalType.SHORT:
            # Check for bearish reversal patterns
            if any(patterns['hanging_man'].iloc[-3:] > 0):
                score += 0.15
            if any(patterns['shooting_star'].iloc[-3:] > 0):
                score += 0.15
            if any(patterns['engulfing_bearish'].iloc[-3:] > 0):
                score += 0.2
            if any(patterns['evening_star'].iloc[-3:] > 0):
                score += 0.25
            if any(patterns['dark_cloud_cover'].iloc[-3:] > 0):
                score += 0.15
                
            # Check for bullish patterns - reduce score
            if any(patterns['hammer'].iloc[-3:] > 0):
                score -= 0.1
            if any(patterns['inverted_hammer'].iloc[-3:] > 0):
                score -= 0.1
            if any(patterns['engulfing_bullish'].iloc[-3:] > 0):
                score -= 0.2
        
        return max(0.0, min(1.0, score))  # Clamp to range [0, 1]
    
    def _score_risk_reward(self, signal: TradingSignalModel) -> float:
        """Score based on quality of risk-reward ratio"""
        # Default to middle score if risk-reward not available
        if not signal.risk_reward_ratio:
            return 0.5
            
        rr = signal.risk_reward_ratio
        
        # Score based on risk-reward quality
        if rr >= 3.0:    # Excellent
            return 1.0
        elif rr >= 2.5:  # Very good
            return 0.9
        elif rr >= 2.0:  # Good
            return 0.8
        elif rr >= 1.8:  # Above average
            return 0.7
        elif rr >= 1.5:  # Average
            return 0.6
        elif rr >= 1.3:  # Below average
            return 0.5
        elif rr >= 1.0:  # Minimum acceptable
            return 0.3
        else:            # Poor
            return 0.0
    
    def _score_volatility_fit(self, signal: TradingSignalModel, latest: pd.Series, regime_type: str) -> float:
        """Score based on how well the signal fits current volatility conditions"""
        score = 0.5  # Neutral starting point
        
        # Get ATR percentage
        atr_pct = latest.get('atr', 0) / latest['close'] * 100 if latest['close'] > 0 else 0
        
        # Get signal stop distance percentage
        if signal.entry_price and signal.stop_loss and signal.entry_price > 0:
            stop_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        else:
            stop_distance_pct = 0
            
        # Check if stop distance is appropriate for volatility
        # Stop should be ~1-2.5x ATR for normal conditions
        if 0.8 * atr_pct <= stop_distance_pct <= 2.5 * atr_pct:
            score += 0.2
        elif stop_distance_pct < 0.5 * atr_pct:  # Stop too tight
            score -= 0.3
        elif stop_distance_pct > 3 * atr_pct:    # Stop too wide
            score -= 0.2
            
        # In volatile regimes, wider stops are okay
        if regime_type == 'volatile' and 2 * atr_pct <= stop_distance_pct <= 4 * atr_pct:
            score += 0.2
            
        # Check BB width (volatility)
        bb_width = latest.get('bb_width', 0)
        
        # For narrow bands (low volatility)
        if bb_width < 0.03:  # Very narrow
            if stop_distance_pct < 1.5 * atr_pct:  # Tighter stops good in low vol
                score += 0.2
            else:
                score -= 0.1
        
        # For wide bands (high volatility)
        elif bb_width > 0.06:  # Very wide
            if stop_distance_pct > 1.5 * atr_pct:  # Wider stops needed in high vol
                score += 0.1
            else:
                score -= 0.2
                
        return max(0.0, min(1.0, score))  # Clamp to range [0, 1]
        
    def rank_signals(self, signals: List[TradingSignalModel], df_dict: Dict[Tuple[str, str], pd.DataFrame]) -> List[TradingSignalModel]:
        """
        Score and rank a list of trading signals.
        
        Args:
            signals: List of trading signals to rank
            df_dict: Dictionary mapping (asset, timeframe) to dataframes with indicator data
            
        Returns:
            List of signals sorted by score (highest first)
        """
        scored_signals = []
        
        for signal in signals:
            # Create a key for the df_dict
            key = (signal.asset.value, signal.timeframe.value)
            
            # Skip if we don't have data for this asset/timeframe
            if key not in df_dict:
                logger.warning(f"No data available for {key}. Skipping signal scoring.")
                signal.metadata = signal.metadata or {}
                signal.metadata['signal_score'] = 0.5  # Default neutral score
                scored_signals.append(signal)
                continue
                
            # Score the signal
            score, factor_scores = self.score_signal(signal, df_dict[key])
            scored_signals.append(signal)
            
        # Sort by score (descending)
        scored_signals.sort(key=lambda s: s.metadata.get('signal_score', 0), reverse=True)
        
        return scored_signals
