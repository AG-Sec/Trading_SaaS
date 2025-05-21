"""
Adaptive Engine for Dynamic Trading Strategy Optimization

This module contains the core adaptive engine that dynamically selects and optimizes
trading strategies based on market conditions, asset characteristics, and historical performance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from enum import Enum

from shared_types.models import (
    AssetSymbol, Timeframe, TradingSignalModel, SignalType,
    TradeModel, TradeStatus
)
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.market_regime_detector import MarketRegimeDetector, MarketRegimeType
from backend.agents.regime_optimizer import RegimeOptimizer

# Import strategy implementations
from backend.strategies.mean_reversion_strategy import MeanReversionStrategy
from backend.strategies.trend_following_strategy import TrendFollowingStrategy

logger = logging.getLogger(__name__)

class StrategyType(str, Enum):
    """Types of trading strategies supported by the adaptive engine"""
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_EXPANSION = "volatility_expansion"
    RANGE_TRADING = "range_trading"


class AdaptiveEngine:
    """
    Core engine that dynamically adapts trading strategies based on market conditions.
    
    The Adaptive Engine:
    1. Analyzes current market conditions using regime detection
    2. Selects the optimal timeframe based on asset volatility
    3. Chooses the best-performing strategy for the current regime
    4. Dynamically tunes parameters for optimal performance
    5. Adjusts risk parameters based on market conditions
    6. Scores and ranks signals to prioritize the highest probability trades
    """
    
    def __init__(
        self,
        market_data_agent: MarketDataAgent,
        signal_scanner_agent: SignalScannerAgent,
        risk_manager_agent: RiskManagerAgent,
        regime_detector: Optional[MarketRegimeDetector] = None,
        regime_optimizer: Optional[RegimeOptimizer] = None,
        backtest_lookback_days: int = 90,
        min_strategy_confidence: float = 0.7,
        enable_auto_optimization: bool = True,
        optimization_frequency_hours: int = 24,
        multi_timeframe_confirmation: bool = True,
        minimum_alert_score: float = 0.7,
    ):
        """
        Initialize the Adaptive Engine
        
        Args:
            market_data_agent: For fetching market data
            signal_scanner_agent: For generating trading signals
            risk_manager_agent: For managing risk parameters
            regime_detector: For detecting market regimes
            regime_optimizer: For optimizing regime detection parameters
            backtest_lookback_days: Days to look back for strategy performance
            min_strategy_confidence: Minimum confidence score (0-1) for a strategy to be used
            enable_auto_optimization: Whether to automatically optimize strategies
            optimization_frequency_hours: How often to run optimization (in hours)
        """
        self.market_data_agent = market_data_agent
        self.signal_scanner_agent = signal_scanner_agent
        self.risk_manager_agent = risk_manager_agent
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.regime_optimizer = regime_optimizer
        
        # Configuration
        self.backtest_lookback_days = backtest_lookback_days
        self.min_strategy_confidence = min_strategy_confidence
        self.enable_auto_optimization = enable_auto_optimization
        self.optimization_frequency_hours = optimization_frequency_hours
        self.multi_timeframe_confirmation = multi_timeframe_confirmation
        self.minimum_alert_score = minimum_alert_score
        
        # Strategy performance tracking
        self.strategy_performance: Dict[Tuple[AssetSymbol, Timeframe, StrategyType, MarketRegimeType], Dict] = {}
        
        # Optimization tracking
        self.last_optimization_time: Dict[AssetSymbol, datetime] = {}
        
        # Expanded asset support - initialize with default timeframes
        self._initialize_asset_preferences()
        
        # Initialize strategy profiles
        self._initialize_strategy_profiles()
        
        # Add expanded asset support for major crypto and forex pairs
        self._expand_asset_support()
        
        # Initialize strategy implementations
        self.mean_reversion_strategy = MeanReversionStrategy()
        self.trend_following_strategy = TrendFollowingStrategy()
        
        logger.info("Adaptive Engine initialized with support for multiple assets and strategies")
    
    def _initialize_asset_preferences(self):
        """Initialize default preferences for each asset type"""
        self.asset_preferences = {
            # Cryptocurrencies
            AssetSymbol.BTC_USD: {
                "preferred_timeframes": [Timeframe.HOUR_4, Timeframe.DAY_1, Timeframe.HOUR_1],
                "volatility_factor": 1.2,  # Higher volatility adjustment
                "volume_threshold": 1.5,
                "default_strategies": [StrategyType.BREAKOUT, StrategyType.TREND_FOLLOWING]
            },
            AssetSymbol.ETH_USD: {
                "preferred_timeframes": [Timeframe.HOUR_4, Timeframe.DAY_1, Timeframe.HOUR_1],
                "volatility_factor": 1.3,  # Higher volatility adjustment
                "volume_threshold": 1.4,
                "default_strategies": [StrategyType.BREAKOUT, StrategyType.TREND_FOLLOWING]
            },
            # Forex
            AssetSymbol.EUR_USD_FX: {
                "preferred_timeframes": [Timeframe.HOUR_1, Timeframe.HOUR_4, Timeframe.DAY_1],
                "volatility_factor": 0.8,  # Lower volatility adjustment
                "volume_threshold": 1.2,
                "default_strategies": [StrategyType.RANGE_TRADING, StrategyType.MEAN_REVERSION]
            },
            # Equities
            AssetSymbol.SPY: {
                "preferred_timeframes": [Timeframe.DAY_1, Timeframe.HOUR_4],
                "volatility_factor": 0.9,  # Lower volatility adjustment
                "volume_threshold": 1.3,
                "default_strategies": [StrategyType.TREND_FOLLOWING, StrategyType.PULLBACK]
            }
        }
    
    def _initialize_strategy_profiles(self):
        """Initialize strategy profiles with default parameters for each regime"""
        self.strategy_profiles = {
            StrategyType.BREAKOUT: {
                MarketRegimeType.BULLISH_TRENDING: {
                    "breakout_period": 15,
                    "atr_multiplier_sl": 1.2,
                    "min_rr_ratio": 1.8,
                    "volume_threshold": 1.5,
                    "allow_pullback_entry": True,
                    "confidence_boost": 0.2  # Boost for this regime
                },
                MarketRegimeType.BEARISH_TRENDING: {
                    "breakout_period": 10,
                    "atr_multiplier_sl": 1.5,
                    "min_rr_ratio": 1.5,
                    "volume_threshold": 1.4,
                    "allow_pullback_entry": True,
                    "confidence_boost": 0.1
                },
                MarketRegimeType.HIGH_VOLATILITY: {
                    "breakout_period": 8,
                    "atr_multiplier_sl": 2.0,
                    "min_rr_ratio": 2.0,
                    "volume_threshold": 1.6,
                    "allow_pullback_entry": False,
                    "confidence_boost": 0.3  # Best in high volatility
                },
                MarketRegimeType.LOW_VOLATILITY: {
                    "breakout_period": 20,
                    "atr_multiplier_sl": 1.0,
                    "min_rr_ratio": 1.2,
                    "volume_threshold": 1.3,
                    "allow_pullback_entry": True,
                    "confidence_boost": -0.1  # Penalize in low volatility
                },
                MarketRegimeType.NEUTRAL_RANGING: {
                    "breakout_period": 12,
                    "atr_multiplier_sl": 1.8,
                    "min_rr_ratio": 1.5,
                    "volume_threshold": 1.5,
                    "allow_pullback_entry": False,
                    "confidence_boost": -0.2  # Penalize in ranging markets (false breakouts)
                }
            },
            StrategyType.TREND_FOLLOWING: {
                # Similar structure for other strategies...
                MarketRegimeType.BULLISH_TRENDING: {
                    "ema_fast": 8,
                    "ema_slow": 21,
                    "atr_multiplier_sl": 1.5,
                    "min_rr_ratio": 1.5,
                    "confidence_boost": 0.3  # Best in trends
                },
                # Other regimes...
            },
            # Other strategies...
        }
        
        # Add remaining strategies with similar structure
        self._initialize_remaining_strategies()
    
    def _initialize_remaining_strategies(self):
        """Initialize the remaining strategy profiles"""
        # Mean reversion strategy - works best in ranging markets
        self.strategy_profiles[StrategyType.MEAN_REVERSION] = {
            MarketRegimeType.NEUTRAL_RANGING: {
                "rsi_period": 14,
                "overbought_threshold": 75,
                "oversold_threshold": 25,
                "atr_multiplier_sl": 1.2,
                "min_rr_ratio": 1.5,
                "confidence_boost": 0.3  # Best in ranging markets
            },
            MarketRegimeType.HIGH_VOLATILITY: {
                "rsi_period": 10,
                "overbought_threshold": 80,
                "oversold_threshold": 20,
                "atr_multiplier_sl": 2.0,
                "min_rr_ratio": 1.8,
                "confidence_boost": -0.1  # Not ideal in high volatility
            },
            MarketRegimeType.BULLISH_TRENDING: {
                "rsi_period": 14,
                "overbought_threshold": 70,
                "oversold_threshold": 35, # Higher oversold threshold in uptrend
                "atr_multiplier_sl": 1.5,
                "min_rr_ratio": 1.3,
                "confidence_boost": -0.2 # Not ideal in strong trends
            },
            MarketRegimeType.BEARISH_TRENDING: {
                "rsi_period": 14,
                "overbought_threshold": 65, # Lower overbought threshold in downtrend
                "oversold_threshold": 30,
                "atr_multiplier_sl": 1.5,
                "min_rr_ratio": 1.3,
                "confidence_boost": -0.2 # Not ideal in strong trends
            },
            MarketRegimeType.LOW_VOLATILITY: {
                "rsi_period": 18, # Longer period in low volatility
                "overbought_threshold": 70,
                "oversold_threshold": 30,
                "atr_multiplier_sl": 1.0,
                "min_rr_ratio": 1.5,
                "confidence_boost": 0.1
            }
        }
        
        # Range trading strategy - works best in sideways markets
        self.strategy_profiles[StrategyType.RANGE_TRADING] = {
            MarketRegimeType.NEUTRAL_RANGING: {
                "lookback_period": 20,
                "support_resistance_touches": 2,
                "range_filter_percent": 5.0,
                "atr_multiplier_sl": 1.0,
                "min_rr_ratio": 1.5,
                "confidence_boost": 0.4  # Best in ranging markets
            },
            MarketRegimeType.LOW_VOLATILITY: {
                "lookback_period": 30,
                "support_resistance_touches": 2,
                "range_filter_percent": 3.0,
                "atr_multiplier_sl": 0.8,
                "min_rr_ratio": 1.6,
                "confidence_boost": 0.3  # Good in low volatility
            },
            # Add other regime settings...
        }
        
        # Pullback strategy - works in trending markets after a retracement
        self.strategy_profiles[StrategyType.PULLBACK] = {
            MarketRegimeType.BULLISH_TRENDING: {
                "trend_ema_period": 50,
                "pullback_minimum": 3.0, # Percent pullback
                "pullback_maximum": 8.0,
                "support_level_lookback": 20,
                "atr_multiplier_sl": 1.2,
                "min_rr_ratio": 1.8,
                "confidence_boost": 0.3  # Strong in bullish trends
            },
            MarketRegimeType.BEARISH_TRENDING: {
                "trend_ema_period": 50,
                "pullback_minimum": 3.0,
                "pullback_maximum": 8.0,
                "resistance_level_lookback": 20,
                "atr_multiplier_sl": 1.2,
                "min_rr_ratio": 1.7,
                "confidence_boost": 0.3  # Strong in bearish trends
            },
            # Add other regime settings...
        }
        
        # Volatility expansion strategy - best during volatility breakouts
        self.strategy_profiles[StrategyType.VOLATILITY_EXPANSION] = {
            MarketRegimeType.HIGH_VOLATILITY: {
                "bollinger_period": 20,
                "bollinger_std": 2.0,
                "min_band_expansion_percent": 15,
                "atr_multiplier_sl": 1.8,
                "min_rr_ratio": 2.0,
                "confidence_boost": 0.4  # Best in volatility expansions
            },
            # Add other regime settings...
        }
    
    def _expand_asset_support(self):
        """
        Expand asset support to include major crypto and forex pairs beyond the initial set
        """
        # Additional cryptocurrencies with default settings
        crypto_additions = {
            "XRP-USD": "XRP",
            "ADA-USD": "Cardano",
            "SOL-USD": "Solana",
            "DOT-USD": "Polkadot",
            "DOGE-USD": "Dogecoin",
            "AVAX-USD": "Avalanche",
            "LINK-USD": "Chainlink",
            "MATIC-USD": "Polygon",
            "LTC-USD": "Litecoin",
            "UNI-USD": "Uniswap"
        }
        
        # Additional forex pairs with default settings
        forex_additions = {
            "GBPUSD=X": "GBP/USD",
            "USDJPY=X": "USD/JPY",
            "AUDUSD=X": "AUD/USD",
            "USDCAD=X": "USD/CAD",
            "NZDUSD=X": "NZD/USD",
            "USDCHF=X": "USD/CHF",
            "EURGBP=X": "EUR/GBP",
            "EURJPY=X": "EUR/JPY"
        }
        
        # Additional equities
        equity_additions = {
            "QQQ": "Nasdaq ETF",
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "GOOGL": "Google",
            "META": "Meta",
            "NVDA": "NVIDIA",
            "TSLA": "Tesla"
        }
        
        # Additional commodities
        commodity_additions = {
            "GC=F": "Gold",
            "SI=F": "Silver",
            "CL=F": "Crude Oil",
            "NG=F": "Natural Gas"
        }
        
        logger.info(f"Expanding asset support to include {len(crypto_additions) + len(forex_additions) + len(equity_additions) + len(commodity_additions)} additional assets")
        
        # TODO: Implement dynamic AssetSymbol enum extension
        # This would involve extending the enum at runtime or using a registration pattern
        # For now, we'll just log the expanded assets that would be supported
        
        # In a production implementation, we would register these with our data sources
        # and ensure the system can handle them correctly

    def select_optimal_timeframe(self, asset: AssetSymbol) -> List[Timeframe]:
        """
        Select the optimal timeframe for an asset based on its volatility and characteristics.
        
        Returns a prioritized list of timeframes from most to least optimal for the current market.
        """
        # Start with the preferred timeframes for this asset
        if asset in self.asset_preferences:
            preferred_timeframes = self.asset_preferences[asset]["preferred_timeframes"].copy()
        else:
            # Default timeframes if asset preferences not explicitly defined
            preferred_timeframes = [Timeframe.HOUR_4, Timeframe.DAY_1, Timeframe.HOUR_1]
        
        # Get recent market data to analyze volatility
        try:
            # Get data for multiple timeframes to assess volatility
            # Calculate the start date as 30 days ago
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            daily_data = self.market_data_agent.fetch_historical_data(
                asset, 
                Timeframe.DAY_1, 
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate recent volatility using ATR as percentage of price
            if daily_data and len(daily_data.candles) > 14:
                # Convert to pandas DataFrame for easier manipulation
                df = pd.DataFrame([
                    {
                        "timestamp": candle.timestamp,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume
                    } for candle in daily_data.candles
                ])
                
                # Calculate ATR(14) as percentage of price
                high_s = df['high']
                low_s = df['low']
                close_s = df['close']
                
                # Use pandas implementation or talib if available
                tr1 = high_s - low_s
                tr2 = abs(high_s - close_s.shift())
                tr3 = abs(low_s - close_s.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                avg_price = close_s.iloc[-1]
                atr_pct = (atr / avg_price) * 100
                
                # Adjust timeframe preference based on volatility
                if atr_pct > 5.0:  # Very high volatility
                    # Prefer shorter timeframes for high volatility
                    if Timeframe.HOUR_1 in preferred_timeframes:
                        preferred_timeframes.remove(Timeframe.HOUR_1)
                        preferred_timeframes.insert(0, Timeframe.HOUR_1)
                    if Timeframe.HOUR_4 in preferred_timeframes:
                        preferred_timeframes.remove(Timeframe.HOUR_4)
                        preferred_timeframes.insert(1, Timeframe.HOUR_4)
                elif atr_pct < 1.0:  # Very low volatility
                    # Prefer longer timeframes for low volatility
                    if Timeframe.DAY_1 in preferred_timeframes:
                        preferred_timeframes.remove(Timeframe.DAY_1)
                        preferred_timeframes.insert(0, Timeframe.DAY_1)
                
                logger.info(f"Selected optimal timeframes for {asset.value}: {[tf.value for tf in preferred_timeframes]} (ATR%: {atr_pct:.2f}%)")
            
        except Exception as e:
            logger.warning(f"Error determining optimal timeframe for {asset.value}: {str(e)}")
        
        return preferred_timeframes

    def select_best_strategy(self, asset: AssetSymbol, timeframe: Timeframe) -> StrategyType:
        """
        Select the best strategy for an asset and timeframe based on current market conditions.
        """
        # Get market data
        market_data = self.market_data_agent.fetch_historical_data(asset, timeframe)
        
        if not market_data or len(market_data.candles) < 50:
            logger.warning(f"Insufficient data for {asset.value} {timeframe.value} to select strategy")
            # Return default strategy for this asset if available
            if asset in self.asset_preferences and "default_strategies" in self.asset_preferences[asset]:
                return self.asset_preferences[asset]["default_strategies"][0]
            return StrategyType.BREAKOUT  # Default fallback
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume
            } for candle in market_data.candles
        ])
        
        # Detect current market regime
        regime = self.regime_detector.detect_regime(df, asset, timeframe)
        regime_type = regime.regime_type
        logger.info(f"Detected {regime_type.value} regime for {asset.value} {timeframe.value} with strength {regime.strength:.2f}")
        
        # Score each strategy based on current regime and historical performance
        strategy_scores = {}
        for strategy_type in StrategyType:
            # Base score from regime-strategy match
            base_score = 0.5  # Neutral starting point
            
            # Add confidence boost from strategy profile if available
            if strategy_type in self.strategy_profiles and regime_type in self.strategy_profiles[strategy_type]:
                profile = self.strategy_profiles[strategy_type][regime_type]
                if "confidence_boost" in profile:
                    base_score += profile["confidence_boost"]
            
            # Add historical performance score if available
            perf_key = (asset, timeframe, strategy_type, regime_type)
            if perf_key in self.strategy_performance:
                win_rate = self.strategy_performance[perf_key].get("win_rate", 0.5)
                profit_factor = self.strategy_performance[perf_key].get("profit_factor", 1.0)
                # Normalize and combine performance metrics
                perf_score = ((win_rate - 0.5) * 2) * 0.3  # Scale win rate to -1 to 1 range and weight it
                pf_score = min((profit_factor - 1) * 0.25, 0.5)  # Cap profit factor contribution at 0.5
                base_score += perf_score + pf_score
            
            # Ensure score is within 0-1 range
            strategy_scores[strategy_type] = max(0.0, min(1.0, base_score))
        
        # Log scores for analysis
        logger.info(f"Strategy scores for {asset.value} {timeframe.value}: {strategy_scores}")
        
        # Select best strategy - must meet minimum confidence threshold
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        if best_strategy[1] >= self.min_strategy_confidence:
            logger.info(f"Selected {best_strategy[0].value} strategy with confidence {best_strategy[1]:.2f}")
            return best_strategy[0]
        
        # If no strategy meets confidence threshold, use default for this asset
        if asset in self.asset_preferences and "default_strategies" in self.asset_preferences[asset]:
            logger.info(f"Using default strategy {self.asset_preferences[asset]['default_strategies'][0].value} due to low confidence scores")
            return self.asset_preferences[asset]["default_strategies"][0]
        
        logger.info(f"No high-confidence strategy found, defaulting to breakout strategy")
        return StrategyType.BREAKOUT
    
    def generate_optimized_signals(self, asset: AssetSymbol) -> List[TradingSignalModel]:
        """
        Generate optimized trading signals for an asset using adaptive strategy selection
        and dynamic parameter tuning.
        
        This is the main entry point for signal generation with the adaptive engine.
        """
        logger.info(f"Generating optimized signals for {asset.value}")
        
        # Get optimal timeframes for this asset
        optimal_timeframes = self.select_optimal_timeframe(asset)
        all_signals = []
        
        # Generate signals for each optimal timeframe
        for timeframe in optimal_timeframes:
            # Select best strategy for this asset/timeframe
            strategy_type = self.select_best_strategy(asset, timeframe)
            
            # Get market data
            market_data = self.market_data_agent.fetch_historical_data(asset, timeframe)
            if not market_data or len(market_data.candles) < 50:
                logger.warning(f"Insufficient data for {asset.value} {timeframe.value}")
                continue
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([
                {
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume
                } for candle in market_data.candles
            ])
            df.set_index("timestamp", inplace=True)
            
            # Detect current market regime
            regime = self.regime_detector.detect_regime(df, asset, timeframe)
            
            # Generate signals based on selected strategy type
            signals = []
            
            try:
                # Use appropriate strategy implementation based on selected strategy type
                if strategy_type == StrategyType.MEAN_REVERSION:
                    # Use our new dedicated MeanReversionStrategy implementation
                    signals = self.mean_reversion_strategy.generate_signals(
                        asset=asset,
                        timeframe=timeframe,
                        historical_data=market_data
                    )
                    
                elif strategy_type == StrategyType.TREND_FOLLOWING:
                    # Use our new dedicated TrendFollowingStrategy implementation
                    signals = self.trend_following_strategy.generate_signals(
                        asset=asset,
                        timeframe=timeframe,
                        historical_data=market_data
                    )
                    
                else:
                    # Fallback to signal scanner with optimized parameters for other strategies
                    scanner_params = self._get_optimized_parameters(strategy_type, regime.regime_type, asset)
                    
                    # Backup current signal scanner parameters
                    original_params = self._backup_signal_scanner_params()
                    
                    # Apply optimized parameters to signal scanner
                    self._apply_params_to_signal_scanner(scanner_params)
                    
                    # Select the appropriate scanning method based on strategy type
                    if strategy_type == StrategyType.BREAKOUT:
                        signals = self.signal_scanner_agent.scan_for_breakout_signals(asset, timeframe)
                    elif strategy_type == StrategyType.RANGE_TRADING:
                        # For range trading, we'll use breakout signals with opposite interpretation
                        signals = self.signal_scanner_agent.scan_for_breakout_signals(asset, timeframe)
                    elif strategy_type == StrategyType.PULLBACK:
                        # For pullback, use breakout signals but adjust for pullback entry
                        signals = self.signal_scanner_agent.scan_for_breakout_signals(asset, timeframe)
                    else:
                        # For any other strategies, default to breakout signals
                        signals = self.signal_scanner_agent.scan_for_breakout_signals(asset, timeframe)
                    
                    # Restore original signal scanner parameters
                    self._restore_signal_scanner_params(original_params)
            except Exception as e:
                logger.error(f"Error generating signals for {asset.value} {timeframe.value} with {strategy_type.value} strategy: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
                
            if not signals:
                logger.info(f"No signals generated for {asset.value} {timeframe.value} with {strategy_type.value} strategy")
                continue
            
            # Enhance signals with optimized TP/SL and other metadata
            for signal in signals:
                self._enhance_signal_parameters(signal, strategy_type, regime)
                signal.metadata["strategy_type"] = strategy_type.value
                # Get strategy confidence based on strategy type
                strategy_confidence = 0.8  # Default confidence for our implemented strategies
                
                # For traditional signal scanner strategies, use params if available
                if strategy_type not in [StrategyType.MEAN_REVERSION, StrategyType.TREND_FOLLOWING] and 'scanner_params' in locals():
                    strategy_confidence = scanner_params.get("strategy_confidence", 0.8)
                
                signal.metadata["adaptive_engine"] = {
                    "timeframe_rank": optimal_timeframes.index(timeframe) + 1,
                    "strategy_confidence": strategy_confidence,
                    "market_regime": regime.regime_type.value,
                    "regime_strength": regime.strength
                }
            
            # Add signals from this timeframe to the overall list
            all_signals.extend(signals)
        
        # Apply multi-timeframe confirmation if enabled
        if self.multi_timeframe_confirmation and len(all_signals) > 1:
            all_signals = self._apply_multi_timeframe_confirmation(all_signals, asset)
        
        # Score and rank signals
        scored_signals = self._score_and_rank_signals(all_signals)
        
        # Only return signals that meet the minimum score threshold
        return [s for s, score in scored_signals if score >= self.minimum_alert_score]
    
    def _backup_signal_scanner_params(self) -> Dict:
        """Backup the current signal scanner parameters for later restoration"""
        return {
            "breakout_period": self.signal_scanner_agent.breakout_period,
            "atr_period": self.signal_scanner_agent.atr_period,
            "atr_multiplier_sl": self.signal_scanner_agent.atr_multiplier_sl,
            "min_rr_ratio": self.signal_scanner_agent.min_rr_ratio,
            "volume_confirmation": self.signal_scanner_agent.volume_confirmation,
            "volume_threshold": self.signal_scanner_agent.volume_threshold,
            "rsi_period": self.signal_scanner_agent.rsi_period,
            "rsi_oversold": self.signal_scanner_agent.rsi_oversold,
            "rsi_overbought": self.signal_scanner_agent.rsi_overbought,
            "allow_pullback_entry": self.signal_scanner_agent.allow_pullback_entry
        }
    
    def _apply_params_to_signal_scanner(self, params: Dict):
        """Apply optimized parameters to the signal scanner"""
        # Apply parameters from the strategy profile to the signal scanner
        # Only set parameters that exist and are provided
        if "breakout_period" in params:
            self.signal_scanner_agent.breakout_period = params["breakout_period"]
        if "atr_multiplier_sl" in params:
            self.signal_scanner_agent.atr_multiplier_sl = params["atr_multiplier_sl"]
        if "min_rr_ratio" in params:
            self.signal_scanner_agent.min_rr_ratio = params["min_rr_ratio"]
        if "volume_threshold" in params:
            self.signal_scanner_agent.volume_threshold = params["volume_threshold"]
        if "allow_pullback_entry" in params:
            self.signal_scanner_agent.allow_pullback_entry = params["allow_pullback_entry"]
        if "rsi_period" in params:
            self.signal_scanner_agent.rsi_period = params["rsi_period"]
        if "rsi_oversold" in params:
            self.signal_scanner_agent.rsi_oversold = params["rsi_oversold"]
        if "rsi_overbought" in params:
            self.signal_scanner_agent.rsi_overbought = params["rsi_overbought"]
    
    def _restore_signal_scanner_params(self, original_params: Dict):
        """Restore the signal scanner parameters to their original values"""
        self._apply_params_to_signal_scanner(original_params)
    
    def _get_optimized_parameters(self, strategy_type: StrategyType, regime_type: MarketRegimeType, asset: AssetSymbol) -> Dict:
        """Get optimized parameters for a specific strategy and market regime"""
        # Start with default parameters for this strategy and regime
        params = {}
        
        # Add strategy-specific parameters from profiles
        if strategy_type in self.strategy_profiles and regime_type in self.strategy_profiles[strategy_type]:
            params.update(self.strategy_profiles[strategy_type][regime_type])
        
        # Apply asset-specific adjustments for volatility etc.
        if asset in self.asset_preferences and "volatility_factor" in self.asset_preferences[asset]:
            vol_factor = self.asset_preferences[asset]["volatility_factor"]
            # Adjust ATR multiplier for stop loss based on asset volatility
            if "atr_multiplier_sl" in params:
                params["atr_multiplier_sl"] = params["atr_multiplier_sl"] * vol_factor
        
        return params
    
    def _enhance_signal_parameters(self, signal: TradingSignalModel, strategy_type: StrategyType, regime):
        """Enhance signal with better TP/SL levels and additional metadata"""
        # Base prices
        entry_price = signal.entry_price
        current_sl = signal.stop_loss
        current_tp = signal.take_profit
        
        # Calculate baseline risk
        if signal.signal_type == SignalType.LONG:
            base_risk = entry_price - current_sl
        else:  # SHORT
            base_risk = current_sl - entry_price
        
        # Adjust TP/SL based on market regime
        regime_type = regime.regime_type
        regime_strength = regime.strength
        
        # Adjust stop loss to be tighter in trending markets, wider in volatile markets
        sl_adjustment = 1.0  # Default: no adjustment
        if regime_type == MarketRegimeType.BULLISH_TRENDING or regime_type == MarketRegimeType.BEARISH_TRENDING:
            # Tighter stops in strong trends (more confident)
            sl_adjustment = max(0.8, 1.0 - (regime_strength * 0.2))
        elif regime_type == MarketRegimeType.HIGH_VOLATILITY:
            # Wider stops in volatile markets
            sl_adjustment = min(1.5, 1.0 + (regime_strength * 0.5))
        
        # Adjust take profit to be further in trending markets, closer in ranging markets
        tp_adjustment = 1.0  # Default: no adjustment
        if regime_type == MarketRegimeType.BULLISH_TRENDING or regime_type == MarketRegimeType.BEARISH_TRENDING:
            # Further take profits in strong trends
            tp_adjustment = min(1.5, 1.0 + (regime_strength * 0.5))
        elif regime_type == MarketRegimeType.NEUTRAL_RANGING:
            # Closer take profits in ranging markets
            tp_adjustment = max(0.8, 1.0 - (regime_strength * 0.2))
        
        # Apply adjustments based on signal type
        if signal.signal_type == SignalType.LONG:
            # LONG: stop_loss below entry, take_profit above entry
            adjusted_sl_distance = base_risk * sl_adjustment
            new_stop_loss = round(entry_price - adjusted_sl_distance, 5)
            
            # Calculate new take profit based on adjusted risk and R:R ratio
            min_rr = 1.5  # Minimum R:R ratio
            if strategy_type in self.strategy_profiles and regime_type in self.strategy_profiles[strategy_type]:
                min_rr = self.strategy_profiles[strategy_type][regime_type].get("min_rr_ratio", 1.5)
            
            # Apply the R:R ratio adjustment from the regime
            adjusted_rr = min_rr * tp_adjustment
            new_take_profit = round(entry_price + (adjusted_sl_distance * adjusted_rr), 5)
            
            # Update the signal
            signal.stop_loss = new_stop_loss
            signal.take_profit = new_take_profit
            
        else:  # SHORT
            # SHORT: stop_loss above entry, take_profit below entry
            adjusted_sl_distance = base_risk * sl_adjustment
            new_stop_loss = round(entry_price + adjusted_sl_distance, 5)
            
            # Calculate new take profit based on adjusted risk and R:R ratio
            min_rr = 1.5  # Minimum R:R ratio
            if strategy_type in self.strategy_profiles and regime_type in self.strategy_profiles[strategy_type]:
                min_rr = self.strategy_profiles[strategy_type][regime_type].get("min_rr_ratio", 1.5)
            
            # Apply the R:R ratio adjustment from the regime
            adjusted_rr = min_rr * tp_adjustment
            new_take_profit = round(entry_price - (adjusted_sl_distance * adjusted_rr), 5)
            
            # Update the signal
            signal.stop_loss = new_stop_loss
            signal.take_profit = new_take_profit
        
        # Update metadata to reflect adjustments
        if "metadata" not in signal.metadata:
            signal.metadata["adaptive_adjustments"] = {}
        
        signal.metadata["adaptive_adjustments"] = {
            "original_stop_loss": current_sl,
            "original_take_profit": current_tp,
            "sl_adjustment_factor": sl_adjustment,
            "tp_adjustment_factor": tp_adjustment,
            "adjusted_min_rr": min_rr * tp_adjustment,
            "regime_based": True
        }
        
        # Calculate and add new risk-reward ratio
        if signal.signal_type == SignalType.LONG:
            new_risk = entry_price - signal.stop_loss
            new_reward = signal.take_profit - entry_price
        else:  # SHORT
            new_risk = signal.stop_loss - entry_price
            new_reward = entry_price - signal.take_profit
        
        if new_risk > 0:  # Avoid division by zero
            signal.risk_reward_ratio = round(new_reward / new_risk, 2)
    
    def _apply_multi_timeframe_confirmation(self, signals: List[TradingSignalModel], asset: AssetSymbol) -> List[TradingSignalModel]:
        """
        Apply multi-timeframe confirmation to filter signals.
        Signals are confirmed if they align with the trend on multiple timeframes.
        """
        # Group signals by type (LONG/SHORT)
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        
        # If we have signals of both types, that's a conflict - keep only the ones from higher timeframes
        if long_signals and short_signals:
            logger.info(f"Conflicting signal directions detected for {asset.value}. Resolving based on timeframe priority.")
            
            # Get timeframes in priority order
            optimal_timeframes = self.select_optimal_timeframe(asset)
            
            # If the highest priority timeframe has a LONG signal, keep LONG signals
            long_timeframes = set(s.timeframe for s in long_signals)
            short_timeframes = set(s.timeframe for s in short_signals)
            
            # Find the highest priority timeframe that has a signal
            for tf in optimal_timeframes:
                if tf in long_timeframes and tf not in short_timeframes:
                    logger.info(f"Keeping LONG signals based on {tf.value} timeframe priority")
                    return long_signals
                elif tf in short_timeframes and tf not in long_timeframes:
                    logger.info(f"Keeping SHORT signals based on {tf.value} timeframe priority")
                    return short_signals
            
            # If both signal types are present in the same timeframe, use additional confirmation
            return self._resolve_signal_conflict(long_signals, short_signals, asset)
        
        # Return all signals if no conflict
        return signals
    
    def _resolve_signal_conflict(self, long_signals: List[TradingSignalModel], short_signals: List[TradingSignalModel], asset: AssetSymbol) -> List[TradingSignalModel]:
        """
        Resolve conflicts between LONG and SHORT signals using additional factors 
        like regime strength, signal count, and other technical indicators.
        """
        # Count signals of each type
        long_count = len(long_signals)
        short_count = len(short_signals)
        
        # Get the average regime strength for each type
        long_regime_strength = sum(float(s.metadata.get("adaptive_engine", {}).get("regime_strength", 0.5)) for s in long_signals) / long_count if long_count > 0 else 0
        short_regime_strength = sum(float(s.metadata.get("adaptive_engine", {}).get("regime_strength", 0.5)) for s in short_signals) / short_count if short_count > 0 else 0
        
        logger.info(f"Signal conflict resolution - LONG: {long_count} signals (avg strength: {long_regime_strength:.2f}), SHORT: {short_count} signals (avg strength: {short_regime_strength:.2f})")
        
        # Use regime strength as a tiebreaker
        if long_regime_strength > short_regime_strength + 0.1:  # Add a threshold
            logger.info(f"Resolving in favor of LONG signals based on regime strength")
            return long_signals
        elif short_regime_strength > long_regime_strength + 0.1:
            logger.info(f"Resolving in favor of SHORT signals based on regime strength")
            return short_signals
        
        # If still tied, check signal counts
        if long_count > short_count * 1.5:  # Significantly more long signals
            logger.info(f"Resolving in favor of LONG signals based on signal count")
            return long_signals
        elif short_count > long_count * 1.5:  # Significantly more short signals
            logger.info(f"Resolving in favor of SHORT signals based on signal count")
            return short_signals
        
        # If still can't resolve, return all but mark as conflicted
        for signal in long_signals + short_signals:
            if "adaptive_engine" not in signal.metadata:
                signal.metadata["adaptive_engine"] = {}
            signal.metadata["adaptive_engine"]["conflicted_signals"] = True
        
        logger.warning(f"Could not resolve signal conflict for {asset.value}. Returning all signals but marked as conflicted.")
        return long_signals + short_signals
    
    def _score_and_rank_signals(self, signals: List[TradingSignalModel]) -> List[Tuple[TradingSignalModel, float]]:
        """
        Score and rank signals based on multiple factors including regime strength,
        timeframe priority, risk-reward ratio, and strategy confidence.
        
        Returns a list of tuples (signal, score) sorted by score from highest to lowest.
        """
        scored_signals = []
        
        for signal in signals:
            # Start with a base score
            score = 0.5
            
            # Get metadata
            adaptive_meta = signal.metadata.get("adaptive_engine", {})
            
            # Factor 1: Timeframe rank (higher timeframes are more reliable)
            # Convert to 0-0.2 range (1st timeframe = 0.2, 2nd = 0.15, etc.)
            tf_rank = adaptive_meta.get("timeframe_rank", 3)  # Default to middle rank
            tf_score = max(0.0, 0.25 - ((tf_rank - 1) * 0.05))
            score += tf_score
            
            # Factor 2: Regime strength (stronger regimes = more reliable signals)
            regime_strength = float(adaptive_meta.get("regime_strength", 0.5))
            score += (regime_strength - 0.5) * 0.4  # Convert 0.5-1.0 to 0-0.2 range
            
            # Factor 3: Risk-reward ratio (better R:R = higher score)
            rr_ratio = signal.risk_reward_ratio or 1.0
            rr_score = min(0.15, (rr_ratio - 1.0) * 0.1)  # Convert 1.0-2.5 to 0-0.15 range
            score += rr_score
            
            # Factor 4: Strategy confidence
            strategy_conf = float(adaptive_meta.get("strategy_confidence", 0.5))
            score += (strategy_conf - 0.5) * 0.3  # Convert 0.5-1.0 to 0-0.15 range
            
            # Factor 5: Conflict penalty
            if adaptive_meta.get("conflicted_signals", False):
                score -= 0.1  # Penalty for conflicted signals
            
            # Ensure score is within 0-1 range
            final_score = max(0.0, min(1.0, score))
            
            # Add to scored signals list
            scored_signals.append((signal, final_score))
            
            # Add score to signal metadata
            signal.metadata["adaptive_engine"]["signal_score"] = round(final_score, 2)
        
        # Sort by score (highest first)
        return sorted(scored_signals, key=lambda x: x[1], reverse=True)
