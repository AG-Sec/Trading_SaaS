"""
Regime Analyzer for Backtesting Module

This module provides tools for analyzing market regimes and their impact on
trading performance, including regime detection, transition analysis, and
optimization of strategy parameters by regime.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import joblib

from shared_types.models import AssetSymbol, Timeframe
from backend.agents.market_regime_detector import MarketRegime, MarketRegimeType
from backend.backtesting.backtester import BacktestTrade, TradeStatus
from backend.backtesting.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class RegimeAnalyzer:
    """
    Analyzes trading performance across different market regimes.
    """
    
    def __init__(self):
        """Initialize the RegimeAnalyzer."""
        pass
    
    def analyze_regime_performance(self, 
                                 trades: List[BacktestTrade], 
                                 equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trading performance across different market regimes.
        
        Args:
            trades: List of backtest trades
            equity_curve: DataFrame with equity curve including regime information
            
        Returns:
            Dictionary of regime-specific performance metrics
        """
        # Check if regime information is available
        if not trades or not any(t.market_regime for t in trades):
            logger.warning("No regime information found in trades")
            return {}
        
        # Calculate metrics by regime
        regime_metrics = PerformanceMetrics.calculate_regime_metrics(trades)
        
        # Analyze regime transitions if equity curve has regime data
        regime_transitions = []
        
        if 'market_regime' in equity_curve and len(equity_curve) > 1:
            # Detect regime changes in the equity curve
            for i in range(1, len(equity_curve)):
                curr_regime = equity_curve['market_regime'].iloc[i]
                prev_regime = equity_curve['market_regime'].iloc[i-1]
                
                if isinstance(curr_regime, MarketRegime) and isinstance(prev_regime, MarketRegime):
                    if curr_regime.regime_type != prev_regime.regime_type:
                        # Regime transition detected
                        transition_time = equity_curve['date'].iloc[i]
                        regime_transitions.append((
                            transition_time,
                            prev_regime.regime_type.value,
                            curr_regime.regime_type.value,
                            curr_regime.strength
                        ))
        
        # Calculate metrics around regime transitions
        transition_metrics = {}
        if regime_transitions:
            transition_metrics = PerformanceMetrics.calculate_regime_transition_metrics(
                trades, regime_transitions
            )
        
        # Calculate equity curve by regime
        regime_equity_curves = self._calculate_regime_equity_curves(equity_curve)
        
        # Compare strategy performance across regimes
        strategy_regime_performance = self._analyze_strategy_regime_performance(trades)
        
        return {
            "regime_metrics": regime_metrics,
            "regime_transitions": regime_transitions,
            "transition_metrics": transition_metrics,
            "regime_equity_curves": regime_equity_curves,
            "strategy_regime_performance": strategy_regime_performance
        }
    
    def _calculate_regime_equity_curves(self, equity_curve: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate equity curves for each market regime.
        
        Args:
            equity_curve: DataFrame with 'date', 'equity', and 'market_regime' columns
            
        Returns:
            Dictionary of regime-specific equity curves
        """
        if 'market_regime' not in equity_curve.columns or len(equity_curve) < 2:
            return {}
        
        # Ensure market_regime is processed to string
        equity_curve = equity_curve.copy()
        
        # Convert MarketRegime objects to string names if needed
        if isinstance(equity_curve['market_regime'].iloc[0], MarketRegime):
            equity_curve['regime_name'] = equity_curve['market_regime'].apply(
                lambda r: r.regime_type.value if isinstance(r, MarketRegime) else str(r)
            )
        else:
            equity_curve['regime_name'] = equity_curve['market_regime'].astype(str)
        
        # Group equity curve by regime
        regime_groups = equity_curve.groupby('regime_name')
        
        # Create equity curves for each regime
        regime_equity_curves = {}
        
        for regime_name, group in regime_groups:
            # Reset equity to start from 100 for each regime
            regime_df = group.copy()
            first_equity = regime_df['equity'].iloc[0]
            regime_df['normalized_equity'] = regime_df['equity'] / first_equity * 100
            
            regime_equity_curves[regime_name] = regime_df
        
        return regime_equity_curves
    
    def _analyze_strategy_regime_performance(self, trades: List[BacktestTrade]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze strategy performance across different market regimes.
        
        Args:
            trades: List of backtest trades
            
        Returns:
            Dictionary of strategy performance by regime
        """
        # Group trades by strategy and regime
        strategy_regime_trades = defaultdict(lambda: defaultdict(list))
        
        for trade in trades:
            if trade.status != TradeStatus.OPEN and trade.strategy_name and trade.market_regime:
                strategy_regime_trades[trade.strategy_name][trade.market_regime].append(trade)
        
        # Calculate metrics for each strategy-regime combination
        strategy_regime_metrics = {}
        
        for strategy, regime_trades in strategy_regime_trades.items():
            strategy_regime_metrics[strategy] = {}
            
            for regime, trades_list in regime_trades.items():
                if not trades_list:
                    continue
                    
                metrics = PerformanceMetrics.calculate_basic_metrics(trades_list)
                strategy_regime_metrics[strategy][regime] = metrics
        
        return strategy_regime_metrics
    
    def detect_regime_from_data(self, 
                              historical_data: pd.DataFrame, 
                              window_size: int = 20) -> List[MarketRegime]:
        """
        Detect market regimes from historical price data.
        
        Args:
            historical_data: DataFrame with OHLCV price data
            window_size: Window size for regime detection in bars
            
        Returns:
            List of detected market regimes
        """
        if len(historical_data) < window_size:
            return [MarketRegime(regime_type=MarketRegimeType.NEUTRAL_RANGING, strength=0.5)]
        
        regimes = []
        
        for i in range(window_size, len(historical_data)):
            # Get data window
            window = historical_data.iloc[i-window_size:i]
            
            # Calculate key metrics
            returns = window['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Simple trend detection
            trend = (window['close'].iloc[-1] / window['close'].iloc[0]) - 1
            
            # Momentum calculation (rate of change)
            momentum = window['close'].pct_change(5).mean()
            
            # Regime determination logic
            regime_type = MarketRegimeType.NEUTRAL_RANGING
            strength = 0.5
            
            if trend > 0.05 and momentum > 0:
                regime_type = MarketRegimeType.BULLISH_TRENDING
                strength = min(1.0, abs(trend) * 10)  # Scale strength based on trend magnitude
            elif trend < -0.05 and momentum < 0:
                regime_type = MarketRegimeType.BEARISH_TRENDING
                strength = min(1.0, abs(trend) * 10)
            elif volatility > 0.03:  # Annualized volatility > 30%
                regime_type = MarketRegimeType.HIGH_VOLATILITY
                strength = min(1.0, volatility / 0.05)  # Scale with volatility
            elif volatility < 0.01:  # Annualized volatility < 10%
                regime_type = MarketRegimeType.LOW_VOLATILITY
                strength = min(1.0, 0.01 / max(0.001, volatility))
            
            regimes.append(MarketRegime(regime_type=regime_type, strength=strength))
        
        # Prepend with neutral regimes for the initial window where we couldn't calculate
        initial_regimes = [MarketRegime(regime_type=MarketRegimeType.NEUTRAL_RANGING, strength=0.5)] * (window_size)
        
        return initial_regimes + regimes
    
    def optimize_regime_thresholds(self, 
                                 historical_data: pd.DataFrame, 
                                 trades: List[BacktestTrade],
                                 param_grid: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Optimize regime detection thresholds based on trading performance.
        
        Args:
            historical_data: DataFrame with OHLCV price data
            trades: List of backtest trades with regime information
            param_grid: Dictionary of parameters and values to test
            
        Returns:
            Dictionary of optimized parameters
        """
        # Prepare dataset for optimization
        trade_outcomes = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time if t.exit_time else t.entry_time + timedelta(days=1),
                'market_regime': t.market_regime,
                'outcome': 1 if t.pnl_absolute > 0 else 0,
                'pnl': t.pnl_absolute
            }
            for t in trades if t.status != TradeStatus.OPEN
        ])
        
        if len(trade_outcomes) < 10:
            logger.warning("Insufficient trades for regime threshold optimization")
            return {
                'trend_threshold': 0.05,
                'volatility_threshold': 0.03,
                'momentum_threshold': 0.01
            }
        
        # Define parameter grid
        if not param_grid:
            param_grid = {
                'trend_threshold': [0.02, 0.03, 0.05, 0.07, 0.1],
                'volatility_threshold': [0.01, 0.02, 0.03, 0.04, 0.05],
                'momentum_threshold': [0.005, 0.01, 0.015, 0.02]
            }
        
        # Grid search for optimal parameters
        best_params = None
        best_score = -float('inf')
        
        import itertools
        param_combinations = list(itertools.product(*param_grid.values()))
        param_keys = list(param_grid.keys())
        
        for params in param_combinations:
            param_dict = {param_keys[i]: params[i] for i in range(len(params))}
            
            # Apply these params to detect regimes
            regimes = self._detect_regimes_with_params(historical_data, param_dict)
            
            # Map regimes to trade dates
            trade_outcomes['detected_regime'] = trade_outcomes['entry_time'].apply(
                lambda dt: self._get_regime_at_time(dt, historical_data.index, regimes)
            )
            
            # Calculate performance metrics by regime
            regime_performance = {}
            
            for regime in set(r.regime_type.value for r in regimes):
                regime_trades = trade_outcomes[trade_outcomes['detected_regime'] == regime]
                
                if len(regime_trades) < 5:
                    continue
                    
                win_rate = regime_trades['outcome'].mean()
                avg_pnl = regime_trades['pnl'].mean()
                
                regime_performance[regime] = {
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'trade_count': len(regime_trades)
                }
            
            # Calculate overall score (can be customized based on objectives)
            if not regime_performance:
                continue
                
            avg_win_rate = np.mean([perf['win_rate'] for perf in regime_performance.values()])
            win_rate_std = np.std([perf['win_rate'] for perf in regime_performance.values()])
            
            # Score favors higher win rates with lower standard deviation across regimes
            score = avg_win_rate - win_rate_std
            
            if score > best_score:
                best_score = score
                best_params = param_dict
        
        if best_params is None:
            logger.warning("Failed to optimize regime thresholds, using defaults")
            return {
                'trend_threshold': 0.05,
                'volatility_threshold': 0.03,
                'momentum_threshold': 0.01
            }
        
        return best_params
    
    def _detect_regimes_with_params(self, 
                                  historical_data: pd.DataFrame, 
                                  params: Dict[str, float],
                                  window_size: int = 20) -> List[MarketRegime]:
        """
        Detect market regimes using specific threshold parameters.
        
        Args:
            historical_data: DataFrame with OHLCV price data
            params: Dictionary of threshold parameters
            window_size: Window size for regime detection
            
        Returns:
            List of detected regimes
        """
        trend_threshold = params.get('trend_threshold', 0.05)
        volatility_threshold = params.get('volatility_threshold', 0.03)
        momentum_threshold = params.get('momentum_threshold', 0.01)
        
        regimes = []
        
        for i in range(window_size, len(historical_data)):
            # Get data window
            window = historical_data.iloc[i-window_size:i]
            
            # Calculate key metrics
            returns = window['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Trend calculation
            trend = (window['close'].iloc[-1] / window['close'].iloc[0]) - 1
            
            # Momentum calculation (rate of change)
            momentum = window['close'].pct_change(5).mean()
            
            # Regime determination logic
            regime_type = MarketRegimeType.NEUTRAL_RANGING
            strength = 0.5
            
            if trend > trend_threshold and momentum > momentum_threshold:
                regime_type = MarketRegimeType.BULLISH_TRENDING
                strength = min(1.0, abs(trend) / (trend_threshold * 2))
            elif trend < -trend_threshold and momentum < -momentum_threshold:
                regime_type = MarketRegimeType.BEARISH_TRENDING
                strength = min(1.0, abs(trend) / (trend_threshold * 2))
            elif volatility > volatility_threshold:
                regime_type = MarketRegimeType.HIGH_VOLATILITY
                strength = min(1.0, volatility / (volatility_threshold * 2))
            elif volatility < volatility_threshold / 3:
                regime_type = MarketRegimeType.LOW_VOLATILITY
                strength = min(1.0, (volatility_threshold / 3) / max(0.001, volatility))
            
            regimes.append(MarketRegime(regime_type=regime_type, strength=strength))
        
        # Prepend with neutral regimes for the initial window
        initial_regimes = [MarketRegime(regime_type=MarketRegimeType.NEUTRAL_RANGING, strength=0.5)] * window_size
        
        return initial_regimes + regimes
    
    def _get_regime_at_time(self, 
                          timestamp: datetime, 
                          data_index: pd.DatetimeIndex, 
                          regimes: List[MarketRegime]) -> str:
        """
        Get the market regime at a specific timestamp.
        
        Args:
            timestamp: Timestamp to check
            data_index: DatetimeIndex of the historical data
            regimes: List of detected regimes
            
        Returns:
            Regime type as string
        """
        # Find the nearest index before or equal to the timestamp
        try:
            idx = data_index.get_indexer([timestamp], method='pad')[0]
            
            if idx >= 0 and idx < len(regimes):
                return regimes[idx].regime_type.value
            else:
                return MarketRegimeType.NEUTRAL_RANGING.value
        except:
            return MarketRegimeType.NEUTRAL_RANGING.value

    def save_regime_model(self, model_path: str, params: Dict[str, Any]):
        """
        Save regime detection parameters to a file.
        
        Args:
            model_path: Path to save the model
            params: Parameters to save
        """
        joblib.dump(params, model_path)
        logger.info(f"Saved regime detection model to {model_path}")
    
    def load_regime_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load regime detection parameters from a file.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            Dictionary of loaded parameters
        """
        try:
            params = joblib.load(model_path)
            logger.info(f"Loaded regime detection model from {model_path}")
            return params
        except Exception as e:
            logger.error(f"Failed to load regime model: {e}")
            return {
                'trend_threshold': 0.05,
                'volatility_threshold': 0.03,
                'momentum_threshold': 0.01
            }
    
    def create_regime_classification_model(self, 
                                         historical_data: pd.DataFrame, 
                                         trades: List[BacktestTrade]) -> Any:
        """
        Create a machine learning model for regime classification.
        
        Args:
            historical_data: DataFrame with OHLCV price data
            trades: List of backtest trades with regime information
            
        Returns:
            Trained machine learning model
        """
        # This is a placeholder for implementing a machine learning model
        # In an actual implementation, this would:
        # 1. Extract features from price data
        # 2. Use trade performance as training labels
        # 3. Train a classifier (e.g., Random Forest) to predict optimal regimes
        # 4. Return the trained model
        
        logger.info("Creating regime classification model (placeholder)")
        
        # For now, we'll just return a simple dictionary with default parameters
        return {
            'model_type': 'placeholder',
            'trend_threshold': 0.05,
            'volatility_threshold': 0.03,
            'momentum_threshold': 0.01
        }
