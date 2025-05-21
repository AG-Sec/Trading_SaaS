"""
Market Regime Optimizer Module for the Trading SaaS platform.
Uses machine learning to optimize the thresholds and parameters for market regime detection.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, timezone
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

from shared_types import AssetSymbol, Timeframe
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.market_regime_detector import MarketRegimeDetector, MarketRegimeType
from backend.agents.backtesting import Backtester, BacktestResult
from backend.agents.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class RegimeThresholds:
    """Class to hold optimized regime detection thresholds"""
    
    def __init__(self):
        # Trend detection thresholds
        self.trend_ema_distance = 0.015  # Minimum distance between fast/slow EMAs to confirm trend
        self.trend_slope_threshold = 0.0003  # Minimum slope of SMA to confirm trend direction
        self.price_sma_distance = 0.01  # Min distance between price and SMA to confirm trend
        
        # Volatility thresholds
        self.high_volatility_atr_pct = 3.0  # ATR % threshold for high volatility
        self.low_volatility_atr_pct = 1.5  # ATR % threshold for low volatility
        self.bb_width_high = 0.06  # BB width threshold for high volatility
        self.bb_width_low = 0.025  # BB width threshold for low volatility
        
        # Momentum thresholds
        self.overbought_rsi = 70  # RSI threshold for overbought
        self.oversold_rsi = 30  # RSI threshold for oversold
        self.neutral_rsi_min = 40  # Minimum RSI for neutral range 
        self.neutral_rsi_max = 60  # Maximum RSI for neutral range
        
        # Breakout thresholds
        self.volume_surge_ratio = 1.5  # Volume ratio threshold for breakouts
        self.bb_position_threshold = 0.8  # BB position threshold for breakouts
        
    def to_dict(self) -> Dict[str, float]:
        """Convert thresholds to dictionary"""
        return {
            'trend_ema_distance': self.trend_ema_distance,
            'trend_slope_threshold': self.trend_slope_threshold,
            'price_sma_distance': self.price_sma_distance,
            'high_volatility_atr_pct': self.high_volatility_atr_pct,
            'low_volatility_atr_pct': self.low_volatility_atr_pct,
            'bb_width_high': self.bb_width_high,
            'bb_width_low': self.bb_width_low,
            'overbought_rsi': self.overbought_rsi,
            'oversold_rsi': self.oversold_rsi,
            'neutral_rsi_min': self.neutral_rsi_min,
            'neutral_rsi_max': self.neutral_rsi_max,
            'volume_surge_ratio': self.volume_surge_ratio,
            'bb_position_threshold': self.bb_position_threshold
        }
        
    def from_dict(self, data: Dict[str, float]) -> None:
        """Load thresholds from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class RegimeOptimizer:
    """
    Class for optimizing market regime detection parameters using machine learning.
    Uses supervised learning to find optimal thresholds based on historical 
    performance in different market conditions.
    """
    
    def __init__(
        self,
        market_data_agent: MarketDataAgent,
        backtester: Backtester,
        regime_detector: Optional[MarketRegimeDetector] = None,
        output_dir: str = "regime_optimizer_output"
    ):
        self.market_data_agent = market_data_agent
        self.backtester = backtester
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.tech_indicators = TechnicalIndicators()
        self.output_dir = output_dir
        self.current_thresholds = RegimeThresholds()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.model_dir = os.path.join(output_dir, "models")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
    def create_training_dataset(
        self,
        assets: List[AssetSymbol],
        timeframes: List[Timeframe],
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """
        Create a training dataset for ML optimization using historical market data
        and backtesting results.
        
        Args:
            assets: List of assets to analyze
            timeframes: List of timeframes to analyze
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with features and regime labels
        """
        logger.info(f"Creating training dataset from {len(assets)} assets across {len(timeframes)} timeframes")
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        
        all_data = []
        
        for asset in assets:
            for timeframe in timeframes:
                logger.info(f"Processing {asset.value} {timeframe.value}...")
                
                # Fetch historical data
                historical_data = self.market_data_agent.fetch_historical_data(
                    asset=asset,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not historical_data or not historical_data.candles:
                    logger.warning(f"No data for {asset.value} {timeframe.value}")
                    continue
                
                # Convert to dataframe
                df = pd.DataFrame([candle.model_dump() for candle in historical_data.candles])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Calculate indicators
                df = self.tech_indicators.calculate_all(df)
                
                # Run backtest to assess performance with different regimes
                backtest_result = self.backtester.backtest_strategy(
                    asset=asset,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_name=f"optimize_{asset.value}_{timeframe.value}",
                    use_adaptive_params=True
                )
                
                # Skip if no regime data
                if not backtest_result.regime_periods:
                    continue
                
                # Process each bar with its regime label
                for i in range(50, len(df)):  # Skip first 50 bars for indicator stability
                    current_bar = df.iloc[i]
                    current_time = current_bar.name
                    
                    # Find the regime for this bar
                    current_regime = None
                    for period in backtest_result.regime_periods:
                        period_start = period['start_date']
                        period_end = period['end_date']
                        
                        if period_start <= current_time <= period_end:
                            current_regime = period['regime_type']
                            break
                    
                    if not current_regime:
                        continue
                    
                    # Extract features for this bar
                    features = self._extract_features(df, i)
                    
                    # Add performance label from backtest for this regime
                    regime_performance = backtest_result.regime_performance.get(current_regime, {})
                    win_rate = regime_performance.get('win_rate', 0)
                    avg_return = regime_performance.get('average_return_per_trade', 0)
                    
                    # Create per-bar data point
                    row = {
                        'asset': asset.value,
                        'timeframe': timeframe.value,
                        'timestamp': current_time,
                        'regime': current_regime,
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        # Add all features
                        **features
                    }
                    
                    all_data.append(row)
        
        # Convert to DataFrame
        dataset = pd.DataFrame(all_data)
        
        # Save dataset
        dataset.to_csv(f"{self.output_dir}/training_dataset.csv", index=False)
        logger.info(f"Training dataset created with {len(dataset)} samples")
        
        return dataset
        
    def _extract_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Extract features from a single bar for ML model"""
        features = {}
        
        # Current bar
        current = df.iloc[idx]
        
        # Get previous bars
        lookback_10 = df.iloc[idx-10:idx]
        lookback_20 = df.iloc[idx-20:idx]
        lookback_50 = df.iloc[idx-50:idx]
        
        # Price action features
        features['close'] = current['close']
        features['high_low_range'] = (current['high'] - current['low']) / current['close']
        features['body_size'] = abs(current['close'] - current['open']) / current['close']
        features['upper_wick'] = (current['high'] - max(current['open'], current['close'])) / current['close']
        features['lower_wick'] = (min(current['open'], current['close']) - current['low']) / current['close']
        
        # Trend features
        features['sma20_dist'] = (current['close'] - current['sma_20']) / current['close']
        features['sma50_dist'] = (current['close'] - current['sma_50']) / current['close']
        features['sma200_dist'] = (current['close'] - current['sma_200']) / current['close']
        features['ema_fast_slope'] = (current['ema_fast'] - lookback_10['ema_fast'].iloc[0]) / lookback_10['ema_fast'].iloc[0]
        features['ema_slow_slope'] = (current['ema_slow'] - lookback_10['ema_slow'].iloc[0]) / lookback_10['ema_slow'].iloc[0]
        features['ema_distance'] = (current['ema_fast'] - current['ema_slow']) / current['close']
        features['above_sma20'] = 1 if current['close'] > current['sma_20'] else 0
        features['above_sma50'] = 1 if current['close'] > current['sma_50'] else 0
        features['above_sma200'] = 1 if current['close'] > current['sma_200'] else 0
        
        # Volatility features
        features['atr'] = current['atr']
        features['atr_pct'] = current['atr'] / current['close']
        features['bb_width'] = (current['bb_upper'] - current['bb_lower']) / current['bb_middle']
        features['bb_position'] = (current['close'] - current['bb_lower']) / (current['bb_upper'] - current['bb_lower']) if (current['bb_upper'] - current['bb_lower']) > 0 else 0.5
        
        # 10-day volatility change
        avg_atr_10d = lookback_10['atr'].mean()
        prev_avg_atr_10d = df.iloc[idx-20:idx-10]['atr'].mean()
        features['volatility_change_10d'] = avg_atr_10d / prev_avg_atr_10d if prev_avg_atr_10d > 0 else 1.0
        
        # Range vs trend features
        high_20d = lookback_20['high'].max()
        low_20d = lookback_20['low'].min()
        range_20d = (high_20d - low_20d) / low_20d
        features['range_20d'] = range_20d
        
        # Price movement
        features['price_change_10d'] = (current['close'] - lookback_10['close'].iloc[0]) / lookback_10['close'].iloc[0]
        features['price_change_20d'] = (current['close'] - lookback_20['close'].iloc[0]) / lookback_20['close'].iloc[0]
        
        # Oscillators
        features['rsi'] = current['rsi']
        features['rsi_slope'] = (current['rsi'] - lookback_10['rsi'].iloc[0])
        features['stoch_k'] = current['stoch_k']
        features['stoch_d'] = current['stoch_d']
        features['stoch_diff'] = current['stoch_k'] - current['stoch_d']
        
        # MACD features
        features['macd'] = current['macd']
        features['macd_signal'] = current['macd_signal']
        features['macd_hist'] = current['macd_hist']
        features['macd_hist_slope'] = (current['macd_hist'] - lookback_10['macd_hist'].iloc[0])
        
        # Volume features
        features['volume'] = current['volume']
        avg_volume_20d = lookback_20['volume'].mean()
        features['volume_ratio_20d'] = current['volume'] / avg_volume_20d if avg_volume_20d > 0 else 1.0
        
        # Detect recent crosses
        features['recent_ema_cross_up'] = 1 if any(lookback_10['ema_cross'] == 1) else 0
        features['recent_ema_cross_down'] = 1 if any(lookback_10['ema_cross'] == -1) else 0
        features['recent_macd_cross_up'] = 1 if any(lookback_10['macd_cross'] == 1) else 0
        features['recent_macd_cross_down'] = 1 if any(lookback_10['macd_cross'] == -1) else 0
        
        return features
    
    def train_regime_classifier(self, dataset: pd.DataFrame) -> None:
        """
        Train ML model to classify market regimes based on technical indicators.
        
        Args:
            dataset: DataFrame with features and regime labels
        """
        logger.info("Training market regime classifier model")
        
        if dataset.empty:
            logger.error("Empty training dataset")
            return
        
        # Prepare features and target
        X = dataset.drop(['asset', 'timeframe', 'timestamp', 'regime', 'win_rate', 'avg_return'], axis=1)
        y = dataset['regime']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        logger.info(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Define models to try
        models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Parameters for grid search
        params = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
        }
        
        best_model = None
        best_score = 0
        best_model_name = ""
        
        # Find best model
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Grid search
            grid = GridSearchCV(model, params[model_name], cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)
            
            # Evaluate
            y_pred = grid.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"{model_name} accuracy: {accuracy:.4f}")
            logger.info(f"Best parameters: {grid.best_params_}")
            
            # Save classification report
            with open(f"{self.output_dir}/{model_name}_classification_report.txt", 'w') as f:
                f.write(classification_report(y_test, y_pred))
            
            # Save confusion matrix visualization
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=sorted(y.unique()),
                       yticklabels=sorted(y.unique()))
            plt.title(f'{model_name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{model_name}_confusion_matrix.png")
            plt.close()
            
            # Check if this is the best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = grid.best_estimator_
                best_model_name = model_name
        
        # Save best model
        if best_model:
            logger.info(f"Best model: {best_model_name} with accuracy {best_score:.4f}")
            joblib.dump(best_model, f"{self.model_dir}/regime_classifier.joblib")
            
            # Save feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_model.feature_importances_
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
                plt.title('Feature Importance')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/feature_importance.png")
                plt.close()
                
                feature_importance.to_csv(f"{self.output_dir}/feature_importance.csv", index=False)
    
    def train_performance_predictor(self, dataset: pd.DataFrame) -> None:
        """
        Train ML model to predict trading performance based on regime and features.
        
        Args:
            dataset: DataFrame with features and performance metrics
        """
        logger.info("Training performance predictor model")
        
        if dataset.empty:
            logger.error("Empty training dataset")
            return
        
        # Prepare features and targets
        X = dataset.drop(['asset', 'timeframe', 'timestamp', 'win_rate', 'avg_return'], axis=1)
        y_win_rate = dataset['win_rate']
        y_avg_return = dataset['avg_return']
        
        # One-hot encode regime
        X = pd.get_dummies(X, columns=['regime'], drop_first=False)
        
        # Split data
        X_train, X_test, y_wr_train, y_wr_test = train_test_split(X, y_win_rate, test_size=0.2, random_state=42)
        _, _, y_ar_train, y_ar_test = train_test_split(X, y_avg_return, test_size=0.2, random_state=42)
        
        # Train win rate model
        wr_model = GradientBoostingClassifier(random_state=42)
        wr_model.fit(X_train, y_wr_train)
        
        # Train avg return model
        ar_model = GradientBoostingClassifier(random_state=42)
        ar_model.fit(X_train, y_ar_train)
        
        # Save models
        joblib.dump(wr_model, f"{self.model_dir}/win_rate_predictor.joblib")
        joblib.dump(ar_model, f"{self.model_dir}/avg_return_predictor.joblib")
        
        logger.info("Performance predictor models trained and saved")
    
    def optimize_thresholds(self, dataset: pd.DataFrame) -> RegimeThresholds:
        """
        Optimize regime detection thresholds based on trading performance.
        Uses ML models to find optimal thresholds for different market conditions.
        
        Args:
            dataset: DataFrame with features and performance data
            
        Returns:
            Optimized thresholds
        """
        logger.info("Optimizing regime detection thresholds")
        
        optimized = RegimeThresholds()
        
        if dataset.empty:
            logger.warning("Empty dataset, using default thresholds")
            return optimized
        
        try:
            # Group by regime
            regime_groups = dataset.groupby('regime')
            
            # Process each regime
            for regime, group in regime_groups:
                # Skip if too few samples
                if len(group) < 20:
                    continue
                    
                logger.info(f"Optimizing thresholds for {regime} regime with {len(group)} samples")
                
                # Filter to successful periods (high win rate or good returns)
                good_periods = group[(group['win_rate'] > 60) | (group['avg_return'] > 0)]
                
                if len(good_periods) < 10:
                    logger.warning(f"Too few good periods for {regime}, skipping optimization")
                    continue
                
                # Calculate optimal thresholds based on successful periods
                if regime in ['bullish_trending', 'bearish_trending']:
                    # Optimize trend thresholds
                    optimized.trend_ema_distance = good_periods['ema_distance'].abs().quantile(0.5)
                    optimized.trend_slope_threshold = good_periods['ema_fast_slope'].abs().quantile(0.3)
                    optimized.price_sma_distance = good_periods['sma20_dist'].abs().quantile(0.5)
                    
                elif regime in ['high_volatility']:
                    # Optimize volatility thresholds
                    optimized.high_volatility_atr_pct = good_periods['atr_pct'].quantile(0.3) * 100
                    optimized.bb_width_high = good_periods['bb_width'].quantile(0.3)
                    
                elif regime in ['low_volatility']:
                    # Optimize low volatility thresholds
                    optimized.low_volatility_atr_pct = good_periods['atr_pct'].quantile(0.7) * 100
                    optimized.bb_width_low = good_periods['bb_width'].quantile(0.7)
                    
                elif regime in ['neutral_ranging']:
                    # Optimize neutral thresholds
                    optimized.neutral_rsi_min = good_periods['rsi'].quantile(0.25)
                    optimized.neutral_rsi_max = good_periods['rsi'].quantile(0.75)
                    
                elif regime in ['bullish_breakout', 'bearish_breakout']:
                    # Optimize breakout thresholds
                    optimized.volume_surge_ratio = good_periods['volume_ratio_20d'].quantile(0.6)
                    optimized.bb_position_threshold = good_periods['bb_position'].quantile(0.7)
        
        except Exception as e:
            logger.error(f"Error optimizing thresholds: {e}", exc_info=True)
            
        # Save optimized thresholds
        with open(f"{self.output_dir}/optimized_thresholds.json", 'w') as f:
            json.dump(optimized.to_dict(), f, indent=2)
            
        logger.info("Threshold optimization completed")
        return optimized
    
    def apply_optimized_thresholds(self, thresholds: RegimeThresholds) -> None:
        """
        Apply optimized thresholds to the regime detector.
        
        Args:
            thresholds: RegimeThresholds object with optimized values
        """
        logger.info("Applying optimized thresholds to regime detector")
        
        # TODO: Update this when the MarketRegimeDetector gets support for custom thresholds
        # For now we'll just log the thresholds that would be applied
        
        logger.info("Optimized thresholds would be applied as follows:")
        logger.info(f"Trend EMA distance: {thresholds.trend_ema_distance:.4f}")
        logger.info(f"Trend slope threshold: {thresholds.trend_slope_threshold:.4f}")
        logger.info(f"Price-SMA distance: {thresholds.price_sma_distance:.4f}")
        logger.info(f"High volatility ATR %: {thresholds.high_volatility_atr_pct:.2f}%")
        logger.info(f"Low volatility ATR %: {thresholds.low_volatility_atr_pct:.2f}%")
        logger.info(f"Bollinger Band width high: {thresholds.bb_width_high:.4f}")
        logger.info(f"Bollinger Band width low: {thresholds.bb_width_low:.4f}")
        logger.info(f"Overbought RSI: {thresholds.overbought_rsi:.1f}")
        logger.info(f"Oversold RSI: {thresholds.oversold_rsi:.1f}")
        logger.info(f"Neutral RSI range: {thresholds.neutral_rsi_min:.1f}-{thresholds.neutral_rsi_max:.1f}")
        logger.info(f"Volume surge ratio: {thresholds.volume_surge_ratio:.2f}")
        logger.info(f"BB position threshold: {thresholds.bb_position_threshold:.2f}")
        
    def run_optimization_pipeline(
        self,
        assets: List[AssetSymbol],
        timeframes: List[Timeframe],
        lookback_days: int = 365
    ) -> RegimeThresholds:
        """
        Run the complete optimization pipeline:
        1. Create training dataset
        2. Train ML models
        3. Optimize thresholds
        4. Apply optimized thresholds
        
        Args:
            assets: List of assets to analyze
            timeframes: List of timeframes to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Optimized thresholds
        """
        logger.info(f"Starting regime optimization pipeline with {len(assets)} assets, {len(timeframes)} timeframes, {lookback_days} days lookback")
        
        # Step 1: Create training dataset
        dataset = self.create_training_dataset(assets, timeframes, lookback_days)
        
        # Step 2: Train ML models
        self.train_regime_classifier(dataset)
        self.train_performance_predictor(dataset)
        
        # Step 3: Optimize thresholds
        optimized_thresholds = self.optimize_thresholds(dataset)
        
        # Step 4: Apply optimized thresholds
        self.apply_optimized_thresholds(optimized_thresholds)
        
        logger.info("Optimization pipeline completed successfully")
        
        return optimized_thresholds

def run_optimizer_demo():
    """Run a demo of the regime optimizer"""
    from backend.agents.market_data_agent import MarketDataAgent
    from backend.agents.signal_scanner_agent import SignalScannerAgent
    from backend.agents.risk_manager_agent import RiskManagerAgent
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agents
    market_data_agent = MarketDataAgent()
    signal_scanner = SignalScannerAgent(market_data_agent)
    risk_manager = RiskManagerAgent()
    backtester = Backtester(market_data_agent, signal_scanner, risk_manager)
    
    # Create optimizer
    optimizer = RegimeOptimizer(market_data_agent, backtester)
    
    # Define assets and timeframes for optimization
    # Use a limited set for the demo to avoid long runtimes
    assets = [
        AssetSymbol.BTC_USD,
        AssetSymbol.SPY
    ]
    
    timeframes = [
        Timeframe.DAILY
    ]
    
    # Run optimization with shorter lookback for the demo
    optimized_thresholds = optimizer.run_optimization_pipeline(assets, timeframes, lookback_days=90)
    
    logger.info(f"Optimizer demo completed. Output saved to {optimizer.output_dir}")
    
if __name__ == "__main__":
    run_optimizer_demo()
