"""
Base Strategy Module for Trading SaaS

This module provides a base class for all trading strategies in the system.
Each strategy will inherit from this class and implement its own signal generation logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pandas as pd

from shared_types.models import (
    AssetSymbol, Timeframe, TradingSignalModel, 
    HistoricalDataModel, SignalType
)
from backend.agents.market_regime_detector import MarketRegimeDetector, MarketRegimeType, MarketRegime

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This abstract class defines the interface that all strategy implementations
    must follow, ensuring consistent behavior across different strategies.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the strategy with its name and description.
        
        Args:
            name: The name of the strategy
            description: A brief description of how the strategy works
        """
        self.name = name
        self.description = description
        self.parameters: Dict[str, Any] = {}
        self.regime_detector = MarketRegimeDetector()
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set strategy parameters.
        
        Args:
            parameters: Dictionary of parameter name-value pairs
        """
        self.parameters = parameters
        
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current strategy parameters.
        
        Returns:
            Dictionary of current parameters
        """
        return self.parameters
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for this strategy.
        
        Returns:
            Dictionary of default parameters
        """
        return {}
    
    def get_regime_adapted_parameters(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get parameters adapted to the current market regime.
        
        Args:
            regime: The current market regime
            
        Returns:
            Dictionary of parameters adapted for the regime
        """
        # Base implementation - subclasses should override this
        return self.get_default_parameters()
    
    @abstractmethod
    def generate_signals(self, 
                         asset: AssetSymbol, 
                         timeframe: Timeframe, 
                         historical_data: HistoricalDataModel) -> List[TradingSignalModel]:
        """
        Generate trading signals for the given asset and timeframe.
        
        Args:
            asset: The asset to generate signals for
            timeframe: The timeframe to use
            historical_data: Historical price data
            
        Returns:
            A list of generated trading signals (may be empty)
        """
        pass
    
    def preprocess_data(self, historical_data: HistoricalDataModel) -> pd.DataFrame:
        """
        Preprocess historical data for strategy use.
        
        Args:
            historical_data: Raw historical data
            
        Returns:
            Preprocessed pandas DataFrame
        """
        # Convert candles to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume
            } for candle in historical_data.candles
        ])
        
        # Make sure the data is sorted by timestamp
        df = df.sort_values("timestamp")
        
        return df
    
    def detect_regime(self, 
                     df: pd.DataFrame, 
                     asset: AssetSymbol, 
                     timeframe: Timeframe) -> MarketRegime:
        """
        Detect the current market regime.
        
        Args:
            df: Preprocessed DataFrame of historical data
            asset: The asset
            timeframe: The timeframe
            
        Returns:
            Detected market regime
        """
        return self.regime_detector.detect_regime(df, asset, timeframe)
    
    def is_suitable_for_regime(self, regime_type: MarketRegimeType) -> bool:
        """
        Determine if this strategy is suitable for the given market regime.
        
        Args:
            regime_type: The market regime type
            
        Returns:
            True if the strategy is suitable, False otherwise
        """
        # Base implementation - subclasses should override this
        return True
    
    def get_confidence_for_regime(self, regime: MarketRegime) -> float:
        """
        Get the confidence score for this strategy in the given market regime.
        
        Args:
            regime: The market regime
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        # Base implementation - subclasses should override this
        if self.is_suitable_for_regime(regime.regime_type):
            return 0.5 + (regime.strength * 0.1)  # Basic scaling with regime strength
        else:
            return 0.3  # Lower baseline for unsuitable regimes
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} - {self.description}"
