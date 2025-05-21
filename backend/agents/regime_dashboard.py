"""
Market Regime Dashboard Module for the Trading SaaS platform.
Provides visualization capabilities for market regimes and adaptive trading strategies.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta, timezone
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import io
import base64
from pathlib import Path

from shared_types import AssetSymbol, Timeframe
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.market_regime_detector import MarketRegimeDetector, MarketRegimeType
from backend.agents.backtesting import Backtester, BacktestResult
from backend.agents.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class RegimeDashboard:
    """
    Class for generating market regime visualizations and analytics dashboard.
    """
    
    def __init__(
        self,
        market_data_agent: MarketDataAgent,
        regime_detector: Optional[MarketRegimeDetector] = None,
        output_dir: str = "dashboard_output"
    ):
        self.market_data_agent = market_data_agent
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.tech_indicators = TechnicalIndicators()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Color mapping for different regimes (consistent with backtesting)
        self.regime_colors = {
            'bullish_trending': '#90EE90',  # Light green
            'bearish_trending': '#FFA07A',  # Light salmon
            'neutral_ranging': '#ADD8E6',  # Light blue
            'high_volatility': '#FFFF99',  # Light yellow
            'low_volatility': '#D8BFD8',  # Thistle
            'bullish_breakout': '#7FFF00',  # Chartreuse
            'bearish_breakout': '#FF6347',  # Tomato
            'unknown': '#E0E0E0'   # Light gray
        }
        
    def generate_regime_overview(
        self, 
        assets: List[AssetSymbol], 
        timeframes: List[Timeframe],
        lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Generate a multi-asset, multi-timeframe market regime overview.
        
        Args:
            assets: List of assets to analyze
            timeframes: List of timeframes to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Dict with overview data and base64-encoded visualizations
        """
        logger.info(f"Generating market regime overview for {len(assets)} assets across {len(timeframes)} timeframes")
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Store results
        overview = {
            'generated_at': end_date.isoformat(),
            'lookback_days': lookback_days,
            'assets': [asset.value for asset in assets],
            'timeframes': [tf.value for tf in timeframes],
            'current_regimes': {},
            'regime_history': {},
            'regime_heatmap': None,
            'regime_correlation': None,
            'assets_by_regime': {}
        }
        
        # Initialize regime counter
        all_regimes = [regime.value for regime in MarketRegimeType]
        regime_counts = {regime: 0 for regime in all_regimes}
        
        # Process each asset and timeframe
        for asset in assets:
            overview['current_regimes'][asset.value] = {}
            overview['regime_history'][asset.value] = {}
            
            for timeframe in timeframes:
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
                
                # Detect current regime
                current_regime = self.regime_detector.detect_regime(df, asset, timeframe)
                
                # Store current regime info
                overview['current_regimes'][asset.value][timeframe.value] = {
                    'regime_type': current_regime.regime_type.value,
                    'strength': current_regime.strength,
                    'timestamp': current_regime.timestamp.isoformat(),
                    'adjusted_parameters': current_regime.get_adjusted_parameters()
                }
                
                # Increment regime counter
                regime_counts[current_regime.regime_type.value] += 1
                
                # Track assets by regime type
                if current_regime.regime_type.value not in overview['assets_by_regime']:
                    overview['assets_by_regime'][current_regime.regime_type.value] = []
                overview['assets_by_regime'][current_regime.regime_type.value].append(
                    f"{asset.value} ({timeframe.value})"
                )
                
                # Get regime history
                regime_history = []
                # Process in chunks to avoid memory issues with large datasets
                chunk_size = 50  # Process 50 bars at a time for regime detection
                overlap = 10     # Overlap between chunks for continuity
                
                for i in range(0, len(df) - overlap, chunk_size - overlap):
                    end_idx = min(i + chunk_size, len(df))
                    chunk = df.iloc[i:end_idx].copy()
                    
                    if len(chunk) < 30:  # Need enough data for reliable detection
                        continue
                        
                    # Detect regime for this chunk
                    regime = self.regime_detector.detect_regime(chunk, asset, timeframe)
                    
                    # Add to history
                    regime_history.append({
                        'timestamp': chunk.index[-1].isoformat(),
                        'regime_type': regime.regime_type.value,
                        'strength': regime.strength
                    })
                
                # Store regime history
                overview['regime_history'][asset.value][timeframe.value] = regime_history
        
        # Generate heatmap visualization
        overview['regime_heatmap'] = self._generate_regime_heatmap(overview['current_regimes'])
        
        # Generate regime distribution visualization
        overview['regime_distribution'] = self._generate_regime_distribution(regime_counts)
        
        # Save overview to file
        with open(f"{self.output_dir}/regime_overview.json", 'w') as f:
            json.dump(overview, f, indent=2)
            
        logger.info(f"Market regime overview generated and saved to {self.output_dir}/regime_overview.json")
        
        return overview
        
    def generate_asset_regime_chart(
        self,
        asset: AssetSymbol,
        timeframe: Timeframe,
        lookback_days: int = 90
    ) -> Optional[str]:
        """
        Generate a chart showing price action with market regime background for a specific asset.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe
            lookback_days: Number of days to look back
            
        Returns:
            Base64-encoded image
        """
        logger.info(f"Generating market regime chart for {asset.value} {timeframe.value}")
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Fetch historical data
        historical_data = self.market_data_agent.fetch_historical_data(
            asset=asset,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not historical_data or not historical_data.candles:
            logger.warning(f"No data for {asset.value} {timeframe.value}")
            return None
        
        # Convert to dataframe
        df = pd.DataFrame([candle.model_dump() for candle in historical_data.candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate indicators
        df = self.tech_indicators.calculate_all(df)
        
        # Detect market regimes
        regime_periods = []
        current_regime = None
        current_regime_start = None
        current_strength = 0.0
        
        # Process in chunks to avoid memory issues with large datasets
        chunk_size = 50  # Process 50 bars at a time for regime detection
        overlap = 10     # Overlap between chunks for continuity
        
        for i in range(0, len(df) - overlap, chunk_size - overlap):
            end_idx = min(i + chunk_size, len(df))
            chunk = df.iloc[i:end_idx].copy()
            
            if len(chunk) < 30:  # Need enough data for reliable detection
                continue
                
            # Detect regime for this chunk
            regime = self.regime_detector.detect_regime(chunk, asset, timeframe)
            
            # Process the chunk (excluding overlap with next chunk)
            process_end = min(end_idx, i + chunk_size - overlap)
            
            for j in range(i, process_end):
                bar_date = df.index[j]
                
                # Check if regime has changed
                if current_regime != regime.regime_type.value or abs(current_strength - regime.strength) > 0.2:
                    # If we had a previous regime, record its end
                    if current_regime and current_regime_start:
                        regime_periods.append({
                            'regime_type': current_regime,
                            'start_date': current_regime_start,
                            'end_date': bar_date,
                            'strength': current_strength
                        })
                    
                    # Start a new regime period
                    current_regime = regime.regime_type.value
                    current_regime_start = bar_date
                    current_strength = regime.strength
        
        # Add the final regime period if exists
        if current_regime and current_regime_start and current_regime_start < df.index[-1]:
            regime_periods.append({
                'regime_type': current_regime,
                'start_date': current_regime_start,
                'end_date': df.index[-1],
                'strength': current_strength
            })
        
        # Generate the chart
        try:
            # Create figure with price and indicators
            fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
            fig.suptitle(f'{asset.value} {timeframe.value} - Market Regime Analysis', fontsize=16)
            
            # Calculate technical indicators if they don't exist
            if 'sma_20' not in df.columns:
                # Add basic indicators
                df['sma_20'] = self.tech_indicators.calculate_sma(df['close'], period=20)
                df['sma_50'] = self.tech_indicators.calculate_sma(df['close'], period=50)
                df['sma_200'] = self.tech_indicators.calculate_sma(df['close'], period=200)
                
                # Add Bollinger Bands using talib directly
                try:
                    import talib
                    upper, middle, lower = talib.BBANDS(
                        df['close'],
                        timeperiod=20,
                        nbdevup=2.0,
                        nbdevdn=2.0
                    )
                    df['bb_upper'] = upper
                    df['bb_middle'] = middle
                    df['bb_lower'] = lower
                except Exception as e:
                    logger.warning(f"Error calculating Bollinger Bands: {e}")
                    # Fallback to simple bands based on SMA
                    df['bb_middle'] = df['sma_20']
                    std = df['close'].rolling(window=20).std()
                    df['bb_upper'] = df['bb_middle'] + (std * 2)
                    df['bb_lower'] = df['bb_middle'] - (std * 2)
            
            # Plot 1: Price with regimes
            ax1 = axs[0]
            
            # Plot candlesticks
            width = 0.6
            width2 = width * 0.8
            
            up = df[df.close >= df.open]
            down = df[df.close < df.open]
            
            # Plot up candles
            ax1.bar(up.index, up.close-up.open, width, bottom=up.open, color='green', alpha=0.5)
            ax1.bar(up.index, up.high-up.close, width2, bottom=up.close, color='green', alpha=0.5)
            ax1.bar(up.index, up.low-up.open, width2, bottom=up.open, color='green', alpha=0.5)
            
            # Plot down candles
            ax1.bar(down.index, down.close-down.open, width, bottom=down.open, color='red', alpha=0.5)
            ax1.bar(down.index, down.high-down.open, width2, bottom=down.open, color='red', alpha=0.5)
            ax1.bar(down.index, down.low-down.close, width2, bottom=down.close, color='red', alpha=0.5)
            
            # Add moving averages (check if they exist first)
            if 'sma_20' in df.columns:
                ax1.plot(df.index, df['sma_20'], label='SMA 20', color='blue', alpha=0.7)
            if 'sma_50' in df.columns:
                ax1.plot(df.index, df['sma_50'], label='SMA 50', color='orange', alpha=0.7)
            if 'sma_200' in df.columns:
                ax1.plot(df.index, df['sma_200'], label='SMA 200', color='purple', alpha=0.7)
            
            # Add Bollinger Bands (check if they exist first)
            if 'bb_upper' in df.columns and 'bb_middle' in df.columns and 'bb_lower' in df.columns:
                ax1.plot(df.index, df['bb_upper'], color='gray', linestyle='--', alpha=0.4)
                ax1.plot(df.index, df['bb_middle'], color='gray', linestyle='-', alpha=0.4)
                ax1.plot(df.index, df['bb_lower'], color='gray', linestyle='--', alpha=0.4)
            
            # Add regime backgrounds
            for period in regime_periods:
                start_date = period['start_date']
                end_date = period['end_date']
                regime_type = period['regime_type']
                strength = period.get('strength', 0.5)
                
                # Get color for regime and adjust alpha based on strength
                color = self.regime_colors.get(regime_type, '#E0E0E0')
                alpha = 0.15 + (strength * 0.2)  # Alpha varies from 0.15 to 0.35 based on strength
                
                # Add colored background
                ax1.axvspan(start_date, end_date, alpha=alpha, color=color)
                
                # Try to add label in the middle of the regime period if it's long enough
                span_days = (end_date - start_date).days
                if span_days > 5:  # Only add text for longer periods
                    mid_point = start_date + (end_date - start_date) / 2
                    y_pos = df['low'].min() + (df['high'].max() - df['low'].min()) * 0.05  # Bottom
                    ax1.text(mid_point, y_pos, regime_type.replace('_', ' ').title(), 
                             ha='center', va='bottom', fontsize=8, alpha=0.7, rotation=0)
            
            ax1.set_ylabel('Price')
            ax1.grid(alpha=0.2)
            ax1.legend(loc='upper left')
            
            # Plot 2: RSI with overbought/oversold lines
            ax2 = axs[1]
            ax2.plot(df.index, df['rsi'], color='blue')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI')
            ax2.grid(alpha=0.2)
            
            # Plot 3: MACD
            ax3 = axs[2]
            ax3.plot(df.index, df['macd'], color='blue', label='MACD')
            ax3.plot(df.index, df['macd_signal'], color='red', label='Signal')
            
            # MACD histogram
            positive = df[df['macd_hist'] >= 0]
            negative = df[df['macd_hist'] < 0]
            ax3.bar(positive.index, positive['macd_hist'], color='green', alpha=0.5, width=width)
            ax3.bar(negative.index, negative['macd_hist'], color='red', alpha=0.5, width=width)
            
            ax3.set_ylabel('MACD')
            ax3.grid(alpha=0.2)
            ax3.legend(loc='upper left')
            
            # Format date axis
            for ax in axs:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            plt.tight_layout()
            
            # Save to file
            file_path = f"{self.output_dir}/{asset.value}_{timeframe.value}_regime_chart.png"
            plt.savefig(file_path, dpi=100)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            b64_image = base64.b64encode(image_png).decode()
            
            logger.info(f"Market regime chart for {asset.value} {timeframe.value} saved to {file_path}")
            
            return b64_image
            
        except Exception as e:
            logger.error(f"Error generating regime chart: {e}", exc_info=True)
            return None
    
    def _generate_regime_heatmap(self, current_regimes: Dict[str, Dict[str, Dict[str, Any]]]) -> Optional[str]:
        """Generate heatmap of current market regimes for all assets and timeframes"""
        try:
            # Extract data for heatmap
            assets = list(current_regimes.keys())
            if not assets:
                return None
                
            # Get all timeframes from the first asset
            first_asset = assets[0]
            timeframes = list(current_regimes[first_asset].keys())
            if not timeframes:
                return None
                
            # Create matrix of regime strengths
            data = []
            # Map of regime types to numeric values for coloring
            regime_values = {
                'bullish_trending': 2,
                'bullish_breakout': 3,
                'neutral_ranging': 0,
                'bearish_trending': -2,
                'bearish_breakout': -3,
                'high_volatility': 1,
                'low_volatility': -1,
                'unknown': 0
            }
            
            for asset in assets:
                row = []
                for tf in timeframes:
                    if tf in current_regimes[asset]:
                        # Get regime type and strength
                        regime_type = current_regimes[asset][tf]['regime_type']
                        strength = current_regimes[asset][tf]['strength']
                        
                        # Calculate value (regime value * strength)
                        value = regime_values.get(regime_type, 0) * strength
                        row.append(value)
                    else:
                        row.append(0)  # No data
                data.append(row)
                
            # Create heatmap
            plt.figure(figsize=(12, 8))
            
            # Create custom colormap: red for bearish, green for bullish, yellow for volatility
            cmap = sns.diverging_palette(10, 133, as_cmap=True)
            
            # Plot heatmap
            ax = sns.heatmap(data, annot=False, cmap=cmap, center=0,
                        xticklabels=timeframes, yticklabels=assets)
            
            # Annotate cells with regime names
            for i, asset in enumerate(assets):
                for j, tf in enumerate(timeframes):
                    if tf in current_regimes[asset]:
                        regime_type = current_regimes[asset][tf]['regime_type']
                        strength = current_regimes[asset][tf]['strength']
                        
                        # Format text
                        text = regime_type.replace('_', ' ').title()
                        
                        # Text color (white for dark backgrounds, black for light)
                        value = abs(regime_values.get(regime_type, 0) * strength)
                        text_color = 'white' if value > 1.5 else 'black'
                        
                        # Add text
                        ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                                color=text_color, fontsize=8)
            
            plt.title('Market Regime Heatmap')
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            logger.error(f"Error generating regime heatmap: {e}", exc_info=True)
            return None
            
    def _generate_regime_distribution(self, regime_counts: Dict[str, int]) -> Optional[str]:
        """Generate pie chart of regime distribution"""
        try:
            # Filter out regimes with zero count
            filtered_counts = {k: v for k, v in regime_counts.items() if v > 0}
            if not filtered_counts:
                return None
                
            # Create pie chart
            plt.figure(figsize=(10, 8))
            
            # Use consistent colors
            colors = [self.regime_colors.get(regime, '#E0E0E0') for regime in filtered_counts.keys()]
            
            # Plot pie chart
            plt.pie(
                filtered_counts.values(),
                labels=[regime.replace('_', ' ').title() for regime in filtered_counts.keys()],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                shadow=False
            )
            plt.axis('equal')
            plt.title('Market Regime Distribution')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()
            
            return base64.b64encode(image_png).decode()
            
        except Exception as e:
            logger.error(f"Error generating regime distribution: {e}", exc_info=True)
            return None

def generate_dashboard_demo():
    """Run a demo of the regime dashboard"""
    from backend.agents.market_data_agent import MarketDataAgent
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agents
    market_data_agent = MarketDataAgent()
    regime_detector = MarketRegimeDetector()
    
    # Create dashboard
    dashboard = RegimeDashboard(market_data_agent, regime_detector)
    
    # Generate overview for major assets
    assets = [
        AssetSymbol.BTC_USD,
        AssetSymbol.ETH_USD,
        AssetSymbol.SPY,
        AssetSymbol.QQQ,
        AssetSymbol.AAPL
    ]
    
    timeframes = [
        Timeframe.DAILY,
        Timeframe.HOURLY_4,
        Timeframe.HOURLY_1
    ]
    
    # Generate overview
    overview = dashboard.generate_regime_overview(assets, timeframes, lookback_days=60)
    
    # Generate individual charts
    for asset in assets:
        dashboard.generate_asset_regime_chart(asset, Timeframe.DAILY, lookback_days=90)
    
    logger.info(f"Dashboard demo completed. Output saved to {dashboard.output_dir}")
    
if __name__ == "__main__":
    generate_dashboard_demo()
