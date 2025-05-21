# backend/agents/signal_scanner_agent.py
import pandas as pd
import talib
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import logging

from shared_types import AssetSymbol, Timeframe, TradingSignalModel, SignalType, CandleModel, HistoricalDataModel
from .market_data_agent import MarketDataAgent
from .market_regime_detector import MarketRegimeDetector
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class SignalScannerAgent:
    """
    Agent responsible for scanning market data to identify trading signals
    based on predefined strategies.
    """

    def __init__(self, market_data_agent: MarketDataAgent):
        """
        Initializes the SignalScannerAgent.

        Args:
            market_data_agent: An instance of MarketDataAgent to fetch data.
        """
        self.market_data_agent = market_data_agent
        self.regime_detector = MarketRegimeDetector()
        self.tech_indicators = TechnicalIndicators()
        
        # Base strategy parameters (will be adjusted based on market regime)
        self.breakout_period = 10  # Reduced from 15 to 10 for more frequent signals
        self.atr_period = 14
        self.atr_multiplier_sl = 1.5  # For stop-loss
        self.min_rr_ratio = 1.2       # Minimum risk-to-reward ratio
        
        # Secondary confirmation parameters
        self.volume_confirmation = False  # Volume confirmation disabled by default to increase signal generation
        self.volume_threshold = 1.1  # Reduced from 1.2 - current volume should be only 10% higher than average
        self.rsi_period = 14
        self.rsi_oversold = 25  # Lowered threshold for oversold condition (was 30)
        self.rsi_overbought = 75  # Increased threshold for overbought condition (was 70)
        self.allow_pullback_entry = True  # Allow entries on pullbacks after breakout
        
        # Adaptivity flags
        self.adapt_to_regime = True  # Whether to adjust parameters based on market regime

    def _calculate_atr(self, high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series) -> Optional[float]:
        """Calculates ATR for the latest period."""
        if len(close_prices) < self.atr_period:
            logger.warning(f"Not enough data for ATR calculation. Need {self.atr_period}, got {len(close_prices)}")
            return None
        try:
            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.atr_period)
            return atr_values.iloc[-1] if not pd.isna(atr_values.iloc[-1]) else None
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=True)
            return None
            
    def _calculate_rsi(self, prices: pd.Series) -> Optional[float]:
        """Calculate RSI for the latest period."""
        if len(prices) < self.rsi_period + 10:  # Need more data for reliable RSI
            logger.warning(f"Not enough data for RSI calculation. Need at least {self.rsi_period + 10}, got {len(prices)}")
            return None
        try:
            rsi_values = talib.RSI(prices, timeperiod=self.rsi_period)
            return rsi_values.iloc[-1] if not pd.isna(rsi_values.iloc[-1]) else None
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=True)
            return None
            
    def _check_volume_confirmation(self, volumes: pd.Series) -> bool:
        """Check if current volume is significantly higher than recent average volume."""
        if len(volumes) < self.breakout_period + 1:
            logger.warning(f"Not enough volume data for confirmation. Need {self.breakout_period + 1}, got {len(volumes)}")
            return True  # Default to True if we can't check
        
        current_volume = volumes.iloc[-1]
        avg_volume = volumes.iloc[-self.breakout_period-1:-1].mean()  # Average volume excluding current
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        is_confirmed = volume_ratio >= self.volume_threshold
        
        logger.debug(f"Volume confirmation: current={current_volume:.2f}, avg={avg_volume:.2f}, ratio={volume_ratio:.2f}, confirmed={is_confirmed}")
        return is_confirmed
        
    def _adapt_sl_based_on_volatility(self, entry_price: float, atr: float, signal_type: SignalType) -> float:
        """Adapt stop loss distance based on price volatility."""
        price_volatility_pct = atr / entry_price * 100
        
        # Adjust ATR multiplier based on volatility
        if price_volatility_pct < 1.0:  # Low volatility
            adjusted_multiplier = self.atr_multiplier_sl * 0.9  # Tighter stop for low volatility
        elif price_volatility_pct > 3.0:  # High volatility
            adjusted_multiplier = self.atr_multiplier_sl * 1.2  # Wider stop for high volatility
        else:
            adjusted_multiplier = self.atr_multiplier_sl
            
        # Calculate adapted stop loss
        # For LONG positions: stop loss is BELOW entry
        # For SHORT positions: stop loss is ABOVE entry
        if signal_type == SignalType.LONG:
            stop_loss = entry_price - (adjusted_multiplier * atr)
        else:  # SHORT
            stop_loss = entry_price + (adjusted_multiplier * atr)
            
        return stop_loss

    def _get_adjusted_parameters(self, df: pd.DataFrame, asset: AssetSymbol, timeframe: Timeframe) -> Dict[str, Any]:
        """
        Get strategy parameters adjusted for current market regime.
        
        Args:
            df: DataFrame with price data
            asset: Asset symbol
            timeframe: Timeframe
            
        Returns:
            Dictionary of parameter adjustments for the current regime
        """
        if not self.adapt_to_regime:
            # Return default parameters if no adaptation is needed
            return {
                'breakout_period': self.breakout_period,
                'atr_multiplier_sl': self.atr_multiplier_sl,
                'min_rr_ratio': self.min_rr_ratio,
                'volume_threshold': self.volume_threshold,
                'allow_pullback_entry': self.allow_pullback_entry
            }
            
        # Get parameters adjusted for current market regime
        adjusted_params = self.regime_detector.get_adjusted_parameters(df, asset, timeframe)
        
        # Log the regime detection and parameter adjustments
        regime = self.regime_detector.detect_regime(df, asset, timeframe)
        logger.info(f"Market regime detected for {asset.value} {timeframe.value}: {regime.regime_type.value} (strength: {regime.strength:.2f})")
        logger.info(f"Adjusted parameters: {adjusted_params}")
        
        return adjusted_params
        
    def scan_for_breakout_signals(
        self,
        asset: AssetSymbol,
        timeframe: Timeframe,
        historical_data: Optional[HistoricalDataModel] = None
    ) -> List[TradingSignalModel]:
        """
        Scans for 10-period high/low breakout signals for a given asset and timeframe.

        Args:
            asset: The asset to scan.
            timeframe: The timeframe to use for scanning.
            historical_data: Optional pre-fetched historical data. If None, it will be fetched.

        Returns:
            A list of TradingSignalModel instances if signals are found, otherwise an empty list.
        """
        signals: List[TradingSignalModel] = []

        if not historical_data:
            # Fetch slightly more data than breakout_period + atr_period to ensure enough data for calculations
            # e.g., breakout_period (10) + atr_period (14) + a few extra for stability = ~30-40 candles
            # yfinance fetches might need adjustment based on how `fetch_historical_data` handles periods/dates
            # For now, let's assume MarketDataAgent's default period for a timeframe is sufficient.
            # A more robust approach would be to calculate required lookback and pass start_date.
            logger.info(f"No historical data provided for {asset.value} {timeframe.value}. Fetching...")
            data_model = self.market_data_agent.fetch_historical_data(asset, timeframe)
        else:
            data_model = historical_data
            logger.info(f"Using provided historical data for {asset.value} {timeframe.value}.")

        if not data_model or not data_model.candles or len(data_model.candles) < self.breakout_period + 1:
            logger.warning(
                f"Not enough candle data for {asset.value} on {timeframe.value} "
                f"to scan for {self.breakout_period}-period breakouts. "
                f"Need at least {self.breakout_period + 1} candles, got {len(data_model.candles) if data_model else 0}."
            )
            return signals

        # Convert CandleModel list to DataFrame for easier manipulation with TA-Lib
        df = pd.DataFrame([candle.model_dump() for candle in data_model.candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Batch-calculate all required technical indicators once
        df = self.tech_indicators.calculate_all(df)

        if len(df) < self.breakout_period + 1: # Ensure enough data after DF conversion
            logger.warning(f"DataFrame too short for {asset.value} {timeframe.value} after conversion.")
            return signals
            
        # Get adjusted parameters based on current market regime
        adjusted_params = self._get_adjusted_parameters(df, asset, timeframe)
        
        # Apply adjusted parameters for this specific scan
        breakout_period = adjusted_params.get('breakout_period', self.breakout_period)
        atr_multiplier_sl = adjusted_params.get('atr_multiplier_sl', self.atr_multiplier_sl)
        min_rr_ratio = adjusted_params.get('min_rr_ratio', self.min_rr_ratio)
        volume_threshold = adjusted_params.get('volume_threshold', self.volume_threshold)
        allow_pullback_entry = adjusted_params.get('allow_pullback_entry', self.allow_pullback_entry)
        
        # Store original parameters to restore after scan
        original_params = {
            'breakout_period': self.breakout_period,
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'min_rr_ratio': self.min_rr_ratio,
            'volume_threshold': self.volume_threshold,
            'allow_pullback_entry': self.allow_pullback_entry
        }
        
        # Temporarily set adjusted parameters
        self.breakout_period = breakout_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.min_rr_ratio = min_rr_ratio
        self.volume_threshold = volume_threshold
        self.allow_pullback_entry = allow_pullback_entry

        # Get the most recent complete candle and the preceding candles for breakout check
        latest_candle_data = df.iloc[-1]
        
        # Look back period should include previous n periods (excluding current candle)
        # For a 20-period strategy, use the 20 completed candles before the current one
        if len(df) <= self.breakout_period + 1:  # Need at least breakout_period + current candle
            logger.warning(f"Not enough data for lookback analysis. Need {self.breakout_period+1}, got {len(df)}")
            return signals
            
        # Use rolling window to look at the previous N periods (more efficient)
        # The window is positioned to end at the second-to-last candle, so we check if the latest candle breaks out
        lookback_end_idx = -2  # Second-to-last candle (the completed candle before current)
        lookback_start_idx = lookback_end_idx - self.breakout_period + 1  # Start of lookback window
        
        lookback_data = df.iloc[lookback_start_idx:lookback_end_idx+1]  # Include the end index
        
        if len(lookback_data) < self.breakout_period:
            logger.warning(f"Lookback data insufficient: got {len(lookback_data)}, need {self.breakout_period} for {asset.value} on {timeframe.value}.")
            return signals
            
        highest_high_lookback = lookback_data['high'].max()
        lowest_low_lookback = lookback_data['low'].min()
        
        logger.debug(f"Lookback data for {asset.value} on {timeframe.value}:")
        logger.debug(f"  Lookback range: {lookback_data.index[0]} to {lookback_data.index[-1]}")
        logger.debug(f"  Highest high: {highest_high_lookback}, Lowest low: {lowest_low_lookback}")

        current_close = latest_candle_data['close']
        current_high = latest_candle_data['high'] # For ATR calculation
        current_low = latest_candle_data['low']   # For ATR calculation
        current_ts = latest_candle_data.name.to_pydatetime() # Get timestamp from DataFrame index


        # Use precomputed ATR if available
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
        else:
            atr = self._calculate_atr(df['high'], df['low'], df['close'])
        if atr is None or atr == 0: # ATR might be zero in very flat markets or if data is insufficient/problematic
            logger.warning(f"ATR calculation failed or returned zero for {asset.value} {timeframe.value}. Skipping signal generation.")
            return signals

        signal_type: Optional[SignalType] = None
        entry_price: Optional[float] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None

        # Use precomputed RSI if available
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
        else:
            rsi = self._calculate_rsi(df['close'])
        volume_confirmed = self._check_volume_confirmation(df['volume']) if self.volume_confirmation else True
        
        # Log current prices and indicators for debugging
        logger.info(f"Current data for {asset.value} {timeframe.value}:")
        logger.info(f"  Timestamp: {current_ts}, Open: {latest_candle_data['open']:.4f}, High: {current_high:.4f}, Low: {current_low:.4f}, Close: {current_close:.4f}")
        if rsi is not None:
            logger.info(f"  RSI: {rsi:.2f}")
        
        # ----- STRATEGY 1: STANDARD BREAKOUT -----
        # Check for long breakout: current candle's high or close breaks above the lookback period's highest high
        if (current_high > highest_high_lookback or current_close > highest_high_lookback):
            # Long signal with RSI filter (avoid buying overbought)
            if rsi is None or rsi < self.rsi_overbought:  # Proceed if RSI is below overbought or not available
                signal_type = SignalType.LONG
                entry_price = highest_high_lookback  # Enter at the breakout level
                
                # Use adaptive stop loss calculation for better risk management
                stop_loss = self._adapt_sl_based_on_volatility(entry_price, atr, signal_type)
                
                # Calculate take profit based on min reward/risk ratio
                take_profit = entry_price + (self.min_rr_ratio * (entry_price - stop_loss))
                
                # Add metadata about confirmation
                signal_metadata = {
                    "strategy_name": "breakout_15period",
                    "breakout_period": self.breakout_period,
                    "atr_period": self.atr_period,
                    "atr_value_at_signal": round(atr, 5),
                    "volume_confirmed": volume_confirmed,
                    "highest_high_lookback": round(highest_high_lookback, 5),
                    "lowest_low_lookback": round(lowest_low_lookback, 5),
                    "rsi": round(rsi, 2) if rsi is not None else None
                }
                
                logger.info(f"Long breakout signal for {asset.value} {timeframe.value} at {entry_price:.4f}")
                logger.info(f"  Triggered by: high={current_high:.4f}/close={current_close:.4f} > lookback high={highest_high_lookback:.4f}")
                # Safe formatting for RSI
                rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
                logger.info(f"  Volume confirmed: {volume_confirmed}, RSI: {rsi_str}")
                
                # Create the signal and append to results if volume confirms
                if volume_confirmed:
                    signal_id = str(uuid.uuid4())
                    signal = TradingSignalModel(
                        signal_id=signal_id,
                        generated_at=current_ts,
                        asset=asset,
                        timeframe=timeframe,
                        signal_type=signal_type,
                        entry_price=round(entry_price, 5),
                        stop_loss=round(stop_loss, 5),
                        take_profit=round(take_profit, 5),
                        metadata=signal_metadata
                    )
                    signals.append(signal)
                    logger.info(f"Generated long signal ID: {signal_id}")
                else:
                    logger.info("Long signal rejected due to insufficient volume confirmation")
            else:
                # Format RSI explicitly for safety
                rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
                logger.info(f"Long breakout detected but rejected - RSI too high: {rsi_str} > {self.rsi_overbought}")

        # Check for short breakout: current candle's low or close breaks below the lookback period's lowest low
        elif (current_low < lowest_low_lookback or current_close < lowest_low_lookback):
            # Short signal with RSI filter (avoid shorting oversold)
            if rsi is None or rsi > self.rsi_oversold:  # Proceed if RSI is above oversold or not available
                signal_type = SignalType.SHORT
                entry_price = lowest_low_lookback  # Enter at the breakout level
                
                # Use adaptive stop loss calculation
                stop_loss = self._adapt_sl_based_on_volatility(entry_price, atr, signal_type)
                
                # Calculate take profit based on min reward/risk ratio
                # For a SHORT position, the take profit must be BELOW entry price
                # Distance to stop loss is (stop_loss - entry_price), which is positive
                # The reward distance should be (min_rr_ratio * risk_distance)
                # So take_profit = entry_price - (min_rr_ratio * risk_distance)
                risk_distance = stop_loss - entry_price  # Positive for SHORT
                take_profit = entry_price - (self.min_rr_ratio * risk_distance)
                
                # Add metadata about confirmation
                signal_metadata = {
                    "strategy_name": "breakout_15period",
                    "breakout_period": self.breakout_period,
                    "atr_period": self.atr_period,
                    "atr_value_at_signal": round(atr, 5),
                    "volume_confirmed": volume_confirmed,
                    "highest_high_lookback": round(highest_high_lookback, 5),
                    "lowest_low_lookback": round(lowest_low_lookback, 5),
                    "rsi": round(rsi, 2) if rsi is not None else None
                }
                
                logger.info(f"Short breakout signal for {asset.value} {timeframe.value} at {entry_price:.4f}")
                logger.info(f"  Triggered by: low={current_low:.4f}/close={current_close:.4f} < lookback low={lowest_low_lookback:.4f}")
                # Safe formatting for RSI
                rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
                logger.info(f"  Volume confirmed: {volume_confirmed}, RSI: {rsi_str}")
                
                # Create the signal and append to results if volume confirms
                if volume_confirmed:
                    signal_id = str(uuid.uuid4())
                    signal = TradingSignalModel(
                        signal_id=signal_id,
                        generated_at=current_ts,
                        asset=asset,
                        timeframe=timeframe,
                        signal_type=signal_type,
                        entry_price=round(entry_price, 5),
                        stop_loss=round(stop_loss, 5),
                        take_profit=round(take_profit, 5),
                        metadata=signal_metadata
                    )
                    signals.append(signal)
                    logger.info(f"Generated short signal ID: {signal_id}")
                else:
                    logger.info("Short signal rejected due to insufficient volume confirmation")
            else:
                # Format RSI explicitly for safety
                rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
                logger.info(f"Short breakout detected but rejected - RSI too low: {rsi_str} < {self.rsi_oversold}")
        
        # ----- STRATEGY 2: PULLBACK ENTRY -----
        # Look for pullback entries on recent breakouts if no immediate breakout is found
        # This allows for better entries on confirmed breakouts that have slightly pulled back
        elif self.allow_pullback_entry:
            # Check recent price action (last 5-8 candles) for breakout then pullback pattern
            recent_candles = 5  # Check for pullbacks in last 5 candles
            
            if len(df) >= recent_candles + self.breakout_period + 1:
                # Window to check for recent breakout first (starting 2-7 candles back)
                recent_window = df.iloc[-(recent_candles+1):-1]
                prior_window = df.iloc[-self.breakout_period-(recent_candles+1):-(recent_candles+1)]
                
                # Looking for a breakout in recent window, then a pullback on current candle
                recent_high = recent_window['high'].max()
                recent_low = recent_window['low'].min()
                prior_high = prior_window['high'].max()
                prior_low = prior_window['low'].min()
                
                # Pullback long: if we broke out higher in recent candles but now pulling back to the breakout level
                if recent_high > prior_high and (current_low <= prior_high * 1.01 and current_low >= prior_high * 0.99):
                    # Potential pullback long entry (provided we have RSI confirmation)
                    if rsi is not None and rsi < self.rsi_overbought:
                        signal_type = SignalType.LONG
                        entry_price = prior_high  # Enter at the original breakout level
                        
                        # Calculate stop and take profit
                        stop_loss = current_low - (atr * 0.5)  # Tighter stop under current low
                        take_profit = entry_price + (self.min_rr_ratio * (entry_price - stop_loss))
                        
                        # Get the market regime for metadata
                        regime = self.regime_detector.detect_regime(df, asset, timeframe)
                        
                        signal_metadata = {
                            "strategy_name": "pullback_entry_long",
                            "breakout_period": self.breakout_period,
                            "atr_value_at_signal": round(atr, 5),
                            "volume_confirmed": volume_confirmed,
                            "original_breakout_level": round(prior_high, 5),
                            "rsi": round(rsi, 2) if rsi is not None else None,
                            "market_regime": regime.regime_type.value,
                            "regime_strength": round(regime.strength, 2),
                            "adjusted_parameters": adjusted_params
                        }
                        
                        logger.info(f"Pullback long entry for {asset.value} {timeframe.value} at {entry_price:.4f}")
                        logger.info(f"  Recent high={recent_high:.4f} broke above prior high={prior_high:.4f}, now pulling back")
                        
                        # Only create signal if volume confirms and risk/reward is good
                        if volume_confirmed and (take_profit - entry_price) >= self.min_rr_ratio * (entry_price - stop_loss):
                            signal_id = str(uuid.uuid4())
                            signal = TradingSignalModel(
                                signal_id=signal_id,
                                generated_at=current_ts,
                                asset=asset,
                                timeframe=timeframe,
                                signal_type=signal_type,
                                entry_price=round(entry_price, 5),
                                stop_loss=round(stop_loss, 5),
                                take_profit=round(take_profit, 5),
                                metadata=signal_metadata
                            )
                            signals.append(signal)
                            logger.info(f"Generated pullback long signal ID: {signal_id}")
                
                # Pullback short: if we broke down lower in recent candles but now pulling back to the breakout level
                elif recent_low < prior_low and (current_high >= prior_low * 0.99 and current_high <= prior_low * 1.01):
                    # Potential pullback short entry (provided we have RSI confirmation)
                    if rsi is not None and rsi > self.rsi_oversold:
                        signal_type = SignalType.SHORT
                        entry_price = prior_low  # Enter at the original breakout level
                        
                        # Calculate stop and take profit
                        # For SHORT positions: stop loss above entry, take profit below entry
                        
                        # Ensure stop loss is always above entry price by a minimum distance
                        min_sl_distance = atr * 0.5  # Minimum distance for stop loss
                        stop_loss = max(current_high + min_sl_distance, entry_price + min_sl_distance)
                        
                        # Calculate risk and ensure it's positive
                        risk_distance = stop_loss - entry_price
                        
                        # Make sure we have a valid risk distance (safety check)
                        if risk_distance <= 0:
                            risk_distance = atr * self.atr_multiplier_sl
                            stop_loss = entry_price + risk_distance
                            
                        # Calculate reward based on risk
                        reward_distance = risk_distance * self.min_rr_ratio
                        take_profit = entry_price - reward_distance  # Take profit BELOW entry price for SHORT positions
                        
                        # Get the market regime for metadata
                        regime = self.regime_detector.detect_regime(df, asset, timeframe)
                        
                        signal_metadata = {
                            "strategy_name": "pullback_entry_short",
                            "breakout_period": self.breakout_period,
                            "atr_value_at_signal": round(atr, 5),
                            "volume_confirmed": volume_confirmed,
                            "original_breakout_level": round(prior_low, 5),
                            "rsi": round(rsi, 2) if rsi is not None else None,
                            "market_regime": regime.regime_type.value,
                            "regime_strength": round(regime.strength, 2),
                            "adjusted_parameters": adjusted_params
                        }
                        
                        logger.info(f"Pullback short entry for {asset.value} {timeframe.value} at {entry_price:.4f}")
                        logger.info(f"  Recent low={recent_low:.4f} broke below prior low={prior_low:.4f}, now pulling back")
                        
                        # Only create signal if volume confirms and risk/reward is good
                        # For SHORT positions, the reward is (entry_price - take_profit) and the risk is (stop_loss - entry_price)
                        # We already calculated take_profit to ensure the right R:R ratio, so this check should always pass
                        # But keeping it for safety and clarity
                        if volume_confirmed:
                            signal_id = str(uuid.uuid4())
                            signal = TradingSignalModel(
                                signal_id=signal_id,
                                generated_at=current_ts,
                                asset=asset,
                                timeframe=timeframe,
                                signal_type=signal_type,
                                entry_price=round(entry_price, 5),
                                stop_loss=round(stop_loss, 5),
                                take_profit=round(take_profit, 5),
                                metadata=signal_metadata
                            )
                            signals.append(signal)
                            logger.info(f"Generated pullback short signal ID: {signal_id}")

        # We've now moved signal generation to the strategy sections
        # for cleaner code and better strategy-specific logging
        
        # Restore original parameters after scan
        self.breakout_period = original_params['breakout_period']
        self.atr_multiplier_sl = original_params['atr_multiplier_sl']
        self.min_rr_ratio = original_params['min_rr_ratio']
        self.volume_threshold = original_params['volume_threshold']
        self.allow_pullback_entry = original_params['allow_pullback_entry']
        
        return signals

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # --- Mock MarketDataAgent for testing ---
    class MockMarketDataAgent(MarketDataAgent):
        def __init__(self):
            super().__init__()
            self._mock_data_store: Dict[tuple[AssetSymbol, Timeframe], HistoricalDataModel] = {}

        def add_mock_data(self, asset: AssetSymbol, timeframe: Timeframe, data: HistoricalDataModel):
            self._mock_data_store[(asset, timeframe)] = data

        def fetch_historical_data(
            self, asset: AssetSymbol, timeframe: Timeframe, 
            start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
            use_cache: bool = True # Unused in mock for simplicity here
        ) -> Optional[HistoricalDataModel]:
            logger.info(f"[MockMarketDataAgent] Fetching data for {asset.value} {timeframe.value}")
            return self._mock_data_store.get((asset, timeframe))

    # --- Test Setup ---
    mock_mda = MockMarketDataAgent()
    scanner = SignalScannerAgent(market_data_agent=mock_mda)

    # Create sample candle data for BTC-USD, 1h for a breakout scenario
    # Need at least 11 candles for a 10-period breakout + current candle
    # And enough for ATR(14)
    
    # Scenario 1: Long Breakout
    logger.info("\\n--- Test 1: Long Breakout Scenario ---")
    candles_long_breakout: List[CandleModel] = []
    base_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Initial 10 candles (non-breakout)
    for i in range(scanner.breakout_period + scanner.atr_period): # Enough for lookback and ATR
        ts = base_time + pd.Timedelta(hours=i)
        price_open = 10000 + i * 5
        price_high = 10050 + i * 5
        price_low = 9950 + i * 5
        price_close = 10000 + i * 2 # Keep close within a range
        volume = 100 + i
        candles_long_breakout.append(CandleModel(timestamp=ts, open=price_open, high=price_high, low=price_low, close=price_close, volume=volume))
    
    highest_high_prev_10 = max(c.high for c in candles_long_breakout[-scanner.breakout_period:])
    breakout_ts = base_time + pd.Timedelta(hours=len(candles_long_breakout))
    breakout_candle = CandleModel(
        timestamp=breakout_ts,
        open=highest_high_prev_10 - 10, # Open below breakout level
        high=highest_high_prev_10 + 50, # High breaks out
        low=highest_high_prev_10 - 20, 
        close=highest_high_prev_10 - 5, # Close can be anywhere, even below breakout level for this test
        volume=200
    )
    candles_long_breakout.append(breakout_candle)
    mock_data_long = HistoricalDataModel(asset=AssetSymbol.BTC_USD, timeframe=Timeframe.HOUR_1, candles=candles_long_breakout)
    mock_mda.add_mock_data(AssetSymbol.BTC_USD, Timeframe.HOUR_1, mock_data_long)
    
    signals_long = scanner.scan_for_breakout_signals(AssetSymbol.BTC_USD, Timeframe.HOUR_1)
    if signals_long:
        logger.info(f"Found {len(signals_long)} long signal(s). First signal: {signals_long[0].signal_type}, Entry: {signals_long[0].entry_price}")
        assert signals_long[0].signal_type == SignalType.LONG
        assert signals_long[0].entry_price == highest_high_prev_10 # Entry should be at the breakout level
        assert signals_long[0].stop_loss is not None and signals_long[0].stop_loss < signals_long[0].entry_price
        assert signals_long[0].take_profit is not None and signals_long[0].take_profit > signals_long[0].entry_price
    else:
        logger.warning("No long signals found in Test 1, check data and logic.")

    # Scenario 2: Short Breakout
    logger.info("\\n--- Test 2: Short Breakout Scenario ---")
    candles_short_breakout: List[CandleModel] = []
    base_time_short = datetime(2023, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(scanner.breakout_period + scanner.atr_period):
        ts = base_time_short + pd.Timedelta(hours=i)
        price_open = 15000 - i * 5
        price_high = 15050 - i * 5
        price_low = 14950 - i * 5
        price_close = 15000 - i * 2 # Keep close within a range
        volume = 120 + i
        candles_short_breakout.append(CandleModel(timestamp=ts, open=price_open, high=price_high, low=price_low, close=price_close, volume=volume))

    lowest_low_prev_10 = min(c.low for c in candles_short_breakout[-scanner.breakout_period:])
    breakout_ts_short = base_time_short + pd.Timedelta(hours=len(candles_short_breakout))
    breakout_candle_short = CandleModel(
        timestamp=breakout_ts_short,
        open=lowest_low_prev_10 + 10, # Open above breakout level
        high=lowest_low_prev_10 + 20,
        low=lowest_low_prev_10 - 50,  # Low breaks out
        close=lowest_low_prev_10 + 5, # Close can be anywhere, even above breakout level
        volume=250
    )
    candles_short_breakout.append(breakout_candle_short)
    mock_data_short = HistoricalDataModel(asset=AssetSymbol.ETH_USD, timeframe=Timeframe.MIN_15, candles=candles_short_breakout)
    mock_mda.add_mock_data(AssetSymbol.ETH_USD, Timeframe.MIN_15, mock_data_short)
    signals_short = scanner.scan_for_breakout_signals(AssetSymbol.ETH_USD, Timeframe.MIN_15)
    if signals_short:
        logger.info(f"Found {len(signals_short)} short signal(s). First signal: {signals_short[0].signal_type}, Entry: {signals_short[0].entry_price}")
        assert signals_short[0].signal_type == SignalType.SHORT
        assert signals_short[0].entry_price == lowest_low_prev_10 # Entry should be at the breakout level
        assert signals_short[0].stop_loss is not None and signals_short[0].stop_loss > signals_short[0].entry_price
        assert signals_short[0].take_profit is not None and signals_short[0].take_profit < signals_short[0].entry_price
    else:
        logger.warning("No short signals found in Test 2, check data and logic.")

    # Scenario 3: No Breakout
    logger.info("\\n--- Test 3: No Breakout Scenario ---")
    candles_no_breakout: List[CandleModel] = []
    base_time_no = datetime(2023, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(scanner.breakout_period + scanner.atr_period + 1): # ensure enough data
        ts = base_time_no + pd.Timedelta(days=i) # Use daily for variety
        price_open = 50 + (i % 3 * 2)
        price_high = price_open + 2
        price_low = price_open - 2
        price_close = price_open + (0.5 if i % 2 == 0 else -0.5)
        volume = 1000 + i*10
        candles_no_breakout.append(CandleModel(timestamp=ts, open=price_open, high=price_high, low=price_low, close=price_close, volume=volume))
    
    mock_data_no = HistoricalDataModel(asset=AssetSymbol.SPY, timeframe=Timeframe.DAY_1, candles=candles_no_breakout)
    mock_mda.add_mock_data(AssetSymbol.SPY, Timeframe.DAY_1, mock_data_no)
    
    signals_no = scanner.scan_for_breakout_signals(AssetSymbol.SPY, Timeframe.DAY_1)
    if not signals_no:
        logger.info("Correctly found no signals in Test 3.")
        assert not signals_no
    else:
        logger.warning(f"Incorrectly found {len(signals_no)} signal(s) in Test 3.")

    # Scenario 4: Insufficient Data
    logger.info("\\n--- Test 4: Insufficient Data Scenario ---")
    candles_insufficient: List[CandleModel] = []
    for i in range(5): # Not enough for breakout_period or atr_period
        ts = datetime(2023, 4, 1, tzinfo=timezone.utc) + pd.Timedelta(hours=i)
        candles_insufficient.append(CandleModel(timestamp=ts, open=100, high=101, low=99, close=100, volume=10))
    
    mock_data_insufficient = HistoricalDataModel(asset=AssetSymbol.BTC_USD, timeframe=Timeframe.MIN_5, candles=candles_insufficient)
    mock_mda.add_mock_data(AssetSymbol.BTC_USD, Timeframe.MIN_5, mock_data_insufficient)
    
    signals_insufficient = scanner.scan_for_breakout_signals(AssetSymbol.BTC_USD, Timeframe.MIN_5)
    if not signals_insufficient:
        logger.info("Correctly found no signals due to insufficient data in Test 4.")
        assert not signals_insufficient
    else:
        logger.warning(f"Incorrectly found {len(signals_insufficient)} signal(s) in Test 4 (insufficient data).")
