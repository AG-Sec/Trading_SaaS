from typing import List, Optional, Tuple, Dict, Any
import logging
import math

from shared_types import TradingSignalModel, SignalType, AssetSymbol, Timeframe
from .journal_agent import JournalAgent
from .market_regime_detector import MarketRegimeDetector, MarketRegimeType

logger = logging.getLogger(__name__)

class RiskManagerAgent:
    """
    Agent responsible for evaluating trading signals against risk management rules
    and calculating position sizes.
    """

    def __init__(
        self,
        account_balance_usd: float = 1000.0, # Default based on memory b099cc16-6929-45bd-8019-a4bb0858de93
        max_risk_per_trade_pct: float = 0.03,  # Increased from 2% to 3% for more signals
        min_reward_to_risk_ratio: float = 1.1,  # Reduced from 1.2 to 1.1 to allow more signals
        max_signals_per_asset: int = 3,  # Increased from 2 to 3 maximum signals per asset
        journal_agent: Optional[JournalAgent] = None,
        adapt_to_regime: bool = True  # Whether to adjust risk parameters based on market regime
    ):
        """
        Initializes the RiskManagerAgent.

        Args:
            account_balance_usd: Current total account balance in USD.
            max_risk_per_trade_pct: Maximum percentage of account balance to risk per trade.
            min_reward_to_risk_ratio: Minimum acceptable reward-to-risk ratio for a trade.
            journal_agent: Optional JournalAgent instance for recording approved signals.
        """
        if not 0 < max_risk_per_trade_pct <= 1:
            raise ValueError("max_risk_per_trade_pct must be between 0 (exclusive) and 1 (inclusive).")
        
        self.account_balance_usd = account_balance_usd
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.min_reward_to_risk_ratio = min_reward_to_risk_ratio
        self.max_signals_per_asset = max_signals_per_asset
        self.journal_agent = journal_agent
        self.adapt_to_regime = adapt_to_regime
        
        # Initialize market regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Store active signals per asset to limit exposure
        self.active_signals_by_asset = {}
        
        # Provide flexibility for different assets
        self.asset_specific_params = {
            # Format: 'ASSET-SYMBOL': {'min_rr': float, 'max_risk_pct': float}
            # More permissive parameters for all assets to generate more signals
            'BTC-USD': {'min_rr': 1.1, 'max_risk_pct': 0.03},
            'ETH-USD': {'min_rr': 1.1, 'max_risk_pct': 0.03},
            # Example for more stable assets
            'SPY': {'min_rr': 0.9, 'max_risk_pct': 0.035},
            'EURUSD=X': {'min_rr': 1.0, 'max_risk_pct': 0.03},
        }
        
        # Placeholder for more advanced state (e.g., current open positions, daily drawdown tracking)
        # self.current_open_positions = [] 
        # self.daily_drawdown_limit = 0.05 # e.g. 5% from memory cfd3f21e-9a17-41b7-a57e-6877e5c8d2b8
        # self.max_concurrent_positions = 3 # from memory cfd3f21e-9a17-41b7-a57e-6877e5c8d2b8

    def _calculate_reward_to_risk(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_type: SignalType
    ) -> Optional[float]:
        """Calculates the reward-to-risk ratio for a signal."""
        if entry_price is None or stop_loss is None or take_profit is None:
            return None

        potential_risk_per_unit = abs(entry_price - stop_loss)
        potential_reward_per_unit = abs(take_profit - entry_price)

        if potential_risk_per_unit == 0: # Avoid division by zero
            return None 

        # Validate direction consistency
        if signal_type == SignalType.LONG:
            if not (take_profit > entry_price > stop_loss):
                logger.warning(f"Inconsistent LONG signal prices: E={entry_price}, SL={stop_loss}, TP={take_profit}")
                return None
        elif signal_type == SignalType.SHORT:
            if not (take_profit < entry_price < stop_loss):
                logger.warning(f"Inconsistent SHORT signal prices: E={entry_price}, SL={stop_loss}, TP={take_profit}")
                return None
        else: # Should not happen for LONG/SHORT signals being evaluated
            return None
            
        return potential_reward_per_unit / potential_risk_per_unit

    def _calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss: float,
        risk_pct_override: Optional[float] = None
    ) -> Optional[Tuple[float, float, float]]:
        """
        Calculates position size based on max risk per trade.
        Returns: (position_size_asset, position_size_usd, risk_amount_usd) or None if invalid.
        """
        if entry_price is None or stop_loss is None or entry_price == stop_loss:
            logger.warning("Cannot calculate position size: entry or stop_loss is None, or entry equals stop_loss.")
            return None

        # Use override risk percentage if provided, otherwise use default
        risk_pct = risk_pct_override if risk_pct_override is not None else self.max_risk_per_trade_pct
        risk_amount_usd_for_trade = self.account_balance_usd * risk_pct
        risk_per_unit_asset = abs(entry_price - stop_loss)

        if risk_per_unit_asset == 0: # Should have been caught, but double check
            logger.warning("Risk per unit asset is zero, cannot calculate position size.")
            return None

        position_size_asset = risk_amount_usd_for_trade / risk_per_unit_asset
        
        # For assets like forex, might need to consider lot sizes and contract specifications.
        # For crypto/stocks, this direct calculation is usually fine.
        # Example: For BTC-USD, position_size_asset is in BTC.
        # We might want to round to a sensible number of decimal places depending on asset.
        # For now, no specific rounding, but broker minimums/steps would apply in reality.
        # position_size_asset = math.floor(position_size_asset * 1e8) / 1e8 # Example for BTC precision

        position_size_usd_notional = position_size_asset * entry_price
        
        return position_size_asset, position_size_usd_notional, risk_amount_usd_for_trade

    def evaluate_signal(self, signal: TradingSignalModel) -> Optional[TradingSignalModel]:
        """
        Evaluates a single trading signal against risk parameters.
        If approved, it updates the signal with risk metrics and position size.

        Args:
            signal: The TradingSignalModel instance to evaluate.

        Returns:
            The updated TradingSignalModel if approved, otherwise None.
        """
        if not all([signal.entry_price, signal.stop_loss, signal.take_profit]):
            logger.warning(f"Signal {signal.signal_id} missing entry, SL, or TP. Rejecting.")
            return None

        # 1. Calculate and check Reward-to-Risk Ratio
        rr_ratio = self._calculate_reward_to_risk(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_type=signal.signal_type
        )

        if rr_ratio is None:
            logger.warning(f"Could not calculate R:R for signal {signal.signal_id} due to inconsistent prices. Rejecting.")
            return None
            
        signal.risk_reward_ratio = round(rr_ratio, 2)

        # Get asset-specific parameters if they exist
        asset_params = self.asset_specific_params.get(signal.asset.value, {})
        min_rr_for_asset = asset_params.get('min_rr', self.min_reward_to_risk_ratio)
        
        # Accept signals that are very close to the minimum R:R (within 15% - increased from 5%)
        acceptance_threshold = min_rr_for_asset * 0.85
        
        if rr_ratio < acceptance_threshold:
            logger.info(
                f"Signal {signal.signal_id} for {signal.asset.value} rejected. "
                f"R:R ({rr_ratio:.2f}) is below minimum threshold ({acceptance_threshold:.2f})."
            )
            return None
            
        # If R:R is close but not quite at target, log a warning but still accept
        if acceptance_threshold <= rr_ratio < min_rr_for_asset:
            logger.warning(
                f"Signal {signal.signal_id} R:R ({rr_ratio:.2f}) is slightly below target ({min_rr_for_asset:.2f}) "
                f"but above minimum threshold ({acceptance_threshold:.2f}). Accepting with caution."
            )
        else:
            logger.info(f"Signal {signal.signal_id} R:R excellent: {rr_ratio:.2f} >= {min_rr_for_asset:.2f}")

        # 2. Calculate Position Size and Risk Amount with asset-specific risk parameters
        asset_max_risk_pct = asset_params.get('max_risk_pct', self.max_risk_per_trade_pct)
        
        # Adjust risk percentage based on signal quality (higher R:R = higher allowed risk)
        quality_boost = min(1.0, (rr_ratio - min_rr_for_asset) / 2) if rr_ratio > min_rr_for_asset else 0
        adjusted_risk_pct = asset_max_risk_pct * (1 + quality_boost * 0.5)  # Up to 50% boost for excellent signals
        
        # Ensure we don't exceed our maximum allowed percentage regardless of boost
        adjusted_risk_pct = min(adjusted_risk_pct, asset_max_risk_pct * 1.5)
        
        size_calculation = self._calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            risk_pct_override=adjusted_risk_pct
        )

        if size_calculation is None:
            logger.warning(f"Could not calculate position size for signal {signal.signal_id}. Rejecting.")
            return None
        
        pos_size_asset, pos_size_usd, risk_usd = size_calculation
        signal.position_size_asset = round(pos_size_asset, 8) # Round to 8 decimal places, common for crypto
        signal.position_size_usd = round(pos_size_usd, 2)
        signal.risk_per_trade_usd = round(risk_usd, 2)
        
        # Add risk quality info to metadata
        if not signal.metadata:
            signal.metadata = {}
        signal.metadata.update({
            "risk_quality": "excellent" if rr_ratio >= min_rr_for_asset else "acceptable",
            "adjusted_risk_pct": round(adjusted_risk_pct * 100, 2),  # Convert to percentage format
            "base_risk_pct": round(asset_max_risk_pct * 100, 2),
            "quality_boost": round(quality_boost * 100, 2)
        })

        logger.info(
            f"Signal {signal.signal_id} approved. Position Asset: {signal.position_size_asset:.8f} {signal.asset.value}, "
            f"Position USD: ${signal.position_size_usd:.2f}, Risk USD: ${signal.risk_per_trade_usd:.2f}"
        )

        # Attempt to record the signal if a journal agent is provided
        if self.journal_agent:
            if not self.journal_agent.record_signal(signal):
                # Log if recording failed, but don't necessarily reject the signal
                # as per current design (journaling is a side effect of approval)
                logger.warning(f"Failed to journal approved signal {signal.signal_id}. Continuing...")

        # Future checks (placeholders for more advanced logic):
        # - Check against daily max drawdown (requires account P&L tracking)
        # - Check against max concurrent positions (requires portfolio state)
        # - Check against available capital vs position_size_usd (if margin not used or limited)
        # - Check for specific asset volatility or liquidity constraints (might need MarketDataAgent input)

        return signal

    def is_valid_signal(self, signal: TradingSignalModel) -> bool:
        # For this check, we don't calculate position size, just if trade is valid
        if signal.entry_price <= 0 or signal.stop_loss <= 0 or signal.take_profit <= 0:
            logger.warning(f"Signal {signal.signal_id} rejected: Invalid prices.")
            return False

        # Get adjusted risk parameters based on market regime
        adjusted_params = self._adjust_risk_for_regime(signal)
        this_min_rr = adjusted_params['min_reward_to_risk_ratio']
        this_max_risk = adjusted_params['max_risk_per_trade_pct']

        # Check if the signal is valid against risk management rules
        rr_ratio = self._calculate_reward_to_risk(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            signal_type=signal.signal_type
        )

        if rr_ratio is None:
            logger.warning(f"Could not calculate R:R for signal {signal.signal_id} due to inconsistent prices. Rejecting.")
            return False

        if rr_ratio < this_min_rr:
            logger.info(
                f"Signal {signal.signal_id} for {signal.asset.value} rejected. "
                f"R:R ({rr_ratio:.2f}) is below minimum threshold ({this_min_rr:.2f})."
            )
            return False

        return True

    def _adjust_risk_for_regime(self, signal: TradingSignalModel) -> Dict[str, float]:
        """
        Adjust risk parameters based on market regime information in the signal metadata.
        
        Args:
            signal: The trading signal with market regime information
            
        Returns:
            Dictionary with adjusted risk parameters
        """
        if not self.adapt_to_regime or 'metadata' not in signal.model_dump() or not signal.metadata:
            # Return default parameters if no adaptation needed or no metadata
            return {
                'max_risk_per_trade_pct': self.max_risk_per_trade_pct,
                'min_reward_to_risk_ratio': self.min_reward_to_risk_ratio
            }
            
        # Extract market regime information from signal metadata
        metadata = signal.metadata
        regime_type = metadata.get('market_regime')
        regime_strength = metadata.get('regime_strength', 0.5)
        
        # Default values to start with
        adjusted_max_risk = self.max_risk_per_trade_pct
        adjusted_min_rr = self.min_reward_to_risk_ratio
        
        # Get asset-specific base parameters if available
        asset_symbol = signal.asset.value
        if asset_symbol in self.asset_specific_params:
            asset_params = self.asset_specific_params[asset_symbol]
            adjusted_max_risk = asset_params.get('max_risk_pct', adjusted_max_risk)
            adjusted_min_rr = asset_params.get('min_rr', adjusted_min_rr)
        
        # Adjust based on market regime
        if regime_type == MarketRegimeType.BULLISH_TRENDING.value:
            if signal.signal_type == SignalType.LONG:
                # More aggressive on longs in bullish trends
                adjusted_max_risk *= 1.2  # Increase risk allowance by 20%
                adjusted_min_rr *= 0.9   # Decrease RR requirement by 10%
            else:  # SHORT
                # More conservative on shorts in bullish trends
                adjusted_max_risk *= 0.8  # Decrease risk allowance by 20%
                adjusted_min_rr *= 1.2   # Increase RR requirement by 20%
                
        elif regime_type == MarketRegimeType.BEARISH_TRENDING.value:
            if signal.signal_type == SignalType.SHORT:
                # More aggressive on shorts in bearish trends
                adjusted_max_risk *= 1.2  # Increase risk allowance by 20%
                adjusted_min_rr *= 0.9   # Decrease RR requirement by 10%
            else:  # LONG
                # More conservative on longs in bearish trends
                adjusted_max_risk *= 0.8  # Decrease risk allowance by 20%
                adjusted_min_rr *= 1.2   # Increase RR requirement by 20%
                
        elif regime_type == MarketRegimeType.HIGH_VOLATILITY.value:
            # More conservative in high volatility
            adjusted_max_risk *= 0.7  # Significantly reduce risk
            adjusted_min_rr *= 1.3   # Require better R:R
            
        elif regime_type == MarketRegimeType.LOW_VOLATILITY.value:
            # Can be slightly more aggressive in low volatility
            adjusted_max_risk *= 1.1  # Slightly increase risk allowance
            adjusted_min_rr *= 0.9   # Slightly decrease RR requirement
            
        elif regime_type in [MarketRegimeType.BULLISH_BREAKOUT.value, MarketRegimeType.BEARISH_BREAKOUT.value]:
            # For breakouts, check if signal aligns with breakout direction
            is_aligned = ((regime_type == MarketRegimeType.BULLISH_BREAKOUT.value and 
                          signal.signal_type == SignalType.LONG) or
                         (regime_type == MarketRegimeType.BEARISH_BREAKOUT.value and
                          signal.signal_type == SignalType.SHORT))
            
            if is_aligned:
                # Aligned with breakout direction - can be more aggressive
                adjusted_max_risk *= 1.3  # Increase risk allowance
                adjusted_min_rr *= 0.85  # Lower RR requirement
            else:
                # Counter to breakout direction - be very conservative
                adjusted_max_risk *= 0.6  # Significantly reduce risk
                adjusted_min_rr *= 1.5   # Require much better R:R
        
        # Scale adjustments based on regime strength
        # If strength = 0.5 (medium), no additional scaling
        # If strength > 0.5, amplify the adjustments
        # If strength < 0.5, reduce the adjustments
        strength_factor = 1.0 + (regime_strength - 0.5)  # 0.5 -> 1.0, 1.0 -> 1.5, 0.0 -> 0.5
        
        # Apply strength scaling, but ensure we don't go below reasonable minimums
        if adjusted_max_risk > self.max_risk_per_trade_pct:  # We increased risk
            risk_adjustment = adjusted_max_risk / self.max_risk_per_trade_pct - 1.0  # How much we increased
            scaled_risk_adjustment = risk_adjustment * strength_factor  # Scale by strength
            adjusted_max_risk = self.max_risk_per_trade_pct * (1.0 + scaled_risk_adjustment)
        else:  # We decreased risk
            risk_adjustment = 1.0 - adjusted_max_risk / self.max_risk_per_trade_pct  # How much we decreased
            scaled_risk_adjustment = risk_adjustment * strength_factor  # Scale by strength
            adjusted_max_risk = self.max_risk_per_trade_pct * (1.0 - scaled_risk_adjustment)
            
        # Similar scaling for reward:risk ratio
        if adjusted_min_rr < self.min_reward_to_risk_ratio:  # We decreased RR requirement
            rr_adjustment = 1.0 - adjusted_min_rr / self.min_reward_to_risk_ratio  # How much we decreased
            scaled_rr_adjustment = rr_adjustment * strength_factor  # Scale by strength
            adjusted_min_rr = self.min_reward_to_risk_ratio * (1.0 - scaled_rr_adjustment)
        else:  # We increased RR requirement
            rr_adjustment = adjusted_min_rr / self.min_reward_to_risk_ratio - 1.0  # How much we increased
            scaled_rr_adjustment = rr_adjustment * strength_factor  # Scale by strength
            adjusted_min_rr = self.min_reward_to_risk_ratio * (1.0 + scaled_rr_adjustment)
        
        # Apply reasonable bounds
        adjusted_max_risk = max(0.005, min(0.05, adjusted_max_risk))  # Keep between 0.5% and 5%
        adjusted_min_rr = max(1.0, min(3.0, adjusted_min_rr))       # Keep between 1.0 and 3.0
        
        logger.info(f"Adjusted risk parameters for {signal.asset.value} based on {regime_type} regime (strength: {regime_strength:.2f})")
        logger.info(f"  Max risk: {self.max_risk_per_trade_pct:.4f} -> {adjusted_max_risk:.4f}, Min R:R: {self.min_reward_to_risk_ratio:.2f} -> {adjusted_min_rr:.2f}")
        
        return {
            'max_risk_per_trade_pct': adjusted_max_risk,
            'min_reward_to_risk_ratio': adjusted_min_rr
        }
        
    def evaluate_signals(self, signals: List[TradingSignalModel]) -> List[TradingSignalModel]:
        """
        Filters a list of trading signals, returning only those that pass risk evaluation.
        Limits signals per asset and prioritizes higher quality signals.
        """
        # First, group signals by asset and sort by R:R ratio to prioritize higher quality signals
        signals_by_asset = {}
        for signal in signals:
            if signal.asset.value not in signals_by_asset:
                signals_by_asset[signal.asset.value] = []
            signals_by_asset[signal.asset.value].append(signal)
        
        # Prioritize signals with better R:R ratio within each asset group
        def calculate_position_size(signal: TradingSignalModel) -> float:
            """Calculate the appropriate position size for a valid signal."""
            # Get adjusted risk parameters based on market regime
            adjusted_params = self._adjust_risk_for_regime(signal)
            this_max_risk = adjusted_params['max_risk_per_trade_pct']
            this_min_rr = adjusted_params['min_reward_to_risk_ratio']
            
            # Calculate position size
            size_calculation = self._calculate_position_size(
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                risk_pct_override=this_max_risk
            )

            if size_calculation is None:
                logger.warning(f"Could not calculate position size for signal {signal.signal_id}. Rejecting.")
                return None
            
            pos_size_asset, pos_size_usd, risk_usd = size_calculation
            signal.position_size_asset = round(pos_size_asset, 8) # Round to 8 decimal places, common for crypto
            signal.position_size_usd = round(pos_size_usd, 2)
            signal.risk_per_trade_usd = round(risk_usd, 2)
            
            return signal
        
        # Now process signals with prioritization
        approved_signals: List[TradingSignalModel] = []
        for asset, asset_signals in signals_by_asset.items():
            # Check how many signals already exist for this asset
            current_active_count = self.active_signals_by_asset.get(asset, 0)
            slots_available = self.max_signals_per_asset - current_active_count
            
            # Process signals until we hit the limit
            approved_count = 0
            for signal in asset_signals:
                if approved_count >= slots_available:
                    logger.info(f"Maximum signals ({self.max_signals_per_asset}) reached for {asset}. Skipping remaining signals.")
                    break
                    
                evaluated_signal = self.evaluate_signal(signal)
                if evaluated_signal:
                    approved_signals.append(evaluated_signal)
                    approved_count += 1
                    
            # Update active signal count for this asset
            if approved_count > 0:
                self.active_signals_by_asset[asset] = current_active_count + approved_count
                
        return approved_signals
        
    def filter_signals(self, signals: List[TradingSignalModel]) -> List[TradingSignalModel]:
        """
        Filter a list of trading signals, returning only those that pass risk management criteria.
        This is used primarily during backtesting.
        
        Args:
            signals: List of trading signals to evaluate
            
        Returns:
            List of approved trading signals
        """
        if not signals:
            return []
            
        # Use the existing evaluate_signals method to process the signals
        return self.evaluate_signals(signals)


if __name__ == '__main__':
    from datetime import datetime, timezone
    from shared_types import AssetSymbol, Timeframe
    import uuid

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Test Setup ---
    risk_agent_default = RiskManagerAgent() # Uses defaults: $1000 balance, 1% risk, 1.5 R:R
    risk_agent_high_risk = RiskManagerAgent(account_balance_usd=10000, max_risk_per_trade_pct=0.02)

    # Example Signal 1: Good Long Signal (BTC-USD)
    signal1_raw = TradingSignalModel(
        signal_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        asset=AssetSymbol.BTC_USD,
        timeframe=Timeframe.HOUR_1,
        signal_type=SignalType.LONG,
        entry_price=20000.0,
        stop_loss=19800.0,  # Risk $200 per unit
        take_profit=20600.0 # Reward $600 per unit -> R:R = 3.0
    )

    logger.info("\n--- Test 1: Good Long Signal (Default Risk Agent) ---")
    approved_signal1 = risk_agent_default.evaluate_signal(signal1_raw.model_copy(deep=True))
    if approved_signal1:
        logger.info(f"Signal 1 Approved: {approved_signal1.model_dump_json(indent=2)}")
        # Expected risk: $1000 * 0.01 = $10
        # Expected pos size asset: $10 / $200 = 0.05 BTC
        # Expected pos size usd: 0.05 BTC * $20000 = $1000
        assert approved_signal1.risk_per_trade_usd == 10.00
        assert approved_signal1.position_size_asset == 0.05
        assert approved_signal1.position_size_usd == 1000.00
        assert approved_signal1.risk_reward_ratio == 3.0
    else:
        logger.error("Signal 1 was unexpectedly rejected.")

    # Example Signal 2: Good Short Signal (ETH-USD), High Risk Agent
    signal2_raw = TradingSignalModel(
        signal_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        asset=AssetSymbol.ETH_USD,
        timeframe=Timeframe.MIN_15,
        signal_type=SignalType.SHORT,
        entry_price=1500.0,
        stop_loss=1530.0,  # Risk $30 per unit
        take_profit=1410.0 # Reward $90 per unit -> R:R = 3.0
    )
    logger.info("\n--- Test 2: Good Short Signal (High Risk Agent) ---")
    approved_signal2 = risk_agent_high_risk.evaluate_signal(signal2_raw.model_copy(deep=True))
    if approved_signal2:
        logger.info(f"Signal 2 Approved: {approved_signal2.model_dump_json(indent=2)}")
        # Expected risk: $10000 * 0.02 = $200
        # Expected pos size asset: $200 / $30 approx 6.66666667 ETH
        # Expected pos size usd: 6.66666667 ETH * $1500 = $10000
        assert approved_signal2.risk_per_trade_usd == 200.00
        assert abs(approved_signal2.position_size_asset - (200/30)) < 1e-6
        assert abs(approved_signal2.position_size_usd - (200/30 * 1500)) < 1e-6
        assert approved_signal2.risk_reward_ratio == 3.0
    else:
        logger.error("Signal 2 was unexpectedly rejected.")

    # Example Signal 3: Bad R:R Signal
    signal3_raw = TradingSignalModel(
        signal_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        asset=AssetSymbol.SPY,
        timeframe=Timeframe.DAY_1,
        signal_type=SignalType.LONG,
        entry_price=400.0,
        stop_loss=395.0,  # Risk $5
        take_profit=405.0 # Reward $5 -> R:R = 1.0 (below default min 1.5)
    )
    logger.info("\n--- Test 3: Bad R:R Signal (Default Risk Agent) ---")
    rejected_signal3 = risk_agent_default.evaluate_signal(signal3_raw.model_copy(deep=True))
    if rejected_signal3 is None:
        logger.info(f"Signal 3 Correctly Rejected due to R:R. Raw R:R was {signal3_raw.risk_reward_ratio if signal3_raw.risk_reward_ratio is not None else 'N/A (updated by agent)'}. Expected ~1.0")
        # We can check the updated R:R on the raw (but modified) signal if needed, but it won't be returned
        # risk_agent_default._calculate_reward_to_risk updates the signal in place for the check. This is a bit of a side effect for a private method.
        # To avoid this, the evaluate_signal method could create a copy before passing to private methods or private methods could not modify.
        # For now, let's verify the R:R if calculated:
        temp_signal_for_rr_check = signal3_raw.model_copy(deep=True) # ensure original raw signal is not modified by this test
        calculated_rr = risk_agent_default._calculate_reward_to_risk(temp_signal_for_rr_check.entry_price, temp_signal_for_rr_check.stop_loss, temp_signal_for_rr_check.take_profit, temp_signal_for_rr_check.signal_type)
        logger.info(f"Calculated R:R for signal 3: {calculated_rr}")
        assert calculated_rr is not None and calculated_rr < risk_agent_default.min_reward_to_risk_ratio
    else:
        logger.error(f"Signal 3 was unexpectedly approved: {rejected_signal3.model_dump_json(indent=2)}")

    # Example Signal 4: Inconsistent prices (SL through entry for LONG)
    signal4_raw = TradingSignalModel(
        signal_id=str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        asset=AssetSymbol.BTC_USD,
        timeframe=Timeframe.HOUR_1,
        signal_type=SignalType.LONG,
        entry_price=20000.0,
        stop_loss=20100.0,  # SL is above entry for a LONG
        take_profit=20500.0 
    )
    logger.info("\n--- Test 4: Inconsistent Prices Signal (Default Risk Agent) ---")
    rejected_signal4 = risk_agent_default.evaluate_signal(signal4_raw.model_copy(deep=True))
    if rejected_signal4 is None:
        logger.info("Signal 4 Correctly Rejected due to inconsistent prices.")
    else:
        logger.error(f"Signal 4 was unexpectedly approved: {rejected_signal4.model_dump_json(indent=2)}")

    # Test filter_signals
    logger.info("\n--- Test 5: Filter Signals (mix of good and bad) ---")
    all_signals = [
        signal1_raw.model_copy(deep=True), 
        signal2_raw.model_copy(deep=True), # This will use default agent's params
        signal3_raw.model_copy(deep=True),
        signal4_raw.model_copy(deep=True)
    ]
    approved_list = risk_agent_default.filter_signals(all_signals)
    logger.info(f"Filter Signals: Got {len(approved_list)} approved signals out of {len(all_signals)}.")
    # Expected: signal1 should pass, signal2 would pass R:R but sized by default agent, signal3 (bad R:R) fail, signal4 (inconsistent) fail.
    # So, expecting 2 signals if signal2 is evaluated by default agent (its R:R is fine).
    # The R:R for Signal2 is 3.0, so it should pass. Position sizing will be based on $1000, 1%.
    assert len(approved_list) == 2
    if len(approved_list) == 2:
        logger.info(f"Approved signal IDs: {[s.signal_id for s in approved_list]}")
        assert approved_list[0].signal_id == signal1_raw.signal_id
        assert approved_list[1].signal_id == signal2_raw.signal_id
        # Check signal2's pos sizing with default agent
        # Expected risk: $10 (1% of $1000). Risk per unit: $30. Pos size: 10/30 = 0.333... ETH
        assert approved_list[1].risk_per_trade_usd == 10.00
        assert abs(approved_list[1].position_size_asset - (10/30)) < 1e-6
