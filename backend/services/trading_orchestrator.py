import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from sqlalchemy.orm import Session

from backend.models.user import User
from backend.models.portfolio import AssetClass, TradeDirection
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.journal_agent import JournalAgent
from backend.agents.market_regime_detector import MarketRegimeDetector
from backend.services.portfolio_service import PortfolioService

# Configure logging
logger = logging.getLogger(__name__)

class TradingOrchestrator:
    """
    Orchestrates the flow of data between different components of the trading system:
    Market Data -> Signal Scanner -> Risk Manager -> Portfolio Management
    """
    
    def __init__(self, 
                 market_data_agent: MarketDataAgent,
                 signal_scanner_agent: SignalScannerAgent,
                 risk_manager_agent: RiskManagerAgent,
                 journal_agent: JournalAgent = None,
                 regime_detector: MarketRegimeDetector = None):
        """Initialize the orchestrator with agent instances"""
        self.market_data_agent = market_data_agent
        self.signal_scanner = signal_scanner_agent
        self.risk_manager = risk_manager_agent
        self.journal_agent = journal_agent
        self.regime_detector = regime_detector or MarketRegimeDetector()
        
        # Dictionary to track signal processing status
        self._signal_status = {}
        
        logger.info("TradingOrchestrator initialized with provided agents")
    
    async def scan_for_signals(
        self,
        assets: List,
        timeframes: List,
        portfolio_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Scan for trading signals across multiple assets and timeframes.
        
        Args:
            assets: List of assets to scan
            timeframes: List of timeframes to scan
            portfolio_id: Optional portfolio ID to associate with the scan
            
        Returns:
            Dictionary with signals and metadata
        """
        logger.info(f"Scanning for signals: assets={[a.value for a in assets]}, timeframes={[t.value for t in timeframes]}")
        
        signals = []
        
        try:
            # Process each asset and timeframe combination
            for asset in assets:
                for timeframe in timeframes:
                    try:
                        # 1. Fetch market data
                        historical_data = self.market_data_agent.fetch_historical_data(asset, timeframe)
                        if not historical_data or not historical_data.candles:
                            logger.warning(f"No market data found for {asset.value} {timeframe.value}")
                            
                            continue
                            
                        # 2. Scan for signals
                        raw_signals = self.signal_scanner.scan_for_breakout_signals(
                            asset, 
                            timeframe, 
                            historical_data=historical_data
                        )
                        
                        if not raw_signals:
                            logger.info(f"No raw signals found for {asset.value} {timeframe.value}")
                            
                            continue
                            
                        # 3. Apply risk management
                        approved_signals = self.risk_manager.filter_signals(raw_signals)
                        
                        if approved_signals:
                            signals.extend(approved_signals)
                            
                    except Exception as e:
                        logger.error(f"Error processing {asset.value} {timeframe.value}: {str(e)}", exc_info=True)
                        
            # Journal the signals if a journal agent is available
            if signals and self.journal_agent:
                for signal in signals:
                    self.journal_agent.record_signal(signal)
                    
            return {
                "signals": signals,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "count": len(signals)
            }
                
        except Exception as e:
            logger.error(f"Error in scan_for_signals: {str(e)}", exc_info=True)
            return {
                "signals": [],
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
            
    def simulate_win_probability(self, signal: Dict[str, Any]) -> float:
        """
        Simulate win probability for a signal based on its characteristics.
        
        Args:
            signal: The trading signal
            
        Returns:
            Simulated win probability as a float between 0 and 1
        """
        # Base win rate from signal type
        base_win_rate = 0.55  # Default win rate
        
        # Adjust based on market regime if available
        regime_bonus = 0.0
        if "market_regime" in signal:
            regime_map = {
                "bullish_trending": 0.1 if signal["signal_type"] == "long" else -0.1,
                "bearish_trending": 0.1 if signal["signal_type"] == "short" else -0.1,
                "bullish_breakout": 0.15 if signal["signal_type"] == "long" else -0.05,
                "bearish_breakout": 0.15 if signal["signal_type"] == "short" else -0.05,
                "high_volatility": 0.05,
                "low_volatility": -0.05,
                "neutral_ranging": 0.0
            }
            regime_bonus = regime_map.get(signal["market_regime"], 0.0)
        
        # Adjust based on asset class
        asset_bonus = 0.0
        if signal["asset"] in ["BTC-USD", "ETH-USD"]:
            asset_bonus = 0.05  # Crypto tends to have clearer breakouts
        
        # Calculate final win probability
        win_probability = base_win_rate + regime_bonus + asset_bonus
        
        # Ensure within bounds
        return max(0.1, min(0.95, win_probability))
    
    def simulate_expected_return(self, signal: Dict[str, Any]) -> float:
        """
        Simulate expected return for a signal.
        
        Args:
            signal: The trading signal
            
        Returns:
            Simulated expected return percentage
        """
        # Base expected return
        base_return = 2.0  # 2% default expected return
        
        # Adjust based on volatility
        volatility_factor = 1.0
        if "volatility" in signal:
            volatility_factor = 0.5 + (signal["volatility"] * 0.5)  # Scale volatility impact
        
        # Adjust based on market regime
        regime_factor = 1.0
        if "market_regime" in signal:
            regime_map = {
                "bullish_trending": 1.2 if signal["signal_type"] == "long" else 0.8,
                "bearish_trending": 1.2 if signal["signal_type"] == "short" else 0.8,
                "bullish_breakout": 1.5 if signal["signal_type"] == "long" else 0.7,
                "bearish_breakout": 1.5 if signal["signal_type"] == "short" else 0.7,
                "high_volatility": 1.3,
                "low_volatility": 0.7,
                "neutral_ranging": 0.9
            }
            regime_factor = regime_map.get(signal["market_regime"], 1.0)
        
        # Calculate final expected return
        expected_return = base_return * volatility_factor * regime_factor
        
        return round(expected_return, 2)
    
    def get_signal_historical_stats(self, asset, signal_type, timeframe) -> Dict[str, Any]:
        """
        Get historical statistics for a particular type of signal.
        
        Args:
            asset: Asset symbol
            signal_type: Type of signal (long/short)
            timeframe: Timeframe
            
        Returns:
            Dictionary with historical stats
        """
        # In a real implementation, this would query a database of historical signals
        # For this example, we'll return simulated data
        if signal_type == "long":
            win_rate = 0.65 if asset in ["BTC-USD", "ETH-USD"] else 0.58
            avg_return = 2.8 if asset in ["BTC-USD", "ETH-USD"] else 1.9
            profit_factor = 2.1 if asset in ["BTC-USD", "ETH-USD"] else 1.6
        else:  # short
            win_rate = 0.62 if asset in ["BTC-USD", "ETH-USD"] else 0.55
            avg_return = 2.5 if asset in ["BTC-USD", "ETH-USD"] else 1.7
            profit_factor = 1.9 if asset in ["BTC-USD", "ETH-USD"] else 1.5
        
        return {
            "win_rate": win_rate,
            "avg_return_pct": avg_return,
            "avg_duration_hours": 36 if (hasattr(timeframe, 'value') and timeframe.value in ["1d"]) or timeframe in ["1d"] else 12,
            "profit_factor": profit_factor,
            "sample_size": 120 if asset in ["BTC-USD", "ETH-USD"] else 85,
            "last_updated": datetime.now().isoformat()
        }
        
    async def process_signals_for_user(
        self, 
        db: Session, 
        user: User, 
        portfolio_id: Optional[int] = None,
        auto_execute: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process trading signals for a specific user
        
        Args:
            db: Database session
            user: User to process signals for
            portfolio_id: Optional portfolio ID to execute trades in
            auto_execute: Whether to automatically execute trades
            
        Returns:
            List of processed signals
        """
        try:
            # Get user's active watchlist or use default assets
            watchlist = await self._get_user_watchlist(db, user.id)
            
            # Get market data for watchlist assets
            market_data = await self._fetch_market_data(watchlist)
            
            # Generate signals from market data
            raw_signals = await self._generate_signals(market_data)
            
            # Detect market regime for each asset
            for signal in raw_signals:
                symbol = signal.get("asset", "")
                asset_class = signal.get("asset_class", "STOCK")
                
                regime_info = await self.regime_detector.detect_regime(symbol, asset_class)
                signal["regime"] = regime_info.get("primary_regime", "unknown")
                signal["regime_strength"] = regime_info.get("strength", 0.5)
                
                # Log regime detection
                logger.info(f"Detected {signal['regime']} regime for {symbol} with strength {signal['regime_strength']}")
            
            # Filter signals through risk manager
            approved_signals = await self._filter_signals(raw_signals)
            
            # Record all signals in journal
            for signal in raw_signals:
                signal["is_approved"] = signal in approved_signals
                await self.journal_agent.record_signal(signal)
            
            # Execute approved signals if auto-execute is enabled and portfolio ID is provided
            if auto_execute and portfolio_id:
                await self._execute_signals(db, user, portfolio_id, approved_signals)
            
            return approved_signals
        
        except Exception as e:
            logger.error(f"Error processing signals: {str(e)}")
            return []
    
    async def _get_user_watchlist(self, db: Session, user_id: int) -> List[Dict[str, Any]]:
        """Get the user's active watchlist or return default assets"""
        # TODO: Implement watchlist functionality
        # For now, return default assets to monitor
        return [
            {"symbol": "BTC-USD", "asset_class": "CRYPTO", "timeframes": ["1h", "4h", "1d"]},
            {"symbol": "ETH-USD", "asset_class": "CRYPTO", "timeframes": ["1h", "4h", "1d"]},
            {"symbol": "SPY", "asset_class": "ETF", "timeframes": ["1h", "4h", "1d"]},
            {"symbol": "AAPL", "asset_class": "STOCK", "timeframes": ["1h", "4h", "1d"]},
            {"symbol": "MSFT", "asset_class": "STOCK", "timeframes": ["1h", "4h", "1d"]},
            {"symbol": "EURUSD=X", "asset_class": "FOREX", "timeframes": ["1h", "4h", "1d"]}
        ]
    
    async def _fetch_market_data(self, watchlist: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fetch market data for all assets in the watchlist"""
        market_data = {}
        
        for asset in watchlist:
            symbol = asset["symbol"]
            asset_class = asset["asset_class"]
            timeframes = asset.get("timeframes", ["1d"])
            
            asset_data = {}
            for timeframe in timeframes:
                try:
                    # Get historical data for the asset
                    data = await self.market_data_agent.get_historical_data(
                        symbol=symbol,
                        asset_class=asset_class,
                        interval=timeframe,
                        periods=100  # Use last 100 periods for analysis
                    )
                    asset_data[timeframe] = data
                except Exception as e:
                    logger.error(f"Error fetching {timeframe} data for {symbol}: {str(e)}")
            
            if asset_data:
                market_data[symbol] = {
                    "data": asset_data,
                    "asset_class": asset_class
                }
        
        return market_data
    
    async def _generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from market data"""
        signals = []
        
        for symbol, data in market_data.items():
            asset_class = data["asset_class"]
            
            for timeframe, ohlcv_data in data["data"].items():
                try:
                    # Scan for signals in this timeframe
                    timeframe_signals = await self.signal_scanner.scan_for_signals(
                        symbol=symbol,
                        asset_class=asset_class,
                        timeframe=timeframe,
                        data=ohlcv_data
                    )
                    
                    # Add metadata to signals
                    for signal in timeframe_signals:
                        signal["timeframe"] = timeframe
                        signal["timestamp"] = datetime.now().isoformat()
                        signal["signal_id"] = f"{symbol}_{timeframe}_{datetime.now().timestamp()}"
                    
                    signals.extend(timeframe_signals)
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol} {timeframe}: {str(e)}")
        
        return signals
    
    async def _filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter signals using risk management rules"""
        approved_signals = []
        
        for signal in signals:
            try:
                # Evaluate signal with risk manager
                risk_result = await self.risk_manager.evaluate_signal(signal)
                
                if risk_result.get("is_approved", False):
                    # Add risk metrics to approved signal
                    approved_signals.append(risk_result)
            except Exception as e:
                logger.error(f"Error filtering signal: {str(e)}")
        
        return approved_signals
    
    async def _execute_signals(
        self, 
        db: Session, 
        user: User, 
        portfolio_id: int, 
        signals: List[Dict[str, Any]]
    ) -> None:
        """Execute approved signals as trades in the specified portfolio"""
        for signal in signals:
            try:
                # Only execute new signals that haven't been processed
                signal_id = signal.get("signal_id")
                if signal_id in self._signal_status:
                    continue
                
                # Mark signal as processing
                self._signal_status[signal_id] = "processing"
                
                # Extract signal details
                symbol = signal.get("asset", "")
                asset_class_str = signal.get("asset_class", "").upper()
                asset_class = self._get_asset_class(asset_class_str, symbol)
                
                signal_type = signal.get("signal_type", "").upper()
                direction = TradeDirection.LONG if "LONG" in signal_type or "BUY" in signal_type else TradeDirection.SHORT
                
                price = signal.get("price", 0)
                if not price:
                    logger.error(f"Signal for {symbol} has no price, skipping execution")
                    self._signal_status[signal_id] = "error"
                    continue
                
                # Get risk parameters
                position_size_pct = signal.get("position_size", 2.0)  # Default to 2% of portfolio
                stop_loss = signal.get("stop_loss")
                take_profit = signal.get("take_profit")
                risk_percent = signal.get("risk_percent", 2.0)
                
                # Get strategy info
                strategy_id = signal.get("strategy_id")
                strategy_name = signal.get("strategy_name", "Trading SaaS")
                
                # Create notes from signal data
                notes = f"Signal from {strategy_name}. "
                if "regime" in signal:
                    notes += f"Market regime: {signal.get('regime')}. "
                if "confidence" in signal:
                    notes += f"Signal confidence: {signal.get('confidence')}%. "
                
                # Create tags for filtering
                tags = ["auto_signal"]
                if "timeframe" in signal:
                    tags.append(f"timeframe:{signal.get('timeframe')}")
                if "strategy_name" in signal:
                    tags.append(f"strategy:{strategy_name}")
                
                # Estimate quantity
                # Get portfolio details to calculate position size
                portfolio = await PortfolioService.get_portfolio(db, portfolio_id, user.id)
                portfolio_value = portfolio.current_value
                position_value = portfolio_value * (position_size_pct / 100)
                quantity = position_value / price
                
                # Create position in portfolio
                position = await PortfolioService.create_position(
                    db=db,
                    portfolio_id=portfolio_id,
                    user_id=user.id,
                    symbol=symbol,
                    asset_class=asset_class,
                    direction=direction,
                    quantity=quantity,
                    entry_price=price,
                    strategy_id=strategy_id,
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                    max_risk_percent=risk_percent,
                    notes=notes,
                    tags=tags
                )
                
                # Update signal status
                self._signal_status[signal_id] = "executed"
                logger.info(f"Executed signal {signal_id} as position {position.id} in portfolio {portfolio_id}")
                
            except Exception as e:
                logger.error(f"Error executing signal: {str(e)}")
                self._signal_status[signal_id] = "error"
    
    def _get_asset_class(self, asset_class_str: str, symbol: str) -> AssetClass:
        """Determine the asset class from string or symbol patterns"""
        if asset_class_str and hasattr(AssetClass, asset_class_str):
            return AssetClass(asset_class_str)
        
        # Try to determine from symbol
        if "-USD" in symbol:
            return AssetClass.CRYPTO
        elif "=" in symbol or symbol.endswith("USD"):
            return AssetClass.FOREX
        else:
            return AssetClass.STOCK  # Default
