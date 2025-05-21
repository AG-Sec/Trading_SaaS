"""
End-to-end tests for the complete trading signal pipeline.
"""
import pytest
import pandas as pd
import json
from datetime import datetime, timedelta
import os

from shared_types.models import AssetSymbol, Timeframe, SignalType, HistoricalDataModel, CandleModel

def test_e2e_signal_generation_pipeline(market_agent, scanner_agent, risk_agent, regime_detector, journal_agent):
    """Test the complete signal generation pipeline from market data to journal entry."""
    # Test assets and timeframes
    assets = [AssetSymbol.BTC_USD, AssetSymbol.SPY]
    timeframes = [Timeframe.HOUR_4, Timeframe.DAY_1]
    
    # Initialize counters
    total_signals = 0
    total_signals_after_regime = 0
    total_approved_signals = 0
    
    # Process each asset and timeframe
    results = {}
    
    for asset in assets:
        results[asset.value] = {}
        for timeframe in timeframes:
            results[asset.value][timeframe.value] = {}
            
            # 1. Fetch market data
            market_data = market_agent.fetch_historical_data(asset, timeframe)
            if not market_data or not market_data.candles:
                continue
            
            # Convert to DataFrame for regime detection
            df = pd.DataFrame([candle.model_dump() for candle in market_data.candles])
            
            # 2. Detect market regime
            regime = regime_detector.detect_regime(df, asset, timeframe)
            
            # 3. Get adjusted parameters based on regime
            default_params = {
                'breakout_period': scanner_agent.breakout_period,
                'rsi_lower_threshold': scanner_agent.rsi_lower_threshold,
                'rsi_upper_threshold': scanner_agent.rsi_upper_threshold,
                'atr_multiplier_sl': scanner_agent.atr_multiplier_sl,
                'atr_multiplier_tp': scanner_agent.atr_multiplier_tp
            }
            
            adjusted_params = regime.get_adjusted_parameters(default_params)
            
            # 4. Generate signals with standard parameters
            standard_signals = scanner_agent.scan_for_breakout_signals(
                asset, timeframe, market_data=market_data
            )
            
            # 5. Create a regime-adapted scanner
            adapted_scanner = scanner_agent.__class__(
                market_data_agent=market_agent,
                breakout_period=adjusted_params.get('breakout_period', scanner_agent.breakout_period),
                rsi_lower_threshold=adjusted_params.get('rsi_lower_threshold', scanner_agent.rsi_lower_threshold),
                rsi_upper_threshold=adjusted_params.get('rsi_upper_threshold', scanner_agent.rsi_upper_threshold),
                atr_multiplier_sl=adjusted_params.get('atr_multiplier_sl', scanner_agent.atr_multiplier_sl),
                atr_multiplier_tp=adjusted_params.get('atr_multiplier_tp', scanner_agent.atr_multiplier_tp)
            )
            
            # 6. Generate signals with adapted parameters
            adapted_signals = adapted_scanner.scan_for_breakout_signals(
                asset, timeframe, market_data=market_data
            )
            
            # 7. Evaluate signals with risk management
            standard_approved = []
            for signal in standard_signals:
                approved = risk_agent.evaluate_signal(signal)
                if approved:
                    standard_approved.append(approved)
                    # Record in journal
                    journal_agent.record_signal(approved)
            
            adapted_approved = []
            for signal in adapted_signals:
                approved = risk_agent.evaluate_signal(signal)
                if approved:
                    adapted_approved.append(approved)
                    # Record in journal
                    journal_agent.record_signal(approved)
            
            # Store results
            results[asset.value][timeframe.value] = {
                "regime": regime.regime_type.value,
                "regime_strength": regime.strength,
                "standard_signals": len(standard_signals),
                "adapted_signals": len(adapted_signals),
                "standard_approved": len(standard_approved),
                "adapted_approved": len(adapted_approved)
            }
            
            # Update totals
            total_signals += len(standard_signals)
            total_signals_after_regime += len(adapted_signals)
            total_approved_signals += len(standard_approved) + len(adapted_approved)
    
    # Log results
    print("\nE2E Test Results:")
    print(f"Total signals generated (standard parameters): {total_signals}")
    print(f"Total signals generated (regime-adapted): {total_signals_after_regime}")
    print(f"Total signals approved: {total_approved_signals}")
    
    # Save detailed results to a file
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"e2e_pipeline_test_{timestamp}.json")
    
    with open(result_file, 'w') as f:
        json.dump({
            "summary": {
                "total_signals": total_signals,
                "total_signals_after_regime": total_signals_after_regime,
                "total_approved_signals": total_approved_signals,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_results": results
        }, f, indent=2)
    
    print(f"Detailed results saved to: {result_file}")
    
    # Verify the pipeline produced meaningful results
    assert total_signals_after_regime > 0, "No signals were generated with regime adaptation"
    assert total_approved_signals > 0, "No signals were approved by risk management"

def test_e2e_historical_backtest(market_agent, scanner_agent, risk_agent, regime_detector, journal_agent):
    """Test a historical backtest over a specific period."""
    # Test parameters
    asset = AssetSymbol.BTC_USD
    timeframe = Timeframe.DAY_1
    
    # Date range (use a shorter period for testing)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # 60 days for testing
    
    # Create backtesting report
    trades = []
    signals = []
    
    # Fetch historical data
    market_data = market_agent.fetch_historical_data(asset, timeframe)
    if not market_data or not market_data.candles:
        pytest.skip("No market data available for backtesting")
    
    # Filter data to our date range
    df = pd.DataFrame([candle.model_dump() for candle in market_data.candles])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    if len(df) < 30:  # Need enough data for indicators
        pytest.skip("Insufficient market data for backtesting")
    
    # Process each day in the date range
    dates = sorted(df['timestamp'].unique())
    
    # Initialize performance tracking
    starting_balance = 10000.0
    current_balance = starting_balance
    max_drawdown = 0.0
    equity_curve = [(dates[0], current_balance)]
    open_positions = []
    
    # Set up risk manager
    risk_agent.account_balance_usd = current_balance
    
    for i, current_date in enumerate(dates[30:], 30):  # Start after warmup period
        # Get data up to current date (simulate "today")
        current_df = df[df['timestamp'] <= current_date].copy()
        
        # Detect regime
        regime = regime_detector.detect_regime(current_df, asset, timeframe)
        
        # Get adjusted parameters
        default_params = {
            'breakout_period': scanner_agent.breakout_period,
            'rsi_lower_threshold': scanner_agent.rsi_lower_threshold,
            'rsi_upper_threshold': scanner_agent.rsi_upper_threshold,
            'atr_multiplier_sl': scanner_agent.atr_multiplier_sl,
            'atr_multiplier_tp': scanner_agent.atr_multiplier_tp
        }
        
        adjusted_params = regime.get_adjusted_parameters(default_params)
        
        # Create adapted scanner
        adapted_scanner = scanner_agent.__class__(
            market_data_agent=market_agent,
            breakout_period=adjusted_params.get('breakout_period', scanner_agent.breakout_period),
            rsi_lower_threshold=adjusted_params.get('rsi_lower_threshold', scanner_agent.rsi_lower_threshold),
            rsi_upper_threshold=adjusted_params.get('rsi_upper_threshold', scanner_agent.rsi_upper_threshold),
            atr_multiplier_sl=adjusted_params.get('atr_multiplier_sl', scanner_agent.atr_multiplier_sl),
            atr_multiplier_tp=adjusted_params.get('atr_multiplier_tp', scanner_agent.atr_multiplier_tp)
        )
        
        # Generate signals for this day
        # Convert current_df back to HistoricalDataModel format
        
        # Convert current_df back to HistoricalDataModel
        candles = [
            CandleModel(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            for _, row in current_df.iterrows()
        ]
        
        current_market_data = HistoricalDataModel(
            asset=asset,
            timeframe=timeframe,
            candles=candles
        )
        
        # Generate signals
        day_signals = adapted_scanner.scan_for_breakout_signals(
            asset, timeframe, market_data=current_market_data
        )
        
        # Risk manager evaluation
        for signal in day_signals:
            approved = risk_agent.evaluate_signal(signal)
            if approved:
                signals.append(approved)
                # Simulate opening a position
                position = {
                    "signal_id": approved.signal_id,
                    "entry_date": current_date,
                    "entry_price": approved.entry_price,
                    "stop_loss": approved.stop_loss,
                    "take_profit": approved.take_profit,
                    "position_size_usd": approved.position_size_usd,
                    "signal_type": approved.signal_type,
                    "active": True
                }
                open_positions.append(position)
        
        # Check if next day exists for price comparison
        if i + 1 < len(dates):
            next_day = df[df['timestamp'] == dates[i + 1]].iloc[0]
            next_high = next_day['high']
            next_low = next_day['low']
            next_close = next_day['close']
            
            # Process open positions
            positions_to_remove = []
            
            for idx, position in enumerate(open_positions):
                if not position["active"]:
                    positions_to_remove.append(idx)
                    continue
                
                # Check for stop loss hit
                if (position["signal_type"] == SignalType.LONG and next_low <= position["stop_loss"]) or \
                   (position["signal_type"] == SignalType.SHORT and next_high >= position["stop_loss"]):
                    # Stop loss hit
                    pnl = position["position_size_usd"] * (position["stop_loss"] / position["entry_price"] - 1) \
                        if position["signal_type"] == SignalType.LONG else \
                        position["position_size_usd"] * (1 - position["stop_loss"] / position["entry_price"])
                    
                    current_balance += pnl
                    
                    trades.append({
                        "signal_id": position["signal_id"],
                        "entry_date": position["entry_date"],
                        "exit_date": dates[i + 1],
                        "entry_price": position["entry_price"],
                        "exit_price": position["stop_loss"],
                        "position_size_usd": position["position_size_usd"],
                        "pnl": pnl,
                        "signal_type": position["signal_type"].value,
                        "exit_type": "stop_loss"
                    })
                    
                    position["active"] = False
                    positions_to_remove.append(idx)
                
                # Check for take profit hit
                elif (position["signal_type"] == SignalType.LONG and next_high >= position["take_profit"]) or \
                     (position["signal_type"] == SignalType.SHORT and next_low <= position["take_profit"]):
                    # Take profit hit
                    pnl = position["position_size_usd"] * (position["take_profit"] / position["entry_price"] - 1) \
                        if position["signal_type"] == SignalType.LONG else \
                        position["position_size_usd"] * (1 - position["take_profit"] / position["entry_price"])
                    
                    current_balance += pnl
                    
                    trades.append({
                        "signal_id": position["signal_id"],
                        "entry_date": position["entry_date"],
                        "exit_date": dates[i + 1],
                        "entry_price": position["entry_price"],
                        "exit_price": position["take_profit"],
                        "position_size_usd": position["position_size_usd"],
                        "pnl": pnl,
                        "signal_type": position["signal_type"].value,
                        "exit_type": "take_profit"
                    })
                    
                    position["active"] = False
                    positions_to_remove.append(idx)
            
            # Remove closed positions
            for idx in sorted(positions_to_remove, reverse=True):
                open_positions.pop(idx)
            
            # Update equity curve
            equity_curve.append((dates[i + 1], current_balance))
            
            # Update max drawdown
            max_balance = max(point[1] for point in equity_curve)
            current_drawdown = (max_balance - current_balance) / max_balance
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Update risk manager balance
            risk_agent.account_balance_usd = current_balance
    
    # Close any remaining positions at the last price
    if open_positions and len(dates) > 0:
        last_close = df[df['timestamp'] == dates[-1]].iloc[0]['close']
        
        for position in open_positions:
            if position["active"]:
                # Calculate PnL
                pnl = position["position_size_usd"] * (last_close / position["entry_price"] - 1) \
                    if position["signal_type"] == SignalType.LONG else \
                    position["position_size_usd"] * (1 - last_close / position["entry_price"])
                
                current_balance += pnl
                
                trades.append({
                    "signal_id": position["signal_id"],
                    "entry_date": position["entry_date"],
                    "exit_date": dates[-1],
                    "entry_price": position["entry_price"],
                    "exit_price": last_close,
                    "position_size_usd": position["position_size_usd"],
                    "pnl": pnl,
                    "signal_type": position["signal_type"].value,
                    "exit_type": "simulation_end"
                })
    
    # Calculate backtest metrics
    winning_trades = [trade for trade in trades if trade["pnl"] > 0]
    losing_trades = [trade for trade in trades if trade["pnl"] <= 0]
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    average_win = sum(trade["pnl"] for trade in winning_trades) / len(winning_trades) if winning_trades else 0
    average_loss = sum(trade["pnl"] for trade in losing_trades) / len(losing_trades) if losing_trades else 0
    profit_factor = abs(sum(trade["pnl"] for trade in winning_trades) / sum(trade["pnl"] for trade in losing_trades)) if losing_trades and sum(trade["pnl"] for trade in losing_trades) != 0 else float('inf')
    
    total_return = (current_balance - starting_balance) / starting_balance * 100
    
    # Log backtest results
    print("\nBacktest Results:")
    print(f"Asset: {asset.value}, Timeframe: {timeframe.value}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Total Signals Generated: {len(signals)}")
    print(f"Total Trades Executed: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Win: ${average_win:.2f}")
    print(f"Average Loss: ${average_loss:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Starting Balance: ${starting_balance:.2f}")
    print(f"Final Balance: ${current_balance:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Save backtest results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"backtest_{asset.value}_{timeframe.value}_{timestamp}.json")
    
    # Convert datetime objects to strings for JSON serialization
    equity_curve_serializable = [(dt.isoformat() if hasattr(dt, 'isoformat') else str(dt), balance) for dt, balance in equity_curve]
    trades_serializable = []
    
    for trade in trades:
        trade_copy = trade.copy()
        if hasattr(trade_copy["entry_date"], 'isoformat'):
            trade_copy["entry_date"] = trade_copy["entry_date"].isoformat()
        if hasattr(trade_copy["exit_date"], 'isoformat'):
            trade_copy["exit_date"] = trade_copy["exit_date"].isoformat()
        trades_serializable.append(trade_copy)
    
    with open(result_file, 'w') as f:
        json.dump({
            "summary": {
                "asset": asset.value,
                "timeframe": timeframe.value,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_signals": len(signals),
                "total_trades": len(trades),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "average_win": average_win,
                "average_loss": average_loss,
                "max_drawdown": max_drawdown,
                "starting_balance": starting_balance,
                "final_balance": current_balance,
                "total_return": total_return
            },
            "trades": trades_serializable,
            "equity_curve": equity_curve_serializable
        }, f, indent=2)
    
    print(f"Backtest results saved to: {result_file}")
    
    # Verify that the backtest produced meaningful results
    assert len(signals) > 0, "No signals were generated during backtest"
    assert len(trades) > 0, "No trades were executed during backtest"
