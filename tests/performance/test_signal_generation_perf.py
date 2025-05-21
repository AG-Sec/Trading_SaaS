"""
Performance tests for signal generation components.
"""
import pytest
import time
import pandas as pd
from datetime import datetime, timedelta

from shared_types.models import AssetSymbol, Timeframe, HistoricalDataModel, CandleModel, SignalType

def test_market_data_performance(market_agent, sample_assets, sample_timeframes):
    """Test the performance of market data fetching."""
    results = {}
    
    for asset in sample_assets:
        results[asset.value] = {}
        for timeframe in sample_timeframes:
            # Time the data fetch
            start_time = time.time()
            data = market_agent.fetch_historical_data(asset, timeframe)
            elapsed = time.time() - start_time
            
            # Record result
            if data and data.candles:
                results[asset.value][timeframe.value] = {
                    "elapsed_seconds": elapsed,
                    "candle_count": len(data.candles)
                }
            else:
                results[asset.value][timeframe.value] = {
                    "elapsed_seconds": elapsed,
                    "candle_count": 0
                }
    
    # Log results
    print("\nMarket Data Fetch Performance:")
    for asset, timeframes in results.items():
        print(f"  {asset}:")
        for timeframe, result in timeframes.items():
            print(f"    {timeframe}: {result['elapsed_seconds']:.4f} seconds, {result['candle_count']} candles")
            
            # Assert that data fetching is reasonably fast
            assert result['elapsed_seconds'] < 5.0, f"Data fetching for {asset} {timeframe} took too long: {result['elapsed_seconds']} seconds"

def test_signal_generation_performance(market_agent, scanner_agent, sample_assets, sample_timeframes):
    """Test the performance of signal generation."""
    results = {}
    
    # Warm up the cache first by fetching all data
    for asset in sample_assets:
        for timeframe in sample_timeframes:
            market_agent.fetch_historical_data(asset, timeframe)
    
    # Now test signal generation performance
    for asset in sample_assets:
        results[asset.value] = {}
        for timeframe in sample_timeframes:
            # Get data
            data = market_agent.fetch_historical_data(asset, timeframe)
            if not data or not data.candles:
                continue
                
            # Time the signal generation process
            start_time = time.time()
            signals = scanner_agent.scan_for_breakout_signals(asset, timeframe, market_data=data)
            elapsed = time.time() - start_time
            
            # Record result
            results[asset.value][timeframe.value] = {
                "elapsed_seconds": elapsed,
                "signal_count": len(signals)
            }
    
    # Log results
    print("\nSignal Generation Performance:")
    for asset, timeframes in results.items():
        print(f"  {asset}:")
        for timeframe, result in timeframes.items():
            print(f"    {timeframe}: {result['elapsed_seconds']:.4f} seconds, {result['signal_count']} signals")
            
            # Assert that signal generation is reasonably fast
            assert result['elapsed_seconds'] < 1.0, f"Signal generation for {asset} {timeframe} took too long: {result['elapsed_seconds']} seconds"

def test_regime_detection_performance(market_agent, regime_detector, sample_assets, sample_timeframes):
    """Test the performance of market regime detection."""
    results = {}
    
    # Warm up the cache first by fetching all data
    for asset in sample_assets:
        for timeframe in sample_timeframes:
            market_agent.fetch_historical_data(asset, timeframe)
    
    # Now test regime detection performance
    for asset in sample_assets:
        results[asset.value] = {}
        for timeframe in sample_timeframes:
            # Get data
            data = market_agent.fetch_historical_data(asset, timeframe)
            if not data or not data.candles:
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame([candle.model_dump() for candle in data.candles])
            
            # Time the regime detection process
            start_time = time.time()
            regime = regime_detector.detect_regime(df, asset, timeframe)
            elapsed = time.time() - start_time
            
            # Record result
            results[asset.value][timeframe.value] = {
                "elapsed_seconds": elapsed,
                "regime_type": regime.regime_type.value,
                "regime_strength": regime.strength
            }
    
    # Log results
    print("\nRegime Detection Performance:")
    for asset, timeframes in results.items():
        print(f"  {asset}:")
        for timeframe, result in timeframes.items():
            print(f"    {timeframe}: {result['elapsed_seconds']:.4f} seconds, regime: {result['regime_type']}")
            
            # Assert that regime detection is reasonably fast
            assert result['elapsed_seconds'] < 0.5, f"Regime detection for {asset} {timeframe} took too long: {result['elapsed_seconds']} seconds"

def test_risk_evaluation_performance(risk_agent):
    """Test the performance of risk evaluation."""
    from uuid import uuid4
    import random
    from shared_types.models import TradingSignalModel
    
    # Create a large batch of test signals
    batch_sizes = [10, 100, 1000]
    results = {}
    
    for size in batch_sizes:
        test_signals = []
        
        # Generate test signals
        for i in range(size):
            # Randomize parameters a bit
            price = 50000.0 + random.uniform(-5000, 5000)
            risk_pct = random.uniform(0.01, 0.05)  # 1-5% risk
            reward_risk = random.uniform(1.0, 3.0)  # 1-3x reward to risk
            
            is_long = random.choice([True, False])
            
            if is_long:
                stop_loss = price * (1 - risk_pct)
                take_profit = price * (1 + risk_pct * reward_risk)
                signal_type = SignalType.LONG
            else:
                stop_loss = price * (1 + risk_pct)
                take_profit = price * (1 - risk_pct * reward_risk)
                signal_type = SignalType.SHORT
            
            signal = TradingSignalModel(
                signal_id=str(uuid4()),
                generated_at=datetime.now(),
                asset=random.choice(list(AssetSymbol)),
                timeframe=random.choice(list(Timeframe)),
                signal_type=signal_type,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={"test": True, "batch_id": i}
            )
            
            test_signals.append(signal)
        
        # Measure evaluation time
        start_time = time.time()
        approved_signals = []
        
        for signal in test_signals:
            approved = risk_agent.evaluate_signal(signal)
            if approved:
                approved_signals.append(approved)
                
        elapsed = time.time() - start_time
        
        # Record results
        results[size] = {
            "total_signals": size,
            "approved_signals": len(approved_signals),
            "total_time": elapsed,
            "avg_time_per_signal": elapsed / size if size > 0 else 0
        }
    
    # Log results
    print("\nRisk Evaluation Performance:")
    for size, result in results.items():
        print(f"  Batch size {size}:")
        print(f"    Total time: {result['total_time']:.4f} seconds")
        print(f"    Avg time per signal: {result['avg_time_per_signal']:.6f} seconds")
        print(f"    Approval rate: {result['approved_signals'] / result['total_signals']:.2%}")
        
        # Assert evaluation speed is reasonable (scales roughly linearly)
        assert result['total_time'] < size * 0.001 + 0.1, f"Risk evaluation for {size} signals took too long: {result['total_time']} seconds"

def test_backtest_performance(market_agent, scanner_agent, risk_agent, test_date_range):
    """Test the performance of backtesting."""
    start_date, end_date = test_date_range
    
    # Different test periods
    periods = [
        {"name": "1 Week", "days": 7},
        {"name": "1 Month", "days": 30},
        {"name": "3 Months", "days": 90}
    ]
    
    # Test with BTC-USD
    asset = AssetSymbol.BTC_USD
    timeframe = Timeframe.DAY_1
    
    results = {}
    
    for period in periods:
        # Set period
        period_end = end_date
        period_start = period_end - timedelta(days=period["days"])
        
        # Fetch data
        data = market_agent.fetch_historical_data(asset, timeframe)
        if not data or not data.candles:
            continue
            
        # Convert to DataFrame and filter to period
        df = pd.DataFrame([candle.model_dump() for candle in data.candles])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= period_start) & (df['timestamp'] <= period_end)]
        
        if len(df) < 10:  # Skip if not enough data
            continue
        
        # Prepare for backtest
        start_time = time.time()
        
        # Simple simulation of backtest
        trades = []
        signals = []
        days_processed = 0
        
        # Process each day in sequence
        for day in sorted(df['timestamp'].unique()):
            days_processed += 1
            
            # Get data up to this day
            current_df = df[df['timestamp'] <= day]
            
            # Generate signals
            # Create HistoricalDataModel from current DataFrame
            
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
            
            day_data = HistoricalDataModel(
                asset=asset,
                timeframe=timeframe,
                candles=candles
            )
            
            # Generate signals
            day_signals = scanner_agent.scan_for_breakout_signals(
                asset, timeframe, market_data=day_data
            )
            
            # Process signals
            for signal in day_signals:
                # Evaluate with risk management
                approved = risk_agent.evaluate_signal(signal)
                if approved:
                    signals.append(approved)
                    
                    # Simulate trade based on signal
                    # (simplified for performance test)
                    trades.append({
                        "signal_id": approved.signal_id,
                        "entry_date": day,
                        "entry_price": approved.entry_price,
                        "signal_type": approved.signal_type,
                        "position_size": approved.position_size_asset
                    })
        
        elapsed = time.time() - start_time
        
        # Record results
        results[period["name"]] = {
            "period_days": period["days"],
            "days_processed": days_processed,
            "signals_generated": len(signals),
            "trades_executed": len(trades),
            "total_time": elapsed,
            "avg_time_per_day": elapsed / days_processed if days_processed > 0 else 0
        }
    
    # Log results
    print("\nBacktest Performance:")
    for period_name, result in results.items():
        print(f"  Period {period_name} ({result['period_days']} days):")
        print(f"    Total time: {result['total_time']:.4f} seconds")
        print(f"    Avg time per day: {result['avg_time_per_day']:.4f} seconds")
        print(f"    Signals generated: {result['signals_generated']}")
        print(f"    Trades executed: {result['trades_executed']}")
        
        # Assert backtest speed is reasonable
        max_allowed_time = result['period_days'] * 0.1  # allow 0.1 second per day on average
        assert result['total_time'] < max_allowed_time, f"Backtest for {period_name} took too long: {result['total_time']} seconds"
