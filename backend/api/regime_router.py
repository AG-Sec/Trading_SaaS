from fastapi import APIRouter, HTTPException, Path, Query
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from shared_types import AssetSymbol, Timeframe
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.market_regime_detector import MarketRegimeDetector, MarketRegimeType
from backend.agents.regime_dashboard import RegimeDashboard
from backend.agents.backtesting import Backtester
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.regime_optimizer import RegimeOptimizer, RegimeThresholds

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate agents globally for this router instance
market_agent = MarketDataAgent()
regime_detector = MarketRegimeDetector()
scanner_agent = SignalScannerAgent(market_data_agent=market_agent)
risk_agent = RiskManagerAgent()
backtester = Backtester(market_agent, scanner_agent, risk_agent)
regime_dashboard = RegimeDashboard(market_agent)
regime_optimizer = RegimeOptimizer(market_agent, backtester, regime_detector)

class TimeRangeEnum(str, Enum):
    ONE_WEEK = "1w"
    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"
    SIX_MONTHS = "6m"
    ONE_YEAR = "1y"
    CUSTOM = "custom"

@router.get(
    "/summary",
    response_model=Dict[str, Any],
    summary="Get market regime summary across multiple assets",
    tags=["Market Regimes"]
)
async def get_market_regime_summary(
    timeframe: str = Query("1d", description="Timeframe to analyze"),
):
    """
    Get a summary of market regimes across all supported assets for a specific timeframe.
    Includes regime distribution, dominant regime, and market health indicators.
    """
    try:
        # Get the timeframe
        try:
            timeframe_enum = Timeframe(timeframe.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid timeframe: {timeframe}. Supported: {', '.join([e.value for e in Timeframe])}"
            )
            
        # Define all supported assets
        supported_assets = [AssetSymbol.BTC_USD, AssetSymbol.ETH_USD, AssetSymbol.EURUSD_FX, AssetSymbol.SPY]
        
        # Get regimes for all assets
        regimes = {}
        regime_counts = {regime_type.value: 0 for regime_type in MarketRegimeType}
        total_assets = 0
        
        for asset in supported_assets:
            try:
                historical_data = market_agent.fetch_historical_data(asset, timeframe_enum)
                if historical_data and historical_data.candles:
                    # Convert candles to DataFrame
                    candles_data = [candle.model_dump() for candle in historical_data.candles]
                    df = pd.DataFrame(candles_data)
                    # Call detect_regime with the required parameters
                    current_regime = regime_detector.detect_regime(df, asset, timeframe_enum)
                    regimes[asset.value] = {
                        "regime_type": current_regime.regime_type.value,
                        "strength": current_regime.strength,
                        "metrics": current_regime.metrics
                    }
                    regime_counts[current_regime.regime_type.value] += 1
                    total_assets += 1
            except Exception as e:
                logger.warning(f"Error getting regime for {asset.value}: {str(e)}")
        
        # Calculate regime distribution percentages
        regime_distribution = {}
        for regime_type, count in regime_counts.items():
            if total_assets > 0:
                regime_distribution[regime_type] = round((count / total_assets) * 100, 2)
            else:
                regime_distribution[regime_type] = 0
        
        # Determine dominant regime
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else "unknown"
        
        # Calculate market health score (0-100)
        bullish_score = regime_distribution.get("bullish_trending", 0) + \
                      regime_distribution.get("bullish_breakout", 0)
        bearish_score = regime_distribution.get("bearish_trending", 0) + \
                       regime_distribution.get("bearish_breakout", 0)
        
        market_health = 50 + ((bullish_score - bearish_score) / 2)
        market_health = min(100, max(0, market_health))
        
        # Compile and return results
        return {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "regimes": regimes,
            "regime_distribution": regime_distribution,
            "dominant_regime": dominant_regime,
            "market_health": market_health,
            "assets_analyzed": total_assets,
            "supported_assets": [asset.value for asset in supported_assets]
        }
        
    except Exception as e:
        logger.error(f"Error in get_market_regime_summary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting market regime summary: {str(e)}")


@router.get(
    "/current/{asset_str}/{timeframe_str}",
    response_model=Dict[str, Any],
    summary="Get current market regime",
    tags=["Market Regimes"]
)
async def get_current_regime(
    asset_str: str = Path(..., description="Asset symbol string (e.g., 'BTC-USD', 'SPY')"),
    timeframe_str: str = Path(..., description="Timeframe string (e.g., '1h', '15m')")
):
    """
    Get the current market regime for a specific asset and timeframe.
    Returns the regime type and related metrics.
    """
    try:
        normalized_asset_str = asset_str.upper()
        
        # Check if this is one of our explicitly supported assets, or handle common stocks gracefully
        if normalized_asset_str in [e.value for e in AssetSymbol]:
            asset = AssetSymbol(normalized_asset_str)
        elif len(normalized_asset_str) <= 5 and normalized_asset_str.isalpha():
            # For common stock tickers like AAPL, MSFT, use SPY as a proxy
            # In a production system, you would implement proper stock data fetching
            logger.info(f"Using SPY as proxy for stock ticker {normalized_asset_str}")
            asset = AssetSymbol.SPY
        else:
            logger.error(f"Invalid asset symbol: {asset_str}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid asset symbol: {asset_str}. Supported: {', '.join([e.value for e in AssetSymbol])}"
            )
    except Exception as e:
        logger.error(f"Error processing asset symbol: {asset_str}, error: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid asset symbol: {asset_str}. Supported: {', '.join([e.value for e in AssetSymbol])}"
        )
    
    try:
        timeframe = Timeframe(timeframe_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe_str}")
    
    # Get market data
    historical_data = market_agent.fetch_historical_data(asset, timeframe)
    if not historical_data or not historical_data.candles:
        raise HTTPException(status_code=404, detail=f"No market data found for {asset.value} {timeframe.value}")
    
    # Convert to DataFrame
    import pandas as pd
    candles_data = [candle.model_dump() for candle in historical_data.candles]
    df = pd.DataFrame(candles_data)
    
    # Detect current regime
    regime_info = regime_detector.detect_regime(df, asset, timeframe)
    
    # Sanitize numpy types in metrics for JSON serialization
    raw_metrics = regime_info.metrics or {}
    clean_metrics = {}
    for key, val in raw_metrics.items():
        if isinstance(val, np.generic):
            clean_metrics[key] = val.item()
        elif isinstance(val, np.ndarray):
            clean_metrics[key] = val.tolist()
        else:
            clean_metrics[key] = val
    
    # Add additional metadata
    now = datetime.now()
    response = {
        "asset": asset.value,
        "timeframe": timeframe.value,
        "timestamp": now.isoformat(),
        "regime_type": regime_info.regime_type.value if hasattr(regime_info.regime_type, 'value') else regime_info.regime_type,
        "regime_strength": getattr(regime_info, "strength", None),
        "regime_metrics": clean_metrics,
        "last_regime_change": getattr(regime_info, "last_regime_change", None),
        "regime_duration_bars": getattr(regime_info, "duration_bars", 0)
    }
    
    return response

@router.get(
    "/history/{asset_str}/{timeframe_str}",
    response_model=Dict[str, Any],
    summary="Get market regime history for a specific asset and timeframe",
    tags=["Market Regimes"]
)
async def get_regime_history(
    asset_str: str = Path(..., description="Asset symbol string (e.g., 'BTC-USD', 'SPY')"),
    timeframe_str: str = Path(..., description="Timeframe string (e.g., '1h', '15m')"),
    time_range: TimeRangeEnum = Query(TimeRangeEnum.ONE_MONTH, description="Time range to fetch"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format) for custom range"),
    end_date: Optional[str] = Query(None, description="End date (ISO format) for custom range")
):
    """
    Get the market regime history for a specific asset and timeframe.
    Returns a list of regime periods with start/end dates and regime types.
    """
    try:
        asset = AssetSymbol(asset_str.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid asset symbol: {asset_str}")
    
    try:
        timeframe = Timeframe(timeframe_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe_str}")
    
    # Set date range based on selection
    end_datetime = datetime.now()
    if time_range == TimeRangeEnum.CUSTOM:
        if not start_date:
            raise HTTPException(status_code=400, detail="Start date required for custom range")
        try:
            start_datetime = datetime.fromisoformat(start_date)
            if end_date:
                end_datetime = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
    else:
        # Calculate start date based on time range
        if time_range == TimeRangeEnum.ONE_WEEK:
            start_datetime = end_datetime - timedelta(days=7)
        elif time_range == TimeRangeEnum.ONE_MONTH:
            start_datetime = end_datetime - timedelta(days=30)
        elif time_range == TimeRangeEnum.THREE_MONTHS:
            start_datetime = end_datetime - timedelta(days=90)
        elif time_range == TimeRangeEnum.SIX_MONTHS:
            start_datetime = end_datetime - timedelta(days=180)
        elif time_range == TimeRangeEnum.ONE_YEAR:
            start_datetime = end_datetime - timedelta(days=365)
    
    # Initialize market data agent
    market_data_agent = MarketDataAgent()
    
    # Fetch the historical data using market_data_agent
    historical_data = market_data_agent.fetch_historical_data(
        asset=asset,
        timeframe=timeframe,
        start_date=start_datetime,
        end_date=end_datetime
    )
    
    # Convert to dataframe for regime detection
    try:
        if historical_data and historical_data.candles and len(historical_data.candles) > 0:
            df = pd.DataFrame([candle.model_dump() for candle in historical_data.candles])
            # Convert timestamps to datetime for index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Get regime history using backtester's detect_market_regimes method
            if len(df) > 0:
                regime_periods = backtester._detect_market_regimes(df, asset, timeframe)
            else:
                regime_periods = []
        else:
            # If no data, return empty result
            regime_periods = []
    except Exception as e:
        logger.error(f"Error processing regime history: {e}")
        regime_periods = []
    
    # Format response
    response = {
        "asset": asset.value,
        "timeframe": timeframe.value,
        "start_date": start_datetime.isoformat(),
        "end_date": end_datetime.isoformat(),
        "regime_periods": regime_periods,
        "regime_count": {
            regime.value: len([p for p in regime_periods if p["regime_type"] == regime.value])
            for regime in MarketRegimeType
        }
    }
    
    return response

@router.get(
    "/visualization/{asset_str}/{timeframe_str}",
    response_model=Dict[str, Any],
    summary="Get market regime visualization for a specific asset and timeframe",
    tags=["Market Regimes"]
)
async def get_regime_visualization(
    asset_str: str = Path(..., description="Asset symbol string (e.g., 'BTC-USD', 'SPY')"),
    timeframe_str: str = Path(..., description="Timeframe string (e.g., '1h', '15m')"),
    time_range: TimeRangeEnum = Query(TimeRangeEnum.ONE_MONTH, description="Time range to fetch"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format) for custom range"),
    end_date: Optional[str] = Query(None, description="End date (ISO format) for custom range"),
    chart_type: str = Query("price_with_regimes", description="Chart type: price_with_regimes, regime_distribution, regime_transitions")
):
    """
    Get market regime visualizations for a specific asset and timeframe.
    Returns base64-encoded images of regime charts.
    """
    try:
        asset = AssetSymbol(asset_str.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid asset symbol: {asset_str}")
    
    try:
        timeframe = Timeframe(timeframe_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe_str}")
    
    # Set date range based on selection
    end_datetime = datetime.now()
    if time_range == TimeRangeEnum.CUSTOM:
        if not start_date:
            raise HTTPException(status_code=400, detail="Start date required for custom range")
        try:
            start_datetime = datetime.fromisoformat(start_date)
            if end_date:
                end_datetime = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
    else:
        # Calculate start date based on time range
        if time_range == TimeRangeEnum.ONE_WEEK:
            start_datetime = end_datetime - timedelta(days=7)
        elif time_range == TimeRangeEnum.ONE_MONTH:
            start_datetime = end_datetime - timedelta(days=30)
        elif time_range == TimeRangeEnum.THREE_MONTHS:
            start_datetime = end_datetime - timedelta(days=90)
        elif time_range == TimeRangeEnum.SIX_MONTHS:
            start_datetime = end_datetime - timedelta(days=180)
        elif time_range == TimeRangeEnum.ONE_YEAR:
            start_datetime = end_datetime - timedelta(days=365)
    
    # Generate the appropriate visualization
    image_data = None
    if chart_type == "price_with_regimes":
        # Calculate lookback days from time range
        days_diff = (end_datetime - start_datetime).days
        lookback_days = max(days_diff, 30)  # Minimum 30 days for good visualization
        
        image_data = regime_dashboard.generate_asset_regime_chart(
            asset=asset,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
    elif chart_type == "regime_distribution":
        image_data = regime_dashboard.generate_regime_distribution_chart(
            asset=asset,
            timeframe=timeframe,
            start_date=start_datetime,
            end_date=end_datetime
        )
    elif chart_type == "regime_transitions":
        image_data = regime_dashboard.generate_regime_transition_chart(
            asset=asset,
            timeframe=timeframe,
            start_date=start_datetime,
            end_date=end_datetime
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid chart type: {chart_type}")
    
    if not image_data:
        raise HTTPException(status_code=500, detail="Failed to generate visualization")
    
    return {
        "asset": asset.value,
        "timeframe": timeframe.value,
        "start_date": start_datetime.isoformat(),
        "end_date": end_datetime.isoformat(),
        "chart_type": chart_type,
        "image_data": image_data  # Base64 encoded image
    }

@router.get(
    "/overview",
    response_model=Dict[str, Any],
    summary="Get market regime overview for multiple assets",
    tags=["Market Regimes"]
)
async def get_regime_overview(
    assets: str = Query(..., description="Comma-separated list of asset symbols (e.g., 'BTC-USD,SPY')"),
    timeframes: str = Query("1d", description="Comma-separated list of timeframes (e.g., '1d,4h')")
):
    """
    Get a market regime overview for multiple assets and timeframes.
    Returns a heatmap of current regimes across assets and timeframes.
    """
    # Parse parameters
    try:
        asset_list = [AssetSymbol(a.strip().upper()) for a in assets.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid asset symbol: {str(e)}")
    
    try:
        timeframe_list = [Timeframe(t.strip().lower()) for t in timeframes.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {str(e)}")
    
    # Generate overview
    overview_image = regime_dashboard.generate_regime_overview(
        assets=asset_list,
        timeframes=timeframe_list
    )
    
    if not overview_image:
        raise HTTPException(status_code=500, detail="Failed to generate regime overview")
    
    return {
        "assets": [a.value for a in asset_list],
        "timeframes": [t.value for t in timeframe_list],
        "timestamp": datetime.now().isoformat(),
        "overview_image": overview_image  # Base64 encoded image
    }

@router.get(
    "/backtest/{asset_str}/{timeframe_str}",
    response_model=Dict[str, Any],
    summary="Run a backtest with regime analysis",
    tags=["Market Regimes"]
)
async def run_regime_backtest(
    asset_str: str = Path(..., description="Asset symbol string (e.g., 'BTC-USD', 'SPY')"),
    timeframe_str: str = Path(..., description="Timeframe string (e.g., '1d', '4h')"),
    strategy_name: str = Query("breakout", description="Strategy name to backtest"),
    time_range: TimeRangeEnum = Query(TimeRangeEnum.ONE_YEAR, description="Time range to backtest"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format) for custom range"),
    end_date: Optional[str] = Query(None, description="End date (ISO format) for custom range"),
    use_adaptive_params: bool = Query(True, description="Whether to use adaptive parameters based on market regime")
):
    """
    Run a backtest with market regime analysis for a specific asset and timeframe.
    Returns backtest results with regime-specific performance metrics.
    """
    try:
        asset = AssetSymbol(asset_str.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid asset symbol: {asset_str}")
    
    try:
        timeframe = Timeframe(timeframe_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe_str}")
    
    # Set date range based on selection
    end_datetime = datetime.now()
    if time_range == TimeRangeEnum.CUSTOM:
        if not start_date:
            raise HTTPException(status_code=400, detail="Start date required for custom range")
        try:
            start_datetime = datetime.fromisoformat(start_date)
            if end_date:
                end_datetime = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
    else:
        # Calculate start date based on time range
        if time_range == TimeRangeEnum.ONE_WEEK:
            start_datetime = end_datetime - timedelta(days=7)
        elif time_range == TimeRangeEnum.ONE_MONTH:
            start_datetime = end_datetime - timedelta(days=30)
        elif time_range == TimeRangeEnum.THREE_MONTHS:
            start_datetime = end_datetime - timedelta(days=90)
        elif time_range == TimeRangeEnum.SIX_MONTHS:
            start_datetime = end_datetime - timedelta(days=180)
        elif time_range == TimeRangeEnum.ONE_YEAR:
            start_datetime = end_datetime - timedelta(days=365)
    
    # Run backtest
    backtest_result = backtester.backtest_strategy(
        asset=asset,
        timeframe=timeframe,
        start_date=start_datetime,
        end_date=end_datetime,
        strategy_name=strategy_name,
        use_adaptive_params=use_adaptive_params
    )
    
    if not backtest_result:
        raise HTTPException(status_code=500, detail="Backtest failed to run")
    
    # Generate regime performance chart
    regime_performance_image = backtest_result.generate_regime_performance_image()
    
    # Generate adaptive vs fixed comparison chart
    adaptive_vs_fixed_image = None
    if use_adaptive_params:
        adaptive_vs_fixed_image = backtest_result.generate_adaptive_vs_fixed_image()
    
    # Format response
    response = {
        "asset": asset.value,
        "timeframe": timeframe.value,
        "strategy_name": strategy_name,
        "start_date": start_datetime.isoformat(),
        "end_date": end_datetime.isoformat(),
        "use_adaptive_params": use_adaptive_params,
        "overall_performance": {
            "total_trades": backtest_result.total_trades,
            "win_rate": backtest_result.win_rate,
            "profit_factor": backtest_result.profit_factor,
            "total_return_pct": backtest_result.total_return_pct,
            "max_drawdown": backtest_result.max_drawdown,
            "sharpe_ratio": backtest_result.sharpe_ratio,
            "average_return_per_trade": backtest_result.average_return_per_trade
        },
        "regime_performance": backtest_result.regime_performance,
        "regime_periods": backtest_result.regime_periods,
        "regime_performance_image": regime_performance_image,
        "adaptive_vs_fixed_image": adaptive_vs_fixed_image
    }
    
    return response

@router.post(
    "/optimize",
    response_model=Dict[str, Any],
    summary="Optimize market regime detection thresholds using machine learning",
    tags=["Market Regimes"]
)
async def optimize_regime_thresholds(
    assets: str = Query(..., description="Comma-separated list of asset symbols (e.g., 'BTC-USD,SPY')"),
    timeframes: str = Query("1d", description="Comma-separated list of timeframes (e.g., '1d,4h')"),
    lookback_days: int = Query(90, description="Number of days to look back for optimization")
):
    """
    Optimize market regime detection thresholds using machine learning.
    This process analyzes historical data to find optimal thresholds for regime detection.
    Note: This is a computationally intensive process that may take some time to complete.
    """
    # Parse parameters
    try:
        asset_list = [AssetSymbol(a.strip().upper()) for a in assets.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid asset symbol: {str(e)}")
    
    try:
        timeframe_list = [Timeframe(t.strip().lower()) for t in timeframes.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {str(e)}")
    
    if lookback_days < 30:
        raise HTTPException(status_code=400, detail="Lookback days must be at least 30")
    
    # Run optimization
    try:
        optimized_thresholds = regime_optimizer.run_optimization_pipeline(
            assets=asset_list,
            timeframes=timeframe_list,
            lookback_days=lookback_days
        )
        
        return {
            "assets": [a.value for a in asset_list],
            "timeframes": [t.value for t in timeframe_list],
            "lookback_days": lookback_days,
            "timestamp": datetime.now().isoformat(),
            "optimized_thresholds": optimized_thresholds.to_dict(),
            "output_dir": regime_optimizer.output_dir
        }
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get(
    "/real-time/alerts",
    response_model=List[Dict[str, Any]],
    summary="Get real-time regime change alerts",
    tags=["Market Regimes"]
)
async def get_regime_alerts(
    assets: str = Query(..., description="Comma-separated list of asset symbols (e.g., 'BTC-USD,SPY')"),
    timeframes: str = Query("1d,4h,1h", description="Comma-separated list of timeframes (e.g., '1d,4h,1h')"),
    min_strength: float = Query(0.7, description="Minimum regime strength to trigger alert (0-1)")
):
    """
    Get real-time alerts for market regime changes.
    Returns a list of recent regime changes that exceed the minimum strength threshold.
    """
    # Parse parameters
    try:
        asset_list = [AssetSymbol(a.strip().upper()) for a in assets.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid asset symbol: {str(e)}")
    
    try:
        timeframe_list = [Timeframe(t.strip().lower()) for t in timeframes.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {str(e)}")
    
    # Check for regime changes
    alerts = []
    for asset in asset_list:
        for timeframe in timeframe_list:
            # Get current regime
            historical_data = market_agent.fetch_historical_data(asset, timeframe)
            if not historical_data or not historical_data.candles:
                continue
            
            # Convert to DataFrame
            import pandas as pd
            candles_data = [candle.model_dump() for candle in historical_data.candles]
            df = pd.DataFrame(candles_data)
            
            # Detect regime
            regime_info = regime_detector.detect_regime(df, asset, timeframe)
            
            # Check if this is a new regime with significant strength
            if (regime_info.get("is_new_regime", False) and 
                regime_info.get("regime_strength", 0) >= min_strength):
                # Create alert
                alerts.append({
                    "asset": asset.value,
                    "timeframe": timeframe.value,
                    "timestamp": datetime.now().isoformat(),
                    "regime_type": regime_info["regime_type"],
                    "regime_strength": regime_info["regime_strength"],
                    "previous_regime": regime_info.get("previous_regime"),
                    "metrics": regime_info["metrics"]
                })
    
    return alerts
