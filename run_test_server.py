from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="backend/templates")

# Override url_for in templates to properly handle static files
@app.middleware("http")
async def add_custom_url_for(request: Request, call_next):
    request.state.url_for = lambda name, **path_params: request.url_for(name, **path_params)
    response = await call_next(request)
    return response

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("strategies.html", {"request": request, "title": "Trading Strategies"})

# Route for login page
@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    context = {
        "request": request, 
        "title": "Login - Trading SaaS Platform"
    }
    return templates.TemplateResponse("login.html", context)

# Mock login API endpoint
@app.post("/api/v1/auth/login")
async def mock_login():
    return {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxNzE5MTc2ODcwfQ.mock-token-for-testing",
        "token_type": "bearer",
        "user": {
            "id": 1,
            "username": "test_user",
            "email": "user@example.com",
            "subscription_tier": "ENTERPRISE"
        }
    }

# Route for strategy editor
@app.get("/strategy-editor", response_class=HTMLResponse)
async def strategy_editor(request: Request, strategy_id: str = None):
    context = {
        "request": request, 
        "title": "Strategy Editor - Trading SaaS Platform",
        "strategy_id": strategy_id
    }
    return templates.TemplateResponse("strategy_editor.html", context)

# Mock API endpoint for strategies
@app.get("/api/v1/strategies")
async def get_strategies():
    return [
        {
            "id": 1,
            "name": "Example Trend Following Strategy",
            "description": "A sample trend following strategy using EMA crossovers",
            "strategy_type": "TREND_FOLLOWING",
            "created_at": "2025-05-10T14:30:00Z",
            "is_active": True,
            "is_public": False
        },
        {
            "id": 2,
            "name": "RSI Mean Reversion Strategy",
            "description": "A strategy that trades overbought and oversold conditions using RSI",
            "strategy_type": "MEAN_REVERSION",
            "created_at": "2025-05-15T10:45:00Z",
            "is_active": True,
            "is_public": False
        }
    ]

# Mock API endpoint for strategy presets
@app.get("/api/v1/strategies/presets")
async def get_strategy_presets():
    return [
        {
            "id": "preset_ema_crossover",
            "name": "EMA Crossover",
            "description": "Basic trend following strategy using EMA crossovers",
            "strategy_type": "TREND_FOLLOWING",
            "tier_required": "BASIC",
            "dsl_content": """strategy_name: ema_crossover
rules:
  - if: ema(9) > ema(21)
    and: ema(9) > ema(9).shift(1)
    then: signal = "long"
  - if: ema(9) < ema(21)
    and: ema(9) < ema(9).shift(1)
    then: signal = "short"
risk:
  entry_size_pct: 1.0
  stop: atr(14) * 2.0
  take_profit: 3R"""
        },
        {
            "id": "preset_rsi_mean_reversion",
            "name": "RSI Mean Reversion",
            "description": "Mean reversion strategy using RSI oversold/overbought conditions",
            "strategy_type": "MEAN_REVERSION",
            "tier_required": "BASIC",
            "dsl_content": """strategy_name: rsi_mean_reversion
rules:
  - if: rsi(14) < 30
    and: close > close.shift(1)
    then: signal = "long"
  - if: rsi(14) > 70
    and: close < close.shift(1)
    then: signal = "short"
risk:
  entry_size_pct: 1.0
  stop: atr(14) * 1.5
  take_profit: 2R"""
        },
        {
            "id": "preset_breakout",
            "name": "Breakout Strategy",
            "description": "Volatility breakout strategy using ATR and previous highs/lows",
            "strategy_type": "BREAKOUT",
            "tier_required": "PRO",
            "dsl_content": """strategy_name: volatility_breakout
rules:
  - if: close > highest(20).shift(1)
    and: atr(14) > atr(14).rolling(5).mean()
    then: signal = "long"
  - if: close < lowest(20).shift(1)
    and: atr(14) > atr(14).rolling(5).mean()
    then: signal = "short"
risk:
  entry_size_pct: 1.0
  stop: atr(14) * 2.0
  take_profit: 2.5R"""
        }
    ]

# Mock API endpoint for strategy validation
@app.post("/api/v1/strategies/validate")
async def validate_strategy():
    return {"valid": True, "parsed": {"strategy_name": "example_strategy"}}

# Mock API endpoint for code generation
@app.post("/api/v1/strategies/generate-code")
async def generate_code():
    python_code = """
# Auto-generated strategy code from DSL
# Strategy: example_strategy
import pandas as pd
import numpy as np
from backend.strategies.strategy_base import StrategyBase
from shared_types import TradingSignalModel, SignalType, AssetSymbol, Timeframe

class ExampleStrategyStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='example_strategy')

    def generate_signals(self, df, asset, timeframe):
        \"\"\"
        Generate trading signals based on the strategy rules.
        \"\"\"
        # Initialize signal column
        df['signal'] = None

        # Calculate indicators
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['rsi_14'] = df['close'].rolling(14).apply(self._calculate_rsi)

        # Apply strategy rules
        for idx, row in df.iterrows():
            if row['ema_9'] > row['ema_21'] and row['rsi_14'] < 70:
                df.at[idx, 'signal'] = 'long'
            elif row['ema_9'] < row['ema_21'] and row['rsi_14'] > 30:
                df.at[idx, 'signal'] = 'short'

        # Process signals to create signal models
        signals = []
        for idx, row in df.iterrows():
            if pd.notna(row['signal']):
                if row['signal'] == 'long':
                    signal_type = SignalType.LONG
                elif row['signal'] == 'short':
                    signal_type = SignalType.SHORT
                else:
                    continue

                signals.append(TradingSignalModel(
                    asset=asset,
                    timeframe=timeframe,
                    timestamp=idx,
                    signal_type=signal_type,
                    price=row['close'],
                    strategy_name=self.name
                ))

        return signals

    @staticmethod
    def _calculate_rsi(prices):
        deltas = np.diff(prices)
        seed = deltas[:14]
        up = seed[seed >= 0].sum()/14
        down = -seed[seed < 0].sum()/14
        rs = up/down
        return 100 - (100/(1 + rs))
"""
    return {"code": python_code}

# Mock API endpoint for backtesting
@app.post("/api/v1/strategies/backtest")
async def backtest_strategy():
    import random
    from datetime import datetime, timedelta
    
    # Generate random equity curve
    start_date = datetime.now() - timedelta(days=365)
    equity_curve = []
    equity = 10000.0
    
    for i in range(365):
        current_date = start_date + timedelta(days=i)
        daily_change = random.uniform(-0.03, 0.03)
        equity *= (1 + daily_change)
        
        equity_curve.append({
            "date": current_date.isoformat(),
            "equity": round(equity, 2)
        })
    
    return {
        "strategy_id": None,
        "asset_symbol": "BTC-USD",
        "timeframe": "1d",
        "start_date": start_date.isoformat(),
        "end_date": datetime.now().isoformat(),
        "win_rate": 58.7,
        "profit_factor": 1.85,
        "max_drawdown": 12.3,
        "total_trades": 42,
        "total_return": 34.5,
        "annualized_return": 28.2,
        "metrics": {
            "sharpe_ratio": 1.8,
            "sortino_ratio": 2.1,
            "average_gain": 3.2,
            "average_loss": 1.7,
            "win_loss_ratio": 1.42,
            "avg_holding_period": 4.5,
            "largest_win": 8.4,
            "largest_loss": 4.2
        },
        "equity_curve": equity_curve
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
