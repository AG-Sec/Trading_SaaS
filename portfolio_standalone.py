from fastapi import FastAPI, HTTPException, Request, Depends, APIRouter
from typing import List, Dict, Any, Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import logging.config
from pathlib import Path
import os
from dotenv import load_dotenv

# Import only the database setup, not any other dependencies
from backend.core.database import engine, Base, get_db
from sqlalchemy.orm import Session

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Create a standalone portfolio router ---
portfolio_router = APIRouter()

# Mock portfolio data for testing
MOCK_PORTFOLIOS = [
    {
        "id": 1,
        "name": "Growth Portfolio",
        "description": "Long-term growth strategy focused on technology and innovation",
        "starting_capital": 10000.0,
        "current_value": 12500.0,
        "cash_balance": 2500.0,
        "total_profit_loss": 2500.0,
        "daily_profit_loss": 150.0,
        "win_rate": 65.0,
        "profit_factor": 2.1,
        "created_at": "2025-01-15T10:00:00",
        "updated_at": "2025-05-20T08:30:00"
    },
    {
        "id": 2,
        "name": "Income Portfolio",
        "description": "Dividend-focused portfolio for passive income",
        "starting_capital": 20000.0,
        "current_value": 21800.0,
        "cash_balance": 5000.0,
        "total_profit_loss": 1800.0,
        "daily_profit_loss": 75.0,
        "win_rate": 72.0,
        "profit_factor": 1.8,
        "created_at": "2025-02-20T14:30:00",
        "updated_at": "2025-05-20T08:30:00"
    }
]

MOCK_POSITIONS = {
    1: [
        {
            "id": 1,
            "symbol": "AAPL",
            "asset_class": "STOCK",
            "direction": "LONG",
            "quantity": 10.0,
            "average_entry_price": 175.5,
            "current_price": 190.25,
            "cost_basis": 1755.0,
            "market_value": 1902.5,
            "unrealized_profit_loss": 147.5,
            "unrealized_profit_loss_percent": 8.4,
            "stop_loss_price": 160.0,
            "take_profit_price": 210.0,
            "position_size_percent": 15.22,
            "opened_at": "2025-03-10T09:30:00",
            "updated_at": "2025-05-20T08:30:00"
        },
        {
            "id": 2,
            "symbol": "MSFT",
            "asset_class": "STOCK",
            "direction": "LONG",
            "quantity": 5.0,
            "average_entry_price": 350.75,
            "current_price": 372.50,
            "cost_basis": 1753.75,
            "market_value": 1862.5,
            "unrealized_profit_loss": 108.75,
            "unrealized_profit_loss_percent": 6.2,
            "stop_loss_price": 330.0,
            "take_profit_price": 400.0,
            "position_size_percent": 14.9,
            "opened_at": "2025-03-15T10:15:00",
            "updated_at": "2025-05-20T08:30:00"
        },
        {
            "id": 3,
            "symbol": "BTC-USD",
            "asset_class": "CRYPTO",
            "direction": "LONG",
            "quantity": 0.05,
            "average_entry_price": 50000.0,
            "current_price": 57000.0,
            "cost_basis": 2500.0,
            "market_value": 2850.0,
            "unrealized_profit_loss": 350.0,
            "unrealized_profit_loss_percent": 14.0,
            "stop_loss_price": 45000.0,
            "take_profit_price": 65000.0,
            "position_size_percent": 22.8,
            "opened_at": "2025-04-01T08:00:00",
            "updated_at": "2025-05-20T08:30:00"
        }
    ],
    2: [
        {
            "id": 4,
            "symbol": "VYM",
            "asset_class": "ETF",
            "direction": "LONG",
            "quantity": 25.0,
            "average_entry_price": 105.25,
            "current_price": 110.50,
            "cost_basis": 2631.25,
            "market_value": 2762.5,
            "unrealized_profit_loss": 131.25,
            "unrealized_profit_loss_percent": 4.99,
            "stop_loss_price": 95.0,
            "take_profit_price": None,
            "position_size_percent": 12.67,
            "opened_at": "2025-02-25T11:30:00",
            "updated_at": "2025-05-20T08:30:00"
        },
        {
            "id": 5,
            "symbol": "JEPI",
            "asset_class": "ETF",
            "direction": "LONG",
            "quantity": 40.0,
            "average_entry_price": 55.75,
            "current_price": 57.25,
            "cost_basis": 2230.0,
            "market_value": 2290.0,
            "unrealized_profit_loss": 60.0,
            "unrealized_profit_loss_percent": 2.69,
            "stop_loss_price": 50.0,
            "take_profit_price": None,
            "position_size_percent": 10.5,
            "opened_at": "2025-03-05T13:45:00",
            "updated_at": "2025-05-20T08:30:00"
        }
    ]
}

MOCK_TRADES = {
    1: [
        {
            "id": 1,
            "symbol": "AAPL",
            "asset_class": "STOCK",
            "direction": "LONG",
            "quantity": 10.0,
            "price": 175.50,
            "timestamp": "2025-03-10T09:30:00",
            "status": "OPEN",
            "fees": 2.50,
            "total_cost": 1757.50,
            "realized_profit_loss": None,
            "realized_profit_loss_percent": None,
            "notes": "Initial position in Apple based on breakout strategy",
            "tags": "tech,breakout,long-term"
        },
        {
            "id": 2,
            "symbol": "MSFT",
            "asset_class": "STOCK",
            "direction": "LONG",
            "quantity": 5.0,
            "price": 350.75,
            "timestamp": "2025-03-15T10:15:00",
            "status": "OPEN",
            "fees": 2.50,
            "total_cost": 1756.25,
            "realized_profit_loss": None,
            "realized_profit_loss_percent": None,
            "notes": "Microsoft position after earnings beat",
            "tags": "tech,earnings,momentum"
        },
        {
            "id": 3,
            "symbol": "BTC-USD",
            "asset_class": "CRYPTO",
            "direction": "LONG",
            "quantity": 0.05,
            "price": 50000.0,
            "timestamp": "2025-04-01T08:00:00",
            "status": "OPEN",
            "fees": 5.0,
            "total_cost": 2505.0,
            "realized_profit_loss": None,
            "realized_profit_loss_percent": None,
            "notes": "Bitcoin position on technical breakout",
            "tags": "crypto,trend,breakout"
        },
        {
            "id": 6,
            "symbol": "NVDA",
            "asset_class": "STOCK",
            "direction": "LONG",
            "quantity": 3.0,
            "price": 580.0,
            "timestamp": "2025-03-20T11:00:00",
            "status": "CLOSED",
            "fees": 4.50,
            "total_cost": 1744.50,
            "realized_profit_loss": 285.0,
            "realized_profit_loss_percent": 16.34,
            "notes": "Closed NVIDIA position after reaching target",
            "tags": "tech,profit,target"
        }
    ],
    2: [
        {
            "id": 4,
            "symbol": "VYM",
            "asset_class": "ETF",
            "direction": "LONG",
            "quantity": 25.0,
            "price": 105.25,
            "timestamp": "2025-02-25T11:30:00",
            "status": "OPEN",
            "fees": 2.50,
            "total_cost": 2633.75,
            "realized_profit_loss": None,
            "realized_profit_loss_percent": None,
            "notes": "Vanguard High Dividend ETF for income strategy",
            "tags": "etf,dividend,income"
        },
        {
            "id": 5,
            "symbol": "JEPI",
            "asset_class": "ETF",
            "direction": "LONG",
            "quantity": 40.0,
            "price": 55.75,
            "timestamp": "2025-03-05T13:45:00",
            "status": "OPEN",
            "fees": 2.50,
            "total_cost": 2232.50,
            "realized_profit_loss": None,
            "realized_profit_loss_percent": None,
            "notes": "JPMorgan Equity Premium Income ETF for monthly dividends",
            "tags": "etf,dividend,income,monthly"
        },
        {
            "id": 7,
            "symbol": "SCHD",
            "asset_class": "ETF",
            "direction": "LONG",
            "quantity": 20.0,
            "price": 78.50,
            "timestamp": "2025-03-10T14:15:00",
            "status": "CLOSED",
            "fees": 2.50,
            "total_cost": 1572.50,
            "realized_profit_loss": 90.0,
            "realized_profit_loss_percent": 5.72,
            "notes": "Closed Schwab Dividend ETF position to rebalance",
            "tags": "etf,dividend,rebalance"
        }
    ]
}

# Portfolio routes
@portfolio_router.get("/", response_model=List[Dict[str, Any]])
async def get_user_portfolios():
    """Get all portfolios for the current user (mock data)"""
    return MOCK_PORTFOLIOS

@portfolio_router.get("/{portfolio_id}", response_model=Dict[str, Any])
async def get_portfolio(portfolio_id: int):
    """Get a specific portfolio by ID (mock data)"""
    portfolio = next((p for p in MOCK_PORTFOLIOS if p["id"] == portfolio_id), None)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio

@portfolio_router.get("/{portfolio_id}/positions", response_model=List[Dict[str, Any]])
async def get_portfolio_positions(portfolio_id: int):
    """Get all positions in a portfolio (mock data)"""
    positions = MOCK_POSITIONS.get(portfolio_id, [])
    return positions

@portfolio_router.get("/{portfolio_id}/trades", response_model=List[Dict[str, Any]])
async def get_portfolio_trades(portfolio_id: int):
    """Get trades for a portfolio (mock data)"""
    trades = MOCK_TRADES.get(portfolio_id, [])
    return trades

@portfolio_router.get("/{portfolio_id}/metrics", response_model=Dict[str, Any])
async def get_portfolio_metrics(portfolio_id: int):
    """Calculate performance metrics for a portfolio (mock data)"""
    # Return mock metrics
    return {
        "starting_capital": 10000.0,
        "current_value": 12500.0,
        "total_return_amount": 2500.0,
        "total_return_percent": 25.0,
        "annual_return": 15.5,
        "sharpe_ratio": 1.2,
        "max_drawdown": 8.5,
        "win_rate": 65.0,
        "profit_factor": 2.1,
        "avg_win_loss_ratio": 1.5,
        "equity_curve": [
            {"date": "2025-01-15", "value": 10000.0, "benchmark_value": 10000.0},
            {"date": "2025-02-15", "value": 10500.0, "benchmark_value": 10200.0},
            {"date": "2025-03-15", "value": 11200.0, "benchmark_value": 10500.0},
            {"date": "2025-04-15", "value": 11800.0, "benchmark_value": 10800.0},
            {"date": "2025-05-15", "value": 12500.0, "benchmark_value": 11000.0}
        ],
        "monthly_returns": [
            {
                "year": 2025,
                "months": [2.5, 3.1, 4.2, 3.8, 2.1, None, None, None, None, None, None, None],
                "ytd": 16.7
            }
        ],
        "regime_performance": [
            {"regime": "bullish_trending", "win_rate": 75.0, "return": 12.5, "trade_count": 5},
            {"regime": "bearish_trending", "win_rate": 40.0, "return": -3.0, "trade_count": 2},
            {"regime": "high_volatility", "win_rate": 60.0, "return": 8.0, "trade_count": 3},
            {"regime": "low_volatility", "win_rate": 70.0, "return": 7.5, "trade_count": 2}
        ]
    }

@portfolio_router.post("/{portfolio_id}/update-prices", response_model=Dict[str, Any])
async def update_portfolio_prices(portfolio_id: int):
    """Update current prices and metrics for all positions in a portfolio (mock)"""
    return {"status": "success", "message": "Prices updated successfully"}

# --- Application Lifecycle (Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup...")
    # Initialize database tables
    logger.info("Creating database tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    
    # Create static and templates directories if they don't exist
    static_dir = Path("backend/static")
    templates_dir = Path("backend/templates")
    static_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    # Shutdown
    logger.info("Application shutdown...")

# --- FastAPI App Instance ---
app = FastAPI(
    title="Trading SaaS Portfolio Management",
    description="Portfolio management functionality for the Trading SaaS platform",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up static files directory
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="backend/templates")

# --- API Routers ---
app.include_router(portfolio_router, prefix="/api/v1/portfolios", tags=["Portfolio Management"])

# --- Web UI Routes ---
@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def home(request: Request):
    """Redirect to portfolios page"""
    return templates.TemplateResponse(
        "portfolios.html", 
        {"request": request, "title": "Portfolio Management - Trading SaaS Platform"}
    )

@app.get("/portfolios", response_class=HTMLResponse, tags=["Web UI"])
async def portfolios(request: Request):
    """Render the portfolio management page"""
    return templates.TemplateResponse(
        "portfolios.html", 
        {"request": request, "title": "Portfolio Management - Trading SaaS Platform"}
    )

@app.get("/templates/portfolio_components/{component_name}", response_class=HTMLResponse)
async def serve_portfolio_component(request: Request, component_name: str):
    """Serve portfolio component templates"""
    return templates.TemplateResponse(
        f"portfolio_components/{component_name}", 
        {"request": request}
    )

@app.get("/health", tags=["General"])
async def health_check():
    logger.info("Health check endpoint was called.")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Portfolio management app")
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on')
    args = parser.parse_args()
    
    # Run the server with the specified port
    print(f"\nStarting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
