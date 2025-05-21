from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException, Request, Depends
from typing import Optional
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import logging.config
from pathlib import Path
from backend.api.trading_router import router as trading_api_router
from backend.api.regime_router import router as regime_api_router
from backend.api.auth_router import router as auth_api_router
from backend.api.subscription_router import router as subscription_api_router
from backend.api.strategy_router import router as strategy_api_router
from backend.api.portfolio_router import router as portfolio_api_router
from backend.api.orchestrator_router import router as orchestrator_api_router
from backend.core.database import engine, Base

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
# Basic config, can be expanded with a dictConfig from a file/dict
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # Clean up resources

# --- FastAPI App Instance ---
app = FastAPI(
    title="Trading SaaS Platform API",
    description="API for managing trading strategies, signals, and execution.",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add development mode authentication middleware
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # Check if we're in development mode
    if os.getenv("ENV", "development") == "development":
        # Skip authentication for all routes
        response = await call_next(request)
        return response
        
    # For production, proceed with normal authentication flow
    path = request.url.path
    
    # Skip auth for public routes
    public_routes = ["/", "/login", "/api/v1/auth/login", "/api/v1/auth/token"]
    public_prefixes = ["/static/", "/docs", "/openapi", "/redoc"]
    
    if path in public_routes or any(path.startswith(prefix) for prefix in public_prefixes):
        response = await call_next(request)
        return response
        
    # Check for auth token in cookies or headers
    token = request.cookies.get("access_token") or request.headers.get("Authorization")
    
    if not token and not path.startswith("/api/"):
        # For web pages redirect to login
        return RedirectResponse(url="/login")
    
    # Continue with request handling
    response = await call_next(request)
    return response

# Set up static files directory
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="backend/templates")

# --- API Routers ---
app.include_router(trading_api_router, prefix="/api/v1/trading", tags=["Trading Services"])
app.include_router(regime_api_router, prefix="/api/v1/market-regimes", tags=["Market Regimes"])
app.include_router(auth_api_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(subscription_api_router, prefix="/api/v1/subscriptions", tags=["Subscriptions"])
app.include_router(strategy_api_router, prefix="/api/v1/strategies", tags=["Strategy Customization"])
app.include_router(portfolio_api_router, prefix="/api/v1/portfolios", tags=["Portfolio Management"])
app.include_router(orchestrator_api_router, prefix="/api/v1/orchestrator", tags=["Trading Orchestration"])

# --- Web UI Routes ---
@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "Trading SaaS Platform"}
    )

@app.get("/login", response_class=HTMLResponse, tags=["Authentication"])
async def login_page(request: Request, error: str = None, registered: bool = None):
    """Render the login page"""
    context = {
        "request": request, 
        "title": "Login - Trading SaaS Platform",
        "error": error
    }
    
    if registered:
        context["success"] = "Registration successful! Please log in."
        
    return templates.TemplateResponse("login.html", context)

@app.get("/register", response_class=HTMLResponse, tags=["Authentication"])
async def register_page(request: Request, error: str = None):
    """Render the registration page"""
    return templates.TemplateResponse(
        "register.html", 
        {
            "request": request, 
            "title": "Register - Trading SaaS Platform",
            "error": error
        }
    )

@app.get("/dashboard", response_class=HTMLResponse, tags=["Web UI"])
async def dashboard(request: Request):
    """Render the main dashboard page"""
    return templates.TemplateResponse(
        "dashboard.html", 
        {"request": request, "title": "Trading Dashboard"}
    )

@app.get("/market-regimes", response_class=HTMLResponse, tags=["Web UI"])
async def market_regimes(request: Request):
    """Render the market regimes page"""
    return templates.TemplateResponse(
        "regimes.html", 
        {"request": request, "title": "Market Regime Analysis"}
    )

@app.get("/backtest", response_class=HTMLResponse, tags=["Web UI"])
async def backtest(request: Request):
    """Render the backtest page"""
    return templates.TemplateResponse(
        "backtest.html", 
        {"request": request, "title": "Backtest Results"}
    )

@app.get("/optimize", response_class=HTMLResponse, tags=["Web UI"])
async def optimize(request: Request):
    """Render the regime optimization page"""
    return templates.TemplateResponse(
        "optimize.html", 
        {"request": request, "title": "Regime Optimization"}
    )

@app.get("/strategies", response_class=HTMLResponse, tags=["Web UI"])
async def strategies(request: Request):
    """Render the strategies page"""
    return templates.TemplateResponse(
        "strategies.html", 
        {"request": request, "title": "Trading Strategies"}
    )

@app.get("/portfolios", response_class=HTMLResponse, tags=["Web UI"])
async def portfolios_page(request: Request):
    """Render the portfolios page"""
    return templates.TemplateResponse(
        "portfolios.html", 
        {"request": request, "title": "Portfolio Management"}
    )

@app.get("/trading-signals", response_class=HTMLResponse, tags=["Web UI"])
async def trading_signals_page(request: Request):
    """Render the trading signals page"""
    return templates.TemplateResponse(
        "trading_signals.html",
        {"request": request, "title": "Trading Signals"}
    )

@app.get("/strategy-editor", response_class=HTMLResponse, tags=["Web UI"])
async def strategy_editor(request: Request, strategy_id: Optional[int] = None):
    """Render the strategy editor page"""
    context = {
        "request": request, 
        "title": "Strategy Editor - Trading SaaS Platform",
        "strategy_id": strategy_id
    }
    return templates.TemplateResponse("strategy_editor.html", context)

@app.get("/subscription", response_class=HTMLResponse, tags=["Web UI"])
async def subscription(request: Request):
    """Render the subscription management page"""
    return templates.TemplateResponse(
        "subscription.html", 
        {"request": request, "title": "Subscription Management - Trading SaaS Platform"}
    )

# --- API Documentation Endpoint ---
@app.get("/api", tags=["General"])
async def api_docs_redirect():
    """Redirect to the API documentation"""
    return {"message": "API documentation is available at /docs or /redoc"}

@app.get("/health", tags=["General"])
async def health_check():
    logger.info("Health check endpoint was called.")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FASTAPI_PORT", 8000)))
