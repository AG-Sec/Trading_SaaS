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
import os
from dotenv import load_dotenv

# Import only what we need for portfolio features
from backend.api.portfolio_router import router as portfolio_api_router
from backend.core.database import engine, Base

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
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
app.include_router(portfolio_api_router, prefix="/api/v1/portfolios", tags=["Portfolio Management"])

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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FASTAPI_PORT", 8000)))
