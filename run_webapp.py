#!/usr/bin/env python3
"""
Trading SaaS Platform Web Application Runner
This script starts the web application with the market regime analysis features.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trading_webapp")

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_web_app():
    """Run the web application"""
    logger.info("Starting the Trading SaaS web application...")
    
    from backend.main import app
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Web application will be available at http://localhost:{port}")
    logger.info("Press Ctrl+C to stop the server")
    
    # Run the web app
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    logger.info("=" * 40)
    logger.info("Trading SaaS Platform Web Application")
    logger.info("=" * 40)
    
    # Check if required packages are installed
    try:
        logger.info("Checking required packages...")
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import sklearn
        logger.info("All required packages are installed")
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        sys.exit(1)
    
    # Start the web app
    run_web_app()
