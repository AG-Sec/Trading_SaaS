"""
Test environment initialization script.
This script sets up the test environment and provides utility functions for testing.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path to ensure imports work correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Disable yfinance warning output to reduce test noise
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Import shared test utilities
from shared_types.models import (
    AssetSymbol, 
    Timeframe, 
    SignalType, 
    HistoricalDataModel, 
    CandleModel
)

def get_test_data_dir():
    """Get the test data directory path."""
    test_data_dir = os.path.join(project_root, "tests", "test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    return test_data_dir

def get_test_output_dir():
    """Get the test output directory path."""
    test_output_dir = os.path.join(project_root, "test_reports")
    os.makedirs(test_output_dir, exist_ok=True)
    return test_output_dir

print(f"Test environment initialized. Project root: {project_root}")
