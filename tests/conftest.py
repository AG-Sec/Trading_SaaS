"""
Common test fixtures for Trading SaaS.
"""
import os
import pytest
import tempfile
from datetime import datetime, timedelta

from shared_types.models import AssetSymbol, Timeframe, SignalType, HistoricalDataModel, CandleModel
from backend.agents.market_data_agent import MarketDataAgent
from backend.agents.signal_scanner_agent import SignalScannerAgent
from backend.agents.risk_manager_agent import RiskManagerAgent
from backend.agents.market_regime_detector import MarketRegimeDetector
from backend.agents.journal_agent import JournalAgent

@pytest.fixture
def market_agent():
    """Return a MarketDataAgent instance for testing."""
    return MarketDataAgent()

@pytest.fixture
def scanner_agent(market_agent):
    """Return a SignalScannerAgent instance for testing."""
    return SignalScannerAgent(market_data_agent=market_agent)

@pytest.fixture
def regime_detector():
    """Return a MarketRegimeDetector instance for testing."""
    return MarketRegimeDetector()

@pytest.fixture
def test_db_path():
    """Create a temporary DB file for testing."""
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix='.sqlite')
    os.close(temp_db_fd)
    yield temp_db_path
    # Cleanup after tests
    if os.path.exists(temp_db_path):
        os.unlink(temp_db_path)

@pytest.fixture
def journal_agent(test_db_path):
    """Return a JournalAgent instance with a test database."""
    return JournalAgent(db_path=test_db_path)

@pytest.fixture
def risk_agent(journal_agent):
    """Return a RiskManagerAgent instance for testing."""
    return RiskManagerAgent(
        account_balance_usd=10000.0,
        journal_agent=journal_agent
    )

@pytest.fixture
def sample_timeframes():
    """Return a list of timeframes for testing."""
    return [
        Timeframe.MIN_15, 
        Timeframe.HOUR_1, 
        Timeframe.HOUR_4, 
        Timeframe.DAY_1
    ]

@pytest.fixture
def sample_assets():
    """Return a list of assets for testing."""
    return [
        AssetSymbol.BTC_USD, 
        AssetSymbol.ETH_USD, 
        AssetSymbol.SPY, 
        AssetSymbol.EUR_USD_FX
    ]

@pytest.fixture
def test_date_range():
    """Return a sample date range for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Use 30 days for faster tests
    return start_date, end_date
