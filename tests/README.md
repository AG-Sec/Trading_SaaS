# Trading SaaS Test Suite

This directory contains comprehensive tests for the Trading SaaS platform.

## Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test complete workflows from signal generation to execution
- **Performance Tests**: Benchmark critical operations

## Running Tests

You can run the tests using the `run_tests.py` script in the project root:

```bash
# Run all tests
./run_tests.py

# Run specific test types
./run_tests.py --type unit
./run_tests.py --type integration
./run_tests.py --type e2e
./run_tests.py --type performance

# Generate dashboard
./run_tests.py --dashboard

# For help
./run_tests.py --help
```

## Test Coverage

The test suite covers:

1. **MarketDataAgent**: Data fetching and error handling for all assets/timeframes
2. **SignalScannerAgent**: Breakout detection, signal generation, and pullback entries
3. **RiskManagerAgent**: Signal filtering, risk-reward checks, and position sizing
4. **MarketRegimeDetector**: Regime classification and parameter adjustment
5. **Journal System**: Signal recording and trade tracking

## Test Reports

Reports and dashboards are saved to the `test_reports/` directory.
