#!/bin/bash
# Script to run validation tests for the Trading SaaS platform

echo "Running Trading SaaS validation tests..."
echo "These tests verify that signals adhere to fundamental principles of trading"

# Run the validation tests with pytest
python -m pytest tests/validation/test_signal_validation.py -v
