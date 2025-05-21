#!/usr/bin/env python3
"""
Test runner for the Trading SaaS platform.
Executes all tests and generates reports.
"""
import os
import sys
import argparse
import pytest
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def run_tests(test_type=None, verbose=True, html_report=True):
    """
    Run tests based on the specified type.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'e2e', 'performance', or None for all)
        verbose: Whether to show verbose output
        html_report: Whether to generate HTML reports
    
    Returns:
        Tuple of (exit code, test duration)
    """
    # Prepare arguments for pytest
    pytest_args = []
    
    # Add test directory based on type
    if test_type == 'unit':
        pytest_args.append('tests/unit/')
    elif test_type == 'integration':
        pytest_args.append('tests/integration/')
    elif test_type == 'e2e':
        pytest_args.append('tests/e2e/')
    elif test_type == 'performance':
        pytest_args.append('tests/performance/')
    else:
        pytest_args.append('tests/')
    
    # Add verbosity flag
    if verbose:
        pytest_args.append('-v')
    
    # Add HTML report generation if requested
    if html_report:
        report_dir = os.path.join(os.getcwd(), 'test_reports')
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"report_{test_type or 'all'}_{timestamp}.html"
        report_path = os.path.join(report_dir, report_name)
        
        pytest_args.extend(['--html', report_path, '--self-contained-html'])
    
    # Measure execution time
    start_time = time.time()
    
    # Run pytest with the collected arguments
    exit_code = pytest.main(pytest_args)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Return results
    return exit_code, duration

def create_dashboard(results):
    """
    Create a dashboard with test results visualization.
    
    Args:
        results: Dictionary containing test results
    """
    # Create output directory
    dashboard_dir = os.path.join(os.getcwd(), 'test_reports', 'dashboard')
    os.makedirs(dashboard_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Test execution times
    test_types = list(results['execution_times'].keys())
    times = list(results['execution_times'].values())
    axs[0, 0].bar(test_types, times)
    axs[0, 0].set_title('Test Execution Times')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].set_xlabel('Test Type')
    
    # Plot 2: Signal counts by asset
    assets = results['signal_counts']['assets']
    counts = results['signal_counts']['counts']
    axs[0, 1].bar(assets, counts)
    axs[0, 1].set_title('Signal Generation by Asset')
    axs[0, 1].set_ylabel('Number of Signals')
    axs[0, 1].set_xlabel('Asset')
    
    # Plot 3: Approval rates
    categories = ['Approved', 'Rejected']
    approved = results['approval_stats']['approved']
    rejected = results['approval_stats']['total'] - approved
    values = [approved, rejected]
    axs[1, 0].pie(values, labels=categories, autopct='%1.1f%%')
    axs[1, 0].set_title('Signal Approval Rate')
    
    # Plot 4: Regime distribution
    regimes = results['regime_distribution']['types']
    regime_counts = results['regime_distribution']['counts']
    axs[1, 1].bar(regimes, regime_counts)
    axs[1, 1].set_title('Market Regime Distribution')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_xlabel('Regime Type')
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(dashboard_dir, 'test_results_dashboard.png'))
    
    # Create HTML dashboard
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading SaaS Test Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .stat {{ font-weight: bold; }}
            .dashboard {{ text-align: center; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Trading SaaS Test Dashboard</h1>
        
        <div class="summary">
            <h2>Test Summary</h2>
            <p>Generated: <span class="stat">{timestamp}</span></p>
            <p>Total Tests: <span class="stat">{results['summary']['total_tests']}</span></p>
            <p>Passed: <span class="stat">{results['summary']['passed']}</span> ({results['summary']['pass_rate']:.1f}%)</p>
            <p>Failed: <span class="stat">{results['summary']['failed']}</span></p>
            <p>Total Execution Time: <span class="stat">{results['summary']['total_duration']:.2f} seconds</span></p>
        </div>
        
        <div class="dashboard">
            <h2>Results Dashboard</h2>
            <img src="test_results_dashboard.png" alt="Test Results Dashboard" style="max-width: 100%;">
        </div>
        
        <div>
            <h2>Signal Generation Statistics</h2>
            <table>
                <tr>
                    <th>Asset</th>
                    <th>Signals Generated</th>
                    <th>Signals Approved</th>
                    <th>Approval Rate</th>
                </tr>
    """
    
    # Add rows for each asset
    for i, asset in enumerate(results['signal_counts']['assets']):
        generated = results['signal_counts']['counts'][i]
        approved = results['signal_counts']['approved'][i]
        rate = (approved / generated * 100) if generated > 0 else 0
        
        html_content += f"""
                <tr>
                    <td>{asset}</td>
                    <td>{generated}</td>
                    <td>{approved}</td>
                    <td>{rate:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div>
            <h2>Test Execution Times</h2>
            <table>
                <tr>
                    <th>Test Type</th>
                    <th>Execution Time (seconds)</th>
                    <th>Number of Tests</th>
                </tr>
    """
    
    # Add rows for each test type
    for test_type, execution_time in results['execution_times'].items():
        html_content += f"""
                <tr>
                    <td>{test_type}</td>
                    <td>{execution_time:.2f}</td>
                    <td>{results['test_counts'].get(test_type, 0)}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(dashboard_dir, 'index.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard created at: {os.path.join(dashboard_dir, 'index.html')}")

def extract_test_results():
    """
    Extract test results from the test output directories.
    This is a simplified version that uses mock data.
    In a real implementation, you would parse actual test results.
    
    Returns:
        Dictionary with test result data
    """
    # This would normally parse real test results
    # For this example, we'll use mock data
    
    results = {
        'summary': {
            'total_tests': 45,
            'passed': 42,
            'failed': 3,
            'pass_rate': 93.3,
            'total_duration': 25.7
        },
        'execution_times': {
            'unit': 5.3,
            'integration': 8.2,
            'e2e': 9.8,
            'performance': 2.4
        },
        'test_counts': {
            'unit': 25,
            'integration': 10,
            'e2e': 5,
            'performance': 5
        },
        'signal_counts': {
            'assets': ['BTC_USD', 'ETH_USD', 'SPY', 'EUR_USD_FX'],
            'counts': [15, 12, 8, 5],
            'approved': [10, 8, 5, 3]
        },
        'approval_stats': {
            'total': 40,
            'approved': 26
        },
        'regime_distribution': {
            'types': ['BULLISH_TRENDING', 'BEARISH_TRENDING', 'RANGING', 'HIGH_VOLATILITY', 'LOW_VOLATILITY', 'NEUTRAL'],
            'counts': [8, 5, 12, 3, 2, 10]
        }
    }
    
    return results

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run Trading SaaS tests')
    parser.add_argument('--type', choices=['unit', 'integration', 'e2e', 'performance', 'all'],
                      default='all', help='Type of tests to run')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    parser.add_argument('--no-html', action='store_true', help='Skip HTML report generation')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard after tests')
    
    args = parser.parse_args()
    
    print(f"Running {'all' if args.type == 'all' else args.type} tests...")
    
    # Convert 'all' to None for the run_tests function
    test_type = None if args.type == 'all' else args.type
    
    # Run tests
    exit_code, duration = run_tests(
        test_type=test_type,
        verbose=not args.quiet,
        html_report=not args.no_html
    )
    
    print(f"Tests completed in {duration:.2f} seconds with exit code {exit_code}")
    
    # Generate dashboard if requested
    if args.dashboard:
        print("Generating test results dashboard...")
        results = extract_test_results()
        create_dashboard(results)
    
    # Return the exit code to the shell
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
