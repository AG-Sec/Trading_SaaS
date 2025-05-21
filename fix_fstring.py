#!/usr/bin/env python
"""
Utility script to fix f-string formatting issues in SignalScannerAgent
"""
import re
import sys

def fix_fstring_issues(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Pattern to match problematic RSI f-string format
    pattern = r'(logger\.info\(f".*?)({rsi:\.2f if rsi is not None else \'N/A\'})(.*?"\))'
    
    # Replacement with correct formatting approach
    replacement = r'\1{rsi_str}\3'
    
    # Add variable declaration before problematic lines
    content = re.sub(pattern, r'# Safe formatting for RSI\n                rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"\n                \1{rsi_str}\3', content)
    
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Fixed f-string issues in {file_path}")

if __name__ == "__main__":
    scanner_agent_path = "/home/eazy/code/Trading_SaaS/backend/agents/signal_scanner_agent.py"
    fix_fstring_issues(scanner_agent_path)
