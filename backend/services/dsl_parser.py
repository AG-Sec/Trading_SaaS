import yaml
import json
import re
import logging
from fastapi import HTTPException, status
from typing import Dict, Any, List, Tuple, Optional, Union
import jsonschema
from pydantic import ValidationError

# Configure logging
logger = logging.getLogger(__name__)

# Define DSL schema as JSON Schema
DSL_SCHEMA = {
    "type": "object",
    "required": ["strategy_name", "rules", "risk"],
    "properties": {
        "strategy_name": {"type": "string"},
        "description": {"type": "string"},
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["then"],
                "properties": {
                    "if": {"type": "string"},
                    "and": {"type": "string"},
                    "or": {"type": "string"},
                    "not": {"type": "string"},
                    "then": {"type": "string"}
                }
            }
        },
        "risk": {
            "type": "object",
            "required": ["entry_size_pct"],
            "properties": {
                "entry_size_pct": {"type": "number", "minimum": 0.01, "maximum": 5.0},
                "stop": {"type": "string"},
                "take_profit": {"type": "string"},
                "max_trades": {"type": "integer", "minimum": 1},
                "max_daily_drawdown_pct": {"type": "number", "minimum": 0.1, "maximum": 10.0}
            }
        },
        "timeframes": {
            "type": "array",
            "items": {"type": "string", "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]}
        },
        "assets": {
            "type": "array",
            "items": {"type": "string"}
        },
        "filters": {
            "type": "object",
            "properties": {
                "min_volume": {"type": "number", "minimum": 0},
                "min_volatility": {"type": "number", "minimum": 0},
                "market_hours_only": {"type": "boolean"}
            }
        }
    }
}

# Allowed technical indicators and functions in expressions
ALLOWED_INDICATORS = {
    # Moving Averages
    "sma": {"params": [("length", int)]},
    "ema": {"params": [("length", int)]},
    "wma": {"params": [("length", int)]},
    
    # Oscillators
    "rsi": {"params": [("length", int)]},
    "macd": {"params": [("fast_length", int), ("slow_length", int), ("signal_length", int)]},
    "stoch": {"params": [("k_length", int), ("d_length", int)]},
    
    # Volatility
    "atr": {"params": [("length", int)]},
    "bollinger": {"params": [("length", int), ("std_dev", float)]},
    
    # Volume
    "obv": {"params": []},
    "volume": {"params": []},
    
    # Price Action
    "highest": {"params": [("length", int)]},
    "lowest": {"params": [("length", int)]},
    "close": {"params": []},
    "open": {"params": []},
    "high": {"params": []},
    "low": {"params": []},
    
    # Candlestick Patterns
    "doji": {"params": []},
    "engulfing": {"params": []},
    "hammer": {"params": []}
}

# Allowed operators in expressions
ALLOWED_OPERATORS = ["<", ">", "<=", ">=", "==", "+", "-", "*", "/", "and", "or", "not", "cross above", "cross below"]

class DSLParser:
    """
    Parser for the Trading SaaS custom strategy DSL format.
    
    The DSL follows a YAML-like syntax:
    
    ```yaml
    strategy_name: ema_rsi_confluence
    rules:
      - if: ema(9) > ema(21)
        and: rsi(14) < 70
        then: signal = "long"
      - if: ema(9) < ema(21)
        and: rsi(14) > 30
        then: signal = "short"
    risk:
      entry_size_pct: 0.5
      stop: atr(14) * 1.5
      take_profit: 2R
    ```
    """
    
    @staticmethod
    def parse(dsl_content: str) -> Dict[str, Any]:
        """
        Parse DSL content string into a structured dictionary.
        
        Args:
            dsl_content: String containing the YAML-like DSL content
            
        Returns:
            Dict representing the parsed strategy
            
        Raises:
            HTTPException: If the DSL content is invalid
        """
        try:
            # Parse YAML content
            parsed_dsl = yaml.safe_load(dsl_content)
            
            # Validate against schema
            jsonschema.validate(instance=parsed_dsl, schema=DSL_SCHEMA)
            
            # Validate expressions in rules
            for rule in parsed_dsl.get("rules", []):
                DSLParser._validate_rule_expressions(rule)
                
            # Validate risk management expressions
            DSLParser._validate_risk_expressions(parsed_dsl.get("risk", {}))
            
            return parsed_dsl
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid YAML format: {str(e)}"
            )
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Schema validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy structure: {str(e.message)}"
            )
        except ValueError as e:
            logger.error(f"Expression validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error parsing DSL: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error parsing strategy DSL: {str(e)}"
            )
    
    @staticmethod
    def _validate_rule_expressions(rule: Dict[str, str]) -> None:
        """
        Validate expressions in a rule to ensure they use allowed indicators and syntax.
        
        Args:
            rule: Dict containing the rule expressions
            
        Raises:
            ValueError: If an expression is invalid
        """
        for key, expr in rule.items():
            if key == "then":
                # Validate signal assignment
                if not re.match(r'^signal\s*=\s*"(long|short|exit)"$', expr):
                    raise ValueError(f"Invalid signal assignment: {expr}. Must be signal = \"long\", signal = \"short\", or signal = \"exit\".")
            elif key in ["if", "and", "or", "not"]:
                # Validate condition expression
                DSLParser._validate_expression(expr)
            else:
                raise ValueError(f"Unknown rule key: {key}")
    
    @staticmethod
    def _validate_risk_expressions(risk: Dict[str, Any]) -> None:
        """
        Validate risk management expressions.
        
        Args:
            risk: Dict containing the risk management settings
            
        Raises:
            ValueError: If a risk expression is invalid
        """
        # Validate stop loss expression if present
        if "stop" in risk and isinstance(risk["stop"], str):
            DSLParser._validate_expression(risk["stop"])
            
        # Validate take profit expression if present
        if "take_profit" in risk and isinstance(risk["take_profit"], str):
            # Allow "R" notation for risk-reward ratio
            if not re.match(r'^\d+(\.\d+)?R$', risk["take_profit"]):
                DSLParser._validate_expression(risk["take_profit"])
    
    @staticmethod
    def _validate_expression(expr: str) -> None:
        """
        Validate technical indicator and operator usage in expressions.
        
        Args:
            expr: String containing the expression
            
        Raises:
            ValueError: If the expression uses disallowed indicators or operators
        """
        # Check for indicator calls
        for indicator_call in re.finditer(r'(\w+)\(([^)]*)\)', expr):
            indicator_name = indicator_call.group(1)
            params_str = indicator_call.group(2)
            
            # Check if indicator is allowed
            if indicator_name not in ALLOWED_INDICATORS:
                raise ValueError(f"Unsupported indicator: {indicator_name}")
                
            # Parse parameters
            if params_str:
                params = [p.strip() for p in params_str.split(',')]
                expected_params = ALLOWED_INDICATORS[indicator_name]["params"]
                
                # Check parameter count
                if len(params) != len(expected_params):
                    raise ValueError(
                        f"Indicator {indicator_name} expects {len(expected_params)} "
                        f"parameters, but got {len(params)}"
                    )
                
                # Validate each parameter type
                for i, (param_value, (param_name, param_type)) in enumerate(zip(params, expected_params)):
                    try:
                        if param_type == int:
                            int(param_value)
                        elif param_type == float:
                            float(param_value)
                    except ValueError:
                        raise ValueError(
                            f"Parameter {i+1} of {indicator_name} ({param_name}) should be a {param_type.__name__}"
                        )
        
        # Check for disallowed operators or functions
        # This is a simplified check - a complete parser would use tokenization
        for operator in ALLOWED_OPERATORS:
            expr = expr.replace(operator, "")
            
        # Remove indicator calls already validated
        expr = re.sub(r'\w+\([^)]*\)', "", expr)
        
        # Remove literals and whitespace
        expr = re.sub(r'"[^"]*"', "", expr)
        expr = re.sub(r'\d+(\.\d+)?', "", expr)
        expr = re.sub(r'\s', "", expr)
        
        # Check for remaining unexpected characters
        expr = re.sub(r'[().,]', "", expr)
        
        if expr.strip():
            unusual_chars = ''.join(set(expr.strip()))
            raise ValueError(f"Expression contains unexpected characters: {unusual_chars}")
    
    @staticmethod
    def to_python_code(parsed_dsl: Dict[str, Any]) -> str:
        """
        Convert parsed DSL to executable Python code.
        
        Args:
            parsed_dsl: Dict containing the parsed DSL
            
        Returns:
            String containing Python code that implements the strategy
        """
        strategy_name = parsed_dsl.get("strategy_name", "custom_strategy")
        
        code = [
            "# Auto-generated strategy code from DSL",
            f"# Strategy: {strategy_name}",
            "import pandas as pd",
            "import numpy as np",
            "from backend.strategies.strategy_base import StrategyBase",
            "from shared_types import TradingSignalModel, SignalType, AssetSymbol, Timeframe",
            "",
            f"class {strategy_name.title().replace('_', '')}Strategy(StrategyBase):",
            "    def __init__(self):",
            f"        super().__init__(name='{strategy_name}')",
            "",
            "    def generate_signals(self, df, asset, timeframe):",
            "        \"\"\"\n        Generate trading signals based on the strategy rules.\n        \"\"\"",
            "        # Initialize signal column",
            "        df['signal'] = None",
            "",
            "        # Calculate indicators"
        ]
        
        # Extract all indicators used in rules
        indicators = set()
        for rule in parsed_dsl.get("rules", []):
            for key, expr in rule.items():
                if key != "then":
                    indicators.update(re.findall(r'(\w+)\(([^)]*)\)', expr))
        
        # Add indicator calculations
        for indicator, params in indicators:
            if indicator in ALLOWED_INDICATORS:
                param_values = [p.strip() for p in params.split(',')] if params else []
                if indicator == "sma":
                    code.append(f"        df['sma_{param_values[0]}'] = df['close'].rolling(window={param_values[0]}).mean()")
                elif indicator == "ema":
                    code.append(f"        df['ema_{param_values[0]}'] = df['close'].ewm(span={param_values[0]}, adjust=False).mean()")
                elif indicator == "rsi":
                    code.append(f"        # Calculate RSI {param_values[0]}")
                    code.append(f"        delta = df['close'].diff()")
                    code.append(f"        gain = (delta.where(delta > 0, 0)).rolling(window={param_values[0]}).mean()")
                    code.append(f"        loss = (-delta.where(delta < 0, 0)).rolling(window={param_values[0]}).mean()")
                    code.append(f"        rs = gain / loss")
                    code.append(f"        df['rsi_{param_values[0]}'] = 100 - (100 / (1 + rs))")
                elif indicator == "atr":
                    code.append(f"        # Calculate ATR {param_values[0]}")
                    code.append(f"        high_low = df['high'] - df['low']")
                    code.append(f"        high_close = (df['high'] - df['close'].shift()).abs()")
                    code.append(f"        low_close = (df['low'] - df['close'].shift()).abs()")
                    code.append(f"        ranges = pd.concat([high_low, high_close, low_close], axis=1)")
                    code.append(f"        true_range = ranges.max(axis=1)")
                    code.append(f"        df['atr_{param_values[0]}'] = true_range.rolling({param_values[0]}).mean()")
        
        code.append("")
        code.append("        # Apply strategy rules")
        
        # Process rules
        for i, rule in enumerate(parsed_dsl.get("rules", [])):
            conditions = []
            for key, expr in rule.items():
                if key == "if":
                    conditions.append(DSLParser._convert_condition_to_python(expr))
                elif key == "and":
                    conditions.append(DSLParser._convert_condition_to_python(expr))
                elif key == "or":
                    conditions.append(DSLParser._convert_condition_to_python(expr))
                elif key == "not":
                    conditions.append(f"~({DSLParser._convert_condition_to_python(expr)})")
                    
            # Combine conditions and apply signal
            if "then" in rule:
                signal_value = re.search(r'"([^"]*)"', rule["then"]).group(1)
                if conditions:
                    condition_code = " & ".join(f"({c})" for c in conditions)
                    code.append(f"        # Rule {i+1}")
                    code.append(f"        mask = {condition_code}")
                    code.append(f"        df.loc[mask, 'signal'] = '{signal_value}'")
        
        # Handle risk management
        risk = parsed_dsl.get("risk", {})
        entry_size_pct = risk.get("entry_size_pct", 1.0)
        
        code.append("")
        code.append("        # Process signals to create signal models")
        code.append("        signals = []")
        code.append("        for idx, row in df.iterrows():")
        code.append("            if pd.notna(row['signal']):")
        code.append("                if row['signal'] == 'long':")
        code.append("                    signal_type = SignalType.LONG")
        code.append("                elif row['signal'] == 'short':")
        code.append("                    signal_type = SignalType.SHORT")
        code.append("                elif row['signal'] == 'exit':")
        code.append("                    signal_type = SignalType.EXIT")
        code.append("                else:")
        code.append("                    continue")
        code.append("")
        code.append("                # Risk management")
        code.append(f"                position_size_pct = {entry_size_pct}")
        
        # Handle stop loss calculation
        if "stop" in risk:
            stop_expr = risk["stop"]
            if "atr" in stop_expr:
                code.append("                # Dynamic stop loss based on ATR")
                code.append("                atr_value = row[f'atr_{re.search(r'atr\\((\\d+)\\)', stop_expr).group(1)}']")
                code.append(f"                stop_distance = atr_value * {re.search(r'atr\\(\\d+\\)\\s*\\*\\s*([\\d.]+)', stop_expr).group(1) if '*' in stop_expr else '1.0'}")
                code.append("                if signal_type == SignalType.LONG:")
                code.append("                    stop_price = row['close'] - stop_distance")
                code.append("                else:  # SHORT")
                code.append("                    stop_price = row['close'] + stop_distance")
            else:
                code.append("                # Fixed percentage stop loss")
                code.append("                stop_pct = 0.02  # Default 2% stop")
                code.append("                if signal_type == SignalType.LONG:")
                code.append("                    stop_price = row['close'] * (1 - stop_pct)")
                code.append("                else:  # SHORT")
                code.append("                    stop_price = row['close'] * (1 + stop_pct)")
        else:
            code.append("                # Default stop loss (2%)")
            code.append("                stop_pct = 0.02")
            code.append("                if signal_type == SignalType.LONG:")
            code.append("                    stop_price = row['close'] * (1 - stop_pct)")
            code.append("                else:  # SHORT")
            code.append("                    stop_price = row['close'] * (1 + stop_pct)")
        
        # Handle take profit calculation
        if "take_profit" in risk:
            take_profit_expr = risk["take_profit"]
            if "R" in take_profit_expr:
                # Risk-reward ratio
                rr_ratio = float(take_profit_expr.replace("R", ""))
                code.append(f"                # Take profit based on {rr_ratio}R risk-reward ratio")
                code.append(f"                risk_amount = abs(row['close'] - stop_price)")
                code.append(f"                take_profit_distance = risk_amount * {rr_ratio}")
                code.append("                if signal_type == SignalType.LONG:")
                code.append("                    take_profit_price = row['close'] + take_profit_distance")
                code.append("                else:  # SHORT")
                code.append("                    take_profit_price = row['close'] - take_profit_distance")
            else:
                code.append("                # Fixed take profit (4%)")
                code.append("                take_profit_pct = 0.04")
                code.append("                if signal_type == SignalType.LONG:")
                code.append("                    take_profit_price = row['close'] * (1 + take_profit_pct)")
                code.append("                else:  # SHORT")
                code.append("                    take_profit_price = row['close'] * (1 - take_profit_pct)")
        else:
            code.append("                # Default take profit (4%)")
            code.append("                take_profit_pct = 0.04")
            code.append("                if signal_type == SignalType.LONG:")
            code.append("                    take_profit_price = row['close'] * (1 + take_profit_pct)")
            code.append("                else:  # SHORT")
            code.append("                    take_profit_price = row['close'] * (1 - take_profit_pct)")
        
        # Create and append signal
        code.append("")
        code.append("                # Create signal model")
        code.append("                signal = TradingSignalModel(")
        code.append("                    asset=asset,")
        code.append("                    timeframe=timeframe,")
        code.append("                    signal_type=signal_type,")
        code.append("                    entry_price=row['close'],")
        code.append("                    stop_loss=stop_price,")
        code.append("                    take_profit=take_profit_price,")
        code.append("                    timestamp=idx.to_pydatetime(),")
        code.append("                    confidence=0.8,  # Default confidence")
        code.append("                    risk_reward_ratio=abs(take_profit_price - row['close']) / abs(stop_price - row['close']),")
        code.append("                    position_size_pct=position_size_pct")
        code.append("                )")
        code.append("                signals.append(signal)")
        
        code.append("")
        code.append("        return signals")
        
        return "\n".join(code)
    
    @staticmethod
    def _convert_condition_to_python(expr: str) -> str:
        """
        Convert a DSL condition expression to Python code.
        
        Args:
            expr: String containing the DSL condition
            
        Returns:
            String containing equivalent Python code
        """
        # Replace indicator calls with dataframe references
        for indicator_call in re.finditer(r'(\w+)\(([^)]*)\)', expr):
            indicator_name = indicator_call.group(1)
            params_str = indicator_call.group(2)
            
            if indicator_name in ALLOWED_INDICATORS:
                # Create pandas column reference
                if params_str:
                    params = [p.strip() for p in params_str.split(',')]
                    column_name = f"{indicator_name}_{params[0]}"
                    expr = expr.replace(f"{indicator_name}({params_str})", f"df['{column_name}']")
                else:
                    expr = expr.replace(f"{indicator_name}()", f"df['{indicator_name}']")
        
        # Replace operators
        expr = expr.replace("cross above", "> df.shift(1)")
        expr = expr.replace("cross below", "< df.shift(1)")
        
        return expr
