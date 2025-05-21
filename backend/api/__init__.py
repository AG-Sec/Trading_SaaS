# This file makes 'api' a Python package

from .trading_router import router as trading_router

__all__ = ["trading_router"]
