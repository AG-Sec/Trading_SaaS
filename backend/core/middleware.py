from fastapi import Request, HTTPException, status
import logging
from typing import Callable, Dict, List, Union
from functools import wraps

from backend.models.user import SubscriptionTier, User
from backend.core.security import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Define feature access by subscription tier
# This maps specific features to the minimum subscription tier required
FEATURE_ACCESS = {
    # Trading signals and assets
    "real_time_signals": SubscriptionTier.PRO,
    "all_assets": SubscriptionTier.PRO,
    "basic_assets": SubscriptionTier.BASIC,  # Basic tier has limited assets (e.g., SPY, BTC-USD)
    
    # Trading capabilities
    "paper_trading": SubscriptionTier.PRO,
    "live_trading": SubscriptionTier.ENTERPRISE,
    
    # Strategy customization
    "strategy_templates": SubscriptionTier.BASIC,
    "strategy_customization": SubscriptionTier.ENTERPRISE,
    "strategy_dsl_editor": SubscriptionTier.ENTERPRISE,
    
    # Risk management
    "basic_risk_management": SubscriptionTier.BASIC,
    "advanced_risk_management": SubscriptionTier.PRO,
    "custom_risk_parameters": SubscriptionTier.ENTERPRISE,
    
    # Notifications
    "basic_notifications": SubscriptionTier.BASIC,  # Delayed by 15-30 minutes
    "priority_notifications": SubscriptionTier.PRO,  # Real-time
    "click_to_execute": SubscriptionTier.ENTERPRISE,  # Execute trades from notifications
    
    # Analytics
    "basic_performance_metrics": SubscriptionTier.BASIC,
    "advanced_analytics": SubscriptionTier.PRO,
    "regime_specific_analytics": SubscriptionTier.ENTERPRISE,
}

# Tier limits for specific resources or features
TIER_LIMITS = {
    SubscriptionTier.BASIC: {
        "max_assets": 3,  # Max number of assets to track
        "max_custom_strategies": 1,  # Max number of custom strategies
        "max_concurrent_signals": 3,  # Max number of active signals
        "signal_delay_minutes": 15,  # Delayed signals
    },
    SubscriptionTier.PRO: {
        "max_assets": 10,
        "max_custom_strategies": 5,
        "max_concurrent_signals": 10,
        "signal_delay_minutes": 0,  # Real-time signals
    },
    SubscriptionTier.ENTERPRISE: {
        "max_assets": float('inf'),  # Unlimited
        "max_custom_strategies": float('inf'),  # Unlimited
        "max_concurrent_signals": float('inf'),  # Unlimited
        "signal_delay_minutes": 0,  # Real-time signals
    }
}

def requires_feature(feature_name: str):
    """
    Decorator to restrict access to endpoints based on required features
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the request object from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request and 'request' in kwargs:
                request = kwargs['request']
            
            if not request:
                logger.error("Could not find request object in function arguments")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )
            
            # Get the authenticated user
            try:
                token = request.headers.get('Authorization', '').replace('Bearer ', '')
                if not token:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Not authenticated",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                # Call FastAPI dependency to get current user
                user = await get_current_user(token)
            except HTTPException as e:
                # Re-raise the exception with the same details
                raise e
            
            # Check if the user has the required subscription tier for this feature
            required_tier = FEATURE_ACCESS.get(feature_name)
            if not required_tier:
                logger.warning(f"Unknown feature requested: {feature_name}")
                # If feature not found in mapping, default to ENTERPRISE tier
                required_tier = SubscriptionTier.ENTERPRISE
            
            # Determine if user's tier is sufficient (using enum values)
            tier_values = {
                SubscriptionTier.BASIC: 1,
                SubscriptionTier.PRO: 2,
                SubscriptionTier.ENTERPRISE: 3
            }
            
            user_tier_value = tier_values.get(user.subscription_tier, 0)
            required_tier_value = tier_values.get(required_tier, 3)
            
            if user_tier_value < required_tier_value:
                logger.warning(
                    f"User {user.username} (tier: {user.subscription_tier}) "
                    f"attempted to access feature requiring {required_tier} tier"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"This feature requires a {required_tier.value} subscription"
                )
            
            # User has access, call the original function
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator

async def get_tier_limit(user: User, limit_name: str) -> Union[int, float]:
    """
    Get the limit value for a specific tier and limit name
    """
    tier_limits = TIER_LIMITS.get(user.subscription_tier, {})
    return tier_limits.get(limit_name, 0)

def is_feature_accessible(user: User, feature_name: str) -> bool:
    """
    Check if a specific feature is accessible by a user based on their subscription tier
    """
    required_tier = FEATURE_ACCESS.get(feature_name, SubscriptionTier.ENTERPRISE)
    
    # Determine if user's tier is sufficient (using enum values)
    tier_values = {
        SubscriptionTier.BASIC: 1,
        SubscriptionTier.PRO: 2,
        SubscriptionTier.ENTERPRISE: 3
    }
    
    user_tier_value = tier_values.get(user.subscription_tier, 0)
    required_tier_value = tier_values.get(required_tier, 3)
    
    return user_tier_value >= required_tier_value
