from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status

from backend.models.user import User, SubscriptionTier
from backend.services.auth_service import AuthService
from backend.core.middleware import TIER_LIMITS

# Configure logging
logger = logging.getLogger(__name__)

class SubscriptionService:
    """Service for managing user subscriptions and tiers"""
    
    # Prices in USD per month for each tier
    SUBSCRIPTION_PRICES = {
        SubscriptionTier.BASIC: 0.00,  # Free tier
        SubscriptionTier.PRO: 29.99,
        SubscriptionTier.ENTERPRISE: 99.99
    }
    
    # Features included in each tier (for display purposes)
    TIER_FEATURES = {
        SubscriptionTier.BASIC: [
            "Basic market signals (delayed 15-30 minutes)",
            "Limited asset selection (SPY, BTC-USD, ETH-USD)",
            "Basic risk management presets",
            "Basic performance metrics"
        ],
        SubscriptionTier.PRO: [
            "Real-time market signals",
            "All supported assets and timeframes",
            "Paper trading capabilities",
            "Advanced risk management",
            "Priority notifications via Telegram and Email",
            "Advanced performance analytics"
        ],
        SubscriptionTier.ENTERPRISE: [
            "All PRO features",
            "Custom strategy creation with DSL editor",
            "Live trading capabilities",
            "Regime-specific analytics",
            "Click-to-execute from notifications",
            "Fully customizable risk parameters"
        ]
    }
    
    @staticmethod
    async def get_subscription_info(user: User) -> Dict[str, Any]:
        """Get information about a user's current subscription"""
        # In a real implementation, this would query a subscriptions table
        # For now, we'll return static data based on the user's current tier
        return {
            "current_tier": user.subscription_tier,
            "price": SubscriptionService.SUBSCRIPTION_PRICES[user.subscription_tier],
            "features": SubscriptionService.TIER_FEATURES[user.subscription_tier],
            "limits": TIER_LIMITS[user.subscription_tier],
            # In a real implementation, these would be pulled from the database
            "renewal_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "payment_method": "None" if user.subscription_tier == SubscriptionTier.BASIC else "Credit Card",
            "subscription_status": "Active"
        }
    
    @staticmethod
    async def get_available_tiers() -> Dict[str, Any]:
        """Get information about all available subscription tiers"""
        tiers = {}
        
        for tier in SubscriptionTier:
            tiers[tier.value] = {
                "price": SubscriptionService.SUBSCRIPTION_PRICES[tier],
                "features": SubscriptionService.TIER_FEATURES[tier],
                "limits": TIER_LIMITS[tier]
            }
        
        return tiers
    
    @staticmethod
    async def upgrade_subscription(
        db: Session, 
        user: User, 
        new_tier: SubscriptionTier,
        payment_method_id: Optional[str] = None
    ) -> User:
        """
        Upgrade a user's subscription to a higher tier
        
        In a production environment, this would integrate with a payment
        processor like Stripe and handle the actual billing.
        """
        # Validate the tier upgrade
        if SubscriptionService.SUBSCRIPTION_PRICES[new_tier] <= SubscriptionService.SUBSCRIPTION_PRICES[user.subscription_tier]:
            logger.warning(f"User {user.username} attempted to upgrade to a lower or same tier: {new_tier}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot upgrade to a lower or same tier"
            )
        
        # In a real implementation, we would:
        # 1. Process payment using the payment_method_id with Stripe/other provider
        # 2. Update subscription details in a subscriptions table
        # 3. Set up recurring billing
        
        # For now, just update the user's tier directly
        user_data = {"subscription_tier": new_tier}
        updated_user = AuthService.update_user(db, user.id, user_data)
        
        logger.info(f"User {user.username} upgraded subscription from {user.subscription_tier} to {new_tier}")
        
        return updated_user
    
    @staticmethod
    async def downgrade_subscription(
        db: Session, 
        user: User, 
        new_tier: SubscriptionTier
    ) -> User:
        """
        Downgrade a user's subscription to a lower tier
        
        Changes take effect at the end of the current billing cycle.
        """
        # Validate the tier downgrade
        if SubscriptionService.SUBSCRIPTION_PRICES[new_tier] >= SubscriptionService.SUBSCRIPTION_PRICES[user.subscription_tier]:
            logger.warning(f"User {user.username} attempted to downgrade to a higher or same tier: {new_tier}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot downgrade to a higher or same tier"
            )
        
        # In a real implementation, we would:
        # 1. Update subscription in payment processor to downgrade at end of billing cycle
        # 2. Update subscription_pending_tier in the database
        # 3. At the end of billing cycle, update the actual tier
        
        # For now, just update the user's tier directly
        user_data = {"subscription_tier": new_tier}
        updated_user = AuthService.update_user(db, user.id, user_data)
        
        logger.info(f"User {user.username} downgraded subscription from {user.subscription_tier} to {new_tier}")
        
        return updated_user
    
    @staticmethod
    async def cancel_subscription(db: Session, user: User) -> User:
        """
        Cancel a user's subscription
        
        Reverts to BASIC tier at the end of the current billing cycle.
        """
        # In a real implementation, we would:
        # 1. Cancel subscription in payment processor
        # 2. Set subscription_end_date in the database
        # 3. At the end of billing cycle, update tier to BASIC
        
        # For now, just update the user's tier directly
        user_data = {"subscription_tier": SubscriptionTier.BASIC}
        updated_user = AuthService.update_user(db, user.id, user_data)
        
        logger.info(f"User {user.username} cancelled subscription, reverted to {SubscriptionTier.BASIC}")
        
        return updated_user
    
    @staticmethod
    async def process_payment_webhook(webhook_data: Dict[str, Any]) -> bool:
        """
        Process a payment webhook from a payment processor
        
        This would handle successful payments, failed payments, etc.
        """
        # In a real implementation, this would:
        # 1. Verify the webhook signature
        # 2. Handle different event types (payment succeeded, failed, etc.)
        # 3. Update the user's subscription status accordingly
        
        logger.info("Payment webhook received")
        logger.debug(f"Webhook data: {webhook_data}")
        
        # For now, just return True to indicate success
        return True
