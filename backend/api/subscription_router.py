from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.user import User, SubscriptionTier
from backend.services.subscription_service import SubscriptionService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class SubscriptionRequest(BaseModel):
    tier: SubscriptionTier
    payment_method_id: Optional[str] = None

class SubscriptionResponse(BaseModel):
    success: bool
    message: str
    subscription_info: Optional[Dict[str, Any]] = None

@router.get("/current", response_model=Dict[str, Any])
async def get_current_subscription(
    current_user: User = Depends(get_current_user)
):
    """Get the current user's subscription details"""
    subscription_info = await SubscriptionService.get_subscription_info(current_user)
    return subscription_info

@router.get("/tiers", response_model=Dict[str, Any])
async def get_subscription_tiers():
    """Get all available subscription tiers and their details"""
    tiers = await SubscriptionService.get_available_tiers()
    return tiers

@router.post("/upgrade", response_model=SubscriptionResponse)
async def upgrade_subscription(
    subscription_data: SubscriptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upgrade to a higher subscription tier"""
    try:
        # Upgrade subscription
        updated_user = await SubscriptionService.upgrade_subscription(
            db, 
            current_user, 
            subscription_data.tier, 
            subscription_data.payment_method_id
        )
        
        # Get updated subscription info
        subscription_info = await SubscriptionService.get_subscription_info(updated_user)
        
        return {
            "success": True,
            "message": f"Successfully upgraded to {subscription_data.tier.value} tier",
            "subscription_info": subscription_info
        }
    except HTTPException as e:
        return {
            "success": False,
            "message": e.detail,
            "subscription_info": await SubscriptionService.get_subscription_info(current_user)
        }
    except Exception as e:
        logger.error(f"Error upgrading subscription: {str(e)}")
        return {
            "success": False,
            "message": "An error occurred while processing your upgrade request",
            "subscription_info": await SubscriptionService.get_subscription_info(current_user)
        }

@router.post("/downgrade", response_model=SubscriptionResponse)
async def downgrade_subscription(
    subscription_data: SubscriptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Downgrade to a lower subscription tier"""
    try:
        # Downgrade subscription
        updated_user = await SubscriptionService.downgrade_subscription(
            db, 
            current_user, 
            subscription_data.tier
        )
        
        # Get updated subscription info
        subscription_info = await SubscriptionService.get_subscription_info(updated_user)
        
        return {
            "success": True,
            "message": f"Successfully downgraded to {subscription_data.tier.value} tier",
            "subscription_info": subscription_info
        }
    except HTTPException as e:
        return {
            "success": False,
            "message": e.detail,
            "subscription_info": await SubscriptionService.get_subscription_info(current_user)
        }
    except Exception as e:
        logger.error(f"Error downgrading subscription: {str(e)}")
        return {
            "success": False,
            "message": "An error occurred while processing your downgrade request",
            "subscription_info": await SubscriptionService.get_subscription_info(current_user)
        }

@router.post("/cancel", response_model=SubscriptionResponse)
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel the current subscription (reverts to basic tier)"""
    try:
        # Cancel subscription
        updated_user = await SubscriptionService.cancel_subscription(db, current_user)
        
        # Get updated subscription info
        subscription_info = await SubscriptionService.get_subscription_info(updated_user)
        
        return {
            "success": True,
            "message": "Successfully cancelled subscription",
            "subscription_info": subscription_info
        }
    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        return {
            "success": False,
            "message": "An error occurred while processing your cancellation request",
            "subscription_info": await SubscriptionService.get_subscription_info(current_user)
        }

@router.post("/webhook", status_code=status.HTTP_200_OK)
async def payment_webhook(webhook_data: Dict[str, Any] = Body(...)):
    """
    Handle payment webhooks from payment processor
    
    This endpoint would be called by a payment processor like Stripe
    to notify the application of subscription events.
    """
    try:
        # Process the webhook
        result = await SubscriptionService.process_payment_webhook(webhook_data)
        
        if result:
            return {"status": "success"}
        else:
            return {"status": "error", "message": "Failed to process webhook"}
    except Exception as e:
        logger.error(f"Error processing payment webhook: {str(e)}")
        return {"status": "error", "message": str(e)}
