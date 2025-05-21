from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import logging
from typing import List
from datetime import timedelta

from backend.core.database import get_db
from backend.core.security import get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES
from backend.models.user import User, UserCreate, UserUpdate, UserResponse, Token, APIKeyCreate, APIKeyResponse, VerificationRequest
from backend.services.auth_service import AuthService
from backend.services.email_service import EmailService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """Register a new user and send verification email"""
    # Create user
    user = AuthService.create_user(db, user_data)
    
    # Generate verification token
    token = AuthService.generate_verification_token()
    
    # Send verification email (async)
    background_tasks.add_task(
        EmailService.send_verification_email,
        user.email,
        user.username,
        token
    )
    
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Authenticate a user and return a JWT token"""
    # Authenticate user
    user = AuthService.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token with the default expiry time
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = AuthService.create_user_token(user, access_token_expires)
    
    return token_data


@router.post("/verify", status_code=status.HTTP_200_OK)
async def verify_email(
    verification_data: VerificationRequest,
    db: Session = Depends(get_db)
):
    """Verify a user's email address using the token sent to their email"""
    # TODO: Implement token verification logic when email service is configured
    # For now, we'll just return a success message
    return {"message": "Email verification functionality will be implemented"}


@router.get("/me", response_model=UserResponse)
async def get_user_info(current_user: User = Depends(get_current_user)):
    """Get the current authenticated user's information"""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_info(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update the current authenticated user's information"""
    updated_user = AuthService.update_user(db, current_user.id, user_data)
    return updated_user


@router.post("/api-keys", response_model=APIKeyResponse)
async def add_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a broker API key for the current user"""
    api_key = AuthService.add_broker_api_key(
        db, 
        current_user.id, 
        api_key_data.provider, 
        api_key_data.api_key, 
        api_key_data.api_secret
    )
    return api_key


@router.get("/api-keys", response_model=List[APIKeyResponse])
async def get_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all API keys for the current user"""
    return AuthService.get_broker_api_keys(db, current_user.id)


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an API key for the current user"""
    AuthService.delete_broker_api_key(db, current_user.id, key_id)
    return None


@router.post("/reset-password/request")
async def request_password_reset(
    email: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Request a password reset link to be sent to the user's email"""
    # Implementation will be completed when email service is fully configured
    return {"message": "Password reset functionality will be implemented"}
