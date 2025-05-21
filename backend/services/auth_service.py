from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
import logging
from datetime import datetime, timedelta
from typing import Optional
import uuid
import secrets
import string

from backend.models.user import User, APIKey, UserCreate, UserUpdate, SubscriptionTier
from backend.core.security import get_password_hash, verify_password, create_access_token, encrypt_api_key, decrypt_api_key

# Configure logging
logger = logging.getLogger(__name__)

class AuthService:
    """Service for authentication and user management"""
    
    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> User:
        """Create a new user"""
        # Validate passwords match
        if user_data.password != user_data.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Passwords do not match"
            )
        
        # Check if email already exists
        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )
        
        # Check if username already exists
        existing_username = db.query(User).filter(User.username == user_data.username).first()
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already taken"
            )
        
        # Hash the password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user object
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            subscription_tier=SubscriptionTier.BASIC  # Default to basic tier
        )
        
        try:
            # Add to database
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            logger.info(f"User created: {db_user.username} (ID: {db_user.id})")
            return db_user
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating user"
            )
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        # Find user by username
        user = db.query(User).filter(User.username == username).first()
        if not user:
            logger.warning(f"Authentication failed: User {username} not found")
            return None
        
        # Verify password
        if not verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: Invalid password for user {username}")
            return None
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Authentication failed: User {username} is not active")
            return None
        
        logger.info(f"User authenticated: {user.username} (ID: {user.id})")
        return user
    
    @staticmethod
    def create_user_token(user: User, expires_delta: Optional[timedelta] = None) -> dict:
        """Generate JWT token for authenticated user"""
        token_data = {
            "user_id": user.id,
            "username": user.username,
            "subscription_tier": user.subscription_tier
        }
        
        # Create access token
        access_token = create_access_token(token_data, expires_delta)
        
        # Calculate expiry time for client
        expires_minutes = expires_delta.total_seconds() / 60 if expires_delta else 30
        expires_at = datetime.utcnow() + timedelta(minutes=expires_minutes)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_at": expires_at
        }
    
    @staticmethod
    def update_user(db: Session, user_id: int, user_data: UserUpdate) -> User:
        """Update user information"""
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields if provided
        if user_data.email is not None:
            # Check if email already exists for another user
            existing_email = db.query(User).filter(User.email == user_data.email, User.id != user_id).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered"
                )
            user.email = user_data.email
        
        if user_data.username is not None:
            # Check if username already exists for another user
            existing_username = db.query(User).filter(User.username == user_data.username, User.id != user_id).first()
            if existing_username:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already taken"
                )
            user.username = user_data.username
        
        if user_data.password is not None:
            user.hashed_password = get_password_hash(user_data.password)
        
        if user_data.is_active is not None:
            user.is_active = user_data.is_active
        
        if user_data.subscription_tier is not None:
            user.subscription_tier = user_data.subscription_tier
        
        try:
            db.commit()
            db.refresh(user)
            logger.info(f"User updated: {user.username} (ID: {user.id})")
            return user
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Error updating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating user"
            )
    
    @staticmethod
    def generate_verification_token() -> str:
        """Generate a random verification token"""
        return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
    
    @staticmethod
    def add_broker_api_key(db: Session, user_id: int, provider: str, api_key: str, api_secret: str) -> APIKey:
        """Add a broker API key for a user"""
        # Verify user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if user already has an API key for this provider
        existing_key = db.query(APIKey).filter(
            APIKey.user_id == user_id, 
            APIKey.provider == provider
        ).first()
        
        # If exists, update it
        if existing_key:
            existing_key.encrypted_api_key = encrypt_api_key(api_key)
            existing_key.encrypted_api_secret = encrypt_api_key(api_secret)
            existing_key.is_active = True
            db_api_key = existing_key
        else:
            # Create new API key
            db_api_key = APIKey(
                user_id=user_id,
                provider=provider,
                encrypted_api_key=encrypt_api_key(api_key),
                encrypted_api_secret=encrypt_api_key(api_secret)
            )
            db.add(db_api_key)
        
        try:
            db.commit()
            db.refresh(db_api_key)
            logger.info(f"API key added for user ID {user_id} and provider {provider}")
            return db_api_key
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Error adding API key: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error adding API key"
            )
    
    @staticmethod
    def get_broker_api_keys(db: Session, user_id: int):
        """Get all broker API keys for a user"""
        return db.query(APIKey).filter(APIKey.user_id == user_id).all()
    
    @staticmethod
    def get_broker_api_key(db: Session, user_id: int, provider: str) -> Optional[tuple]:
        """Get a specific broker API key and secret for a user"""
        api_key = db.query(APIKey).filter(
            APIKey.user_id == user_id,
            APIKey.provider == provider,
            APIKey.is_active == True
        ).first()
        
        if not api_key:
            return None
        
        # Decrypt the keys
        decrypted_key = decrypt_api_key(api_key.encrypted_api_key)
        decrypted_secret = decrypt_api_key(api_key.encrypted_api_secret)
        
        return (decrypted_key, decrypted_secret)
    
    @staticmethod
    def delete_broker_api_key(db: Session, user_id: int, key_id: int) -> bool:
        """Delete a broker API key"""
        api_key = db.query(APIKey).filter(
            APIKey.id == key_id,
            APIKey.user_id == user_id
        ).first()
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        try:
            db.delete(api_key)
            db.commit()
            logger.info(f"API key deleted: ID {key_id} for user ID {user_id}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting API key: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting API key"
            )
