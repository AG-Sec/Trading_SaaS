from datetime import datetime, timedelta
from typing import Optional, Union
import os
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import logging
from dotenv import load_dotenv
from cryptography.fernet import Fernet

from backend.core.database import get_db
from backend.models.user import User, TokenData, SubscriptionTier

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Security constants
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    # For development only - in production, always use a secure environment variable
    SECRET_KEY = "DEVELOPMENT_SECRET_KEY_CHANGE_ME_IN_PRODUCTION"
    logger.warning("Using development SECRET_KEY. Set a secure SECRET_KEY in environment variables for production.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Encryption key for broker API keys
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")
if not ENCRYPTION_KEY:
    # Generate a key for development - in production, use a secure environment variable
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    logger.warning("Using generated ENCRYPTION_KEY. Set a secure ENCRYPTION_KEY in environment variables for production.")

# Initialize Fernet cipher for API key encryption
cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer token for FastAPI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a password hash"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_token_expiry(token: str) -> datetime:
    """Get the expiry date from a JWT token"""
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    exp = payload.get("exp")
    return datetime.fromtimestamp(exp)


# Create a modified oauth2 scheme that's optional for development mode
class OptionalOAuth2PasswordBearer(OAuth2PasswordBearer):
    async def __call__(self, request: Request = None) -> Optional[str]:
        try:
            return await super().__call__(request)
        except HTTPException:
            if os.getenv("ENV", "development") == "development":
                return None
            raise

# Use the optional OAuth2 scheme for development
optional_oauth2_scheme = OptionalOAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

async def get_current_user(token: Optional[str] = Depends(optional_oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Dependency to get the current authenticated user"""
    # Development mode bypass for easy testing
    USE_DEV_MODE = True
    
    if (USE_DEV_MODE and os.getenv("ENV", "development") == "development") or token is None:
        logger.debug("Using development mode authentication bypass")
        # Create a mock user for development
        user = User(
            id=1,
            username="dev_user",
            email="dev@example.com",
            hashed_password="dev_password_hash",  # Not used but required by model
            subscription_tier=SubscriptionTier.ENTERPRISE,  # Give full access in dev mode
            is_active=True,
            is_verified=True
        )
        return user
    
    # Normal authentication flow
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        
        token_data = TokenData(
            user_id=user_id,
            username=payload.get("username"),
            subscription_tier=payload.get("subscription_tier"),
            exp=datetime.fromtimestamp(payload.get("exp"))
        )
    except JWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise credentials_exception
    
    # Get the user from the database
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        logger.warning(f"User with ID {token_data.user_id} not found in database")
        raise credentials_exception
    
    if not user.is_active:
        logger.warning(f"User {user.username} (ID: {user.id}) is not active")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    return user


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key or secret"""
    return cipher_suite.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key or secret"""
    return cipher_suite.decrypt(encrypted_key.encode()).decode()
