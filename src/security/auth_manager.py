"""
Enterprise Authentication and Authorization Manager for CrisisMapper.

This module provides comprehensive authentication, authorization,
and user management capabilities for enterprise deployment.
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class UserRole(Enum):
    """User roles in the system."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    VIEWER = "viewer"
    EMERGENCY_RESPONDER = "emergency_responder"

class Permission(Enum):
    """System permissions."""
    # Detection permissions
    RUN_DETECTION = "run_detection"
    BATCH_DETECTION = "batch_detection"
    REAL_TIME_DETECTION = "real_time_detection"
    
    # Data permissions
    UPLOAD_DATA = "upload_data"
    DOWNLOAD_DATA = "download_data"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"
    
    # Model permissions
    TRAIN_MODEL = "train_model"
    DEPLOY_MODEL = "deploy_model"
    MANAGE_MODELS = "manage_models"
    
    # System permissions
    MANAGE_USERS = "manage_users"
    VIEW_LOGS = "view_logs"
    SYSTEM_CONFIG = "system_config"
    EMERGENCY_ACCESS = "emergency_access"
    
    # Research permissions
    CREATE_EXPERIMENTS = "create_experiments"
    VIEW_EXPERIMENTS = "view_experiments"
    MANAGE_EXPERIMENTS = "manage_experiments"

@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[UserRole]
    permissions: Set[Permission]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class Session:
    """User session data."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool

class AuthManager:
    """
    Comprehensive authentication and authorization manager.
    
    Provides user management, role-based access control, session management,
    and security features for enterprise deployment.
    """
    
    def __init__(self, 
                 secret_key: str,
                 token_expiry_hours: int = 24,
                 max_login_attempts: int = 5,
                 lockout_duration_minutes: int = 30,
                 users_db_path: str = "data/users.json"):
        """
        Initialize the AuthManager.
        
        Args:
            secret_key: Secret key for JWT tokens
            token_expiry_hours: Token expiry time in hours
            max_login_attempts: Maximum login attempts before lockout
            lockout_duration_minutes: Lockout duration in minutes
            users_db_path: Path to users database file
        """
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.max_login_attempts = max_login_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        self.users_db_path = Path(users_db_path)
        
        # User and session storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.login_attempts: Dict[str, List[datetime]] = {}
        
        # Role-permission mapping
        self.role_permissions = self._initialize_role_permissions()
        
        # Load users from database
        self._load_users()
        
        logger.info("AuthManager initialized")
    
    def create_user(self, 
                   username: str,
                   email: str,
                   password: str,
                   roles: List[UserRole],
                   created_by: str = "system") -> str:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            roles: List of user roles
            created_by: User who created this user
            
        Returns:
            User ID
        """
        # Validate input
        if self._user_exists(username):
            raise ValueError(f"User {username} already exists")
        
        if self._email_exists(email):
            raise ValueError(f"Email {email} already exists")
        
        # Generate user ID
        user_id = secrets.token_urlsafe(16)
        
        # Hash password
        password_hash = self._hash_password(password)
        
        # Determine permissions based on roles
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        # Create user
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles,
            permissions=permissions,
            is_active=True,
            created_at=datetime.now(),
            last_login=None,
            metadata={"created_by": created_by}
        )
        
        # Store user
        self.users[user_id] = user
        self._save_users()
        
        logger.info(f"Created user: {username} (ID: {user_id})")
        return user_id
    
    def authenticate_user(self, 
                         username: str, 
                         password: str,
                         ip_address: str = "",
                         user_agent: str = "") -> Optional[str]:
        """
        Authenticate a user.
        
        Args:
            username: Username or email
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID if successful, None otherwise
        """
        # Check for lockout
        if self._is_user_locked_out(username):
            logger.warning(f"Login attempt for locked user: {username}")
            return None
        
        # Find user
        user = self._find_user_by_username_or_email(username)
        if not user:
            self._record_failed_login(username)
            return None
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive user: {username}")
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            self._record_failed_login(username)
            return None
        
        # Clear failed login attempts
        self._clear_failed_logins(username)
        
        # Update last login
        user.last_login = datetime.now()
        
        # Create session
        session_id = self._create_session(user.user_id, ip_address, user_agent)
        
        logger.info(f"User authenticated: {username} (Session: {session_id})")
        return session_id
    
    def logout_user(self, session_id: str) -> bool:
        """
        Logout a user by invalidating their session.
        
        Args:
            session_id: Session ID to invalidate
            
        Returns:
            True if successful
        """
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            logger.info(f"User logged out: {session_id}")
            return True
        
        return False
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """
        Validate a session and return the associated user.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            User if session is valid, None otherwise
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is active and not expired
        if not session.is_active or datetime.now() > session.expires_at:
            return None
        
        # Return user
        return self.users.get(session.user_id)
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user: User object
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        return permission in user.permissions
    
    def check_role(self, user: User, role: UserRole) -> bool:
        """
        Check if a user has a specific role.
        
        Args:
            user: User object
            role: Role to check
            
        Returns:
            True if user has role
        """
        return role in user.roles
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all permissions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of permissions
        """
        user = self.users.get(user_id)
        if not user:
            return set()
        
        return user.permissions
    
    def update_user_roles(self, 
                         user_id: str, 
                         roles: List[UserRole],
                         updated_by: str) -> bool:
        """
        Update user roles.
        
        Args:
            user_id: User ID
            roles: New roles
            updated_by: User making the change
            
        Returns:
            True if successful
        """
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        user.roles = roles
        
        # Update permissions based on new roles
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        user.permissions = permissions
        user.metadata["last_role_update"] = datetime.now().isoformat()
        user.metadata["updated_by"] = updated_by
        
        self._save_users()
        
        logger.info(f"Updated roles for user {user_id}: {[r.value for r in roles]}")
        return True
    
    def deactivate_user(self, user_id: str, deactivated_by: str) -> bool:
        """
        Deactivate a user.
        
        Args:
            user_id: User ID
            deactivated_by: User making the change
            
        Returns:
            True if successful
        """
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        user.is_active = False
        user.metadata["deactivated_at"] = datetime.now().isoformat()
        user.metadata["deactivated_by"] = deactivated_by
        
        # Invalidate all sessions for this user
        for session in self.sessions.values():
            if session.user_id == user_id:
                session.is_active = False
        
        self._save_users()
        
        logger.info(f"Deactivated user: {user_id}")
        return True
    
    def change_password(self, 
                       user_id: str, 
                       old_password: str, 
                       new_password: str) -> bool:
        """
        Change user password.
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            
        Returns:
            True if successful
        """
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            return False
        
        # Update password
        user.password_hash = self._hash_password(new_password)
        user.metadata["password_changed_at"] = datetime.now().isoformat()
        
        self._save_users()
        
        logger.info(f"Password changed for user: {user_id}")
        return True
    
    def generate_jwt_token(self, user: User) -> str:
        """
        Generate JWT token for a user.
        
        Args:
            user: User object
            
        Returns:
            JWT token
        """
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token and return payload.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Session]:
        """
        Get active sessions.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of active sessions
        """
        active_sessions = [
            session for session in self.sessions.values()
            if session.is_active and datetime.now() <= session.expires_at
        ]
        
        if user_id:
            active_sessions = [
                session for session in active_sessions
                if session.user_id == user_id
            ]
        
        return active_sessions
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Initialize role-permission mappings."""
        return {
            UserRole.ADMIN: {
                Permission.RUN_DETECTION,
                Permission.BATCH_DETECTION,
                Permission.REAL_TIME_DETECTION,
                Permission.UPLOAD_DATA,
                Permission.DOWNLOAD_DATA,
                Permission.DELETE_DATA,
                Permission.EXPORT_DATA,
                Permission.TRAIN_MODEL,
                Permission.DEPLOY_MODEL,
                Permission.MANAGE_MODELS,
                Permission.MANAGE_USERS,
                Permission.VIEW_LOGS,
                Permission.SYSTEM_CONFIG,
                Permission.EMERGENCY_ACCESS,
                Permission.CREATE_EXPERIMENTS,
                Permission.VIEW_EXPERIMENTS,
                Permission.MANAGE_EXPERIMENTS
            },
            UserRole.RESEARCHER: {
                Permission.RUN_DETECTION,
                Permission.BATCH_DETECTION,
                Permission.UPLOAD_DATA,
                Permission.DOWNLOAD_DATA,
                Permission.EXPORT_DATA,
                Permission.TRAIN_MODEL,
                Permission.MANAGE_MODELS,
                Permission.CREATE_EXPERIMENTS,
                Permission.VIEW_EXPERIMENTS,
                Permission.MANAGE_EXPERIMENTS
            },
            UserRole.ANALYST: {
                Permission.RUN_DETECTION,
                Permission.BATCH_DETECTION,
                Permission.UPLOAD_DATA,
                Permission.DOWNLOAD_DATA,
                Permission.EXPORT_DATA,
                Permission.VIEW_EXPERIMENTS
            },
            UserRole.VIEWER: {
                Permission.DOWNLOAD_DATA,
                Permission.VIEW_EXPERIMENTS
            },
            UserRole.EMERGENCY_RESPONDER: {
                Permission.RUN_DETECTION,
                Permission.REAL_TIME_DETECTION,
                Permission.DOWNLOAD_DATA,
                Permission.EXPORT_DATA,
                Permission.EMERGENCY_ACCESS
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, hash_part = password_hash.split(":", 1)
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == hash_part
        except ValueError:
            return False
    
    def _user_exists(self, username: str) -> bool:
        """Check if a user exists by username."""
        return any(user.username == username for user in self.users.values())
    
    def _email_exists(self, email: str) -> bool:
        """Check if an email exists."""
        return any(user.email == email for user in self.users.values())
    
    def _find_user_by_username_or_email(self, identifier: str) -> Optional[User]:
        """Find user by username or email."""
        for user in self.users.values():
            if user.username == identifier or user.email == identifier:
                return user
        return None
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new session for a user."""
        session_id = secrets.token_urlsafe(32)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.token_expiry_hours),
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def _record_failed_login(self, username: str):
        """Record a failed login attempt."""
        if username not in self.login_attempts:
            self.login_attempts[username] = []
        
        self.login_attempts[username].append(datetime.now())
        
        # Clean up old attempts
        cutoff = datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
        self.login_attempts[username] = [
            attempt for attempt in self.login_attempts[username]
            if attempt > cutoff
        ]
    
    def _is_user_locked_out(self, username: str) -> bool:
        """Check if a user is locked out due to too many failed attempts."""
        if username not in self.login_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.login_attempts[username]
            if attempt > datetime.now() - timedelta(minutes=self.lockout_duration_minutes)
        ]
        
        return len(recent_attempts) >= self.max_login_attempts
    
    def _clear_failed_logins(self, username: str):
        """Clear failed login attempts for a user."""
        if username in self.login_attempts:
            del self.login_attempts[username]
    
    def _load_users(self):
        """Load users from database file."""
        if not self.users_db_path.exists():
            # Create default admin user
            self._create_default_admin()
            return
        
        try:
            with open(self.users_db_path, 'r') as f:
                data = json.load(f)
            
            for user_data in data.get('users', []):
                user = User(
                    user_id=user_data['user_id'],
                    username=user_data['username'],
                    email=user_data['email'],
                    password_hash=user_data['password_hash'],
                    roles=[UserRole(role) for role in user_data['roles']],
                    permissions={Permission(perm) for perm in user_data['permissions']},
                    is_active=user_data['is_active'],
                    created_at=datetime.fromisoformat(user_data['created_at']),
                    last_login=datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None,
                    metadata=user_data.get('metadata', {})
                )
                self.users[user.user_id] = user
            
            logger.info(f"Loaded {len(self.users)} users from database")
            
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            self._create_default_admin()
    
    def _save_users(self):
        """Save users to database file."""
        self.users_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'users': [
                {
                    'user_id': user.user_id,
                    'username': user.username,
                    'email': user.email,
                    'password_hash': user.password_hash,
                    'roles': [role.value for role in user.roles],
                    'permissions': [perm.value for perm in user.permissions],
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None,
                    'metadata': user.metadata
                }
                for user in self.users.values()
            ]
        }
        
        with open(self.users_db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_id = self.create_user(
            username="admin",
            email="admin@crisismapper.com",
            password="admin123",  # Should be changed in production
            roles=[UserRole.ADMIN]
        )
        
        logger.info("Created default admin user (username: admin, password: admin123)")
