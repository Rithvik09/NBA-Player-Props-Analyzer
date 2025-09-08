"""
Advanced User Authentication and Management System
Supports OAuth, JWT, subscription management, and user analytics
"""
import jwt
import bcrypt
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import redis
import sqlite3
import json
import logging
import re
from functools import wraps
import asyncio
import aiohttp
from urllib.parse import urlencode, parse_qs
import base64
import hashlib
import hmac

# User data classes
@dataclass
class User:
    id: str
    email: str
    username: str
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    subscription_tier: str = 'free'
    subscription_expires: Optional[datetime] = None
    profile_data: Dict[str, Any] = None
    preferences: Dict[str, Any] = None

@dataclass
class UserSession:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

@dataclass
class SubscriptionTier:
    name: str
    price: float
    features: List[str]
    limits: Dict[str, int]
    description: str

class UserManager:
    """Manages user authentication, registration, and profiles"""
    
    def __init__(self, db_path='users.db', redis_host='localhost', redis_port=6379):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # JWT configuration
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = 'HS256'
        self.jwt_expiration = timedelta(hours=24)
        
        # Redis for session management
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
        except:
            self.logger.warning("Redis not available, using memory-based sessions")
            self.redis_client = None
        
        # In-memory session store (fallback)
        self.sessions = {}
        
        # Subscription tiers
        self.subscription_tiers = {
            'free': SubscriptionTier(
                name='Free',
                price=0.0,
                features=['3 predictions per day', 'Basic analysis'],
                limits={'predictions_per_day': 3, 'saved_picks': 5},
                description='Basic NBA props analysis'
            ),
            'pro': SubscriptionTier(
                name='Pro',
                price=19.99,
                features=['Unlimited predictions', 'Bankroll management', 'Confidence scoring'],
                limits={'predictions_per_day': -1, 'saved_picks': 50},
                description='Advanced analysis and tools'
            ),
            'elite': SubscriptionTier(
                name='Elite',
                price=39.99,
                features=['All Pro features', 'Parlay optimizer', 'Live odds', 'Real-time alerts'],
                limits={'predictions_per_day': -1, 'saved_picks': 500},
                description='Complete betting intelligence platform'
            ),
            'vip': SubscriptionTier(
                name='VIP',
                price=79.99,
                features=['All features', 'Expert picks', 'Discord access', 'Priority support'],
                limits={'predictions_per_day': -1, 'saved_picks': -1},
                description='Premium experience with expert insights'
            )
        }
        
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for user management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active INTEGER DEFAULT 1,
                is_verified INTEGER DEFAULT 0,
                subscription_tier TEXT DEFAULT 'free',
                subscription_expires TEXT,
                profile_data TEXT,
                preferences TEXT
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Password reset tokens
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Email verification tokens
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User activity logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                activity_data TEXT,
                timestamp TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Subscription history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscription_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                amount REAL,
                payment_method TEXT,
                transaction_id TEXT,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_user(self, email: str, username: str, password: str, 
                     profile_data: Dict = None) -> Tuple[bool, str, Dict]:
        """Register a new user"""
        try:
            # Validate input
            if not self._validate_email(email):
                return False, "Invalid email format", {}
            
            if not self._validate_username(username):
                return False, "Username must be 3-20 characters, alphanumeric only", {}
            
            if not self._validate_password(password):
                return False, "Password must be at least 8 characters with uppercase, lowercase, and number", {}
            
            # Check if user exists
            if self._user_exists(email, username):
                return False, "User already exists", {}
            
            # Create user
            user_id = secrets.token_urlsafe(16)
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            user = User(
                id=user_id,
                email=email.lower(),
                username=username,
                password_hash=password_hash,
                created_at=datetime.now(timezone.utc),
                profile_data=profile_data or {},
                preferences=self._get_default_preferences()
            )
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (id, email, username, password_hash, created_at, 
                                 profile_data, preferences)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.id, user.email, user.username, user.password_hash,
                user.created_at.isoformat(),
                json.dumps(user.profile_data),
                json.dumps(user.preferences)
            ))
            
            conn.commit()
            conn.close()
            
            # Send verification email
            verification_token = self._create_verification_token(user_id)
            self._send_verification_email(email, verification_token)
            
            # Log activity
            self._log_user_activity(user_id, 'registration', {'email': email})
            
            return True, "User registered successfully", {
                'user_id': user_id,
                'verification_required': True
            }
            
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False, "Registration failed", {}
    
    def authenticate_user(self, email_or_username: str, password: str, 
                         ip_address: str = '', user_agent: str = '') -> Tuple[bool, str, Dict]:
        """Authenticate user login"""
        try:
            # Get user from database
            user = self._get_user_by_email_or_username(email_or_username)
            
            if not user:
                return False, "Invalid credentials", {}
            
            # Check password
            if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                self._log_user_activity(user.id, 'failed_login', {'reason': 'invalid_password'})
                return False, "Invalid credentials", {}
            
            # Check if user is active
            if not user.is_active:
                return False, "Account is deactivated", {}
            
            # Update last login
            self._update_last_login(user.id)
            
            # Create session
            session = self._create_session(user.id, ip_address, user_agent)
            
            # Generate JWT token
            jwt_token = self._generate_jwt_token(user.id, session.session_id)
            
            # Log successful login
            self._log_user_activity(user.id, 'login', {'ip': ip_address})
            
            return True, "Login successful", {
                'user_id': user.id,
                'session_id': session.session_id,
                'jwt_token': jwt_token,
                'subscription_tier': user.subscription_tier,
                'is_verified': user.is_verified
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False, "Authentication failed", {}
    
    def verify_jwt_token(self, token: str) -> Tuple[bool, Dict]:
        """Verify JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            user_id = payload.get('user_id')
            session_id = payload.get('session_id')
            
            # Verify session is still active
            if not self._is_session_active(session_id):
                return False, {}
            
            # Get user data
            user = self._get_user_by_id(user_id)
            if not user or not user.is_active:
                return False, {}
            
            return True, {
                'user_id': user_id,
                'session_id': session_id,
                'subscription_tier': user.subscription_tier,
                'is_verified': user.is_verified
            }
            
        except jwt.ExpiredSignatureError:
            return False, {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return False, {'error': 'Invalid token'}
        except Exception as e:
            self.logger.error(f"Token verification error: {e}")
            return False, {}
    
    def logout_user(self, session_id: str) -> bool:
        """Logout user by invalidating session"""
        try:
            # Deactivate session
            if self.redis_client:
                self.redis_client.delete(f"session:{session_id}")
            else:
                if session_id in self.sessions:
                    del self.sessions[session_id]
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions SET is_active = 0 WHERE session_id = ?
            ''', (session_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Logout error: {e}")
            return False
    
    def update_subscription(self, user_id: str, tier: str, expires_at: datetime = None) -> bool:
        """Update user subscription"""
        try:
            if tier not in self.subscription_tiers:
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update user subscription
            cursor.execute('''
                UPDATE users 
                SET subscription_tier = ?, subscription_expires = ?
                WHERE id = ?
            ''', (tier, expires_at.isoformat() if expires_at else None, user_id))
            
            # Add to subscription history
            cursor.execute('''
                INSERT INTO subscription_history (user_id, tier, start_date, end_date, amount)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id, tier, datetime.now().isoformat(),
                expires_at.isoformat() if expires_at else None,
                self.subscription_tiers[tier].price
            ))
            
            conn.commit()
            conn.close()
            
            # Log activity
            self._log_user_activity(user_id, 'subscription_update', {'tier': tier})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Subscription update error: {e}")
            return False
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_username(self, username: str) -> bool:
        """Validate username"""
        return 3 <= len(username) <= 20 and username.isalnum()
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_upper and has_lower and has_digit
    
    def _user_exists(self, email: str, username: str) -> bool:
        """Check if user already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id FROM users WHERE email = ? OR username = ?
        ''', (email.lower(), username))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def _get_user_by_email_or_username(self, identifier: str) -> Optional[User]:
        """Get user by email or username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM users WHERE email = ? OR username = ?
        ''', (identifier.lower(), identifier))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                id=row[0],
                email=row[1],
                username=row[2],
                password_hash=row[3],
                created_at=datetime.fromisoformat(row[4]),
                last_login=datetime.fromisoformat(row[5]) if row[5] else None,
                is_active=bool(row[6]),
                is_verified=bool(row[7]),
                subscription_tier=row[8],
                subscription_expires=datetime.fromisoformat(row[9]) if row[9] else None,
                profile_data=json.loads(row[10]) if row[10] else {},
                preferences=json.loads(row[11]) if row[11] else {}
            )
        
        return None
    
    def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                id=row[0],
                email=row[1],
                username=row[2],
                password_hash=row[3],
                created_at=datetime.fromisoformat(row[4]),
                last_login=datetime.fromisoformat(row[5]) if row[5] else None,
                is_active=bool(row[6]),
                is_verified=bool(row[7]),
                subscription_tier=row[8],
                subscription_expires=datetime.fromisoformat(row[9]) if row[9] else None,
                profile_data=json.loads(row[10]) if row[10] else {},
                preferences=json.loads(row[11]) if row[11] else {}
            )
        
        return None
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> UserSession:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=7)  # Session expires in 7 days
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store in Redis or memory
        session_data = asdict(session)
        session_data['created_at'] = session_data['created_at'].isoformat()
        session_data['expires_at'] = session_data['expires_at'].isoformat()
        
        if self.redis_client:
            self.redis_client.setex(
                f"session:{session_id}",
                int(timedelta(days=7).total_seconds()),
                json.dumps(session_data)
            )
        else:
            self.sessions[session_id] = session_data
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (session_id, user_id, created_at, expires_at, 
                                ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session.session_id, session.user_id,
            session.created_at.isoformat(), session.expires_at.isoformat(),
            session.ip_address, session.user_agent
        ))
        
        conn.commit()
        conn.close()
        
        return session
    
    def _generate_jwt_token(self, user_id: str, session_id: str) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'session_id': session_id,
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + self.jwt_expiration
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _is_session_active(self, session_id: str) -> bool:
        """Check if session is active"""
        try:
            if self.redis_client:
                session_data = self.redis_client.get(f"session:{session_id}")
                return session_data is not None
            else:
                return session_id in self.sessions
        except:
            return False
    
    def _update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now(timezone.utc).isoformat(), user_id))
        
        conn.commit()
        conn.close()
    
    def _log_user_activity(self, user_id: str, activity_type: str, 
                          activity_data: Dict = None, ip_address: str = '', 
                          user_agent: str = ''):
        """Log user activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_activity_logs (user_id, activity_type, activity_data,
                                          timestamp, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id, activity_type, json.dumps(activity_data or {}),
            datetime.now(timezone.utc).isoformat(), ip_address, user_agent
        ))
        
        conn.commit()
        conn.close()
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            'notifications': {
                'email': True,
                'push': False,
                'line_movements': True,
                'arbitrage_alerts': False
            },
            'display': {
                'theme': 'dark',
                'odds_format': 'american',
                'timezone': 'UTC'
            },
            'betting': {
                'default_unit_size': 10.0,
                'risk_tolerance': 'medium',
                'auto_bankroll_management': False
            }
        }
    
    def _create_verification_token(self, user_id: str) -> str:
        """Create email verification token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_verification_tokens (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (token, user_id, datetime.now(timezone.utc).isoformat(), expires_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        return token
    
    def _send_verification_email(self, email: str, token: str):
        """Send email verification (placeholder implementation)"""
        # In a real implementation, you would send an actual email
        self.logger.info(f"Verification email would be sent to {email} with token {token}")
        # verification_url = f"https://your-domain.com/verify-email?token={token}"

class OAuth2Manager:
    """Manages OAuth2 authentication with external providers"""
    
    def __init__(self):
        self.providers = {
            'google': {
                'client_id': 'your-google-client-id',
                'client_secret': 'your-google-client-secret',
                'auth_url': 'https://accounts.google.com/o/oauth2/auth',
                'token_url': 'https://oauth2.googleapis.com/token',
                'user_info_url': 'https://www.googleapis.com/oauth2/v2/userinfo',
                'scope': 'openid email profile'
            },
            'discord': {
                'client_id': 'your-discord-client-id',
                'client_secret': 'your-discord-client-secret',
                'auth_url': 'https://discord.com/api/oauth2/authorize',
                'token_url': 'https://discord.com/api/oauth2/token',
                'user_info_url': 'https://discord.com/api/users/@me',
                'scope': 'identify email'
            }
        }
    
    def get_auth_url(self, provider: str, redirect_uri: str, state: str = None) -> str:
        """Generate OAuth2 authorization URL"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        config = self.providers[provider]
        
        params = {
            'client_id': config['client_id'],
            'redirect_uri': redirect_uri,
            'scope': config['scope'],
            'response_type': 'code',
            'access_type': 'offline',
            'state': state or secrets.token_urlsafe(16)
        }
        
        return f"{config['auth_url']}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, provider: str, code: str, 
                                    redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        config = self.providers[provider]
        
        data = {
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': redirect_uri
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config['token_url'], data=data) as response:
                return await response.json()
    
    async def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Get user information from OAuth provider"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        config = self.providers[provider]
        
        headers = {'Authorization': f'Bearer {access_token}'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(config['user_info_url'], headers=headers) as response:
                return await response.json()

# Decorator for authentication required
def auth_required(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # This would be implemented in your Flask app
        # Check for JWT token in request headers
        pass
    return decorated_function

# Example integration with Flask
def create_auth_routes(app, user_manager: UserManager):
    """Create Flask routes for authentication"""
    from flask import request, jsonify
    
    @app.route('/auth/register', methods=['POST'])
    def register():
        data = request.get_json()
        success, message, result = user_manager.register_user(
            data.get('email'),
            data.get('username'),
            data.get('password'),
            data.get('profile_data')
        )
        
        return jsonify({
            'success': success,
            'message': message,
            'data': result
        }), 200 if success else 400
    
    @app.route('/auth/login', methods=['POST'])
    def login():
        data = request.get_json()
        success, message, result = user_manager.authenticate_user(
            data.get('email_or_username'),
            data.get('password'),
            request.remote_addr,
            request.headers.get('User-Agent', '')
        )
        
        return jsonify({
            'success': success,
            'message': message,
            'data': result
        }), 200 if success else 401
    
    @app.route('/auth/logout', methods=['POST'])
    def logout():
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        is_valid, token_data = user_manager.verify_jwt_token(token)
        
        if is_valid:
            success = user_manager.logout_user(token_data.get('session_id'))
            return jsonify({'success': success})
        else:
            return jsonify({'success': False, 'message': 'Invalid token'}), 401