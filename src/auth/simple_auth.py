"""
Simplified Authentication System for Development
Basic user management without heavy dependencies
"""
import sqlite3
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

class SimpleAuth:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.secret_key = 'dev-secret-key-change-in-production'
        self.init_db()
    
    def init_db(self):
        """Initialize user database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
    
    def hash_password(self, password: str) -> str:
        """Simple password hashing"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return salt + ':' + password_hash.hex()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split(':')
            password_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_check.hex() == hash_hex
        except:
            return False
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user"""
        try:
            password_hash = self.hash_password(password)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                    (username, email, password_hash)
                )
                user_id = cursor.lastrowid
            
            return {
                'success': True,
                'user_id': user_id,
                'message': 'User registered successfully'
            }
        except sqlite3.IntegrityError:
            return {
                'success': False,
                'error': 'Username or email already exists'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Login user and create session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT id, username, password_hash FROM users WHERE email = ? AND is_active = 1',
                    (email,)
                )
                user = cursor.fetchone()
            
            if not user or not self.verify_password(password, user[2]):
                return {
                    'success': False,
                    'error': 'Invalid credentials'
                }
            
            # Create session token
            token = secrets.token_urlsafe(32)
            expires_at = datetime.now() + timedelta(hours=24)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)',
                    (token, user[0], expires_at)
                )
            
            return {
                'success': True,
                'token': token,
                'user_id': user[0],
                'username': user[1],
                'expires_at': expires_at.isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify session token"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT s.user_id, u.username, u.email, s.expires_at
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.token = ? AND s.expires_at > datetime('now')
                ''', (token,))
                result = cursor.fetchone()
            
            if result:
                return {
                    'user_id': result[0],
                    'username': result[1],
                    'email': result[2],
                    'expires_at': result[3]
                }
            return None
        except:
            return None
    
    def logout_user(self, token: str) -> bool:
        """Logout user by invalidating token"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM sessions WHERE token = ?', (token,))
            return True
        except:
            return False

# Create a default instance
simple_auth = SimpleAuth()