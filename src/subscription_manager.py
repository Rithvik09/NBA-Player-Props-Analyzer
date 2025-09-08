from datetime import datetime, timedelta
import sqlite3
import hashlib
import secrets

class SubscriptionManager:
    def __init__(self, db_name='basketball_data.db'):
        self.db_name = db_name
        self.create_tables()
        
        # Define subscription tiers and features
        self.subscription_tiers = {
            'FREE': {
                'price': 0,
                'daily_predictions': 3,
                'features': [
                    'basic_predictions',
                    'confidence_scores',
                    'basic_stats'
                ],
                'description': 'Basic NBA props analysis'
            },
            'BASIC': {
                'price': 9.99,
                'daily_predictions': -1,  # Unlimited
                'features': [
                    'basic_predictions',
                    'confidence_scores', 
                    'basic_stats',
                    'advanced_analytics',
                    'lineup_impact',
                    'injury_tracking'
                ],
                'description': 'Unlimited predictions with advanced analytics'
            },
            'PRO': {
                'price': 19.99,
                'daily_predictions': -1,
                'features': [
                    'basic_predictions',
                    'confidence_scores',
                    'basic_stats', 
                    'advanced_analytics',
                    'lineup_impact',
                    'injury_tracking',
                    'bankroll_management',
                    'kelly_criterion',
                    'parlay_optimizer',
                    'situational_analysis',
                    'real_time_alerts'
                ],
                'description': 'Professional betting tools with bankroll management'
            },
            'VIP': {
                'price': 39.99,
                'daily_predictions': -1,
                'features': [
                    'all_pro_features',
                    'expert_picks',
                    'discord_access',
                    'api_access',
                    'custom_models',
                    'priority_support',
                    'early_access',
                    'monthly_strategy_calls'
                ],
                'description': 'VIP access with expert picks and community'
            }
        }
    
    def create_tables(self):
        """Create subscription and user management tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                email TEXT,
                username TEXT,
                password_hash TEXT,
                subscription_tier TEXT DEFAULT 'FREE',
                subscription_start DATE,
                subscription_end DATE,
                payment_status TEXT DEFAULT 'ACTIVE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                api_key TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                usage_date DATE,
                predictions_used INTEGER DEFAULT 0,
                api_calls_used INTEGER DEFAULT 0,
                features_accessed TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subscription_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                tier TEXT,
                start_date DATE,
                end_date DATE,
                amount_paid REAL,
                payment_method TEXT,
                transaction_id TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                feature_name TEXT,
                access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN,
                tier_required TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, user_id, email, username, password):
        """Create new user account"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Generate API key
        api_key = secrets.token_urlsafe(32)
        
        try:
            cursor.execute('''
                INSERT INTO users (user_id, email, username, password_hash, api_key)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, email, username, password_hash, api_key))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'user_id': user_id,
                'api_key': api_key,
                'subscription_tier': 'FREE'
            }
            
        except sqlite3.IntegrityError:
            conn.close()
            return {'success': False, 'error': 'User already exists'}
    
    def authenticate_user(self, user_id, password):
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute('''
            SELECT user_id, subscription_tier, api_key, subscription_end
            FROM users 
            WHERE user_id = ? AND password_hash = ?
        ''', (user_id, password_hash))
        
        result = cursor.fetchone()
        
        if result:
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
            
            # Check subscription status
            subscription_active = True
            if result[3]:  # subscription_end
                end_date = datetime.strptime(result[3], '%Y-%m-%d').date()
                subscription_active = end_date >= datetime.now().date()
            
            conn.close()
            
            return {
                'success': True,
                'user_id': result[0],
                'subscription_tier': result[1],
                'api_key': result[2],
                'subscription_active': subscription_active
            }
        
        conn.close()
        return {'success': False, 'error': 'Invalid credentials'}
    
    def get_user_info(self, user_id):
        """Get comprehensive user information"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM users WHERE user_id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return None
        
        # Get today's usage
        today = datetime.now().date()
        cursor.execute('''
            SELECT predictions_used, api_calls_used 
            FROM daily_usage 
            WHERE user_id = ? AND usage_date = ?
        ''', (user_id, today))
        
        usage = cursor.fetchone()
        predictions_used = usage[0] if usage else 0
        api_calls_used = usage[1] if usage else 0
        
        # Get subscription info
        tier_info = self.subscription_tiers.get(user[5], self.subscription_tiers['FREE'])
        
        conn.close()
        
        return {
            'user_id': user[1],
            'email': user[2],
            'username': user[3],
            'subscription_tier': user[5],
            'subscription_start': user[6],
            'subscription_end': user[7],
            'payment_status': user[8],
            'api_key': user[10],
            'tier_info': tier_info,
            'daily_usage': {
                'predictions_used': predictions_used,
                'api_calls_used': api_calls_used,
                'predictions_remaining': max(0, tier_info['daily_predictions'] - predictions_used) if tier_info['daily_predictions'] != -1 else -1
            }
        }
    
    def check_feature_access(self, user_id, feature_name):
        """Check if user has access to specific feature"""
        user_info = self.get_user_info(user_id)
        
        if not user_info:
            return {'access': False, 'reason': 'User not found'}
        
        tier = user_info['subscription_tier']
        tier_features = self.subscription_tiers.get(tier, {}).get('features', [])
        
        # Check subscription expiry
        if user_info['subscription_end']:
            end_date = datetime.strptime(user_info['subscription_end'], '%Y-%m-%d').date()
            if end_date < datetime.now().date():
                return {'access': False, 'reason': 'Subscription expired'}
        
        # Check feature access
        if feature_name in tier_features or 'all_pro_features' in tier_features:
            # Check daily limits for free tier
            if tier == 'FREE' and feature_name == 'basic_predictions':
                daily_limit = self.subscription_tiers['FREE']['daily_predictions']
                if user_info['daily_usage']['predictions_used'] >= daily_limit:
                    return {
                        'access': False, 
                        'reason': f'Daily limit reached ({daily_limit} predictions)',
                        'upgrade_required': True
                    }
            
            return {'access': True, 'reason': 'Access granted'}
        
        # Determine required tier
        required_tier = self._get_required_tier(feature_name)
        
        return {
            'access': False,
            'reason': f'Feature requires {required_tier} subscription',
            'required_tier': required_tier,
            'upgrade_required': True
        }
    
    def log_feature_access(self, user_id, feature_name, success=True):
        """Log feature access for analytics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        required_tier = self._get_required_tier(feature_name)
        
        cursor.execute('''
            INSERT INTO feature_access_log (user_id, feature_name, success, tier_required)
            VALUES (?, ?, ?, ?)
        ''', (user_id, feature_name, success, required_tier))
        
        conn.commit()
        conn.close()
    
    def increment_usage(self, user_id, usage_type='predictions'):
        """Increment daily usage counters"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Check if usage record exists for today
        cursor.execute('''
            SELECT id FROM daily_usage 
            WHERE user_id = ? AND usage_date = ?
        ''', (user_id, today))
        
        if cursor.fetchone():
            # Update existing record
            if usage_type == 'predictions':
                cursor.execute('''
                    UPDATE daily_usage 
                    SET predictions_used = predictions_used + 1
                    WHERE user_id = ? AND usage_date = ?
                ''', (user_id, today))
            elif usage_type == 'api_calls':
                cursor.execute('''
                    UPDATE daily_usage 
                    SET api_calls_used = api_calls_used + 1
                    WHERE user_id = ? AND usage_date = ?
                ''', (user_id, today))
        else:
            # Create new record
            predictions = 1 if usage_type == 'predictions' else 0
            api_calls = 1 if usage_type == 'api_calls' else 0
            
            cursor.execute('''
                INSERT INTO daily_usage (user_id, usage_date, predictions_used, api_calls_used)
                VALUES (?, ?, ?, ?)
            ''', (user_id, today, predictions, api_calls))
        
        conn.commit()
        conn.close()
    
    def upgrade_subscription(self, user_id, new_tier, payment_info=None):
        """Upgrade user subscription"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        if new_tier not in self.subscription_tiers:
            return {'success': False, 'error': 'Invalid subscription tier'}
        
        # Calculate subscription dates
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=30)  # Monthly subscription
        
        # Update user subscription
        cursor.execute('''
            UPDATE users 
            SET subscription_tier = ?, subscription_start = ?, subscription_end = ?, payment_status = 'ACTIVE'
            WHERE user_id = ?
        ''', (new_tier, start_date, end_date, user_id))
        
        # Log subscription history
        amount_paid = self.subscription_tiers[new_tier]['price']
        cursor.execute('''
            INSERT INTO subscription_history 
            (user_id, tier, start_date, end_date, amount_paid, status)
            VALUES (?, ?, ?, ?, ?, 'ACTIVE')
        ''', (user_id, new_tier, start_date, end_date, amount_paid))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'new_tier': new_tier,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'amount_paid': amount_paid
        }
    
    def get_subscription_analytics(self):
        """Get subscription analytics for admin dashboard"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # User count by tier
        cursor.execute('''
            SELECT subscription_tier, COUNT(*) as count
            FROM users
            GROUP BY subscription_tier
        ''')
        tier_distribution = dict(cursor.fetchall())
        
        # Revenue analytics
        cursor.execute('''
            SELECT tier, SUM(amount_paid) as revenue, COUNT(*) as transactions
            FROM subscription_history
            WHERE created_at >= date('now', '-30 days')
            GROUP BY tier
        ''')
        revenue_data = cursor.fetchall()
        
        # Feature usage analytics
        cursor.execute('''
            SELECT feature_name, COUNT(*) as usage_count, 
                   COUNT(CASE WHEN success = 1 THEN 1 END) as successful_access
            FROM feature_access_log
            WHERE access_time >= datetime('now', '-7 days')
            GROUP BY feature_name
            ORDER BY usage_count DESC
        ''')
        feature_usage = cursor.fetchall()
        
        # Daily active users
        cursor.execute('''
            SELECT usage_date, COUNT(DISTINCT user_id) as active_users
            FROM daily_usage
            WHERE usage_date >= date('now', '-30 days')
            GROUP BY usage_date
            ORDER BY usage_date
        ''')
        daily_active_users = cursor.fetchall()
        
        conn.close()
        
        return {
            'tier_distribution': tier_distribution,
            'monthly_revenue': revenue_data,
            'feature_usage': feature_usage,
            'daily_active_users': daily_active_users,
            'total_users': sum(tier_distribution.values()),
            'paid_users': sum(count for tier, count in tier_distribution.items() if tier != 'FREE')
        }
    
    def _get_required_tier(self, feature_name):
        """Determine minimum tier required for feature"""
        for tier, info in self.subscription_tiers.items():
            if feature_name in info['features'] or 'all_pro_features' in info['features']:
                return tier
        return 'VIP'  # Default to highest tier if feature not found
    
    def get_upgrade_recommendations(self, user_id):
        """Get personalized upgrade recommendations"""
        user_info = self.get_user_info(user_id)
        
        if not user_info:
            return []
        
        current_tier = user_info['subscription_tier']
        recommendations = []
        
        # Analyze usage patterns
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Check feature access attempts that failed
        cursor.execute('''
            SELECT feature_name, tier_required, COUNT(*) as attempts
            FROM feature_access_log
            WHERE user_id = ? AND success = 0 AND access_time >= datetime('now', '-7 days')
            GROUP BY feature_name, tier_required
            ORDER BY attempts DESC
        ''', (user_id,))
        
        failed_attempts = cursor.fetchall()
        
        # Check daily limit hits for free users
        if current_tier == 'FREE':
            cursor.execute('''
                SELECT COUNT(*) as days_hit_limit
                FROM daily_usage
                WHERE user_id = ? AND predictions_used >= ? AND usage_date >= date('now', '-7 days')
            ''', (user_id, self.subscription_tiers['FREE']['daily_predictions']))
            
            limit_hits = cursor.fetchone()[0]
            
            if limit_hits >= 3:
                recommendations.append({
                    'type': 'daily_limit',
                    'message': f'You\'ve hit the daily limit {limit_hits} times this week',
                    'recommended_tier': 'BASIC',
                    'benefit': 'Unlimited predictions'
                })
        
        # Feature-based recommendations
        for feature, tier_required, attempts in failed_attempts:
            if attempts >= 2:  # Multiple attempts to access premium features
                recommendations.append({
                    'type': 'feature_access',
                    'message': f'You\'ve tried to access {feature} {attempts} times',
                    'recommended_tier': tier_required,
                    'benefit': f'Get access to {feature} and more'
                })
        
        conn.close()
        
        return recommendations[:3]  # Return top 3 recommendations