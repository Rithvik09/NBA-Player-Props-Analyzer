"""
Advanced Real-Time Notification System
Supports push notifications, emails, SMS, webhooks, and real-time alerts
"""
import asyncio
import json
import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import aiohttp
import websockets
import redis
import requests
from twilio.rest import Client as TwilioClient
from pusher import Pusher
import firebase_admin
from firebase_admin import credentials, messaging
from jinja2 import Template
import threading
from queue import Queue, Empty
import time

@dataclass
class Notification:
    id: str
    user_id: str
    type: str  # 'line_movement', 'arbitrage', 'prediction', 'account'
    title: str
    message: str
    data: Dict[str, Any]
    channels: List[str]  # 'email', 'push', 'sms', 'webhook'
    priority: str  # 'low', 'medium', 'high', 'urgent'
    scheduled_for: Optional[datetime] = None
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    status: str = 'pending'  # 'pending', 'sent', 'failed', 'cancelled'

@dataclass
class NotificationPreferences:
    user_id: str
    email_enabled: bool = True
    push_enabled: bool = True
    sms_enabled: bool = False
    webhook_enabled: bool = False
    line_movements: bool = True
    arbitrage_alerts: bool = True
    prediction_results: bool = True
    account_updates: bool = True
    marketing: bool = False
    frequency_limit: int = 10  # Max notifications per hour

class NotificationManager:
    """Main notification management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.email_service = EmailService(config.get('email', {}))
        self.sms_service = SMSService(config.get('sms', {}))
        self.push_service = PushNotificationService(config.get('push', {}))
        self.webhook_service = WebhookService(config.get('webhook', {}))
        self.websocket_service = WebSocketService(config.get('websocket', {}))
        
        # Redis for caching and rate limiting
        self.redis_client = redis.Redis(
            host=config.get('redis', {}).get('host', 'localhost'),
            port=config.get('redis', {}).get('port', 6379),
            decode_responses=True
        )
        
        # Notification queue for async processing
        self.notification_queue = Queue()
        self.worker_threads = []
        self.running = False
        
        # Templates
        self.templates = self._load_templates()
        
    def start(self, num_workers: int = 3):
        """Start notification processing workers"""
        self.running = True
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        
        self.logger.info(f"Started {num_workers} notification workers")
    
    def stop(self):
        """Stop notification processing"""
        self.running = False
        
        # Add poison pills to stop workers
        for _ in self.worker_threads:
            self.notification_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join()
        
        self.logger.info("Stopped notification workers")
    
    async def send_notification(self, notification: Notification) -> bool:
        """Send notification through specified channels"""
        try:
            # Check user preferences
            preferences = await self._get_user_preferences(notification.user_id)
            
            if not self._should_send_notification(notification, preferences):
                self.logger.info(f"Notification filtered out for user {notification.user_id}")
                return False
            
            # Update notification
            notification.created_at = datetime.now()
            
            # Add to queue for async processing
            self.notification_queue.put(notification)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            return False
    
    def _worker_loop(self):
        """Worker loop for processing notifications"""
        while self.running:
            try:
                # Get notification from queue
                notification = self.notification_queue.get(timeout=1)
                
                # Check for poison pill
                if notification is None:
                    break
                
                # Process notification
                self._process_notification(notification)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
    
    def _process_notification(self, notification: Notification):
        """Process individual notification"""
        try:
            success_channels = []
            failed_channels = []
            
            for channel in notification.channels:
                try:
                    if channel == 'email' and hasattr(self, 'email_service'):
                        success = self.email_service.send_email(notification)
                    elif channel == 'sms' and hasattr(self, 'sms_service'):
                        success = self.sms_service.send_sms(notification)
                    elif channel == 'push' and hasattr(self, 'push_service'):
                        success = asyncio.run(self.push_service.send_push(notification))
                    elif channel == 'webhook' and hasattr(self, 'webhook_service'):
                        success = self.webhook_service.send_webhook(notification)
                    elif channel == 'websocket' and hasattr(self, 'websocket_service'):
                        success = asyncio.run(self.websocket_service.send_websocket(notification))
                    else:
                        success = False
                    
                    if success:
                        success_channels.append(channel)
                    else:
                        failed_channels.append(channel)
                        
                except Exception as e:
                    self.logger.error(f"Error sending to {channel}: {e}")
                    failed_channels.append(channel)
            
            # Update notification status
            if success_channels and not failed_channels:
                notification.status = 'sent'
            elif success_channels:
                notification.status = 'partial'
            else:
                notification.status = 'failed'
            
            notification.sent_at = datetime.now()
            
            # Log result
            self._log_notification_result(notification, success_channels, failed_channels)
            
        except Exception as e:
            self.logger.error(f"Error processing notification: {e}")
            notification.status = 'failed'
    
    async def _get_user_preferences(self, user_id: str) -> NotificationPreferences:
        """Get user notification preferences"""
        try:
            # Try Redis cache first
            cached = self.redis_client.get(f"notif_prefs:{user_id}")
            if cached:
                data = json.loads(cached)
                return NotificationPreferences(**data)
            
            # Fallback to database or default
            # In a real implementation, you'd query the database
            prefs = NotificationPreferences(user_id=user_id)
            
            # Cache for 1 hour
            self.redis_client.setex(
                f"notif_prefs:{user_id}",
                3600,
                json.dumps(asdict(prefs))
            )
            
            return prefs
            
        except Exception as e:
            self.logger.error(f"Error getting preferences: {e}")
            return NotificationPreferences(user_id=user_id)
    
    def _should_send_notification(self, notification: Notification, 
                                preferences: NotificationPreferences) -> bool:
        """Check if notification should be sent based on preferences"""
        
        # Check if notification type is enabled
        type_enabled = {
            'line_movement': preferences.line_movements,
            'arbitrage': preferences.arbitrage_alerts,
            'prediction': preferences.prediction_results,
            'account': preferences.account_updates
        }.get(notification.type, True)
        
        if not type_enabled:
            return False
        
        # Check rate limiting
        if self._is_rate_limited(notification.user_id, preferences.frequency_limit):
            return False
        
        # Check channel availability
        available_channels = []
        if preferences.email_enabled and 'email' in notification.channels:
            available_channels.append('email')
        if preferences.push_enabled and 'push' in notification.channels:
            available_channels.append('push')
        if preferences.sms_enabled and 'sms' in notification.channels:
            available_channels.append('sms')
        if preferences.webhook_enabled and 'webhook' in notification.channels:
            available_channels.append('webhook')
        
        # Update notification channels to only enabled ones
        notification.channels = available_channels
        
        return len(available_channels) > 0
    
    def _is_rate_limited(self, user_id: str, limit: int) -> bool:
        """Check if user has exceeded rate limit"""
        try:
            key = f"rate_limit:{user_id}"
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            rate_key = f"{key}:{current_hour}"
            
            current_count = self.redis_client.get(rate_key)
            if current_count and int(current_count) >= limit:
                return True
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(rate_key)
            pipe.expire(rate_key, 3600)  # Expire after 1 hour
            pipe.execute()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Rate limiting error: {e}")
            return False
    
    def _log_notification_result(self, notification: Notification, 
                               success_channels: List[str], failed_channels: List[str]):
        """Log notification results"""
        self.logger.info(
            f"Notification {notification.id} - "
            f"Success: {success_channels}, Failed: {failed_channels}"
        )
        
        # Store in Redis for analytics
        result_data = {
            'notification_id': notification.id,
            'user_id': notification.user_id,
            'type': notification.type,
            'success_channels': success_channels,
            'failed_channels': failed_channels,
            'timestamp': datetime.now().isoformat()
        }
        
        self.redis_client.lpush('notification_results', json.dumps(result_data))
        self.redis_client.ltrim('notification_results', 0, 9999)  # Keep last 10k results
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load notification templates"""
        templates = {}
        
        # Email templates
        templates['line_movement_email'] = Template("""
        <h2>🚨 Line Movement Alert</h2>
        <p>{{ player_name }} {{ prop_type }} line has moved {{ movement }} points!</p>
        <p><strong>New Line:</strong> {{ new_line }}</p>
        <p><strong>Sportsbook:</strong> {{ sportsbook }}</p>
        <p><strong>Time:</strong> {{ timestamp }}</p>
        """)
        
        templates['arbitrage_email'] = Template("""
        <h2>💰 Arbitrage Opportunity</h2>
        <p><strong>Player:</strong> {{ player_name }}</p>
        <p><strong>Prop:</strong> {{ prop_type }} {{ line }}</p>
        <p><strong>Profit:</strong> {{ profit_percentage }}%</p>
        <p><strong>Over:</strong> {{ over_sportsbook }} ({{ over_odds }})</p>
        <p><strong>Under:</strong> {{ under_sportsbook }} ({{ under_odds }})</p>
        """)
        
        # Push notification templates
        templates['line_movement_push'] = Template(
            "🚨 {{ player_name }} {{ prop_type }} moved {{ movement }} pts to {{ new_line }}"
        )
        
        templates['arbitrage_push'] = Template(
            "💰 {{ profit_percentage }}% arbitrage on {{ player_name }} {{ prop_type }}"
        )
        
        return templates

class EmailService:
    """Email notification service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', self.username)
        
    def send_email(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = notification.title
            msg['From'] = self.from_email
            msg['To'] = notification.data.get('email', '')
            
            # Create HTML content
            html_content = self._generate_html_content(notification)
            msg.attach(MimeText(html_content, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.from_email, [notification.data.get('email')], msg.as_string())
            server.quit()
            
            return True
            
        except Exception as e:
            logging.error(f"Email sending error: {e}")
            return False
    
    def _generate_html_content(self, notification: Notification) -> str:
        """Generate HTML email content"""
        template_name = f"{notification.type}_email"
        
        # Use template if available, otherwise use basic format
        if hasattr(self, 'templates') and template_name in self.templates:
            return self.templates[template_name].render(**notification.data)
        else:
            return f"""
            <html>
            <body>
                <h2>{notification.title}</h2>
                <p>{notification.message}</p>
                <hr>
                <small>NBA Props Analyzer Pro</small>
            </body>
            </html>
            """

class SMSService:
    """SMS notification service using Twilio"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        if config.get('twilio_sid') and config.get('twilio_token'):
            self.client = TwilioClient(config['twilio_sid'], config['twilio_token'])
            self.from_number = config.get('from_number')
        else:
            self.client = None
    
    def send_sms(self, notification: Notification) -> bool:
        """Send SMS notification"""
        if not self.client:
            return False
        
        try:
            message = self.client.messages.create(
                body=f"{notification.title}: {notification.message}",
                from_=self.from_number,
                to=notification.data.get('phone_number')
            )
            
            return message.status != 'failed'
            
        except Exception as e:
            logging.error(f"SMS sending error: {e}")
            return False

class PushNotificationService:
    """Push notification service using Firebase"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Firebase Admin SDK
        if config.get('firebase_credentials'):
            cred = credentials.Certificate(config['firebase_credentials'])
            firebase_admin.initialize_app(cred)
            self.firebase_enabled = True
        else:
            self.firebase_enabled = False
    
    async def send_push(self, notification: Notification) -> bool:
        """Send push notification"""
        if not self.firebase_enabled:
            return False
        
        try:
            # Create message
            message = messaging.Message(
                notification=messaging.Notification(
                    title=notification.title,
                    body=notification.message,
                ),
                data=notification.data,
                token=notification.data.get('fcm_token')
            )
            
            # Send message
            response = messaging.send(message)
            return bool(response)
            
        except Exception as e:
            logging.error(f"Push notification error: {e}")
            return False

class WebhookService:
    """Webhook notification service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get('timeout', 10)
    
    def send_webhook(self, notification: Notification) -> bool:
        """Send webhook notification"""
        try:
            webhook_url = notification.data.get('webhook_url')
            if not webhook_url:
                return False
            
            payload = {
                'id': notification.id,
                'type': notification.type,
                'title': notification.title,
                'message': notification.message,
                'data': notification.data,
                'timestamp': notification.created_at.isoformat() if notification.created_at else None
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logging.error(f"Webhook sending error: {e}")
            return False

class WebSocketService:
    """WebSocket real-time notification service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connections = {}  # user_id -> websocket connections
    
    async def send_websocket(self, notification: Notification) -> bool:
        """Send WebSocket notification"""
        try:
            user_connections = self.connections.get(notification.user_id, [])
            
            if not user_connections:
                return False
            
            message = json.dumps({
                'type': 'notification',
                'id': notification.id,
                'title': notification.title,
                'message': notification.message,
                'data': notification.data,
                'timestamp': notification.created_at.isoformat() if notification.created_at else None
            })
            
            # Send to all user connections
            active_connections = []
            for ws in user_connections:
                try:
                    await ws.send(message)
                    active_connections.append(ws)
                except websockets.exceptions.ConnectionClosed:
                    # Remove closed connections
                    pass
            
            # Update active connections
            if active_connections:
                self.connections[notification.user_id] = active_connections
            else:
                self.connections.pop(notification.user_id, None)
            
            return len(active_connections) > 0
            
        except Exception as e:
            logging.error(f"WebSocket sending error: {e}")
            return False
    
    def add_connection(self, user_id: str, websocket):
        """Add WebSocket connection for user"""
        if user_id not in self.connections:
            self.connections[user_id] = []
        self.connections[user_id].append(websocket)
    
    def remove_connection(self, user_id: str, websocket):
        """Remove WebSocket connection for user"""
        if user_id in self.connections:
            try:
                self.connections[user_id].remove(websocket)
                if not self.connections[user_id]:
                    del self.connections[user_id]
            except ValueError:
                pass

# High-level notification functions
class NotificationBuilder:
    """Builder class for creating notifications"""
    
    @staticmethod
    def line_movement_alert(user_id: str, player_name: str, prop_type: str, 
                          old_line: float, new_line: float, sportsbook: str) -> Notification:
        """Create line movement notification"""
        movement = abs(new_line - old_line)
        
        return Notification(
            id=f"line_mov_{user_id}_{int(time.time())}",
            user_id=user_id,
            type='line_movement',
            title=f"Line Movement: {player_name}",
            message=f"{prop_type} moved {movement} points to {new_line}",
            data={
                'player_name': player_name,
                'prop_type': prop_type,
                'old_line': old_line,
                'new_line': new_line,
                'movement': movement,
                'sportsbook': sportsbook,
                'timestamp': datetime.now().isoformat()
            },
            channels=['push', 'websocket'],
            priority='high'
        )
    
    @staticmethod
    def arbitrage_opportunity(user_id: str, player_name: str, prop_type: str,
                           line: float, profit_pct: float, over_book: str, 
                           under_book: str) -> Notification:
        """Create arbitrage opportunity notification"""
        return Notification(
            id=f"arb_{user_id}_{int(time.time())}",
            user_id=user_id,
            type='arbitrage',
            title=f"Arbitrage: {player_name}",
            message=f"{profit_pct:.1f}% profit on {prop_type}",
            data={
                'player_name': player_name,
                'prop_type': prop_type,
                'line': line,
                'profit_percentage': profit_pct,
                'over_sportsbook': over_book,
                'under_sportsbook': under_book,
                'timestamp': datetime.now().isoformat()
            },
            channels=['push', 'email', 'websocket'],
            priority='urgent'
        )

# Example usage
async def example_usage():
    """Example of how to use the notification system"""
    
    # Configuration
    config = {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'from_email': 'nba-props@yourapp.com'
        },
        'sms': {
            'twilio_sid': 'your-twilio-sid',
            'twilio_token': 'your-twilio-token',
            'from_number': '+1234567890'
        },
        'push': {
            'firebase_credentials': 'path/to/firebase-credentials.json'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379
        }
    }
    
    # Initialize notification manager
    notif_manager = NotificationManager(config)
    notif_manager.start()
    
    # Create and send notification
    notification = NotificationBuilder.line_movement_alert(
        user_id='user123',
        player_name='LeBron James',
        prop_type='Points',
        old_line=25.5,
        new_line=27.5,
        sportsbook='DraftKings'
    )
    
    await notif_manager.send_notification(notification)
    
    # Stop manager
    notif_manager.stop()

if __name__ == "__main__":
    asyncio.run(example_usage())