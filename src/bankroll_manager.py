"""
Advanced Bankroll Management System for NBA Props
Implements Kelly Criterion, unit sizing, profit tracking, and risk management
"""
import json
import numpy as np
from datetime import datetime, timedelta
import sqlite3

class BankrollManager:
    def __init__(self, db_name='basketball_data.db'):
        self.db_name = db_name
        self.create_bankroll_tables()
        
    def create_bankroll_tables(self):
        """Create tables for bankroll management"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Bankroll settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bankroll_settings (
                id INTEGER PRIMARY KEY,
                user_id TEXT DEFAULT 'default',
                total_bankroll REAL,
                unit_size REAL,
                max_bet_percentage REAL DEFAULT 0.05,
                kelly_multiplier REAL DEFAULT 0.25,
                risk_tolerance TEXT DEFAULT 'Medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bet tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bet_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default',
                player_name TEXT,
                prop_type TEXT,
                line REAL,
                bet_side TEXT,
                confidence_score INTEGER,
                recommended_units REAL,
                actual_units REAL,
                odds REAL,
                bet_amount REAL,
                result TEXT,
                profit_loss REAL,
                bet_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                game_date TEXT,
                notes TEXT
            )
        ''')
        
        # Performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default',
                period_start DATE,
                period_end DATE,
                total_bets INTEGER,
                winning_bets INTEGER,
                total_profit REAL,
                total_wagered REAL,
                roi REAL,
                avg_confidence INTEGER,
                sharpe_ratio REAL,
                max_drawdown REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def initialize_bankroll(self, bankroll_amount, unit_size=None, risk_tolerance='Medium'):
        """Initialize or update bankroll settings"""
        if unit_size is None:
            unit_size = bankroll_amount * 0.01  # Default 1% units
            
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Check if settings exist
        cursor.execute('SELECT id FROM bankroll_settings WHERE user_id = ?', ('default',))
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute('''
                UPDATE bankroll_settings 
                SET total_bankroll = ?, unit_size = ?, risk_tolerance = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (bankroll_amount, unit_size, risk_tolerance, 'default'))
        else:
            cursor.execute('''
                INSERT INTO bankroll_settings (total_bankroll, unit_size, risk_tolerance)
                VALUES (?, ?, ?)
            ''', (bankroll_amount, unit_size, risk_tolerance))
            
        conn.commit()
        conn.close()
        
        return {
            'bankroll': bankroll_amount,
            'unit_size': unit_size,
            'risk_tolerance': risk_tolerance,
            'units_available': bankroll_amount / unit_size if unit_size > 0 else 0
        }
    
    def get_bankroll_info(self):
        """Get current bankroll information"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bankroll_settings WHERE user_id = ? ORDER BY id DESC LIMIT 1', ('default',))
        result = cursor.fetchone()
        
        if not result:
            # Initialize default bankroll
            conn.close()
            return self.initialize_bankroll(1000, 10, 'Medium')
            
        columns = ['id', 'user_id', 'total_bankroll', 'unit_size', 'max_bet_percentage', 
                  'kelly_multiplier', 'risk_tolerance', 'created_at', 'updated_at']
        bankroll_info = dict(zip(columns, result))
        
        # Get recent performance
        recent_performance = self.get_recent_performance()
        bankroll_info.update(recent_performance)
        
        conn.close()
        return bankroll_info
    
    def calculate_optimal_bet_size(self, confidence_score, edge, odds=-110):
        """Calculate optimal bet size using modified Kelly Criterion"""
        try:
            # Get bankroll settings
            bankroll_info = self.get_bankroll_info()
            
            # Convert American odds to decimal
            if odds < 0:
                decimal_odds = (100 / abs(odds)) + 1
            else:
                decimal_odds = (odds / 100) + 1
            
            # Calculate win probability from confidence
            win_prob = confidence_score / 100.0
            
            # Kelly Criterion: f = (bp - q) / b
            # Where b = odds-1, p = win probability, q = 1-p
            b = decimal_odds - 1
            q = 1 - win_prob
            
            # Basic Kelly fraction
            if b > 0 and win_prob > (1 / decimal_odds):
                kelly_fraction = (b * win_prob - q) / b
            else:
                kelly_fraction = 0
            
            # Apply Kelly multiplier for safety (fractional Kelly)
            kelly_multiplier = bankroll_info.get('kelly_multiplier', 0.25)
            adjusted_kelly = kelly_fraction * kelly_multiplier
            
            # Apply confidence and edge adjustments
            confidence_adjustment = self._get_confidence_adjustment(confidence_score)
            edge_adjustment = self._get_edge_adjustment(edge)
            
            final_fraction = adjusted_kelly * confidence_adjustment * edge_adjustment
            
            # Apply maximum bet limits
            max_bet_pct = bankroll_info.get('max_bet_percentage', 0.05)
            final_fraction = min(final_fraction, max_bet_pct)
            
            # Convert to units
            unit_size = bankroll_info.get('unit_size', 10)
            bankroll = bankroll_info.get('total_bankroll', 1000)
            
            recommended_amount = bankroll * final_fraction
            recommended_units = recommended_amount / unit_size if unit_size > 0 else 0
            
            # Risk assessment
            risk_level = self._assess_bet_risk(recommended_units, confidence_score, edge)
            
            return {
                'recommended_units': round(recommended_units, 2),
                'recommended_amount': round(recommended_amount, 2),
                'kelly_fraction': round(kelly_fraction, 4),
                'adjusted_kelly': round(adjusted_kelly, 4),
                'final_fraction': round(final_fraction, 4),
                'confidence_adjustment': round(confidence_adjustment, 3),
                'edge_adjustment': round(edge_adjustment, 3),
                'risk_level': risk_level,
                'bankroll_percentage': round(final_fraction * 100, 2)
            }
            
        except Exception as e:
            print(f"Error calculating bet size: {e}")
            return {
                'recommended_units': 0,
                'recommended_amount': 0,
                'kelly_fraction': 0,
                'risk_level': 'HIGH',
                'bankroll_percentage': 0
            }
    
    def _get_confidence_adjustment(self, confidence_score):
        """Adjust bet size based on confidence level"""
        if confidence_score >= 90:
            return 1.2
        elif confidence_score >= 80:
            return 1.1
        elif confidence_score >= 70:
            return 1.0
        elif confidence_score >= 60:
            return 0.8
        elif confidence_score >= 50:
            return 0.6
        else:
            return 0.3
    
    def _get_edge_adjustment(self, edge):
        """Adjust bet size based on edge strength"""
        abs_edge = abs(edge)
        if abs_edge >= 0.15:
            return 1.3
        elif abs_edge >= 0.10:
            return 1.2
        elif abs_edge >= 0.05:
            return 1.0
        elif abs_edge >= 0.03:
            return 0.8
        else:
            return 0.5
    
    def _assess_bet_risk(self, units, confidence, edge):
        """Assess overall risk level of the bet"""
        if units <= 0.5 and confidence >= 70:
            return 'LOW'
        elif units <= 1.0 and confidence >= 60:
            return 'LOW'
        elif units <= 2.0 and confidence >= 65:
            return 'MEDIUM'
        elif units <= 3.0 and confidence >= 75:
            return 'MEDIUM'
        elif units <= 5.0 and confidence >= 80:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def log_bet(self, bet_data):
        """Log a bet to the tracking system"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bet_history 
            (player_name, prop_type, line, bet_side, confidence_score, 
             recommended_units, actual_units, odds, bet_amount, game_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bet_data.get('player_name'),
            bet_data.get('prop_type'),
            bet_data.get('line'),
            bet_data.get('bet_side'),
            bet_data.get('confidence_score'),
            bet_data.get('recommended_units'),
            bet_data.get('actual_units'),
            bet_data.get('odds'),
            bet_data.get('bet_amount'),
            bet_data.get('game_date'),
            bet_data.get('notes', '')
        ))
        
        conn.commit()
        bet_id = cursor.lastrowid
        conn.close()
        
        return bet_id
    
    def update_bet_result(self, bet_id, result, profit_loss):
        """Update bet with result and profit/loss"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE bet_history 
            SET result = ?, profit_loss = ?
            WHERE id = ?
        ''', (result, profit_loss, bet_id))
        
        conn.commit()
        conn.close()
    
    def get_recent_performance(self, days=30):
        """Get recent performance metrics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT COUNT(*) as total_bets,
                   SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(profit_loss) as total_profit,
                   SUM(bet_amount) as total_wagered,
                   AVG(confidence_score) as avg_confidence
            FROM bet_history 
            WHERE bet_date >= ? AND result IS NOT NULL
        ''', (cutoff_date,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result or result[0] == 0:
            return {
                'total_bets': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_confidence': 0
            }
        
        total_bets, wins, total_profit, total_wagered, avg_confidence = result
        win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        return {
            'total_bets': total_bets,
            'win_rate': round(win_rate, 1),
            'total_profit': round(total_profit or 0, 2),
            'roi': round(roi, 1),
            'avg_confidence': round(avg_confidence or 0, 1)
        }
    
    def get_bet_history(self, limit=50):
        """Get recent bet history"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM bet_history 
            ORDER BY bet_date DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        bets = []
        for result in results:
            bet = dict(zip(columns, result))
            bets.append(bet)
        
        conn.close()
        return bets
    
    def calculate_performance_metrics(self, period_days=30):
        """Calculate comprehensive performance metrics"""
        bets = self.get_bet_history(1000)  # Get more data for analysis
        
        if not bets:
            return {}
        
        # Filter by period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_bets = [
            bet for bet in bets 
            if bet['result'] and datetime.strptime(bet['bet_date'], '%Y-%m-%d %H:%M:%S') >= cutoff_date
        ]
        
        if not recent_bets:
            return {}
        
        # Calculate metrics
        profits = [bet['profit_loss'] or 0 for bet in recent_bets]
        
        metrics = {
            'total_bets': len(recent_bets),
            'wins': len([b for b in recent_bets if b['result'] == 'WIN']),
            'losses': len([b for b in recent_bets if b['result'] == 'LOSS']),
            'total_profit': sum(profits),
            'average_profit_per_bet': np.mean(profits) if profits else 0,
            'win_rate': len([b for b in recent_bets if b['result'] == 'WIN']) / len(recent_bets) * 100,
            'largest_win': max(profits) if profits else 0,
            'largest_loss': min(profits) if profits else 0,
            'profit_factor': self._calculate_profit_factor(recent_bets),
            'sharpe_ratio': self._calculate_sharpe_ratio(profits),
            'max_drawdown': self._calculate_max_drawdown(profits)
        }
        
        return metrics
    
    def _calculate_profit_factor(self, bets):
        """Calculate profit factor (gross profit / gross loss)"""
        wins = [bet['profit_loss'] for bet in bets if bet['result'] == 'WIN' and bet['profit_loss']]
        losses = [abs(bet['profit_loss']) for bet in bets if bet['result'] == 'LOSS' and bet['profit_loss']]
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 1
        
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def _calculate_sharpe_ratio(self, profits):
        """Calculate Sharpe ratio for betting performance"""
        if not profits or len(profits) < 2:
            return 0
        
        mean_return = np.mean(profits)
        std_return = np.std(profits)
        
        return (mean_return / std_return) if std_return > 0 else 0
    
    def _calculate_max_drawdown(self, profits):
        """Calculate maximum drawdown"""
        if not profits:
            return 0
        
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        
        return abs(min(drawdowns)) if len(drawdowns) > 0 else 0