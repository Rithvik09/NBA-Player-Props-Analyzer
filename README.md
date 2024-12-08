

NBA Player Props Analyzer
Welcome to the NBA Player Props Analyzer project! 

This tool is designed to assist sports bettors in making data-driven decisions by leveraging advanced machine learning models, real-time data updates, and comprehensive basketball analytics.


The NBA Player Props Analyzer transforms sports betting from an intuition-driven activity into a data-driven process. Using historical and real-time basketball data, this system predicts player performance and provides betting recommendations tailored to specific games and players.

This project integrates data management, machine learning, and database-driven analytics to create a recommendation engine that goes beyond traditional betting tools.

Key Features
Real-Time Data Updates: Fetches player and team stats directly from the NBA.com API.
Predictive Modeling:
Random Forest Classifier: Predicts whether a player’s stats will exceed a given betting line.
Random Forest Regressor: Estimates player performance metrics (e.g., points, assists, rebounds).
Comprehensive Analytics: Incorporates player trends, team dynamics, and matchup specifics for enhanced predictions.
User-Friendly API: Provides endpoints for querying player stats and generating predictions.
System Architecture
Backend: Built with Flask, connecting to a SQLite database for structured data management.
Database: Relational tables store player, team, and game data for efficient querying.
Machine Learning: Models are trained on historical game logs, leveraging features like season averages, recent form, and matchup statistics.
Logging: Tracks system events and ensures reliability during API calls and data processing.
Data Collection and Integration
Source: NBA.com API (via nba_api library).
Data Points:
Player stats: Points, assists, rebounds, shooting percentages.
Team metrics: Wins, losses, and game outcomes.
Database Schema:
Players table: Stores player information.
Game Logs table: Tracks performance metrics.
Machine Learning Models
Random Forest Classifier:
Binary predictions: “Over” or “Under” a given betting line.
Features: Recent performance trends, matchup-specific stats.
Random Forest Regressor:
Continuous predictions: Expected values for player stats.
Features: Season averages, defensive matchups, team context
