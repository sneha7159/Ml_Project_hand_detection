import sqlite3
import json
from datetime import datetime

def init_db():
    conn = sqlite3.connect('shikshakhel.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id TEXT PRIMARY KEY, 
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create game progress table
    c.execute('''CREATE TABLE IF NOT EXISTS game_progress
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  game_name TEXT,
                  score INTEGER,
                  level INTEGER,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    # Create learning analytics table
    c.execute('''CREATE TABLE IF NOT EXISTS learning_analytics
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  game_name TEXT,
                  time_spent INTEGER,
                  correct_answers INTEGER,
                  total_questions INTEGER,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (user_id))''')
    
    conn.commit()
    conn.close()

def get_user_data(user_id):
    conn = sqlite3.connect('shikshakhel.db')
    c = conn.cursor()
    
    # Get user progress
    c.execute('''SELECT game_name, MAX(score), MAX(level) 
                 FROM game_progress 
                 WHERE user_id = ? 
                 GROUP BY game_name''', (user_id,))
    progress = c.fetchall()
    
    # Calculate total score and games played
    total_score = sum(p[1] for p in progress) if progress else 0
    games_played = len(progress)
    
    # Determine learning level based on total score
    if total_score < 100:
        level = "Beginner"
    elif total_score < 500:
        level = "Intermediate"
    else:
        level = "Advanced"
    
    conn.close()
    
    return {
        'games_played': games_played,
        'total_score': total_score,
        'level': level,
        'detailed_progress': progress
    }

def save_game_progress(user_id, game_name, score, level):
    conn = sqlite3.connect('shikshakhel.db')
    c = conn.cursor()
    
    # Ensure user exists
    c.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    
    # Save game progress
    c.execute('''INSERT INTO game_progress (user_id, game_name, score, level)
                 VALUES (?, ?, ?, ?)''', (user_id, game_name, score, level))
    
    conn.commit()
    conn.close()

def save_learning_analytics(user_id, game_name, time_spent, correct_answers, total_questions):
    conn = sqlite3.connect('shikshakhel.db')
    c = conn.cursor()
    
    c.execute('''INSERT INTO learning_analytics (user_id, game_name, time_spent, correct_answers, total_questions)
                 VALUES (?, ?, ?, ?, ?)''', (user_id, game_name, time_spent, correct_answers, total_questions))
    
    conn.commit()
    conn.close()