import sqlite3
import threading
from datetime import datetime, timedelta
from config import PROGRESS_DB, REVIEW_INTERVALS, MASTERY_THRESHOLD

class ProgressTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ProgressTracker, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.conn = sqlite3.connect(PROGRESS_DB, check_same_thread=False)
        self._db_lock = threading.Lock()
        self._init_db()
        self._initialized = True
    
    def _init_db(self):
        with self._db_lock:
            c = self.conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS knowledge_progress 
                (knowledge_id TEXT PRIMARY KEY, subject TEXT, content_preview TEXT, 
                review_count INTEGER DEFAULT 0, correct_count INTEGER DEFAULT 0, 
                mastery_level REAL DEFAULT 0.0, last_review_date TEXT, 
                next_review_date TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS learning_logs 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, knowledge_id TEXT, action TEXT, 
                score REAL, user_explanation TEXT, ai_feedback TEXT, 
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')
            c.execute('''CREATE TABLE IF NOT EXISTS daily_stats 
                (date TEXT PRIMARY KEY, total_reviews INTEGER DEFAULT 0, 
                correct_reviews INTEGER DEFAULT 0)''')
            self.conn.commit()
    
    def record_review(self, kid, sub, prev, score, expl, fb):
        now = datetime.now().isoformat()
        today = datetime.now().date().isoformat()
        passed = 1 if score >= 0.6 else 0
        with self._db_lock:
            c = self.conn.cursor()
            c.execute("SELECT review_count FROM knowledge_progress WHERE knowledge_id=?", (kid,))
            res = c.fetchone()
            curr = res[0] if res else 0
            
            next_d = REVIEW_INTERVALS[min(curr+1, len(REVIEW_INTERVALS)-1)] if score >= MASTERY_THRESHOLD else 1
            next_date = (datetime.now() + timedelta(days=next_d)).isoformat()

            c.execute('''INSERT INTO knowledge_progress VALUES (?,?,?,1,?,?,?,?,?) 
                ON CONFLICT(knowledge_id) DO UPDATE SET 
                review_count=review_count+1, correct_count=correct_count+?, 
                mastery_level=(correct_count+?)*1.0/(review_count+1), 
                last_review_date=?, next_review_date=?''', 
                (kid, sub, prev[:100], passed, score, now, next_date, now, passed, passed, now, next_date))
            
            c.execute("INSERT INTO learning_logs (knowledge_id,action,score,user_explanation,ai_feedback) VALUES (?, 'review', ?, ?, ?)", (kid, score, expl, fb))
            c.execute("INSERT INTO daily_stats VALUES (?,1,?) ON CONFLICT(date) DO UPDATE SET total_reviews=total_reviews+1, correct_reviews=correct_reviews+?", (today, passed, passed))
            self.conn.commit()

    def get_due_reviews(self, limit=10):
        now = datetime.now().isoformat()
        c = self.conn.cursor()
        c.execute("SELECT knowledge_id, subject, content_preview, mastery_level FROM knowledge_progress WHERE next_review_date <= ? ORDER BY next_review_date ASC LIMIT ?", (now, limit))
        return c.fetchall()

    def get_statistics(self):
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*), AVG(mastery_level) FROM knowledge_progress")
        overall = c.fetchone()
        c.execute("SELECT subject, COUNT(*) FROM knowledge_progress GROUP BY subject")
        by_subj = c.fetchall()
        return {"total_knowledge": overall[0] or 0, "avg_mastery": round((overall[1] or 0)*100, 1), "by_subject": by_subj, "weekly_stats": []}
