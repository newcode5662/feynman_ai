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
        if self._initialized:
            return
        self.conn = sqlite3.connect(PROGRESS_DB, check_same_thread=False)
        self._db_lock = threading.Lock()
        self._init_db()
        self._initialized = True

    def _init_db(self):
        with self._db_lock:
            c = self.conn.cursor()

            # 知识点进度表
            c.execute('''CREATE TABLE IF NOT EXISTS knowledge_progress
                (knowledge_id TEXT PRIMARY KEY,
                 subject TEXT,
                 content_preview TEXT,
                 review_count INTEGER DEFAULT 0,
                 correct_count INTEGER DEFAULT 0,
                 mastery_level REAL DEFAULT 0.0,
                 best_score REAL DEFAULT 0.0,
                 last_score REAL DEFAULT 0.0,
                 last_review_date TEXT,
                 next_review_date TEXT,
                 created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

            # 详细学习日志
            c.execute('''CREATE TABLE IF NOT EXISTS learning_logs
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 knowledge_id TEXT,
                 action TEXT,
                 score REAL,
                 user_explanation TEXT,
                 ai_feedback TEXT,
                 timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')

            # 每日统计
            c.execute('''CREATE TABLE IF NOT EXISTS daily_stats
                (date TEXT PRIMARY KEY,
                 total_reviews INTEGER DEFAULT 0,
                 correct_reviews INTEGER DEFAULT 0,
                 total_time_minutes INTEGER DEFAULT 0)''')

            # 检查并添加新列（兼容旧数据库）
            try:
                c.execute("ALTER TABLE knowledge_progress ADD COLUMN best_score REAL DEFAULT 0.0")
            except:
                pass
            try:
                c.execute("ALTER TABLE knowledge_progress ADD COLUMN last_score REAL DEFAULT 0.0")
            except:
                pass

            self.conn.commit()

    def record_review(self, kid: str, sub: str, content: str, score: float, expl: str, fb: str):
        """记录一次学习/复习"""
        now = datetime.now().isoformat()
        today = datetime.now().date().isoformat()
        passed = 1 if score >= 0.6 else 0
        preview = content[:100] if content else ""

        with self._db_lock:
            c = self.conn.cursor()

            # 获取当前状态
            c.execute("SELECT review_count, best_score FROM knowledge_progress WHERE knowledge_id=?", (kid,))
            res = c.fetchone()
            curr_count = res[0] if res else 0
            curr_best = res[1] if res else 0.0

            # 计算下次复习时间（艾宾浩斯）
            if score >= MASTERY_THRESHOLD:
                interval_idx = min(curr_count, len(REVIEW_INTERVALS) - 1)
                next_days = REVIEW_INTERVALS[interval_idx]
            else:
                next_days = 1
            next_date = (datetime.now() + timedelta(days=next_days)).isoformat()

            # 更新最高分
            new_best = max(curr_best, score)

            # 更新或插入进度
            c.execute('''INSERT INTO knowledge_progress
                (knowledge_id, subject, content_preview, review_count, correct_count,
                 mastery_level, best_score, last_score, last_review_date, next_review_date)
                VALUES (?,?,?,1,?,?,?,?,?,?)
                ON CONFLICT(knowledge_id) DO UPDATE SET
                review_count = review_count + 1,
                correct_count = correct_count + ?,
                mastery_level = (correct_count + ?) * 1.0 / (review_count + 1),
                best_score = MAX(best_score, ?),
                last_score = ?,
                last_review_date = ?,
                next_review_date = ?''',
                (kid, sub, preview, passed, score, new_best, score, now, next_date,
                 passed, passed, score, score, now, next_date))

            # 插入日志
            c.execute('''INSERT INTO learning_logs
                (knowledge_id, action, score, user_explanation, ai_feedback)
                VALUES (?, 'review', ?, ?, ?)''',
                (kid, score, expl, fb))

            # 更新每日统计
            c.execute('''INSERT INTO daily_stats (date, total_reviews, correct_reviews)
                VALUES (?, 1, ?)
                ON CONFLICT(date) DO UPDATE SET
                total_reviews = total_reviews + 1,
                correct_reviews = correct_reviews + ?''',
                (today, passed, passed))

            self.conn.commit()

    def get_due_reviews(self, limit: int = 10) -> list:
        """获取到期需要复习的知识点"""
        now = datetime.now().isoformat()
        c = self.conn.cursor()
        c.execute('''SELECT knowledge_id, subject, content_preview, mastery_level
            FROM knowledge_progress
            WHERE next_review_date <= ?
            ORDER BY mastery_level ASC, next_review_date ASC
            LIMIT ?''', (now, limit))
        return c.fetchall()

    def get_learned_ids(self, subject: str = None) -> list:
        """【新增】获取已学习过的知识点ID列表"""
        c = self.conn.cursor()
        if subject:
            c.execute("SELECT knowledge_id FROM knowledge_progress WHERE subject = ?", (subject,))
        else:
            c.execute("SELECT knowledge_id FROM knowledge_progress")
        return [row[0] for row in c.fetchall()]

    def get_weak_knowledge(self, subject: str, limit: int = 5) -> list:
        """【新增】获取薄弱知识点（低分优先）"""
        c = self.conn.cursor()
        if subject:
            c.execute('''SELECT knowledge_id, last_score
                FROM knowledge_progress
                WHERE subject = ? AND review_count > 0
                ORDER BY last_score ASC
                LIMIT ?''', (subject, limit))
        else:
            c.execute('''SELECT knowledge_id, last_score
                FROM knowledge_progress
                WHERE review_count > 0
                ORDER BY last_score ASC
                LIMIT ?''', (limit,))
        return c.fetchall()

    def get_subject_progress(self, subject: str) -> list:
        """【新增】获取某学科所有知识点的进度"""
        c = self.conn.cursor()
        c.execute('''SELECT knowledge_id, mastery_level, review_count, best_score, last_score
            FROM knowledge_progress
            WHERE subject = ?''', (subject,))
        rows = c.fetchall()
        return [
            {
                "knowledge_id": r[0],
                "mastery_level": r[1],
                "review_count": r[2],
                "best_score": r[3],
                "last_score": r[4]
            }
            for r in rows
        ]

    def get_statistics(self) -> dict:
        """获取整体统计"""
        c = self.conn.cursor()

        # 总体统计
        c.execute('''SELECT COUNT(*), AVG(mastery_level),
            SUM(CASE WHEN mastery_level >= 0.8 THEN 1 ELSE 0 END)
            FROM knowledge_progress''')
        overall = c.fetchone()

        # 近7天统计
        c.execute('''SELECT date, total_reviews, correct_reviews
            FROM daily_stats
            ORDER BY date DESC LIMIT 7''')
        weekly = c.fetchall()

        # 按学科统计
        c.execute('''SELECT subject, COUNT(*), AVG(mastery_level),
            SUM(CASE WHEN mastery_level >= 0.8 THEN 1 ELSE 0 END)
            FROM knowledge_progress
            GROUP BY subject''')
        by_subject = c.fetchall()

        return {
            "total_knowledge": overall[0] or 0,
            "avg_mastery": round((overall[1] or 0) * 100, 1),
            "mastered_count": overall[2] or 0,
            "weekly_stats": weekly,
            "by_subject": [{"subject": r[0], "total": r[1], "mastery": r[2], "mastered": r[3]} for r in by_subject]
}
