import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
PROGRESS_DB = os.path.join(DATA_DIR, "progress.db")

# ========== 模型配置 ==========
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# ========== 文档处理配置 ==========
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# ========== 费曼学习配置 ==========
MASTERY_THRESHOLD = 0.8
REVIEW_INTERVALS = [1, 3, 7, 14, 30]

# ========== 【新增】评估等级体系 ==========
MASTERY_LEVELS = {
    "master": {"min": 0.9, "label": "🏆 精通", "color": "#FFD700", "desc": "可以教别人了！"},
    "proficient": {"min": 0.75, "label": "✅ 熟练", "color": "#4CAF50", "desc": "理解到位，继续保持"},
    "developing": {"min": 0.6, "label": "📈 进步中", "color": "#2196F3", "desc": "核心概念已懂，细节待加强"},
    "learning": {"min": 0.4, "label": "📖 学习中", "color": "#FF9800", "desc": "有基础理解，需要多练习"},
    "beginner": {"min": 0.0, "label": "🌱 起步", "color": "#9E9E9E", "desc": "刚开始接触，别担心"}
}

# ========== 【新增】学习路径配置 ==========
LEARNING_MODES = {
    "sequential": "📚 顺序学习（从头到尾，适合新手）",
    "review": "🔄 复习模式（艾宾浩斯算法）",
    "random": "🎲 随机挑战（已学过的随机抽取）",
    "weak": "🎯 薄弱点强化（专攻低分知识点）"
}

# ========== 初始化目录 ==========
for dir_path in [DATA_DIR, DOCUMENTS_DIR, CHROMA_DIR]:
    os.makedirs(dir_path, exist_ok=True)
