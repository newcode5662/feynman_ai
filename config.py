import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
PROGRESS_DB = os.path.join(DATA_DIR, "progress.db")

# ========== 模型配置 ==========
# 这里的地址主要用于打印日志，实际连接通过环境变量控制
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# 【显存保护配置】
# 与服务器日志中的 OLLAMA_CONTEXT_LENGTH:4096 保持一致
OLLAMA_CONTEXT_WINDOW = 4096
# 批处理大小：设为 5 以防爆显存 (特别是 4050/3060 等显卡)
EMBEDDING_BATCH_SIZE = 5

# ========== 学习配置 ==========
# 关键：Chunk Size 800 < 上下文窗口，确保 Embedding 不会报错
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MASTERY_THRESHOLD = 0.8
REVIEW_INTERVALS = [1, 3, 7, 14, 30]

# ========== 初始化目录 ==========
for dir_path in [DATA_DIR, DOCUMENTS_DIR, CHROMA_DIR]:
    os.makedirs(dir_path, exist_ok=True)
