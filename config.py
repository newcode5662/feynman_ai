import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
PROGRESS_DB = os.path.join(DATA_DIR, "progress.db")

# 模型配置
# 关键修改：直接指向本机默认地址
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LLM_MODEL = "qwen2.5:3b"   # 你的 4050 跑这个最快
EMBEDDING_MODEL = "nomic-embed-text"

# 学习配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 费曼学习配置
MASTERY_THRESHOLD = 0.8
REVIEW_INTERVALS = [1, 3, 7, 14, 30]

# 自动创建目录
for dir_path in [DATA_DIR, DOCUMENTS_DIR, CHROMA_DIR]:
    os.makedirs(dir_path, exist_ok=True)
