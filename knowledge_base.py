import os
import time
import random
import gc  # 引入垃圾回收
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, EMBEDDING_BATCH_SIZE
from document_processor import DocumentProcessor

class KnowledgeBase:
    def __init__(self):
        # 【核心修复：连接报错】
        # 不在构造函数传 base_url，而是直接设置环境变量
        # LangChain 底层会自动读取这些变量，绕过 Pydantic 的参数校验错误
        os.environ['OLLAMA_BASE_URL'] = OLLAMA_BASE_URL
        os.environ['OLLAMA_HOST'] = OLLAMA_BASE_URL

        try:
            # 仅传递 model，其他配置全靠环境变量
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            # 冒烟测试：尝试 Embedding 一个单词
            self.embeddings.embed_query("test")
        except Exception as e:
            print(f"Embedding 初始化失败: {e}")
            raise RuntimeError(f"无法连接 Ollama ({OLLAMA_BASE_URL})。请检查：1. ollama serve 是否运行 2. 是否已 pull nomic-embed-text")

        self.processor = DocumentProcessor()
        # 版本号升级 v8，避免旧数据结构冲突
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="feynman_knowledge_v8"
        )

    def add_document(self, file_path: str, subject: str = "默认"):
        """
        生成器函数：逐步处理文档并 yield 进度
        Yields: (progress_float, total_docs, status_msg)
        Returns: (total_count, preview_text)
        """
        yield 0.0, 0, "正在读取并切分文档..."

        documents = self.processor.process_document(file_path, subject)
        if not documents:
            return 0, ""

        total_docs = len(documents)

        # 提取预览文本（用于摘要），严格限制长度防止 LLM 溢出
        preview_text = "\n".join([d.page_content for d in documents[:3]])[:2000]

        # 分批处理，防止爆显存
        for i in range(0, total_docs, EMBEDDING_BATCH_SIZE):
            batch = documents[i : i + EMBEDDING_BATCH_SIZE]

            try:
                self.vectorstore.add_documents(batch)

                # 【显存保护】强制垃圾回收
                gc.collect()

                # 计算进度
                current_count = min(i + EMBEDDING_BATCH_SIZE, total_docs)
                progress = current_count / total_docs
                yield progress, total_docs, f"正在向量化: {current_count}/{total_docs}"

                # 显卡冷却时间，防止过热降频
                time.sleep(0.05)

            except Exception as e:
                print(f"⚠️ Batch {i} 失败: {e}")
                # 不抛出异常，继续处理下一批，保证大部分数据可用
                yield progress, total_docs, f"⚠️ 批次 {i} 出错，跳过..."

        return total_docs, preview_text

    # --- 标准查询方法 (保持 V2.0 逻辑) ---
    def get_all_subjects(self) -> list:
        try:
            data = self.vectorstore.get()
            subjects = set()
            for metadata in data.get('metadatas', []):
                if metadata and 'subject' in metadata:
                    subjects.add(metadata['subject'])
            return sorted(list(subjects))
        except: return []

    def get_course_structure(self, subject: str) -> dict:
        collection = self.vectorstore._collection
        where = {"subject": subject} if subject else None
        try:
            results = collection.get(where=where, include=['metadatas', 'documents'])
        except: return {}

        structure = {}
        if results['ids']:
            for i, meta in enumerate(results['metadatas']):
                src = meta.get('source', '未知来源')
                if src not in structure: structure[src] = []
                structure[src].append({
                    "id": results['ids'][i],
                    "chunk_id": meta.get('chunk_id', 0),
                    "preview": results['documents'][i][:60].replace('\n', ' ') + "..."
                })
        for src in structure: structure[src].sort(key=lambda x: x['chunk_id'])
        return structure

    def get_knowledge_by_id(self, k_id: str) -> dict:
        res = self.vectorstore.get(ids=[k_id], include=['metadatas', 'documents'])
        if res['ids']:
            return {"id": res['ids'][0], "metadata": res['metadatas'][0], "content": res['documents'][0]}
        return None

    def get_random_knowledge(self, subject: str = None) -> dict:
        collection = self.vectorstore._collection
        where = {"subject": subject} if subject and subject != "全部" else None
        results = collection.get(where=where, include=['metadatas', 'documents'])
        if not results['ids']: return None
        idx = random.randint(0, len(results['ids']) - 1)
        return {"content": results['documents'][idx], "metadata": results['metadatas'][idx], "id": results['ids'][idx]}
