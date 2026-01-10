import os
import time
import random
# 优先尝试导入新的 HuggingFace 库，如果不存在则使用 community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL
from document_processor import DocumentProcessor

class KnowledgeBase:
    def __init__(self):
        # === 修复核心 1：通过环境变量配置 Ollama ===
        # LangChain 的 OllamaEmbeddings 会自动读取这些环境变量
        # 这样可以绕过构造函数的参数验证错误
        os.environ['OLLAMA_BASE_URL'] = OLLAMA_BASE_URL
        os.environ['OLLAMA_HOST'] = OLLAMA_BASE_URL

        try:
            print(f"🔌 正在连接 Ollama ({OLLAMA_BASE_URL})...")
            # === 修复核心 2：初始化时不传 extra args ===
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL
            )
            # 简单的冒烟测试，确保服务真的通了
            self.embeddings.embed_query("test")
            print("✅ Ollama Embedding 服务连接成功")

        except Exception as e:
            print(f"❌ Ollama 连接失败: {e}")
            print("💡 请检查：1. Ollama 是否已启动 (ollama serve)？ 2. 模型是否已下载 (ollama pull nomic-embed-text)？")

            # === 修复核心 3：不再自动 fallback 到 HuggingFace ===
            # 因为国内网络通常连不上 HF，自动下载会导致长时间卡死
            # 直接抛出异常，让用户先去修好 Ollama
            raise RuntimeError("无法连接到本地 Ollama Embedding 服务，且无法连接 HuggingFace。请优先确保 Ollama 正常运行。")

        self.processor = DocumentProcessor()
        # 初始化向量库
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="feynman_knowledge_v3"
        )

    def add_document(self, file_path: str, subject: str = "默认") -> int:
        documents = self.processor.process_document(file_path, subject)
        if not documents:
            return 0

        # 分批次插入，防止爆显存
        BATCH_SIZE = 10
        total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"正在导入 {len(documents)} 个知识块，分 {total_batches} 批处理...")

        success_count = 0
        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i : i + BATCH_SIZE]
            try:
                self.vectorstore.add_documents(batch)
                success_count += len(batch)
                print(f"进度: {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")
                # 稍微暂停，给显卡喘息时间
                time.sleep(0.1)
            except Exception as e:
                print(f"❌ 批次 {i//BATCH_SIZE + 1} 导入失败: {e}")

        return success_count

    def get_all_subjects(self) -> list:
        try:
            data = self.vectorstore.get()
            subjects = set()
            for metadata in data.get('metadatas', []):
                if metadata and 'subject' in metadata:
                    subjects.add(metadata['subject'])
            return sorted(list(subjects))
        except:
            return []

    def get_course_structure(self, subject: str) -> dict:
        collection = self.vectorstore._collection
        try:
            results = collection.get(
                where={"subject": subject},
                include=['metadatas', 'documents']
            )
        except Exception as e:
            print(f"数据库读取错误: {e}")
            return {"subject": subject, "total_chunks": 0, "chapters": []}

        if not results['ids']:
            return {"subject": subject, "total_chunks": 0, "chapters": []}

        # 清洗数据
        valid_items = []
        ids = results['ids']
        metas = results['metadatas']
        docs = results['documents']

        for i in range(len(ids)):
            if not metas[i] or not docs[i]:
                continue
            valid_items.append((ids[i], metas[i], docs[i]))

        # 排序
        valid_items.sort(key=lambda x: x[1].get('chunk_id', 0))

        chapters = {}
        for pid, meta, doc in valid_items:
            source = meta.get('source', '未知章节')
            if source not in chapters:
                chapters[source] = {
                    "chapter_id": len(chapters),
                    "title": source.replace('.pdf', '').replace('.docx', '').replace('.md', ''),
                    "source": source,
                    "chunks": []
                }

            chapters[source]["chunks"].append({
                "id": pid,
                "chunk_id": meta.get('chunk_id', 0),
                "preview": doc[:80].replace('\n', ' ') + "...",
                "content": doc,
                "metadata": meta
            })

        return {
            "subject": subject,
            "total_chunks": len(valid_items),
            "chapters": list(chapters.values())
        }

    def get_chapter_progress(self, subject: str, tracker) -> list:
        structure = self.get_course_structure(subject)
        progress_data = tracker.get_subject_progress(subject)
        progress_map = {p['knowledge_id']: p for p in progress_data}

        for chapter in structure['chapters']:
            completed = 0
            mastered = 0
            for chunk in chapter['chunks']:
                kid = chunk['id']
                if kid in progress_map:
                    chunk['progress'] = progress_map[kid]
                    completed += 1
                    if progress_map[kid]['mastery_level'] >= 0.8:
                        mastered += 1
                else:
                    chunk['progress'] = None

            chapter['stats'] = {
                'total': len(chapter['chunks']),
                'completed': completed,
                'mastered': mastered,
                'progress_pct': round(completed / len(chapter['chunks']) * 100) if chapter['chunks'] else 0
            }
        return structure

    def get_knowledge_by_id(self, k_id: str) -> dict:
        collection = self.vectorstore._collection
        results = collection.get(ids=[k_id], include=['metadatas', 'documents'])
        if results['ids']:
            return {
                "content": results['documents'][0],
                "metadata": results['metadatas'][0],
                "id": results['ids'][0]
            }
        return None

    def get_next_unlearned(self, subject: str, tracker) -> dict:
        structure = self.get_course_structure(subject)
        learned_ids = set(tracker.get_learned_ids(subject))

        for chapter in structure['chapters']:
            for chunk in chapter['chunks']:
                if chunk['id'] not in learned_ids:
                    return {
                        "content": chunk['content'],
                        "metadata": chunk['metadata'],
                        "id": chunk['id'],
                        "chapter": chapter['title'],
                        "position": f"第{chapter['chapter_id']+1}章 - 第{chunk['chunk_id']+1}节"
                    }
        return None

    def get_weak_points(self, subject: str, tracker, limit: int = 5) -> list:
        weak_ids = tracker.get_weak_knowledge(subject, limit)
        result = []
        for kid, score in weak_ids:
            k = self.get_knowledge_by_id(kid)
            if k:
                k['last_score'] = score
                result.append(k)
        return result

    def get_random_knowledge(self, subject: str = None) -> dict:
        try:
            collection = self.vectorstore._collection
            where_filter = {"subject": subject} if subject else None
            results = collection.get(where=where_filter, include=['metadatas', 'documents'])

            if not results['ids']:
                return None

            idx = random.randint(0, len(results['ids']) - 1)
            return {
                "content": results['documents'][idx],
                "metadata": results['metadatas'][idx],
                "id": results['ids'][idx]
            }
        except:
            return None
