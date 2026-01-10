import os
import random
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL
from document_processor import DocumentProcessor


class KnowledgeBase:
    def __init__(self):
        os.environ['OLLAMA_BASE_URL'] = OLLAMA_BASE_URL

        try:
            self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            self.embeddings.embed_query("test")
            self._using_ollama = True
        except Exception as e:
            print(f"Ollama Embedding 连接失败: {e}")
            from langchain.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._using_ollama = False

        self.processor = DocumentProcessor()
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="feynman_knowledge_v2"
        )

    def add_document(self, file_path: str, subject: str = "默认") -> int:
        """导入文档并返回知识块数量"""
        documents = self.processor.process_document(file_path, subject)
        if documents:
            self.vectorstore.add_documents(documents)
        return len(documents)

    def get_all_subjects(self) -> list:
        """获取所有学科"""
        data = self.vectorstore.get()
        subjects = set()
        for metadata in data.get('metadatas', []):
            if metadata and 'subject' in metadata:
                subjects.add(metadata['subject'])
        return sorted(list(subjects))

    def get_course_structure(self, subject: str) -> dict:
        """
        【新增】获取课程结构（按章节组织）
        返回: {
            "subject": "xxx",
            "total_chunks": 10,
            "chapters": [
                {"chapter_id": 0, "title": "...", "chunks": [...]}
            ]
        }
        """
        collection = self.vectorstore._collection
        results = collection.get(
            where={"subject": subject},
            include=['metadatas', 'documents']
        )

        if not results['ids']:
            return {"subject": subject, "total_chunks": 0, "chapters": []}

        # 组合并排序
        items = list(zip(results['ids'], results['metadatas'], results['documents']))
        items.sort(key=lambda x: x[1].get('chunk_id', 0))

        # 按来源文件分章节
        chapters = {}
        for pid, meta, doc in items:
            source = meta.get('source', '未知文件')
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
            "total_chunks": len(items),
            "chapters": list(chapters.values())
        }

    def get_chapter_progress(self, subject: str, tracker) -> list:
        """
        【新增】获取带进度的章节列表
        """
        structure = self.get_course_structure(subject)
        progress_data = tracker.get_subject_progress(subject)

        # 构建进度映射
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
        """根据ID获取知识点"""
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
        """
        【新增】获取下一个未学习的知识点（顺序学习）
        """
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
        """
        【新增】获取薄弱知识点（低分优先）
        """
        weak_ids = tracker.get_weak_knowledge(subject, limit)
        result = []
        for kid, score in weak_ids:
            k = self.get_knowledge_by_id(kid)
            if k:
                k['last_score'] = score
                result.append(k)
        return result

    def get_random_knowledge(self, subject: str = None) -> dict:
        """随机获取知识点"""
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
