import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL
from document_processor import DocumentProcessor

class KnowledgeBase:
    def __init__(self):
        # 设置环境变量，OllamaEmbeddings会自动使用OLLAMA_BASE_URL环境变量
        import os
        old_url = os.environ.get('OLLAMA_BASE_URL')
        os.environ['OLLAMA_BASE_URL'] = OLLAMA_BASE_URL
        try:
            # 测试Ollama连接
            import requests
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                print(f"警告: 无法连接到Ollama服务 {OLLAMA_BASE_URL}")
            else:
                print(f"成功连接到Ollama服务 {OLLAMA_BASE_URL}")
            
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL
            )
            # 保持环境变量设置，不恢复
            self._using_ollama = True
        except Exception as e:
            print(f"Ollama连接错误: {str(e)}")
            print(f"请确保Ollama服务正在运行在 {OLLAMA_BASE_URL}")
            print("您可以通过运行 'ollama serve' 来启动Ollama服务")
            print("尝试使用备用本地嵌入模型...")
            # 使用本地嵌入模型作为备用方案
            from langchain.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self._using_ollama = False
        finally:
            # 只有在不使用Ollama时才恢复环境变量
            if not getattr(self, '_using_ollama', False):
                if old_url is not None:
                    os.environ['OLLAMA_BASE_URL'] = old_url
                else:
                    os.environ.pop('OLLAMA_BASE_URL', None)
        self.processor = DocumentProcessor()
        # 显式指定集合名称，防止冲突
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name="feynman_knowledge_native"
        )
    
    def add_document(self, file_path: str, subject: str = "默认") -> int:
        documents = self.processor.process_document(file_path, subject)
        if documents:
            self.vectorstore.add_documents(documents)
        return len(documents)
    
    def get_all_subjects(self) -> list:
        data = self.vectorstore.get()
        subjects = set()
        for metadata in data.get('metadatas', []):
            if metadata and 'subject' in metadata:
                subjects.add(metadata['subject'])
        return list(subjects)
    
    def get_random_knowledge(self, subject: str = None) -> dict:
        import random
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
