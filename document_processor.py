import os
import fitz
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDoc
from config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )

    def load_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = ""
        for page in doc: text += page.get_text()
        doc.close()
        return text

    def load_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def load_markdown(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()

    def load_document(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {'.pdf': self.load_pdf, '.docx': self.load_docx, '.md': self.load_markdown, '.txt': self.load_markdown}
        if ext not in loaders: raise ValueError(f"不支持的格式: {ext}")
        return loaders[ext](file_path)

    def process_document(self, file_path: str, subject: str = "默认") -> list:
        try:
            text = self.load_document(file_path)
            filename = os.path.basename(file_path)
            chunks = self.text_splitter.split_text(text)
            documents = []
            for i, chunk in enumerate(chunks):
                doc = LangchainDoc(page_content=chunk, metadata={"source": filename, "subject": subject, "chunk_id": i})
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"Error: {e}")
            return []
