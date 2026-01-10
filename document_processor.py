import os
import fitz  # PyMuPDF
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDoc
from config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        # 优化切分逻辑：
        # 1. 优先按段落切
        # 2. 中文标点
        # 3. 英文标点 (注意 . 后面带空格，避免切断 3.14 这种数字)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n", "\n",
                "。", "！", "？",
                ". ", "! ", "? ",
                " ", ""
            ]
        )

    def load_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = []
        for page in doc:
            # 获取文本并去除过多空白，但保留段落感
            t = page.get_text()
            if t.strip():
                text.append(t)
        doc.close()
        return "\n\n".join(text)

    def load_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def load_markdown(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_document(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.pdf': self.load_pdf,
            '.docx': self.load_docx,
            '.doc': self.load_docx,
            '.md': self.load_markdown,
            '.txt': self.load_markdown
        }
        if ext not in loaders:
            raise ValueError(f"不支持的文件格式: {ext}")
        return loaders[ext](file_path)

    def process_document(self, file_path: str, subject: str = "默认") -> list:
        try:
            text = self.load_document(file_path)
            filename = os.path.basename(file_path)
            chunks = self.text_splitter.split_text(text)

            documents = []
            for i, chunk in enumerate(chunks):
                # 过滤掉过短的碎片（例如页码、页眉）
                if len(chunk.strip()) < 20:
                    continue

                doc = LangchainDoc(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "subject": subject,
                        "chunk_id": int(i)  # 强制转为整数
                    }
                )
                documents.append(doc)
            return documents
        except Exception as e:
            print(f"处理文档 {file_path} 失败: {e}")
            return []
