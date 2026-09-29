import os
from typing import List

import pdfplumber
import tiktoken

from config import CHUNK_OVERLAP, CHUNK_SIZE
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FileProcessorHelper:
    def __init__(
            self,
            file_path: str,
            file_name: str = None,
            file_extension: str = None,
            file_md5: str = None,
    ):
        self.file_path = file_path
        self.file_name = file_name
        self.file_extension = file_extension
        self.file_md5 = file_md5

    # 获取docs
    def file_to_docs(self) -> List:

        strategy_mapping = {
            '.pdf': self.pdf_file_to_docs,
            '.txt': self.txt_file_to_docs,
            # '.doc': self.word_file_to_docs,
            # '.docx': self.word_file_to_docs,
            # '.md': self.md_file_to_docs,
            # '.xls': self.excel_file_to_docs,
            # '.xlsx': self.excel_file_to_docs,
            # '.csv': self.csv_file_to_docs,
            # '.jpg': self.image_file_to_docs,
            # '.jpeg': self.image_file_to_docs,
            # '.png': self.image_file_to_docs,
            # '.gif': self.image_file_to_docs,
            # '.ico': self.image_file_to_docs,
            # '.svg': self.image_file_to_docs,
            # '.bmp': self.image_file_to_docs,
            # '.ppt': self.ppt_file_to_docs,
            # '.pptx': self.ppt_file_to_docs,
            # '.zip': self.zip_file_to_docs,
            # '.mp3': self.audio_file_to_docs,
            # '.wav': self.audio_file_to_docs,
            # '.mp4': self.video_file_to_docs,
        }
        func = strategy_mapping.get(self.file_extension)
        return func(self.file_path)

    @staticmethod
    def pdf_file_to_docs(file_path: str) -> List[dict]:
        """
            从PDF文件中解析数据并将其转换为包含每页文本和元数据的文档列表。

            Args:
                pdf_path (str): 要解析的PDF文件的路径。

            Returns:
                list: 包含每页文本和详细元数据的文档列表。
                  - 每个文档是一个字典，包括：
                    - 'page_content': 页面上的文本内容。
                    - 'metadata': 该页面的元数据，包括来源、页码、总页数、尺寸和PDF的其他元数据。
            """
        file_name = os.path.basename(file_path)

        docs = []  # 初始化一个空列表来存储每页的文档数据

        # 使用pdfplumber打开PDF文件
        with pdfplumber.open(file_path) as pdf:
            # 遍历PDF中的每一页
            for page in pdf.pages:
                # 提取当前页的文本
                page_content = page.extract_text()

                # 如果当前页有文本，则处理并存储相关数据
                if page_content:  # 仅处理含有文本的页面
                    doc = {
                        "page_content": page_content,  # 页面文本
                        "metadata": {
                            "source": file_path,  # 文档来源路径
                            "file_name": file_name, # 文档名
                            "page_number": page.page_number,  # 当前页码
                            "total_pages": len(pdf.pages),  # PDF文档的总页数
                            "width": page.width,  # 页面宽度
                            "height": page.height,  # 页面高度
                            "rotation": page.rotation,  # 页面旋转角度
                            **pdf.metadata  # 从PDF提取的其他元数据
                        }
                    }
                    # 将构建的文档添加到列表中
                    docs.append(doc)

        return docs  # 返回构建的文档列表

    @staticmethod
    def txt_file_to_docs(file_path: str) -> List[dict]:
        file_name = os.path.basename(file_path)

        with open(file_path, 'r') as file:
            text = file.read()
        if not text:
            return []
        doc = {
            "page_content": text,
            "metadata": {
                "file_name": file_name
            }
        }
        return [doc]

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """
        Returns the number of tokens in a text string.
        https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
        """
        encoding = tiktoken.get_encoding(encoding_name)
        # encoding = tiktoken.encoding_for_model('gpt-4-turbo')
        tokens = encoding.encode(string)
        # print(tokens)
        # print(encoding.decode([83, 1609, 5963, 374, 2294, 0]))
        num_tokens = len(tokens)
        return num_tokens

    def split_text_into_texts(self, text, chunk_size, chunk_overlap):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.num_tokens_from_string,
        )
        texts = text_splitter.split_text(text)
        return texts


    def split_docs_into_chunks(self, docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """
        将文档列表中的每个文档按指定规则分割成更小的文本块。

        Args:
            docs (list of dicts): 包含文本和元数据的文档列表。
            chunk_size (int): 每个文本块的最大大小。
            chunk_overlap (int): 文本块之间的重叠大小。

        Returns:
            list of dicts: 包含分割后的文本块和更新后元数据的列表。
        """
        chunks = []
        chunk_id = 0  # 初始化块ID，用于跟踪和标识分割后的每个文本块
        for doc in docs:
            page_content_texts = self.split_text_into_texts(doc['page_content'], chunk_size, chunk_overlap)
            for page_content_text in page_content_texts:
                chunk = {
                    "page_content": page_content_text,
                    "metadata": {
                        "chunk_id": chunk_id,  # 将块ID添加到元数据中
                        **doc['metadata'],
                    }
                }
                chunks.append(chunk)
                chunk_id += 1
        return chunks


if __name__ == "__main__":
    # 测试
    file_path = "data/translated_article.pdf"
    file_name = "translated_article.pdf"
    file_extension = ".pdf"
    file_md5 = "0"  # 'f59d5ebd9c2cb8161651f5e44ecd6c9f'

    file_processor_helper = FileProcessorHelper(
        file_path=file_path,
        file_name=file_name,
        file_extension=file_extension,
        file_md5=file_md5,
    )
    docs = file_processor_helper.file_to_docs()
    print(docs)

    # 测试 num_tokens_from_string
    print(FileProcessorHelper.num_tokens_from_string("tiktoken is great!"))