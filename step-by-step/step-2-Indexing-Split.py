import pdfplumber
import tiktoken
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_pdf_to_docs(pdf_path):
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
    docs = []  # 初始化一个空列表来存储每页的文档数据

    # 使用pdfplumber打开PDF文件
    with pdfplumber.open(pdf_path) as pdf:
        # 遍历PDF中的每一页
        for page in pdf.pages:
            # 提取当前页的文本
            page_content = page.extract_text()

            # 如果当前页有文本，则处理并存储相关数据
            if page_content:  # 仅处理含有文本的页面
                doc = {
                    "page_content": page_content,  # 页面文本
                    "metadata": {
                        "source": pdf_path,  # 文档来源路径
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


def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
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


def split_text_into_texts(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=num_tokens_from_string,
    )
    texts = text_splitter.split_text(text)
    return texts


def split_docs_into_chunks(docs, chunk_size=500, chunk_overlap=50):
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
        page_content_texts = split_text_into_texts(doc['page_content'], chunk_size, chunk_overlap)
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

def save_chunks_to_json(chunks, filename="output_chunks.json"):
    """
    将提取的文本块及其元数据保存到 JSON 文件中。

    Args:
        chunks (list of dicts): 包含文本块和元数据的列表。
        filename (str): 要保存的 JSON 文件名。
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)
    print(f"成功保存了 {len(chunks)} 个文本块到文件 '{filename}'.")


if __name__ == "__main__":

    # load
    file_path = "data/translated_article.pdf"
    docs = parse_pdf_to_docs(file_path)

    print("提取到的文档总数:", len(docs))

    for i, doc in enumerate(docs):
        print(f"第 {i + 1} 页的字符数:", len(doc['page_content']))

    # split
    chunks = split_docs_into_chunks(docs)
    print("分割后的总文本块数:", len(chunks))
    if chunks:
        print("第一个文本块的详细信息:", chunks[0])

    save_chunks_to_json(chunks, "data/chunks.json")