import pdfplumber

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


if __name__ == "__main__":
    file_path = "data/translated_article.pdf"
    docs = parse_pdf_to_docs(file_path)

    print(f"提取到的文档总数: {len(docs)}")

    for i, doc in enumerate(docs):
        print(f"第 {i + 1} 页的字符数:", len(doc['page_content']))

    if docs:
        first_doc = docs[0]
        print("第一页的详细信息:")
        print(f"文本内容: {first_doc['page_content'][:200]}...")  # 打印前200个字符，防止输出过长
        print("元数据:")
        for key, value in first_doc['metadata'].items():
            print(f"{key}: {value}")