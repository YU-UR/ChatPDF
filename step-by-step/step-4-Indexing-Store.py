from dotenv import load_dotenv
from openai import OpenAI
import json
import faiss
import numpy as np


# 加载环境变量
load_dotenv()

client = OpenAI()

def load_data_from_json(filename):
    """
    从指定的 JSON 文件中加载数据。

    Args:
        filename (str): JSON 文件的路径。

    Returns:
        Any: 从 JSON 文件中加载的数据。
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"成功从 '{filename}' 加载了数据。")
        return data
    except json.JSONDecodeError as e:
        print(f"解析 JSON 时出错: {e}")
        return None
    except FileNotFoundError:
        print(f"文件未找到: {filename}")
        return None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None

def get_embeddings(texts, model="text-embedding-3-small"):
    """
    Embedding：https://platform.openai.com/docs/guides/embeddings
    Embedding model：https://platform.openai.com/docs/models/embeddings
    Embedding API：https://platform.openai.com/docs/api-reference/embeddings

    Creates an embedding vector representing the input text.
    创建表示输入文本的嵌入向量。

    Args:
        texts: 要嵌入的输入文本，编码为字符串或token数组。

    Returns:
        嵌入对象的列表。
    """
    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    embeddings = [data.embedding for data in response.data]
    return embeddings

def create_and_save_faiss_index(embeddings, dimension=1536, index_path="test.faiss"):
    """创建FAISS索引并保存到文件。

    Args:
        embeddings (list): 嵌入向量列表。
        dimension (int): 向量的维数。
        index_path (str): 索引文件路径。

    Returns:
        faiss.Index: 创建的FAISS索引。
    """
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, index_path)
    print("成功创建并保存FAISS索引到文件:", index_path)
    return index


if __name__ == "__main__":
    # 从 JSON 文件中加载chunks
    chunks = load_data_from_json("data/chunks.json")

    # 从chunks中提取文本进行embedding
    chunks_texts = [chunk['page_content'] for chunk in chunks]
    chunks_embeddings = get_embeddings(chunks_texts)
    print(len(chunks_embeddings))

    index_path = "data/chunks_index.faiss"
    # 创建FAISS索引
    index = create_and_save_faiss_index(chunks_embeddings, index_path=index_path)