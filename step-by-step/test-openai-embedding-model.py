from dotenv import load_dotenv
from openai import OpenAI


# 加载环境变量
load_dotenv()

client = OpenAI()

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


# 测试函数
if __name__ == "__main__":
    # 单个文本字符串
    text = "hello world"
    embeddings = get_embeddings(text)
    if embeddings:
        print("Embedding vector for single input:", embeddings)
        print("Embedding Length:", len(embeddings[0]))
        print()

    # 文本列表
    texts = ["hello world", "hi"]
    embeddings = get_embeddings(texts)
    if embeddings:
        print("Embedding vectors for multiple inputs:")
        for i, emb in enumerate(embeddings):
            print(f"Embedding vector {i + 1}: {emb}")
            print(f"Embedding Length: {len(emb)}")