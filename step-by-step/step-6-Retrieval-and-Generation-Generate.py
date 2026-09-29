from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss
import json
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

def load_faiss_index(index_path):
    """加载FAISS索引文件"""
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        print(f"加载 FAISS 索引时出错: {e}")
        return None

def search_in_index(query_vector, index, top_k=5):
    """在FAISS索引中查找最近的向量。

    Args:
        query_vector (list): 查询向量。
        index (faiss.Index): FAISS索引。
        top_k (int): 返回的最近向量数。

    Returns:
        tuple: 距离和索引的元组。
    """
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    # 过滤掉 -1 的索引
    filtered_indices = [idx for idx in indices[0] if idx != -1]
    filtered_distances = [dist for dist, idx in zip(distances[0], indices[0]) if idx != -1]
    return filtered_distances, filtered_indices

# 变量绑定函数
def bind_variables(prompt_template, **kwargs):
    for key, value in kwargs.items():
        prompt_template = prompt_template.replace(f"{{{key}}}", value)
    return prompt_template

def get_response(prompt, model="gpt-4o", max_tokens=1000, temperature=0.7):
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    print(f"总token数: {completion.usage.total_tokens}\n")
    return completion.choices[0].message.content


if __name__ == "__main__":
    # 从 JSON 文件中加载chunks
    chunks = load_data_from_json("data/chunks.json")

    index_path = "data/chunks_index.faiss"

    # 加载FAISS索引进行搜索测试
    loaded_index = load_faiss_index(index_path)

    # 创建查询向量
    question = "什么是zero shot prompting？"
    question_embedding = get_embeddings(question)

    # 查找与查询向量最相近的向量
    distances, indices = search_in_index(question_embedding, loaded_index, top_k=2)

    # 从结果索引中获取文本和metadata信息
    print("查询结果索引：", indices)
    print("查询结果距离：", distances)

    # for idx in indices:
    #     print(f"chunk idx:", idx)
    #     print("文本：\n", chunks[idx]['page_content'])
    #     print("元数据：\n", chunks[idx]['metadata'])
    #     print("="*30)

    # 根据索引找到对应的文本
    top_k_chunks = [chunks[idx]['page_content'] for idx in indices]

    # 将 top_k_chunks 用分隔符拼接
    separator = "\n" + "=" * 30 + "\n"
    contextual_text = separator.join(top_k_chunks)
    print(contextual_text)


#     question_answering_prompt_template = """您是一个用于问答任务的助手。使用以下检索到的上下文片段来回答问题。如果您不知道答案，只需说您不知道。最多使用三个句子，并保持回答简洁。
# 问题：{question}
# 上下文：{context}
# 答案："""
    question_answering_prompt_template = """根据文章回答问题: 
文章：
'''
{article}
'''

问题：
'''
{question}
'''
输出："""
    question_answering_prompt = bind_variables(question_answering_prompt_template,
                                               question=question,
                                               article=contextual_text)
    print(question_answering_prompt)

    # 输出响应
    response = get_response(question_answering_prompt, max_tokens=1000)
    print(response)

