import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 获取 OpenAI API 密钥
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 获取 OpenAI API 服务地址
# OPENAI_API_BASE = os.environ['OPENAI_API_BASE'] # 使用国内API服务

# 官方文档 - Models：https://platform.openai.com/docs/models
MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]
DEFAULT_MODEL = MODELS[0]
MODEL_TO_MAX_TOKENS = {
    "gpt-4o": 4096,
    "gpt-4-turbo": 4096,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
}


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DEFAULT_MAX_TOKENS = 2000