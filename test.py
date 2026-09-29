# 上传
from utils import upload_files

file_path = "data/translated_article.pdf"
result = upload_files(file_path)
print(result)


# 问答
from utils import build_chat_document_prompt
from MyGPT import MyGPT

uploaded_file_paths = ['C:\\Users\\92047\\AppData\\Local\\Temp\\gradio\\841cac71a4b836b9d45911b4d495e1a8499c2f83\\translated_article.pdf']
user_input = "这篇文章讲了什么？"
chat_history = [['这篇文章讲了什么？', None]]
top_n = 2
chat_document_prompt = build_chat_document_prompt(
            uploaded_file_paths, user_input, chat_history, top_n)

mygpt = MyGPT()
response = mygpt.get_response(chat_document_prompt)
print(response)