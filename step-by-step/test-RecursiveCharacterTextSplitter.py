# https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/


# This is a long document we can split up.
with open("data/translated_article.txt", encoding="utf8") as f:
    state_of_the_union = f.read()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_text(state_of_the_union)
print(texts)