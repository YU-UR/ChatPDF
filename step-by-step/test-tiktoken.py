import tiktoken

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

if __name__ == "__main__":

    print(num_tokens_from_string("tiktoken is great!"))
