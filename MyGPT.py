from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

class MyGPT:
    def __init__(self):
        self.client = OpenAI()

    def get_response(
            self,
            messages,
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.7,
            stream=False,
    ):
        """
        Chat API：https://platform.openai.com/docs/api-reference/chat

        Creates a model response for the given chat conversation.
        为给定的聊天对话创建模型响应。

        Args:
            messages: 到目前为止，构成对话的消息列表。
            model: 要使用的模型的 ID。
            max_tokens: 聊天完成时可以生成的最大token数。
            temperature: 使用什么采样温度，介于 0 和 2 之间。
                较高的值（如 0.8）将使输出更加随机，而较低的值（如 0.2）将使其更具集中性和确定性。
            stream: 是否流式输出。

        Returns:
            如果请求是流式处理的，则返回chat completion chunk（聊天完成区块对象）的流序列。
            如果请求是非流式的，只返回响应文本。
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            return "无效的 'messages' 类型。它应该是一个字符串或消息列表。"

        completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
        )

        if stream:
            # 流式输出
            return completion

        # 非流式输出
        logger.debug(completion.choices[0].message.content)
        logger.info(f"总token数: {completion.usage.total_tokens}")
        return completion.choices[0].message.content

    def generate_image(self, prompt, model="dall-e-3", quality="standard", size="1024x1024", style="vivid", n=1, response_format="url"):
        """
        Images API: https://platform.openai.com/docs/api-reference/images

        Creates an image given a prompt.
        在给定提示的情况下创建图像。

        Args:
            prompt: 所需图像的文本描述。dall-e-2 的最大长度为 1000 个字符，dall-e-3 的最大长度为 4000 个字符。
            model: 用于图像生成的模型。
            quality: 将生成的图像的质量。hd 创建具有更精细细节和更高一致性的图像。此参数仅支持 dall-e-3。
            size: 生成图像的大小。
                对于 dall-e-2，必须是 256x256、512x512 或 1024x1024 之一。
                对于 dall-e-3 模型，必须是 1024x1024、1792x1024 或 1024x1792 之一。
            style: 生成图像的风格。必须是生动或自然。生动 "使模型偏向于生成超真实和戏剧化的图像。自然 "会使模型生成更自然、不那么超真实的图像。此参数仅支持 dall-e-3。
            n: 要生成的图像数。必须介于 1 和 10 之间。对于 dall-e-3，仅 n=1 受支持。
            response_format: 返回生成的图像的格式。必须是 url 或 b64_json 之一。URL 仅在图像生成后的 60 分钟内有效。

        Returns:
            返回图像对象的列表。
        """
        response = self.client.images.generate(
            prompt=prompt,
            model=model,
            quality=quality,
            size=size,
            style=style,
            n=n,
            response_format=response_format,
        )
        logger.info(f"优化后的prompt: {response.data[0].revised_prompt}")
        return response

    def get_embeddings(self, texts, model="text-embedding-3-small"):
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
        response = self.client.embeddings.create(
            input=texts,
            model=model,
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings

if __name__ == "__main__":
    # 测试
    mygpt = MyGPT()

    # 对话
    # prompt
    prompt = '你好'
    response = mygpt.get_response(prompt, temperature=1)
    print(response)
    #
    # # messages
    # messages = [
    #     {'role': 'user', 'content': '什么是大模型'},
    # ]
    # response = mygpt.get_completion(messages, temperature=1)
    # print(response)

    # # 图像生成
    # prompt = 'cartoon of 蓝天白云，动物园'
    # response = mygpt.generate_image(prompt)
    # print(response)

    # # 嵌入
    # vectors = mygpt.get_embeddings("hello world")
    # print(len(vectors), len(vectors[0]))
    # # 1 1536
    #
    # vectors = mygpt.get_embeddings(["input text 1", "input text 2"])
    # print(len(vectors), len(vectors[0]))
    # # 2 1536

    # # 计算流式输出的 tokens
    # prompt = '你好'
    # response = mygpt.get_response(prompt, stream=True)
    # bot_response = ""
    # for character in response:
    #     character_content = character.choices[0].delta.content
    #     if character_content is not None:
    #         bot_response += character_content
    #         # print(character_content)
    #         # print(result)
    #     else:
    #         from file_processor_helper import FileProcessorHelper
    #         prompt_tokens = FileProcessorHelper.tiktoken_len(prompt)
    #         completion_tokens = FileProcessorHelper.tiktoken_len(bot_response)
    #         total_tokens = prompt_tokens + completion_tokens
    #         logger.success(f"流式输出 | bot_response: {bot_response}")
    #         logger.success(f"流式输出 | total_tokens: {total_tokens} "
    #                        f"= prompt_tokens:{prompt_tokens} + completion_tokens: {completion_tokens}")
