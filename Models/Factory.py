import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOllama

_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings


class ChatModelFactory:
    model_params = {
        "temperature": 0,
        "seed": 42,
    }

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if model_name.startswith("ollama:"):
            ollama_model = model_name.split(":")[1]
            return ChatOllama(
                model=ollama_model,
                temperature=cls.model_params["temperature"],
            )
        elif "gpt" in model_name or "o1" in model_name:
            if not use_azure:
                return ChatOpenAI(model=model_name, **cls.model_params)
            else:
                return AzureChatOpenAI(
                    azure_deployment=model_name,
                    api_version="2024-05-01-preview",
                    **cls.model_params
                )
        elif model_name.startswith("qwen"):
            return ChatOpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=model_name,  # qwen-turbo, qwen-plus, qwen-max
                **cls.model_params
            )
        elif model_name == "deepseek":
            # 换成开源模型试试
            # https://siliconflow.cn/
            # 一个 Model-as-a-Service 平台
            # 可以通过与 OpenAI API 兼容的方式调用各种开源语言模型。
            return ChatOpenAI(
                model="deepseek-ai/DeepSeek-V2.5",  # 模型名称
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),  # 在平台注册账号后获取
                openai_api_base="https://api.siliconflow.cn/v1",  # 平台 API 地址
                **cls.model_params,
            )
        elif model_name == "deepseek-chat":
            # Official DeepSeek API implementation
            return ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                **cls.model_params,
            )

    @classmethod
    def get_default_model(cls):
        return cls.get_model("ollama:qwen2.5")


class EmbeddingModelFactory:

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if model_name.startswith("text-embedding"):
            if not use_azure:
                return OpenAIEmbeddings(model=model_name)
            else:
                return AzureOpenAIEmbeddings(
                    azure_deployment=model_name,
                    openai_api_version="2024-05-01-preview",
                )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("text-embedding-ada-002")
