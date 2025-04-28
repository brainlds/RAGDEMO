from abc import ABC, abstractmethod
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

load_dotenv()

class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    @abstractmethod
    def get_completion(self, messages):
        """获取模型回复"""
        pass



class DeepSeekClient(BaseLLMClient):
    """DeepSeek API客户端"""
    client: Any = Field(default=None)
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        
    def get_completion(self, messages):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        # 打印响应
        logger.info(f"\nOpenAI响应:\n{response.choices[0].message.content}\n")
        return response.choices[0].message.content
    

class DashScopeClient(BaseLLMClient, BaseModel):
    """DashScope API客户端"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def get_completion(self, messages):
        response = self.client.chat.completions.create(
            model="dashscope/dashscope-1-dev",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        # 打印响应
        logger.info(f"\nDashScope响应:\n{response.choices[0].message.content}\n")
        return response.choices[0].message.content

def create_llm_client(provider="deepseek")->BaseLLMClient:
    """LLM客户端工厂函数"""
    try:
        logger.info(f"选择的模型提供商: {provider}")
        
        clients = {
            "deepseek": DeepSeekClient,
            "dashscope": DashScopeClient
        }
        
        client_class = clients.get(provider.lower())
        if not client_class:
            logger.error(f"不支持的模型提供商: {provider}")
            raise ValueError(f"不支持的模型提供商: {provider}")
            
        client = client_class()
        logger.info(f"成功创建客户端: {client.__class__.__name__}")
        return client
        
    except Exception as e:
        logger.error(f"创建LLM客户端失败: {str(e)}")
        raise 