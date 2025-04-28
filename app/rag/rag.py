from app.llm import create_llm_client
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional
import requests
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# 配置日志
logger = logging.getLogger(__name__)

# Milvus配置
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")

# 阿里云DashScope配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_EMBEDDING_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

class Document(BaseModel):
    """文档模型"""
    title: str
    author: Optional[str] = None
    content: str
    similarity: float = 0.0

class DashScopeEmbedding:
    """阿里云DashScope的Embedding模型封装"""
    
    def __init__(self, api_key: str = None):
        """初始化Embedding模型
        """
        self.api_key = api_key or DASHSCOPE_API_KEY
        if not self.api_key:
            raise ValueError("DashScope API Key未设置")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_query(self, text: str) -> List[float]:
        """将查询文本转换为向量表示
        """
        try:
            payload = {
                "model": "text-embedding-v3",
                "input": {
                    "texts": [text]
                },
                "parameters": {
                    "dimension": 1024
                }
            }
            
            response = requests.post(
                DASHSCOPE_EMBEDDING_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"获取嵌入向量失败: {response.text}")
                return [0.0] * 1024  # 返回默认向量
            
            result = response.json()
            embedding = result["output"]["embeddings"][0]["embedding"]
            return embedding
        
        except Exception as e:
            logger.error(f"获取嵌入向量异常: {str(e)}")
            return [0.0] * 1024  # 返回默认向量
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """将多个文档转换为向量表示
        """
        if not documents:
            return []
        
        try:
            payload = {
                "model": "text-embedding-v3",
                "input": {
                    "texts": documents
                },
                "parameters": {
                    "dimension": 1024
                }
            }
            
            response = requests.post(
                DASHSCOPE_EMBEDDING_URL,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"获取嵌入向量失败: {response.text}")
                return [[0.0] * 1024 for _ in documents]  # 返回默认向量
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["output"]["embeddings"]]
            return embeddings
        
        except Exception as e:
            logger.error(f"获取嵌入向量异常: {str(e)}")
            return [[0.0] * 1024 for _ in documents]  # 返回默认向量

class RAG:
    """检索增强生成系统"""
    
    def __init__(self):
        """初始化RAG系统"""
        # 创建LLM客户端
        self.llm_client = create_llm_client()
        logger.info(f"成功创建LLM客户端: {self.llm_client.__class__.__name__}")
        
        # 创建嵌入模型 - 使用阿里云DashScope
        self.embeddings = DashScopeEmbedding()
        
        # 连接Milvus
        try:
            self._connect_milvus()
            logger.info("Milvus连接成功")
        except Exception as e:
            logger.error(f"Milvus连接失败: {str(e)}")
            raise ValueError("Milvus连接失败")
    
    def _connect_milvus(self):
        """连接Milvus向量数据库"""
        try:
            logger.info(f"尝试连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
            
            # 连接Milvus服务器
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            
            # 检查集合是否存在
            if not utility.has_collection(COLLECTION_NAME):
                logger.info(f"集合 {COLLECTION_NAME} 不存在，创建新集合")
                raise ValueError("Milvus集合不存在")
            
            # 加载集合
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            
            logger.info(f"成功连接到Milvus集合: {COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"Milvus连接失败: {str(e)}")
            raise ValueError("Milvus连接失败")
    
    
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """搜索相关文档"""
        try:
            if not self.collection:
                raise ValueError("Milvus连接不可用")
            
            # 获取查询的嵌入向量
            query_embedding = self.embeddings.embed_query(query)
            
            # 搜索参数
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["title", "author", "content"]
            )
            
            # 处理结果
            documents = []
            for hits in results:
                for hit in hits:
                    # 计算相似度
                    similarity = (hit.distance + 1) / 2
                    
                    doc = Document(
                        title=hit.entity.get("title", "未知标题"),
                        author=hit.entity.get("author", "未知作者"),
                        content=hit.entity.get("content", ""),
                        similarity=similarity
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"搜索文档时出错: {str(e)}")
            return [
                Document(
                    title="查询错误",
                    author="系统",
                    content=f"查询出错: {str(e)}",
                    similarity=0.0
                )
            ]
    
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """根据查询和检索到的文档生成回答
        """
        try:
            # 准备上下文
            context = "\n\n".join([
                f"标题: {doc.title}\n作者: {doc.author}\n内容: {doc.content}"
                for doc in documents
            ])
            
            logger.info(f"上下文: {context}")
            
            # 构建提示
            messages = [
                {"role": "system", "content": "你是一个智能助手，基于提供的文档内容创作诗歌。如果文档中没有相关信息，请诚实地告知用户。"},
                {"role": "user", "content": f"根据以下文档内容:\n\n{context}\n\n和问题: {query}，生成一首诗新的歌。"}
            ]

 
                    # 调用LLM生成回答
            answer = self.llm_client.get_completion(
                        messages=messages)
            return answer

        except Exception as e:
            logger.error(f"生成回答时出错: {str(e)}")
            return "抱歉，生成回答时遇到问题，请稍后再试。"
    
    def query(self, query: str, top_k: int = 3):
        """执行完整的RAG流程：检索+生成"""
        # 搜索相关文档
        documents = self.search(query, top_k)
        
        # 生成回答
        answer = self.generate_answer(query, documents)
        
        return {
            "query": query,
            "documents": documents,
            "answer": answer
        }

