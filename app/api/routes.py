"""
API路由定义
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import logging
from app.rag.rag import RAG, Document

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()

# 创建RAG实例
rag = RAG()

# 定义API模型
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class DocumentResponse(BaseModel):
    title: str
    author: Optional[str] = None
    content: str
    similarity: float

class QueryResponse(BaseModel):
    query: str
    documents: List[DocumentResponse]
    answer: str

@router.get("/")
def root():
    """
    根路径，返回欢迎信息
    """
    return {"message": "test"}

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    根据查询文本，返回相关文档和生成的答案
    """
    try:
        logger.info(f"收到查询请求: {request.query}")
        
        # 调用RAG进行查询
        result = rag.query(query=request.query, top_k=request.top_k)
        
        
        response = QueryResponse(
            query=result["query"],
            documents=[
                DocumentResponse(
                    title=doc.title,
                    author=doc.author,
                    content=doc.content,
                    similarity=doc.similarity
                ) for doc in result["documents"]
            ],
            answer=result["answer"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"处理查询时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
