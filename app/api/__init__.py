"""
API包初始化
"""
from fastapi import APIRouter
from app.api.routes import router as rag_router
from app.api.scheduler import router as scheduler_router

# 创建主路由
api_router = APIRouter()

# 包含子路由
api_router.include_router(rag_router, prefix="/api", tags=["rag"])
api_router.include_router(scheduler_router, prefix="/api", tags=["scheduler"])
