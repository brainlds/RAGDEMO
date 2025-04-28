"""
FastAPI 应用程序入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import logging
from app.api import api_router
from app.scheduler.tasks import scheduler_manager
import sys

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 设置为DEBUG级别以获取更详细的日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler('debug.log')    # 同时输出到文件
    ]
)
logger = logging.getLogger(__name__)

# 设置其他模块的日志级别
logging.getLogger('uvicorn').setLevel(logging.DEBUG)
logging.getLogger('fastapi').setLevel(logging.DEBUG)

app = FastAPI(
    title="RAG Demo API",
    description="基于FastAPI的检索增强生成(RAG)系统",
    version="1.0.0",
    debug=True  # 启用FastAPI的调试模式
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 包含API路由
app.include_router(api_router)

# 初始化定时任务调度器
scheduler_manager.init_app(app)

if __name__ == "__main__":
    # 使用uvicorn运行应用，启用调试模式
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 启用热重载
        log_level="debug",  # 设置uvicorn的日志级别为debug
        workers=1  # 使用单工作进程，便于调试
    )
