import logging
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import sys
import os
import asyncio
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.crawler.crawler_scripy import crawl_and_save_to_milvus

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SchedulerManager:
    """定时任务管理器，用于管理FastAPI应用中的定时任务"""
    
    def __init__(self):
        """初始化调度器"""
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        self.last_run_results = {}
    
    def init_app(self, app: FastAPI):
        """初始化FastAPI应用
        
        Args:
            app: FastAPI应用实例
        """
        # 添加启动事件
        @app.on_event("startup")
        async def startup_event():
            await self.start()
            logger.info("定时任务调度器已启动")
        
        # 添加关闭事件
        @app.on_event("shutdown")
        async def shutdown_event():
            await self.shutdown()
            logger.info("定时任务调度器已关闭")
    
    async def start(self):
        """启动调度器"""
        if not self.is_running:
            # 添加爬虫任务，每天凌晨2点执行
            self.scheduler.add_job(
                self._run_crawler_task,
                'cron',
                hour=2,  # 凌晨2点
                minute=0,  # 0分
                id='crawler_task',
                name='古诗文爬虫任务',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info("定时任务调度器已启动")
    
    async def shutdown(self):
        """关闭调度器"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("定时任务调度器已关闭")
    
    async def _run_crawler_task(self):
        """执行爬虫任务的异步包装器"""
        logger.info("开始执行古诗文爬虫定时任务")
        
        # 在事件循环中运行阻塞的爬虫任务
        loop = asyncio.get_event_loop()
        try:
            
            result = await loop.run_in_executor(
                None, 
                lambda: crawl_and_save_to_milvus()
            )
            
            # 记录执行结果
            self.last_run_results["crawler_task"] = {
                "last_run": datetime.now().isoformat(),
                "result": result
            }
            
            logger.info(f"古诗文爬虫定时任务执行完成: {result}")
            return result
        except Exception as e:
            error_msg = f"执行古诗文爬虫任务时出错: {str(e)}"
            logger.error(error_msg)
            
            # 记录错误结果
            self.last_run_results["crawler_task"] = {
                "last_run": datetime.now().isoformat(),
                "result": {"status": "error", "message": error_msg, "data": None}
            }
            
            return {"status": "error", "message": error_msg, "data": None}
    
    async def run_crawler_now(self):
        return await self._run_crawler_task()
    
    def get_task_info(self):
        """获取所有定时任务的信息"""
        tasks = []
        for job in self.scheduler.get_jobs():
            task_info = {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "last_result": self.last_run_results.get(job.id, {})
            }
            tasks.append(task_info)
        return tasks


# 创建调度器管理器实例
scheduler_manager = SchedulerManager() 