from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from app.scheduler.tasks import scheduler_manager
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/scheduler",
    tags=["scheduler"],
    responses={404: {"description": "Not found"}},
)

@router.get("/tasks")
async def get_tasks():
    """获取所有定时任务的信息"""
    tasks = scheduler_manager.get_task_info()
    return {"tasks": tasks}

@router.post("/run-crawler-now")
async def run_crawler_now(background_tasks: BackgroundTasks):
    """
    立即运行爬虫任务（异步执行）
    """
    try:
        background_tasks.add_task(scheduler_manager.run_crawler_now)
        return {
            "status": "success", 
            "message": "爬虫任务已在后台启动，请稍后查看任务状态",
        }
    except Exception as e:
        logger.error(f"启动爬虫任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动爬虫任务失败: {str(e)}")
