"""YOLO 检测服务主入口"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.logging_config import setup_logging, get_logger
from services.detection import router as detection_router
from services.model_manager import get_model_manager

    setup_logging(
    log_level=settings.LOG_LEVEL,
    log_dir=settings.LOG_DIR,
    app_name="multi-yolo-service",
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    try:
        logger.info("正在初始化 YOLO 检测服务...")
        model_manager = get_model_manager()
        models = model_manager.list_models()
        
        if models:
            logger.info(f"✓ 服务初始化成功，已加载 {len(models)} 个模型")
        else:
            logger.warning("⚠ 服务初始化完成，但未加载任何模型")
    except Exception as e:
        logger.error(f"✗ YOLO 检测服务初始化失败: {str(e)}")
        raise
    
    yield
    
    logger.info("YOLO 检测服务正在关闭...")


app = FastAPI(
    title="MultiYOLO Service",
    description="通用 YOLO 多模型检测服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router, prefix="/api/v1", tags=["detection"])


@app.get("/")
async def root():
    """返回服务信息"""
    model_manager = get_model_manager()
    models = model_manager.list_models()
    return {
        "service": "MultiYOLO Service",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": len(models),
        "models": models,
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "multi-yolo-service"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        limit_concurrency=1000,
        timeout_keep_alive=5,
    )
