"""
YOLO 服务日志配置模块
基于 loguru 的日志系统配置
"""

import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional, Any


# 获取环境变量，确定是否为生产环境
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

# 日志格式配置
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 文件日志格式（更详细）
FILE_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message} | "
    "{extra}"
)


def setup_logging(
    *,
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    log_dir: Path = Path("./logs"),
    app_name: str = "yolo-service",
) -> None:
    """
    配置 loguru 日志系统

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        log_dir: 日志目录
        app_name: 应用名称，用于日志文件命名
    """
    # 移除默认的日志处理器
    logger.remove()

    # 控制台日志输出
    if enable_console:
        logger.add(
            sys.stdout,
            format=LOG_FORMAT,
            level=log_level,
            colorize=True,
            diagnose=not IS_PRODUCTION,
            backtrace=True,
        )

    if enable_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        # 应用主日志文件（所有级别）
        logger.add(
            log_dir / f"{app_name}.log",
            format=FILE_LOG_FORMAT,
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            diagnose=not IS_PRODUCTION,
            backtrace=True,
        )

        # 错误日志文件（只记录ERROR和CRITICAL）
        logger.add(
            log_dir / "error.log",
            format=FILE_LOG_FORMAT,
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            diagnose=not IS_PRODUCTION,
            backtrace=True,
        )


def get_logger(name: str, **extra_context) -> Any:
    """
    获取带有上下文的logger实例

    Args:
        name: 模块或类名
        **extra_context: 额外的上下文信息

    Returns:
        配置好的logger实例
    """
    return logger.bind(name=name, **extra_context)

