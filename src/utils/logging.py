"""
日志配置模块
"""

import logging
import logging.handlers
import os
from pathlib import Path
from config.settings import settings

def setup_logging():
    """设置日志配置"""
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 日志格式 - 添加更详细的时间戳
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 配置根日志器
    root_logger = logging.getLogger()
    
    # 清除现有处理器，避免重复
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # 文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    root_logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """获取命名日志器"""
    return logging.getLogger(name)
