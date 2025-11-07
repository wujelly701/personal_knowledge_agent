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
    
    # 使用TimedRotatingFileHandler - 每天创建新日志文件
    from logging.handlers import TimedRotatingFileHandler
    
    file_handler = TimedRotatingFileHandler(
        log_dir / "app.log",
        when='midnight',  # 每天午夜轮转
        interval=1,  # 每1天
        backupCount=7,  # 保留7天的日志
        encoding='utf-8'
    )
    file_handler.suffix = "%Y-%m-%d"  # 备份文件名格式: app.log.2025-11-08
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
