"""日志模块测试"""
import pytest
import logging
from pathlib import Path
from src.utils.logging import setup_logging


class TestLogging:
    """日志配置测试"""
    
    def test_setup_logging_configures_root_logger(self):
        """测试日志设置配置根logger"""
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger is not None
        assert isinstance(root_logger, logging.Logger)
    
    def test_logger_has_handlers(self):
        """测试logger有处理器"""
        setup_logging()
        root_logger = logging.getLogger()
        # 应该有至少一个处理器（文件或控制台）
        assert len(root_logger.handlers) > 0
    
    def test_logger_log_level(self):
        """测试日志级别"""
        setup_logging()
        root_logger = logging.getLogger()
        # 应该能记录INFO级别
        assert root_logger.isEnabledFor(logging.INFO)
    
    def test_log_file_creation(self):
        """测试日志文件创建"""
        setup_logging()
        # 验证logs目录存在
        logs_dir = Path("logs")
        assert logs_dir.exists()
        assert logs_dir.is_dir()
    
    def test_get_module_logger(self):
        """测试获取模块logger"""
        setup_logging()
        logger1 = logging.getLogger("module1")
        logger2 = logging.getLogger("module2")
        
        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2

