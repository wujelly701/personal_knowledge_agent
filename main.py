#!/usr/bin/env python3
"""
个人知识管理Agent系统
主启动脚本
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 创建必要的目录
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

from config.settings import settings
from src.api.gradio_app import create_app
from src.utils.logging import setup_logging

# 配置日志
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        # 创建必要的目录
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info(f"启动 {settings.APP_NAME} v{settings.APP_VERSION}")
        logger.info(f"配置: DEBUG={settings.DEBUG}, LOG_LEVEL={settings.LOG_LEVEL}")
        
        # 验证配置
        logger.info("验证配置...")
        settings.validate()
        
        # 创建应用
        logger.info("初始化应用...")
        app = create_app()
        
        # 启动服务器
        logger.info(f"启动服务器: http://{settings.GRADIO_SERVER_HOST}:{settings.GRADIO_SERVER_PORT}")
        logger.info("按 Ctrl+C 停止服务器")
        
        app.launch(
            server_name=settings.GRADIO_SERVER_HOST,
            server_port=settings.GRADIO_SERVER_PORT,
            share=settings.GRADIO_SHARE,
            debug=settings.GRADIO_DEBUG,
            show_error=True,
            quiet=False,
            inbrowser=True
        )
        
    except KeyboardInterrupt:
        logger.info("用户中断，停止服务器")
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        logger.error("完整的错误堆栈:")
        logger.error(traceback.format_exc())  # 这行会显示错误发生的具体位置
        sys.exit(1)

if __name__ == "__main__":
    main()
