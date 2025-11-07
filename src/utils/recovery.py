"""系统异常恢复工具"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class RecoveryManager:
    """异常恢复管理器
    
    用于保存操作检查点，在系统异常时可以恢复操作状态。
    
    使用示例:
        recovery = RecoveryManager()
        
        try:
            # 保存检查点
            recovery.save_checkpoint("upload_documents", {
                "files": ["file1.txt", "file2.pdf"],
                "timestamp": datetime.now().isoformat()
            })
            
            # 执行操作
            result = do_something()
            
            # 成功后清除检查点
            recovery.clear_checkpoint()
            
        except Exception as e:
            # 失败时可以恢复
            checkpoint = recovery.load_last_checkpoint()
            logger.error(f"操作失败，检查点数据: {checkpoint}")
    """
    
    def __init__(self, recovery_file: str = "data/recovery.json"):
        """初始化恢复管理器
        
        Args:
            recovery_file: 恢复文件路径
        """
        self.recovery_file = Path(recovery_file)
        self.recovery_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, operation: str, data: dict):
        """保存检查点
        
        Args:
            operation: 操作名称
            data: 操作数据（可序列化为JSON的字典）
        """
        try:
            checkpoint = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            with open(self.recovery_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
            logger.info(f"检查点已保存: {operation}")
            
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def load_last_checkpoint(self) -> Optional[Dict]:
        """加载最后的检查点
        
        Returns:
            检查点数据字典，如果不存在则返回None
        """
        try:
            if not self.recovery_file.exists():
                return None
            
            with open(self.recovery_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            logger.info(f"加载检查点: {checkpoint.get('operation', 'unknown')}")
            return checkpoint
            
        except json.JSONDecodeError as e:
            logger.error(f"检查点文件格式错误: {e}")
            return None
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None
    
    def clear_checkpoint(self):
        """清除检查点文件"""
        try:
            if self.recovery_file.exists():
                self.recovery_file.unlink()
                logger.info("检查点已清除")
        except Exception as e:
            logger.error(f"清除检查点失败: {e}")
    
    def has_checkpoint(self) -> bool:
        """检查是否存在检查点
        
        Returns:
            如果存在检查点返回True，否则返回False
        """
        return self.recovery_file.exists()
    
    def get_checkpoint_age(self) -> Optional[float]:
        """获取检查点的年龄（秒数）
        
        Returns:
            检查点创建后经过的秒数，如果不存在返回None
        """
        checkpoint = self.load_last_checkpoint()
        if not checkpoint:
            return None
        
        try:
            checkpoint_time = datetime.fromisoformat(checkpoint['timestamp'])
            age = (datetime.now() - checkpoint_time).total_seconds()
            return age
        except Exception as e:
            logger.error(f"计算检查点年龄失败: {e}")
            return None
    
    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """检查检查点是否过期
        
        Args:
            max_age_seconds: 最大年龄（秒），默认1小时
        
        Returns:
            如果检查点过期返回True，否则返回False
        """
        age = self.get_checkpoint_age()
        if age is None:
            return True
        return age > max_age_seconds
