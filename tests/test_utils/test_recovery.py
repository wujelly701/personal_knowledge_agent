"""恢复管理器测试"""
import pytest
import json
from pathlib import Path
from src.utils.recovery import RecoveryManager


class TestRecoveryManager:
    """RecoveryManager测试"""
    
    def test_save_checkpoint(self):
        """测试保存检查点"""
        manager = RecoveryManager()
        
        test_data = {
            "files": ["test1.txt", "test2.txt"],
            "status": "processing"
        }
        
        # save_checkpoint doesn't return anything
        manager.save_checkpoint("upload_files", test_data)
        
        # Verify by loading
        loaded = manager.load_last_checkpoint()
        assert loaded is not None, "检查点保存应该成功"
        assert loaded["operation"] == "upload_files"
        
        # Cleanup
        manager.clear_checkpoint()

    
    def test_load_checkpoint(self):
        """测试加载检查点"""
        manager = RecoveryManager()
        
        # 保存一个检查点
        test_data = {
            "operation": "test_op",
            "value": 42
        }
        manager.save_checkpoint("test_operation", test_data)
        
        # 加载检查点
        loaded = manager.load_last_checkpoint()
        assert loaded is not None, "应该能加载检查点"
        assert loaded["operation"] == "test_operation"
        assert loaded["data"]["value"] == 42
    
    def test_clear_checkpoint(self):
        """测试清除检查点"""
        manager = RecoveryManager()
        
        # 保存后清除
        manager.save_checkpoint("temp_op", {"test": "data"})
        manager.clear_checkpoint()  # doesn't return anything
        
        # 验证已清除
        loaded = manager.load_last_checkpoint()
        assert loaded is None, "清除后不应有检查点"

    
    def test_checkpoint_persistence(self):
        """测试检查点持久化"""
        # 第一个实例保存
        manager1 = RecoveryManager()
        manager1.save_checkpoint("persist_test", {"key": "value"})
        
        # 第二个实例加载
        manager2 = RecoveryManager()
        loaded = manager2.load_last_checkpoint()
        
        assert loaded is not None
        assert loaded["operation"] == "persist_test"
        assert loaded["data"]["key"] == "value"
        
        # 清理
        manager2.clear_checkpoint()
    
    def test_multiple_checkpoints(self):
        """测试多次保存检查点"""
        manager = RecoveryManager()
        
        # 保存多次，应该保留最新的
        manager.save_checkpoint("op1", {"step": 1})
        manager.save_checkpoint("op2", {"step": 2})
        manager.save_checkpoint("op3", {"step": 3})
        
        loaded = manager.load_last_checkpoint()
        assert loaded["operation"] == "op3"
        assert loaded["data"]["step"] == 3
        
        # 清理
        manager.clear_checkpoint()
    
    def test_checkpoint_with_complex_data(self):
        """测试复杂数据的检查点"""
        manager = RecoveryManager()
        
        complex_data = {
            "files": [
                {"name": "file1.txt", "size": 1024},
                {"name": "file2.md", "size": 2048}
            ],
            "metadata": {
                "user": "test",
                "timestamp": "2024-01-01"
            },
            "flags": [True, False, True]
        }
        
        manager.save_checkpoint("complex_op", complex_data)
        loaded = manager.load_last_checkpoint()
        
        assert loaded["data"]["files"][0]["name"] == "file1.txt"
        assert loaded["data"]["metadata"]["user"] == "test"
        assert loaded["data"]["flags"] == [True, False, True]
        
        # 清理
        manager.clear_checkpoint()
