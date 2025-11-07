"""辅助函数测试"""
import pytest
from pathlib import Path
from src.utils.helpers import (
    generate_document_id,
    format_file_size,
    truncate_text,
    safe_json_loads,
    safe_json_dumps,
    extract_keywords,
    validate_file_path,
    sanitize_filename,
    format_duration,
    create_response_dict
)


class TestHelpers:
    """辅助函数测试"""
    
    def test_generate_document_id(self):
        """测试生成文档ID"""
        doc_id1 = generate_document_id("content1", "file.txt")
        doc_id2 = generate_document_id("content1", "file.txt")
        doc_id3 = generate_document_id("content2", "file.txt")
        
        assert doc_id1 == doc_id2  # 相同内容应该生成相同ID
        assert doc_id1 != doc_id3  # 不同内容应该生成不同ID
    
    def test_format_file_size_bytes(self):
        """测试格式化字节大小"""
        assert "500" in format_file_size(500)
        assert "B" in format_file_size(500)
    
    def test_format_file_size_kb(self):
        """测试格式化KB"""
        size_str = format_file_size(1024 * 50)  # 50KB
        assert "KB" in size_str
    
    def test_format_file_size_mb(self):
        """测试格式化MB"""
        size_str = format_file_size(1024 * 1024 * 5)  # 5MB
        assert "MB" in size_str
    
    def test_format_file_size_zero(self):
        """测试零大小"""
        assert "0B" == format_file_size(0)
    
    def test_truncate_text_short(self):
        """测试短文本不截断"""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text
    
    def test_truncate_text_long(self):
        """测试长文本截断"""
        text = "a" * 300
        result = truncate_text(text, max_length=200)
        assert len(result) == 200
        assert result.endswith("...")
    
    def test_safe_json_loads_valid(self):
        """测试安全JSON解析（有效）"""
        json_str = '{"key": "value"}'
        result = safe_json_loads(json_str)
        assert result == {"key": "value"}
    
    def test_safe_json_loads_invalid(self):
        """测试安全JSON解析（无效）"""
        json_str = "invalid json"
        result = safe_json_loads(json_str, default={})
        assert result == {}
    
    def test_safe_json_dumps_valid(self):
        """测试安全JSON序列化"""
        obj = {"key": "value", "number": 42}
        result = safe_json_dumps(obj)
        assert "key" in result
        assert "value" in result
    
    def test_extract_keywords_basic(self):
        """测试提取关键词"""
        text = "Python是一门编程语言"
        keywords = extract_keywords(text)
        assert isinstance(keywords, list)
    
    def test_extract_keywords_empty(self):
        """测试空文本"""
        keywords = extract_keywords("")
        assert isinstance(keywords, list)
        assert len(keywords) == 0
    
    def test_validate_file_path_invalid(self):
        """测试验证无效文件路径"""
        assert validate_file_path("nonexistent.txt") == False
    
    def test_sanitize_filename(self):
        """测试文件名清理"""
        dirty = "file<name>with:special/chars.txt"
        clean = sanitize_filename(dirty)
        assert "<" not in clean
        assert ">" not in clean
        assert "/" not in clean
    
    def test_format_duration_seconds(self):
        """测试格式化秒数"""
        result = format_duration(45)
        assert "45" in result or "s" in result
    
    def test_format_duration_minutes(self):
        """测试格式化分钟"""
        result = format_duration(120)
        assert "2" in result or "m" in result
    
    def test_create_response_dict_success(self):
        """测试创建响应字典（成功）"""
        response = create_response_dict("success", "操作成功", data={"result": 123})
        assert response["status"] == "success"
        assert response["message"] == "操作成功"
        assert response["data"]["result"] == 123
    
    def test_create_response_dict_error(self):
        """测试创建响应字典（错误）"""
        response = create_response_dict("error", "操作失败")
        assert response["status"] == "error"
        assert response["message"] == "操作失败"

