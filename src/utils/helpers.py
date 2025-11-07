"""
工具函数模块
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def generate_document_id(content: str, filename: str) -> str:
    """生成文档唯一ID"""
    # 结合内容和文件名生成唯一hash
    combined = f"{filename}:{content}"
    hash_obj = hashlib.md5(combined.encode('utf-8'))
    return hash_obj.hexdigest()

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f}{size_names[i]}"

def truncate_text(text: str, max_length: int = 200) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """安全的JSON解析"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """安全的JSON序列化"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default

def measure_time(func):
    """性能测量装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"{func.__name__} 耗时: {duration:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"{func.__name__} 失败，耗时: {duration:.2f}秒, 错误: {str(e)}")
            raise
    return wrapper

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """提取关键词（简单实现）"""
    # 停用词列表
    stop_words = {
        "的", "了", "是", "在", "有", "和", "与", "或", "但", "而", 
        "以及", "对于", "关于", "这个", "那个", "这些", "那些",
        "一个", "一种", "一些", "可以", "需要", "应该", "必须",
        "因为", "所以", "如果", "虽然", "然而", "但是", "不过",
        "我们", "你们", "他们", "她们", "它们", "自己", "本人",
        "很", "非常", "比较", "最", "更", "还", "也", "都", "又",
        "上", "下", "中", "内", "外", "前", "后", "左", "右"
    }
    
    # 分词（简单实现）
    words = []
    for word in text.replace("\n", " ").split():
        word = word.strip('，。！？；：""''()（）[]【】{}《》-—')
        if len(word) > 1 and word not in stop_words and word.isalnum():
            words.append(word)
    
    # 统计词频
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序并返回前N个
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]

def validate_file_path(file_path: str) -> bool:
    """验证文件路径"""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    # 移除或替换不安全的字符
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # 限制长度
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    
    return filename

def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def create_response_dict(status: str, message: str, data: Any = None, **kwargs) -> Dict[str, Any]:
    """创建标准响应字典"""
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    
    # 添加额外字段
    response.update(kwargs)
    
    return response

def batch_process(items: List[Any], batch_size: int, process_func, *args, **kwargs) -> List[Any]:
    """批处理"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            batch_result = process_func(batch, *args, **kwargs)
            results.append(batch_result)
        except Exception as e:
            logger.error(f"批处理失败，批次 {i//batch_size}: {str(e)}")
            results.append(None)
    return results

class ErrorHandler:
    """错误处理器"""
    
    @staticmethod
    def handle_api_error(func):
        """API错误处理装饰器"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"API调用失败 {func.__name__}: {str(e)}")
                return create_response_dict(
                    status="error",
                    message=f"API调用失败: {str(e)}",
                    error=str(e)
                )
        return wrapper
    
    @staticmethod
    def handle_file_error(func):
        """文件操作错误处理装饰器"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError:
                logger.error(f"文件未找到 {func.__name__}")
                return create_response_dict(
                    status="error",
                    message="文件未找到"
                )
            except PermissionError:
                logger.error(f"文件权限错误 {func.__name__}")
                return create_response_dict(
                    status="error",
                    message="文件权限不足"
                )
            except Exception as e:
                logger.error(f"文件操作失败 {func.__name__}: {str(e)}")
                return create_response_dict(
                    status="error",
                    message=f"文件操作失败: {str(e)}"
                )
        return wrapper
