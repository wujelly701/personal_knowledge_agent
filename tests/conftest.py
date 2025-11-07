"""pytest全局配置"""
import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def temp_user_files_dir():
    """创建临时user_files目录"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def temp_vector_db_dir():
    """创建临时vector_db目录"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents():
    """提供测试文档样本"""
    return {
        "python_notes.txt": "Python装饰器是一种设计模式，可以扩展函数功能。装饰器广泛应用于日志记录、性能测试、事务处理等场景。",
        "java_notes.txt": "Java注解类似于Python装饰器，但实现机制不同。Java注解基于反射机制实现。",
        "ml_paper.md": "# 机器学习论文\n\n神经网络是深度学习的基础。卷积神经网络(CNN)特别适合处理图像数据。"
    }


@pytest.fixture
def sample_test_files(temp_user_files_dir, sample_documents):
    """创建样本测试文件"""
    file_paths = []
    for filename, content in sample_documents.items():
        file_path = temp_user_files_dir / filename
        file_path.write_text(content, encoding='utf-8')
        file_paths.append(file_path)
    return file_paths


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """设置测试环境变量"""
    # 设置测试模式
    monkeypatch.setenv("TESTING", "1")
    # 禁用日志输出（可选）
    # monkeypatch.setenv("LOG_LEVEL", "ERROR")


@pytest.fixture(scope="function", autouse=True)
def cleanup_test_data():
    """每个测试后清理ChromaDB测试数据"""
    yield
    # 测试后清理
    try:
        from src.storage.vector_store_simple import VectorStore
        vector_store = VectorStore()
        
        # 获取所有文档
        all_docs = vector_store.collection.get()
        
        if all_docs and all_docs['ids']:
            # 识别测试数据（分类为"未分类"或文件名包含test/temp）
            test_ids = []
            for i, metadata in enumerate(all_docs['metadatas']):
                filename = metadata.get('filename', '').lower()
                category = metadata.get('category', '未分类')
                
                # 删除测试文件
                if ('test' in filename or 
                    'temp' in filename or 
                    'java' in filename or 
                    'python' in filename or 
                    'cooking' in filename or
                    category == '未分类'):
                    test_ids.append(all_docs['ids'][i])
            
            # 批量删除测试数据
            if test_ids:
                vector_store.collection.delete(ids=test_ids)
                print(f"\n✓ 清理了 {len(test_ids)} 条测试数据")
    except Exception as e:
        print(f"\n警告: 清理测试数据失败: {e}")
