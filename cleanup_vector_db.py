#!/usr/bin/env python3
"""
清理 ChromaDB 向量数据库
删除所有旧的集合文件夹，只保留当前使用的集合

使用方法：
    python cleanup_vector_db.py info   # 查看数据库信息
    python cleanup_vector_db.py clean  # 清理数据库（危险！会删除所有数据）
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 检查依赖
try:
    import chromadb
except ImportError:
    print("❌ 错误: chromadb 未安装")
    print("\n请在项目虚拟环境中运行此脚本，或先安装依赖:")
    print("  pip install chromadb")
    print("\n提示: 请确保你在正确的 Python 环境中运行此脚本")
    sys.exit(1)

try:
    from config.settings import settings
except ImportError:
    print("❌ 错误: 无法导入 settings 模块")
    print("\n请确保:")
    print("  1. 在项目根目录下运行此脚本")
    print("  2. config/settings.py 文件存在")
    sys.exit(1)

import shutil

def cleanup_vector_db():
    """清理向量数据库"""
    
    print("⚠️  警告：此操作将删除所有向量数据！")
    print(f"数据库路径: {settings.VECTOR_DB_PATH}")
    
    confirm = input("确认要清理吗？(输入 'YES' 确认): ")
    if confirm != "YES":
        print("❌ 已取消")
        return
    
    try:
        # 1. 连接到数据库
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        
        # 2. 删除所有集合
        collections = client.list_collections()
        print(f"\n找到 {len(collections)} 个集合:")
        for col in collections:
            print(f"  - {col.name}")
            client.delete_collection(col.name)
        
        # 3. 重新创建主集合
        client.create_collection(
            name="knowledge_base",
            metadata={"description": "个人知识库向量存储"}
        )
        
        print("\n✅ 清理完成！")
        print("已重新创建空的 knowledge_base 集合")
        
    except Exception as e:
        print(f"\n❌ 清理失败: {e}")

def list_vector_db_info():
    """查看向量数据库信息"""
    try:
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        collections = client.list_collections()
        
        print(f"\n📊 向量数据库信息:")
        print(f"路径: {settings.VECTOR_DB_PATH}")
        print(f"集合数量: {len(collections)}\n")
        
        for col in collections:
            count = col.count()
            print(f"集合名称: {col.name}")
            print(f"  文档数量: {count}")
            print(f"  元数据: {col.metadata}")
            print()
            
        # 列出物理文件夹
        db_path = Path(settings.VECTOR_DB_PATH)
        folders = [f for f in db_path.iterdir() if f.is_dir()]
        print(f"物理文件夹数量: {len(folders)}")
        for folder in folders:
            size_mb = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()) / (1024*1024)
            print(f"  {folder.name}: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "info":
        list_vector_db_info()
    elif len(sys.argv) > 1 and sys.argv[1] == "clean":
        cleanup_vector_db()
    else:
        print("使用方法:")
        print("  python cleanup_vector_db.py info   # 查看数据库信息")
        print("  python cleanup_vector_db.py clean  # 清理数据库（危险！）")
