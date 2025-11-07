#!/usr/bin/env python3
"""
简单的向量数据库清理工具
不需要导入 chromadb，直接操作文件系统

使用方法：
    python simple_cleanup.py info   # 查看数据库信息
    python simple_cleanup.py clean  # 清理所有旧文件夹（危险！）
"""

import sys
import shutil
from pathlib import Path

def get_db_path():
    """获取数据库路径"""
    db_path = Path("data/vector_db")
    if not db_path.exists():
        print(f"❌ 数据库路径不存在: {db_path}")
        return None
    return db_path

def get_folder_size(folder_path):
    """计算文件夹大小（MB）"""
    total_size = 0
    try:
        for item in folder_path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
    except Exception as e:
        print(f"  ⚠️ 计算大小失败: {e}")
    return total_size / (1024 * 1024)  # 转换为 MB

def list_db_info():
    """显示数据库信息"""
    db_path = get_db_path()
    if not db_path:
        return
    
    # 获取所有 UUID 文件夹
    uuid_folders = [f for f in db_path.iterdir() if f.is_dir() and len(f.name) == 36]
    
    print(f"\n📊 向量数据库信息")
    print(f"=" * 60)
    print(f"数据库路径: {db_path.absolute()}")
    print(f"UUID 文件夹数量: {len(uuid_folders)}")
    
    if not uuid_folders:
        print("\n✅ 数据库是干净的，没有旧文件夹")
        return
    
    print(f"\n📁 文件夹详情:")
    print(f"{'序号':<4} {'文件夹名':<38} {'大小 (MB)':<12} {'文件数':<8}")
    print("-" * 60)
    
    total_size = 0
    for i, folder in enumerate(uuid_folders, 1):
        size_mb = get_folder_size(folder)
        file_count = sum(1 for _ in folder.rglob('*') if _.is_file())
        total_size += size_mb
        
        print(f"{i:<4} {folder.name:<38} {size_mb:>10.2f}  {file_count:>6}")
    
    print("-" * 60)
    print(f"{'总计':<42} {total_size:>10.2f} MB")
    
    # 检查 chroma.sqlite3
    sqlite_file = db_path / "chroma.sqlite3"
    if sqlite_file.exists():
        sqlite_size = sqlite_file.stat().st_size / (1024 * 1024)
        print(f"\n💾 数据库文件: chroma.sqlite3 ({sqlite_size:.2f} MB)")
    
    print(f"\n💡 提示:")
    print(f"  • 通常只需要1个 UUID 文件夹（当前集合）")
    print(f"  • 多余的文件夹是旧版本，可以安全删除")
    print(f"  • 运行 'python simple_cleanup.py clean' 可清理所有文件夹")

def clean_db():
    """清理数据库（删除所有 UUID 文件夹和数据库文件）"""
    db_path = get_db_path()
    if not db_path:
        return
    
    uuid_folders = [f for f in db_path.iterdir() if f.is_dir() and len(f.name) == 36]
    sqlite_file = db_path / "chroma.sqlite3"
    
    if not uuid_folders and not sqlite_file.exists():
        print("\n✅ 数据库已经是干净的")
        return
    
    print(f"\n⚠️  警告: 此操作将删除以下内容:")
    print(f"  • {len(uuid_folders)} 个 UUID 文件夹")
    if sqlite_file.exists():
        print(f"  • chroma.sqlite3 数据库文件")
    print(f"\n❌ 所有向量数据将被永久删除！")
    print(f"🔄 删除后需要重新上传所有文档")
    
    confirm = input(f"\n确认要清理吗？输入 'DELETE' 确认: ")
    if confirm != "DELETE":
        print("❌ 已取消清理")
        return
    
    # 删除所有 UUID 文件夹
    deleted_count = 0
    for folder in uuid_folders:
        try:
            shutil.rmtree(folder)
            print(f"✅ 已删除: {folder.name}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ 删除失败 {folder.name}: {e}")
    
    # 删除 SQLite 文件
    if sqlite_file.exists():
        try:
            sqlite_file.unlink()
            print(f"✅ 已删除: chroma.sqlite3")
        except Exception as e:
            print(f"❌ 删除失败 chroma.sqlite3: {e}")
    
    print(f"\n✅ 清理完成！")
    print(f"  删除了 {deleted_count} 个文件夹")
    print(f"\n💡 下次启动应用时，系统会自动创建新的空数据库")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "info":
        list_db_info()
    elif command == "clean":
        clean_db()
    else:
        print(f"❌ 未知命令: {command}")
        print(__doc__)

if __name__ == "__main__":
    main()
