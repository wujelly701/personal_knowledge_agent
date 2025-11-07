#!/usr/bin/env python3
"""
测试分类器修复
验证 metadata 不再包含嵌套字典
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.documents import Document as LangChainDocument
from src.ingestion.document_loader_simple import DocumentClassifier

def test_classifier():
    """测试分类器"""
    print("🧪 测试文档分类器\n")
    
    # 创建分类器
    classifier = DocumentClassifier()
    
    # 测试文档
    test_cases = [
        {
            "content": "这是一份工作计划，包含项目任务和会议安排",
            "filename": "work_plan.md",
            "expected_category": "工作"
        },
        {
            "content": "Python 学习笔记：变量、函数、类的使用方法和练习",
            "filename": "python_learning_notes.md",
            "expected_category": "学习"
        },
        {
            "content": "日程安排：早上工作，下午学习，晚上运动",
            "filename": "daily_schedule.md",
            "expected_category": "个人"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"测试 {i}: {test['filename']}")
        print(f"内容: {test['content'][:50]}...")
        
        # 创建文档
        doc = LangChainDocument(
            page_content=test['content'],
            metadata={"filename": test['filename']}
        )
        
        # 分类
        result = classifier.classify_document(doc)
        
        # 验证结果
        print(f"✅ 分类结果:")
        print(f"   类别: {result['category']} (预期: {test['expected_category']})")
        print(f"   优先级: {result['priority']}")
        print(f"   置信度: {result['confidence']}")
        print(f"   得分: {result['classification_scores']}")
        
        # 检查 metadata 类型
        print(f"\n🔍 Metadata 类型检查:")
        for key, value in result.items():
            value_type = type(value).__name__
            is_valid = value_type in ['str', 'int', 'float', 'bool', 'NoneType']
            status = "✅" if is_valid else "❌"
            print(f"   {status} {key}: {value_type}")
            
            if not is_valid:
                print(f"      ⚠️  值: {value}")
        
        print("-" * 60 + "\n")
    
    print("✅ 所有测试完成！")

if __name__ == "__main__":
    test_classifier()
