"""BM25关键词搜索测试"""
import pytest
from pathlib import Path
from langchain_core.documents import Document as LangChainDocument
from src.storage.vector_store_simple import VectorStore, HybridRetriever


class TestBM25Search:
    """BM25搜索功能测试"""
    
    def test_keyword_search_basic(self, sample_documents):
        """测试基本关键词搜索"""
        # 1. 初始化向量库
        vector_store = VectorStore()
        
        # 2. 添加测试文档
        docs = [
            LangChainDocument(
                page_content=content,
                metadata={"filename": filename, "file_type": filename.split('.')[-1]}
            )
            for filename, content in sample_documents.items()
        ]
        vector_store.add_documents(docs)
        
        # 3. 执行BM25搜索
        retriever = HybridRetriever(vector_store)
        results = retriever._keyword_search("装饰器", k=3, filter_dict=None)
        
        # 4. 验证结果
        assert len(results) > 0, "应该返回搜索结果"
        assert all('bm25_score' in r.metadata for r in results), "结果应包含BM25分数"
        
        # 验证排序
        if len(results) > 1:
            assert results[0].metadata['bm25_score'] >= results[-1].metadata['bm25_score'], \
                "结果应按BM25分数降序排列"
    
    def test_keyword_search_with_filter(self):
        """测试带元数据过滤的关键词搜索"""
        vector_store = VectorStore()
        
        # 添加带分类的文档
        docs = [
            LangChainDocument(
                page_content="Python教程内容",
                metadata={"filename": "python.txt", "category": "编程"}
            ),
            LangChainDocument(
                page_content="烹饪食谱教程",
                metadata={"filename": "cooking.txt", "category": "生活"}
            ),
            LangChainDocument(
                page_content="Java编程教程",
                metadata={"filename": "java.txt", "category": "编程"}
            )
        ]
        vector_store.add_documents(docs)
        
        # 仅搜索"编程"分类
        retriever = HybridRetriever(vector_store)
        results = retriever._keyword_search("教程", k=5, filter_dict={"category": "编程"})
        
        # 验证
        assert len(results) > 0, "应该返回结果"
        for result in results:
            assert result.metadata.get('category') == "编程", "所有结果应该属于编程分类"

    
    def test_keyword_search_empty_query(self):
        """测试空查询"""
        vector_store = VectorStore()
        retriever = HybridRetriever(vector_store)
        results = retriever._keyword_search("", k=5, filter_dict=None)
        assert len(results) == 0, "空查询应返回空结果"
    
    def test_keyword_search_no_results(self, sample_documents):
        """测试无匹配结果的查询"""
        vector_store = VectorStore()
        
        # 添加文档
        docs = [
            LangChainDocument(page_content=content, metadata={"filename": filename})
            for filename, content in sample_documents.items()
        ]
        vector_store.add_documents(docs)
        
        # 搜索不存在的关键词
        retriever = HybridRetriever(vector_store)
        results = retriever._keyword_search("区块链技术", k=5, filter_dict=None)
        
        # BM25可能返回低分结果
        if len(results) > 0:
            assert results[0].metadata['bm25_score'] < 1.0
    
    def test_keyword_search_chinese_text(self):
        """测试中文文本搜索"""
        vector_store = VectorStore()
        
        # 添加中文文档
        docs = [
            LangChainDocument(
                page_content="机器学习是人工智能的一个分支",
                metadata={"filename": "ml_intro.txt"}
            ),
            LangChainDocument(
                page_content="深度学习是机器学习的一个子集",
                metadata={"filename": "dl_intro.txt"}
            )
        ]
        vector_store.add_documents(docs)
        
        # 搜索中文关键词
        retriever = HybridRetriever(vector_store)
        results = retriever._keyword_search("机器学习", k=2, filter_dict=None)
        
        assert len(results) > 0, "应该找到中文结果"
        assert any("机器学习" in r.page_content for r in results)
