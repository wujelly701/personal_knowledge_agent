"""Embedding缓存测试"""
import pytest
import time
from src.storage.embedding_manager import EmbeddingManager


class TestEmbeddingCache:
    """Embedding缓存功能测试"""
    
    def test_cache_hit_performance(self):
        """测试缓存命中性能提升"""
        manager = EmbeddingManager("text-hash")
        
        # 第一次查询（无缓存）
        query_text = "测试文本内容用于验证embedding缓存功能"
        start = time.time()
        emb1 = manager.embed_query(query_text)
        time_no_cache = time.time() - start
        
        # 第二次查询（有缓存）
        start = time.time()
        emb2 = manager.embed_query(query_text)
        time_with_cache = time.time() - start
        
        # 验证
        assert emb1 == emb2, "缓存的embedding应与原始相同"
        assert time_with_cache < time_no_cache, \
            f"缓存查询({time_with_cache:.4f}s)应快于首次查询({time_no_cache:.4f}s)"
        
        # 缓存应该至少快10倍
        assert time_with_cache < time_no_cache / 10, \
            "缓存查询应显著快于首次查询"
    
    def test_cache_different_queries(self):
        """测试不同查询生成不同embedding"""
        manager = EmbeddingManager("text-hash")
        
        emb1 = manager.embed_query("查询文本1")
        emb2 = manager.embed_query("查询文本2")
        
        assert emb1 != emb2, "不同查询应生成不同embedding"
        assert len(emb1) == len(emb2), "embedding维度应该相同"
    
    def test_cache_persistence_across_instances(self):
        """测试缓存在实例间是否保持（Singleton测试）"""
        manager1 = EmbeddingManager("text-hash")
        manager2 = EmbeddingManager("text-hash")
        
        # 应该是同一个实例（Singleton模式）
        assert manager1 is manager2, "EmbeddingManager应该是单例"
        
        # 在第一个实例中添加缓存
        query = "单例模式测试"
        emb1 = manager1.embed_query(query)
        
        # 第二个实例应该能访问相同的缓存
        emb2 = manager2.embed_query(query)
        assert emb1 == emb2, "单例实例应共享缓存"
    
    def test_cache_content(self):
        """测试缓存内容是否正确存储"""
        manager = EmbeddingManager("text-hash")
        
        # 清空可能存在的缓存
        if hasattr(manager, 'embedding_cache'):
            manager.embedding_cache.clear()
        
        # 添加几个查询
        queries = ["查询A", "查询B", "查询C"]
        embeddings = []
        
        for query in queries:
            emb = manager.embed_query(query)
            embeddings.append(emb)
        
        # 验证缓存大小
        if hasattr(manager, 'embedding_cache'):
            assert len(manager.embedding_cache) >= len(queries), \
                "缓存应包含所有查询"
    
    def test_cache_with_same_content_different_object(self):
        """测试相同内容的不同字符串对象"""
        manager = EmbeddingManager("text-hash")
        
        text1 = "相同的文本内容"
        text2 = "相同" + "的文本" + "内容"  # 不同对象但内容相同
        
        start1 = time.time()
        emb1 = manager.embed_query(text1)
        time1 = time.time() - start1
        
        start2 = time.time()
        emb2 = manager.embed_query(text2)
        time2 = time.time() - start2
        
        # 应该使用相同的缓存
        assert emb1 == emb2, "相同内容应返回相同embedding"
        assert time2 < time1, "第二次查询应使用缓存"
    
    def test_embedding_dimensions(self):
        """测试embedding维度"""
        manager = EmbeddingManager("text-hash")
        
        emb = manager.embed_query("测试embedding维度")
        
        assert isinstance(emb, list), "Embedding应该是列表类型"
        assert len(emb) > 0, "Embedding不应为空"
        assert all(isinstance(x, (int, float)) for x in emb), \
            "Embedding所有元素应为数值类型"
    
    def test_empty_query(self):
        """测试空查询"""
        manager = EmbeddingManager("text-hash")
        
        # 空字符串应该也能生成embedding
        emb = manager.embed_query("")
        assert isinstance(emb, list), "空查询也应返回embedding"
        assert len(emb) > 0, "空查询的embedding不应为空"
    
    def test_batch_embedding(self):
        """测试批量embedding"""
        manager = EmbeddingManager("text-hash")
        
        texts = ["文本1", "文本2", "文本3", "文本4", "文本5"]
        
        # 测试embed_documents方法
        if hasattr(manager, 'embed_documents'):
            embeddings = manager.embed_documents(texts)
            
            assert len(embeddings) == len(texts), \
                "批量embedding应返回相同数量的结果"
            
            # 验证每个embedding的维度相同
            dims = [len(emb) for emb in embeddings]
            assert len(set(dims)) == 1, "所有embedding维度应相同"
