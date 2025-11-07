"""
多种Embedding方案管理器
支持多种免费的embedding模型，无需API密钥
"""

import logging
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """多方案Embedding管理器 - 使用单例模式"""
    
    _instance = None  # 单例实例
    _initialized = False  # 是否已初始化
    
    def __new__(cls, preferred_method: str = "all-MiniLM-L6-v2"):
        """单例模式：确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, preferred_method: str = "all-MiniLM-L6-v2"):
        """
        初始化Embedding管理器
        
        Args:
            preferred_method: 首选embedding方法
            - "all-MiniLM-L6-v2": Sentence Transformers (免费)
            - "text-hash": 文本哈希 (最简单)
            - "sentence-transformers-local": 本地Sentence Transformers
            - "huggingface-embeddings": Hugging Face免费模型
            - "bert-base-nli-mean-tokens": BERT嵌入
        """
        # 单例模式：只初始化一次
        if EmbeddingManager._initialized:
            return
            
        self.preferred_method = preferred_method
        self.method = None
        self.model = None
        self.embedding_dim = 0
        
        # 添加缓存机制
        self.embedding_cache = {}  # 用于缓存已计算的embedding
        self.cache_enabled = True  # 缓存开关
        self.max_cache_size = 1000  # 最大缓存数量
        
        self._initialize_embedding_method()
        EmbeddingManager._initialized = True
    
    def _initialize_embedding_method(self):
        """初始化embedding方法"""
        try:
            # 尝试不同的embedding方法
            methods_tried = []
            
            # 方法1: Sentence Transformers (免费，推荐)
            if self._try_sentence_transformers():
                logger.info("✅ 使用 Sentence Transformers (all-MiniLM-L6-v2)")
                return
            
            methods_tried.append("Sentence Transformers")
            
            # 方法2: 简化文本哈希
            if self._try_text_hash():
                logger.info("✅ 使用文本哈希embedding")
                return
                
            methods_tried.append("文本哈希")
            
            # 方法3: 词袋模型
            if self._try_bag_of_words():
                logger.info("✅ 使用词袋模型embedding")
                return
                
            methods_tried.append("词袋模型")
            
            logger.warning(f"所有embedding方法尝试失败: {methods_tried}")
            logger.info("🔄 默认使用384维文本哈希方法")
            self.method = "text-hash"
            self.embedding_dim = 384  # 升级到384维，匹配Sentence Transformers
            
        except Exception as e:
            logger.error(f"Embedding初始化失败: {str(e)}")
            # 确保至少有一个工作方法
            self.method = "text-hash"
            self.embedding_dim = 384  # 升级到384维，匹配Sentence Transformers
    
    def _try_sentence_transformers(self) -> bool:
        """尝试Sentence Transformers"""
        try:
            import urllib.request
            import urllib.error
            
            # 快速检查网络连接 - 只尝试一次
            try:
                urllib.request.urlopen("https://huggingface.co", timeout=5)
            except (urllib.error.URLError, OSError):
                logger.debug("网络不可用，跳过Sentence Transformers")
                return False
            
            from sentence_transformers import SentenceTransformer
            
            # 尝试加载免费模型
            model_name = "all-MiniLM-L6-v2"  # Hugging Face免费模型
            self.model = SentenceTransformer(model_name)
            self.method = "sentence-transformers"
            self.embedding_dim = 384
            logger.info("✅ 成功加载Sentence Transformers模型")
            return True
            
        except Exception as e:
            logger.debug(f"Sentence Transformers 不可用: {e}")
            return False
    
    def _try_text_hash(self) -> bool:
        """使用文本哈希 - 384维高质量版本"""
        self.method = "text-hash"
        self.embedding_dim = 384  # 升级到384维，匹配Sentence Transformers
        return True
    
    def _try_bag_of_words(self) -> bool:
        """尝试词袋模型"""
        try:
            # 检查是否需要sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.method = "bow-tfidf"
            self.embedding_dim = 1000
            return True
            
        except Exception as e:
            logger.debug(f"BOW方法不可用: {e}")
            return False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文档embedding
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding向量列表
        """
        try:
            if self.method == "sentence-transformers":
                return self._embed_sentence_transformers(texts)
            elif self.method == "text-hash":
                return self._embed_text_hash(texts)
            elif self.method == "bow-tfidf":
                return self._embed_bow_tfidf(texts)
            else:
                # 回退到文本哈希
                return self._embed_text_hash(texts)
                
        except Exception as e:
            logger.error(f"文档embedding生成失败: {str(e)}")
            return self._embed_text_hash(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        生成查询文本embedding (带缓存)
        
        Args:
            text: 查询文本
            
        Returns:
            embedding向量
        """
        # 检查缓存
        if self.cache_enabled:
            cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            if cache_key in self.embedding_cache:
                logger.debug(f"使用缓存的embedding: {text[:30]}...")
                return self.embedding_cache[cache_key]
        
        # 生成embedding
        embeddings = self.embed_documents([text])
        result = embeddings[0] if embeddings else [0.0] * self.embedding_dim
        
        # 缓存结果
        if self.cache_enabled:
            # 如果缓存已满，清理最旧的一半
            if len(self.embedding_cache) >= self.max_cache_size:
                keys_to_remove = list(self.embedding_cache.keys())[:self.max_cache_size // 2]
                for key in keys_to_remove:
                    del self.embedding_cache[key]
                logger.debug(f"缓存已满，清理了 {len(keys_to_remove)} 个旧条目")
            
            self.embedding_cache[cache_key] = result
            logger.debug(f"缓存了新的embedding (缓存大小: {len(self.embedding_cache)})")
        
        return result
        return embeddings[0] if embeddings else [0.0] * self.embedding_dim
    
    def _embed_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """使用Sentence Transformers生成embedding"""
        try:
            return self.model.encode(texts, show_progress_bar=False).tolist()
        except Exception as e:
            logger.error(f"Sentence Transformers embedding失败: {e}")
            return self._embed_text_hash(texts)
    
    def _embed_text_hash(self, texts: List[str]) -> List[List[float]]:
        """使用文本哈希生成embedding"""
        embeddings = []
        for text in texts:
            # 生成多个哈希特征
            hash_features = []
            for i in range(self.embedding_dim):
                # 使用不同种子生成哈希
                hash_obj = hashlib.md5(f"{text}_{i}".encode())
                # 转换为0-1之间的浮点数
                hash_value = int(hash_obj.hexdigest(), 16) % 1000
                hash_features.append(hash_value / 1000.0)
            
            # 确保向量长度正确
            if len(hash_features) != self.embedding_dim:
                hash_features = hash_features[:self.embedding_dim]
                while len(hash_features) < self.embedding_dim:
                    hash_features.append(hash_features[len(hash_features) % len(hash_features)])
            
            embeddings.append(hash_features)
        
        return embeddings
    
    def _embed_bow_tfidf(self, texts: List[str]) -> List[List[float]]:
        """使用词袋模型生成embedding"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 创建或更新TF-IDF向量器
            if not hasattr(self, '_tfidf_vectorizer'):
                self._tfidf_vectorizer = TfidfVectorizer(
                    max_features=self.embedding_dim,
                    stop_words='english',
                    lowercase=True
                )
                self._fitted = False
            
            if not self._fitted:
                tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
                self._fitted = True
            else:
                tfidf_matrix = self._tfidf_vectorizer.transform(texts)
            
            # 转换为密集矩阵并标准化
            embeddings = tfidf_matrix.toarray().tolist()
            return embeddings
            
        except Exception as e:
            logger.error(f"TF-IDF embedding失败: {e}")
            return self._embed_text_hash(texts)
    
    def get_method_info(self) -> Dict[str, Any]:
        """获取当前embedding方法信息"""
        return {
            "method": self.method,
            "dimension": self.embedding_dim,
            "description": self._get_method_description(),
            "is_free": True,
            "privacy_protected": True
        }
    
    def _get_method_description(self) -> str:
        """获取方法描述"""
        descriptions = {
            "sentence-transformers": "Sentence Transformers (all-MiniLM-L6-v2) - 免费高质量语义embedding",
            "text-hash": "文本哈希嵌入 - 简单快速，基于内容哈希",
            "bow-tfidf": "词袋TF-IDF模型 - 基于词频的统计embedding",
        }
        return descriptions.get(self.method, "未知方法")
    
    @staticmethod
    def get_available_methods() -> List[Dict[str, Any]]:
        """获取所有可用的embedding方法"""
        methods = [
            {
                "name": "Sentence Transformers",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "quality": "高",
                "speed": "快",
                "is_free": True,
                "description": "Hugging Face免费模型，语义理解能力强",
                "install": "pip install sentence-transformers"
            },
            {
                "name": "文本哈希",
                "model": "custom",
                "dimension": 384,
                "quality": "中等",
                "speed": "极快",
                "is_free": True,
                "description": "基于内容哈希，零依赖",
                "install": "无需安装"
            },
            {
                "name": "TF-IDF词袋",
                "model": "sklearn",
                "dimension": 1000,
                "quality": "中等",
                "speed": "快",
                "is_free": True,
                "description": "基于词频统计，适合关键词搜索",
                "install": "pip install scikit-learn"
            }
        ]
        return methods
    
    def clear_cache(self):
        """清空embedding缓存"""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"已清空embedding缓存 (清理了 {cache_size} 个条目)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": "N/A"  # 可以后续添加命中率统计
        }
    
    def set_cache_enabled(self, enabled: bool):
        """设置缓存开关"""
        self.cache_enabled = enabled
        logger.info(f"Embedding缓存{'已启用' if enabled else '已禁用'}")