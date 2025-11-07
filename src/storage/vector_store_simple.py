"""
简化版向量存储模块
不使用langchain_chroma，直接使用chromadb
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document as LangChainDocument
from config.settings import settings

# 尝试导入embedding管理器
try:
    from .embedding_manager import EmbeddingManager
    EMBEDDING_MANAGER_AVAILABLE = True
except ImportError as e:
    EMBEDDING_MANAGER_AVAILABLE = False
    logging.warning(f"Embedding管理器不可用: {e}")

# 尝试导入OpenAI（可选）
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorStore:
    """向量存储管理器"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embeddings = None
        
        self._initialize_store()
    
    def _initialize_store(self):
        """初始化向量存储"""
        try:
            # 确保数据目录存在
            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            
            # 创建Chroma客户端
            self.client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH
            )
            
            # 获取或创建集合
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"使用现有集合: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "个人知识库向量存储"}
                )
                logger.info(f"创建新集合: {self.collection_name}")
            
            # 初始化embeddings系统（优先级：OpenAI > 免费方案）
            self._initialize_embeddings()
            
            logger.info(f"向量存储初始化成功: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"向量存储初始化失败: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """初始化embedding系统"""
        # 优先级1: OpenAI（如果有API密钥且可用）
        if settings.OPENAI_API_KEY and OPENAI_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                logger.info("✅ 使用OpenAI Embeddings")
                return
            except Exception as e:
                logger.warning(f"OpenAI Embeddings初始化失败: {e}")
        
        # 优先级2: 免费Embedding管理器
        if EMBEDDING_MANAGER_AVAILABLE:
            try:
                # 使用智能选择最佳embedding方法
                optimal_method = settings.get_optimal_embedding_method()
                self.embedding_manager = EmbeddingManager(optimal_method)
                self.embeddings = self.embedding_manager
                method_info = self.embedding_manager.get_method_info()
                logger.info(f"✅ 使用{'' if 'openai' not in method_info['method'] else '免费'}Embedding方案: {method_info['description']}")
                return
            except Exception as e:
                logger.warning(f"免费Embedding管理器初始化失败: {e}")

        # 降级: 简单文本哈希
        logger.info("🔄 使用简单文本哈希embedding")
        self.embeddings = None

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成embedding向量"""
        try:
            # 方法1: 使用Embedding管理器
            if hasattr(self, 'embedding_manager'):
                return self.embedding_manager.embed_documents(texts)

            # 方法2: 使用OpenAI
            elif hasattr(self, 'embeddings') and self.embeddings is not None and not isinstance(self.embeddings, type(None)):
                return self.embeddings.embed_documents(texts)

            # 降级方法3: 简单文本哈希
            else:
                logger.debug("使用简单文本哈希embedding")
                return [[hash(text) % 1000 for _ in range(3)] for text in texts]

        except Exception as e:
            logger.warning(f"embedding生成失败，使用文本哈希: {e}")
            return [[hash(text) % 1000 for _ in range(3)] for text in texts]

    def _generate_query_embedding(self, query: str) -> List[float]:
        """生成查询embedding"""
        try:
            # 方法1: 使用Embedding管理器
            if hasattr(self, 'embedding_manager'):
                return self.embedding_manager.embed_query(query)

            # 方法2: 使用OpenAI
            elif hasattr(self, 'embeddings') and self.embeddings is not None and not isinstance(self.embeddings, type(None)):
                return self.embeddings.embed_query(query)

            # 降级方法3: 简单文本哈希
            else:
                logger.debug("使用简单文本哈希查询embedding")
                return [hash(query) % 1000 for _ in range(3)]

        except Exception as e:
            logger.warning(f"查询embedding生成失败，使用文本哈希: {e}")
            return [hash(query) % 1000 for _ in range(3)]

    def add_documents(self, documents: List[LangChainDocument]) -> bool:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表

        Returns:
            是否添加成功
        """
        try:
            if not documents:
                logger.warning("没有文档需要添加")
                return False

            # 生成embedding
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"doc_{hash(doc.page_content)}_{i}" for i, doc in enumerate(documents)]

            embeddings = self._generate_embeddings(texts)

            # 添加到集合
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )

            logger.info(f"成功添加 {len(documents)} 个文档块")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[LangChainDocument]:
        """
        搜索相关文档

        Args:
            query: 搜索查询
            k: 返回文档数量
            filter_dict: 元数据过滤条件

        Returns:
            搜索结果文档列表
        """
        try:
            # 生成查询embedding
            query_embedding = self._generate_query_embedding(query)

            # 搜索
            where_clause = None
            if filter_dict:
                where_clause = {}
                for key, value in filter_dict.items():
                    where_clause[key] = {"$eq": value}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, settings.TOP_K * 2),
                where=where_clause
            )

            # 转换为LangChain文档格式
            documents = []
            if results['documents'] and results['documents'][0]:
                # 获取所有距离值用于归一化
                all_distances = results['distances'][0] if results['distances'] else [0.0] * len(results['documents'][0])
                min_distance = min(all_distances) if all_distances else 0.0
                max_distance = max(all_distances) if all_distances else 1.0

                for i, (doc_content, metadata, doc_id) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [],
                    results['ids'][0] if results['ids'] else []
                )):
                    # 改进的相似度分数计算
                    distance = all_distances[i] if i < len(all_distances) else 0.0

                    # 动态归一化：使用相对距离计算相关性
                    if max_distance > min_distance:
                        # 相对距离归一化到[0,1]范围
                        relative_distance = (distance - min_distance) / (max_distance - min_distance)
                        relevance_score = 1.0 - relative_distance
                    else:
                        # 所有距离相同，设为中等相关性
                        relevance_score = 0.5

                    # 确保分数在合理范围内，避免过度乐观
                    if distance > 2.0:  # 对于距离很远的文档
                        relevance_score = max(0.0, min(0.3, relevance_score))
                    elif distance > 1.5:  # 距离较远的文档
                        relevance_score = max(0.1, min(0.5, relevance_score))
                    elif distance < 0.3:  # 非常相似的文档
                        relevance_score = max(0.7, min(1.0, relevance_score))
                    else:  # 中等距离
                        relevance_score = max(0.2, min(0.8, relevance_score))

                    doc = LangChainDocument(
                        page_content=doc_content,
                        metadata={
                            **metadata,
                            "search_score": distance,
                            "relevance_score": round(relevance_score, 3),  # 保留3位小数
                            "doc_id": doc_id
                        }
                    )
                    documents.append(doc)

            # 如果结果少于需要的数量，尝试更多结果
            if len(documents) < k and self.collection.count() > 0:
                logger.info(f"搜索结果不足，尝试获取更多结果")
                more_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(self.collection.count(), k * 3),
                    where=where_clause
                )

                # 重新处理结果
                if more_results['documents'] and more_results['documents'][0]:
                    additional_docs = []
                    all_more_distances = more_results['distances'][0] if more_results['distances'] else [0.0] * len(more_results['documents'][0])
                    min_more_distance = min(all_more_distances) if all_more_distances else 0.0
                    max_more_distance = max(all_more_distances) if all_more_distances else 1.0

                    for i, (doc_content, metadata, doc_id) in enumerate(zip(
                        more_results['documents'][0],
                        more_results['metadatas'][0] if more_results['metadatas'] else [],
                        more_results['ids'][0] if more_results['ids'] else []
                    )):
                        # 检查是否已经存在
                        if not any(doc.page_content == doc_content for doc in documents):
                            distance = all_more_distances[i] if i < len(all_more_distances) else 0.0

                            # 使用相同的归一化逻辑
                            if max_more_distance > min_more_distance:
                                relative_distance = (distance - min_more_distance) / (max_more_distance - min_more_distance)
                                relevance_score = 1.0 - relative_distance
                            else:
                                relevance_score = 0.5

                            if distance > 2.0:
                                relevance_score = max(0.0, min(0.3, relevance_score))
                            elif distance > 1.5:
                                relevance_score = max(0.1, min(0.5, relevance_score))
                            elif distance < 0.3:
                                relevance_score = max(0.7, min(1.0, relevance_score))
                            else:
                                relevance_score = max(0.2, min(0.8, relevance_score))

                            doc = LangChainDocument(
                                page_content=doc_content,
                                metadata={
                                    **metadata,
                                    "search_score": distance,
                                    "relevance_score": round(relevance_score, 3),
                                    "doc_id": doc_id
                                }
                            )
                            additional_docs.append(doc)
                    
                    documents.extend(additional_docs)
            
            logger.info(f"搜索完成: 查询='{query[:50]}...', 结果数量={len(documents)}")
            return documents[:k]
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
    
    def delete_documents(self, filter_dict: Dict[str, Any]) -> bool:
        """
        删除文档
        
        Args:
            filter_dict: 删除条件
            
        Returns:
            是否删除成功
        """
        try:
            where_clause = {}
            for key, value in filter_dict.items():
                where_clause[key] = {"$eq": value}
            
            self.collection.delete(where=where_clause)
            logger.info(f"删除文档成功: {filter_dict}")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_documents": self.collection.count(),
                "vector_db_path": settings.VECTOR_DB_PATH
            }
            
            # 添加embedding信息
            if hasattr(self, 'embedding_manager'):
                method_info = self.embedding_manager.get_method_info()
                stats.update({
                    "embedding_method": method_info["method"],
                    "embeddings_dimension": method_info["dimension"],
                    "embedding_description": method_info["description"],
                    "is_free_embedding": method_info["is_free"],
                    "privacy_protected": method_info["privacy_protected"]
                })
            elif hasattr(self, 'embeddings') and self.embeddings is not None:
                stats.update({
                    "embedding_method": "openai",
                    "embeddings_dimension": 1536,
                    "embedding_description": "OpenAI Embeddings",
                    "is_free_embedding": False,
                    "privacy_protected": False
                })
            else:
                stats.update({
                    "embedding_method": "text-hash",
                    "embeddings_dimension": 384,
                    "embedding_description": "简单文本哈希",
                    "is_free_embedding": True,
                    "privacy_protected": True
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {}
    
    def reset(self) -> bool:
        """清空向量存储"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "个人知识库向量存储"}
            )
            logger.info("向量存储重置成功")
            return True
            
        except Exception as e:
            logger.error(f"向量存储重置失败: {str(e)}")
            return False

class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25_cache = {}
    
    def hybrid_search(self, query: str, k: int = 5, 
                     vector_weight: float = 0.7, keyword_weight: float = 0.3,
                     filter_dict: Optional[Dict] = None) -> List[LangChainDocument]:
        """
        混合检索：向量搜索 + 关键词搜索
        
        Args:
            query: 搜索查询
            k: 返回文档数量
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重
            filter_dict: 过滤条件
            
        Returns:
            混合检索结果
        """
        try:
            # 向量语义搜索
            vector_results = self.vector_store.search(
                query, 
                k=k*2, 
                filter_dict=filter_dict
            )
            
            # 简单关键词搜索（基于文件名和内容）
            keyword_results = self._keyword_search(
                query, 
                k=k*2, 
                filter_dict=filter_dict
            )
            
            # 融合结果
            combined_results = self._fusion_results(
                query,
                vector_results,
                keyword_results,
                vector_weight,
                keyword_weight
            )
            
            logger.info(f"混合检索完成: 查询='{query[:30]}...', 结果数量={len(combined_results)}")
            return combined_results[:k]
            
        except Exception as e:
            logger.error(f"混合检索失败: {str(e)}")
            # 降级到向量搜索
            return self.vector_store.search(query, k=k, filter_dict=filter_dict)
    
    def _keyword_search(self, query: str, k: int, filter_dict: Optional[Dict]) -> List[LangChainDocument]:
        """简单的关键词搜索"""
        # TODO: 实现BM25或其他关键词搜索算法
        # 暂时返回空列表
        return []
    
    def _fusion_results(self, query: str, vector_results: List[LangChainDocument], 
                       keyword_results: List[LangChainDocument], 
                       vector_weight: float, keyword_weight: float) -> List[LangChainDocument]:
        """融合搜索结果"""
        # 简单融合策略：按权重组合结果
        all_results = []
        
        # 添加向量搜索结果
        for doc in vector_results:
            doc.metadata = doc.metadata or {}
            doc.metadata['vector_score'] = doc.metadata.get('relevance_score', 0.5)
            doc.metadata['keyword_score'] = 0
            doc.metadata['combined_score'] = vector_weight * doc.metadata['vector_score']
            all_results.append(doc)
        
        # 添加关键词搜索结果
        for doc in keyword_results:
            doc.metadata = doc.metadata or {}
            doc.metadata['vector_score'] = 0
            doc.metadata['keyword_score'] = 0.5  # 简化处理
            doc.metadata['combined_score'] = keyword_weight * doc.metadata['keyword_score']
            all_results.append(doc)
        
        # 按综合得分排序
        all_results.sort(key=lambda x: x.metadata.get('combined_score', 0), reverse=True)
        
        # 去重（基于文档ID或内容hash）
        seen = set()
        unique_results = []
        for doc in all_results:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_results.append(doc)
        
        return unique_results
