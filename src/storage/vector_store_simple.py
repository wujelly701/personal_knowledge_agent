"""
简化版向量存储模块
不使用langchain_chroma，直接使用chromadb
"""

import os
import logging
import time
import sqlite3
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document as LangChainDocument
from config.settings import settings
from rank_bm25 import BM25Okapi

# 数据库重试配置
DB_MAX_RETRIES = 5
DB_RETRY_DELAY = 0.5  # 秒
DB_TIMEOUT = 10.0  # 秒

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
                logger.info(f"✅ 使用Embedding方案: {method_info['description']}")
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
        添加文档到向量存储（带数据库锁重试）

        Args:
            documents: 文档列表

        Returns:
            是否添加成功
        """
        retries = 0
        last_error = None
        
        while retries < DB_MAX_RETRIES:
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

            except sqlite3.OperationalError as e:
                last_error = e
                error_msg = str(e).lower()
                
                # 检查是否是数据库锁定错误
                if 'locked' in error_msg or 'database is locked' in error_msg:
                    retries += 1
                    if retries < DB_MAX_RETRIES:
                        delay = DB_RETRY_DELAY * retries
                        logger.warning(f"数据库锁定，{delay}秒后重试 ({retries}/{DB_MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"数据库锁定重试失败: {str(e)}")
                        break
                else:
                    # 其他数据库错误，不重试
                    logger.error(f"数据库操作错误: {str(e)}")
                    break
                    
            except Exception as e:
                last_error = e
                logger.error(f"添加文档失败: {str(e)}")
                break
        
        logger.error(f"添加文档最终失败: {str(last_error)}")
        return False

    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[LangChainDocument]:
        """
        搜索相关文档（带数据库锁重试）

        Args:
            query: 搜索查询
            k: 返回文档数量
            filter_dict: 元数据过滤条件

        Returns:
            搜索结果文档列表
        """
        retries = 0
        last_error = None
        
        while retries < DB_MAX_RETRIES:
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
                
            except sqlite3.OperationalError as e:
                last_error = e
                error_msg = str(e).lower()
                
                # 检查是否是数据库锁定错误
                if 'locked' in error_msg or 'database is locked' in error_msg:
                    retries += 1
                    if retries < DB_MAX_RETRIES:
                        delay = DB_RETRY_DELAY * retries
                        logger.warning(f"数据库锁定，{delay}秒后重试 ({retries}/{DB_MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"数据库锁定重试失败: {str(e)}")
                        break
                else:
                    logger.error(f"数据库操作错误: {str(e)}")
                    break
                    
            except Exception as e:
                last_error = e
                logger.error(f"搜索失败: {str(e)}")
                break
        
        logger.error(f"搜索最终失败: {str(last_error)}")
        return []
    
    def delete_documents(self, filter_dict: Dict[str, Any]) -> bool:
        """
        删除文档
        
        Args:
            filter_dict: 删除条件（如 {"filename": "test.md"}）
            
        Returns:
            是否删除成功
        """
        try:
            # 构建查询条件
            where_clause = {}
            for key, value in filter_dict.items():
                where_clause[key] = {"$eq": value}
            
            # 先查询要删除的文档数量
            try:
                existing_docs = self.collection.get(where=where_clause)
                doc_count = len(existing_docs['ids']) if existing_docs and 'ids' in existing_docs else 0
                logger.info(f"找到 {doc_count} 个匹配的文档块准备删除")
            except Exception as e:
                logger.warning(f"查询待删除文档失败: {e}")
                doc_count = 0
            
            # 执行删除
            if doc_count > 0:
                self.collection.delete(where=where_clause)
                logger.info(f"删除文档成功: {filter_dict}, 删除了 {doc_count} 个文档块")
            else:
                logger.warning(f"未找到匹配的文档: {filter_dict}")
            
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
        """BM25关键词搜索（增强文件名匹配）"""
        try:
            # 空查询返回空结果
            if not query or not query.strip():
                return []
            
            # 1. 获取所有文档
            all_docs = self.vector_store.collection.get()

            if not all_docs['documents']:
                return []

            # 2. 应用元数据过滤
            filtered_docs = []
            filtered_metadatas = []
            filtered_ids = []

            for i, (doc, metadata, doc_id) in enumerate(zip(
                all_docs['documents'],
                all_docs['metadatas'],
                all_docs['ids']
            )):
                # 元数据过滤
                if filter_dict:
                    match = all(
                        metadata.get(key) == value
                        for key, value in filter_dict.items()
                    )
                    if not match:
                        continue

                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
                filtered_ids.append(doc_id)
            
            if not filtered_docs:
                return []

            # 3. 构建BM25索引（包含文件名信息以提高匹配）
            tokenized_docs = []
            for doc, metadata in zip(filtered_docs, filtered_metadatas):
                # 将文件名也加入搜索内容，提高文件名匹配权重
                filename = metadata.get('filename', '')
                # 文件名重复3次以提高其权重
                enhanced_content = f"{filename} {filename} {filename} {doc}"
                tokenized_docs.append(enhanced_content.lower().split())
            
            bm25 = BM25Okapi(tokenized_docs)

            # 4. 搜索
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
            
            # 5. 文件名精确匹配加分
            query_lower = query.lower()
            for idx, metadata in enumerate(filtered_metadatas):
                filename = metadata.get('filename', '').lower()
                # 如果查询词在文件名中，大幅提升分数
                if query_lower in filename:
                    scores[idx] *= 3.0  # 文件名匹配3倍加分
                # 如果文件名包含查询词的任一token
                elif any(token in filename for token in query_tokens):
                    scores[idx] *= 2.0  # 部分匹配2倍加分

            # 6. 返回Top-K
            top_indices = np.argsort(scores)[::-1][:k]

            # 7. 构建结果
            results = []
            for idx in top_indices:
                doc = LangChainDocument(
                    page_content=filtered_docs[idx],
                    metadata={
                        **filtered_metadatas[idx],
                        "bm25_score": float(scores[idx]),
                        "keyword_relevance": min(scores[idx] / 10.0, 1.0),
                        "doc_id": filtered_ids[idx]
                    }
                )
                results.append(doc)

            logger.info(f"BM25关键词搜索完成: 查询='{query}', 结果数量={len(results)}")
            logger.debug(f"BM25得分: {[r.metadata['bm25_score'] for r in results]}")
            return results
        except Exception as e:
            logger.error(f"BM25搜索失败: {str(e)}")
            return []
    
    def _fusion_results(self, query: str, vector_results: List[LangChainDocument], 
                       keyword_results: List[LangChainDocument], 
                       vector_weight: float, keyword_weight: float) -> List[LangChainDocument]:
        """融合搜索结果（改进算法）"""
        # 使用字典来合并相同文档的分数
        doc_scores = {}
        
        # 处理向量搜索结果
        for doc in vector_results:
            doc_hash = hash(doc.page_content)
            vector_score = doc.metadata.get('relevance_score', 0.5)
            
            if doc_hash not in doc_scores:
                doc_scores[doc_hash] = {
                    'doc': doc,
                    'vector_score': vector_score,
                    'keyword_score': 0.0
                }
            else:
                doc_scores[doc_hash]['vector_score'] = max(
                    doc_scores[doc_hash]['vector_score'], 
                    vector_score
                )
        
        # 处理关键词搜索结果（使用真实的BM25分数）
        for doc in keyword_results:
            doc_hash = hash(doc.page_content)
            # 改进BM25分数归一化
            bm25_score = doc.metadata.get('bm25_score', 0)
            
            # 动态归一化：如果分数很高，说明匹配度好
            if bm25_score > 10:
                keyword_score = min(bm25_score / 15.0, 1.0)  # 高分情况
            elif bm25_score > 5:
                keyword_score = min(bm25_score / 10.0, 1.0)  # 中等分数
            elif bm25_score > 0:
                keyword_score = min(bm25_score / 5.0, 0.8)   # 低分情况
            else:
                keyword_score = 0.0
            
            # 文件名精确匹配时大幅提升分数
            filename = doc.metadata.get('filename', '').lower()
            query_lower = query.lower()
            
            if query_lower in filename:
                # 文件名包含查询词，给予高分
                keyword_score = max(keyword_score * 2.0, 0.85)
                keyword_score = min(keyword_score, 1.0)
            elif any(token in filename for token in query_lower.split()):
                # 文件名包含部分查询词
                keyword_score = max(keyword_score * 1.5, 0.7)
                keyword_score = min(keyword_score, 0.95)
            
            if doc_hash not in doc_scores:
                doc_scores[doc_hash] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score
                }
            else:
                doc_scores[doc_hash]['keyword_score'] = max(
                    doc_scores[doc_hash]['keyword_score'],
                    keyword_score
                )
        
        # 计算综合得分并构建结果列表
        results = []
        for doc_hash, info in doc_scores.items():
            doc = info['doc']
            
            # 动态调整权重策略
            actual_keyword_weight = keyword_weight
            actual_vector_weight = vector_weight
            
            # 策略1: 文件名精确匹配 + 高关键词得分 → 极高权重
            if info['keyword_score'] > 0.8 and info['vector_score'] < 0.3:
                # 关键词匹配很好但向量匹配差（技术文档场景）
                # 给予关键词搜索主导权
                actual_keyword_weight = 0.9
                actual_vector_weight = 0.1
            elif info['keyword_score'] > 0.8:
                # 文件名匹配度高时，提升关键词权重
                actual_keyword_weight = min(keyword_weight * 1.5, 0.5)
                actual_vector_weight = 1.0 - actual_keyword_weight
            elif info['vector_score'] > 0.8 and info['keyword_score'] < 0.1:
                # 向量匹配很好但关键词不匹配（语义相关但文件名无关）
                actual_vector_weight = 0.8
                actual_keyword_weight = 0.2
            
            combined_score = (
                actual_vector_weight * info['vector_score'] + 
                actual_keyword_weight * info['keyword_score']
            )
            
            # 更新metadata
            doc.metadata['vector_score'] = round(info['vector_score'], 3)
            doc.metadata['keyword_score'] = round(info['keyword_score'], 3)
            doc.metadata['combined_score'] = round(combined_score, 3)
            doc.metadata['relevance_score'] = round(combined_score, 3)  # 更新总相关性
            
            results.append(doc)
        
        # 按综合得分排序
        results.sort(key=lambda x: x.metadata.get('combined_score', 0), reverse=True)
        
        return results
