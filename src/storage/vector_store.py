"""
向量存储模块
实现Chroma向量数据库的封装和操作
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document as LangChainDocument
from langchain_community.vectorstores import Chroma as LangChainChroma
from langchain_openai import OpenAIEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """向量存储管理器"""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.client = None
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self._initialize_store()
    
    def _initialize_store(self):
        """初始化向量存储"""
        try:
            # 确保数据目录存在
            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            
            # 创建Chroma客户端和向量存储
            self.client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_PATH
            )
            
            self.vector_store = LangChainChroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                client=self.client,
                persist_directory=settings.VECTOR_DB_PATH
            )
            
            logger.info(f"向量存储初始化成功: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"向量存储初始化失败: {str(e)}")
            raise
    
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
            
            # 准备文档数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"doc_{hash(doc.page_content)}_{i}" for i, doc in enumerate(documents)]
            
            # 添加到向量存储
            self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
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
            search_kwargs = {
                "query_texts": [query],
                "n_results": min(k, settings.TOP_K * 2),  # 获取更多候选
            }
            
            if filter_dict:
                search_kwargs["where"] = filter_dict
            
            results = self.vector_store.query(**search_kwargs)
            
            # 转换为LangChain文档格式
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_content, metadata, doc_id) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0] if results['metadatas'] else [],
                    results['ids'][0] if results['ids'] else []
                )):
                    doc = LangChainDocument(
                        page_content=doc_content,
                        metadata={
                            **metadata,
                            "search_score": results['distances'][0][i] if results['distances'] else 0.0,
                            "relevance_score": 1.0 - (results['distances'][0][i] if results['distances'] else 0.0)
                        }
                    )
                    documents.append(doc)
            
            # 如果结果少于需要的数量，使用semantic search补充
            if len(documents) < k:
                semantic_docs = self.vector_store.similarity_search(
                    query, 
                    k=k-len(documents),
                    filter=filter_dict
                )
                documents.extend(semantic_docs)
            
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
            self.vector_store.delete(where=filter_dict)
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
            collection = self.client.get_collection(self.collection_name)
            
            stats = {
                "collection_name": self.collection_name,
                "total_documents": collection.count(),
                "embeddings_dimension": len(self.embeddings.embed_query("test")),
                "vector_db_path": settings.VECTOR_DB_PATH
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {}
    
    def reset(self) -> bool:
        """清空向量存储"""
        try:
            self.client.reset()
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
        # 这里可以实现BM25或其他关键词搜索算法
        # 暂时使用简单的文本匹配
        query_words = query.lower().split()
        
        # 获取所有文档（这里需要实际的实现）
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
