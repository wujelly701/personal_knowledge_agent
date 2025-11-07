"""
LLM集成模块
实现混合架构：OpenAI embedding + DeepSeek生成
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document as LangChainDocument
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMManager:
    """LLM管理器"""
    
    def __init__(self):
        self.embedding_model = None
        self.chat_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        try:
            # 智能选择embedding方法
            optimal_method = settings.get_optimal_embedding_method()

            if optimal_method == "openai" and settings.OPENAI_API_KEY:
                # 使用OpenAI embedding
                from langchain_openai import OpenAIEmbeddings
                self.embedding_model = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY
                )
                logger.info("✅ 使用OpenAI embedding模型")
            else:
                # 使用本地embedding管理器
                from src.storage.embedding_manager import EmbeddingManager
                self.embedding_model = EmbeddingManager(optimal_method)
                method_info = self.embedding_model.get_method_info()
                logger.info(f"✅ 使用本地embedding模型: {method_info['description']}")

            # 初始化聊天模型（DeepSeek）
            if settings.DEEPSEEK_API_KEY:
                self.chat_model = ChatDeepSeek(
                    model=settings.LLM_MODEL,
                    api_key=settings.DEEPSEEK_API_KEY,
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )
                logger.info("✅ 使用DeepSeek聊天模型")
            else:
                self.chat_model = None
                logger.warning("⚠️ 未配置DeepSeek API密钥，将使用简化回答模式")

            logger.info("LLM模型初始化成功")

        except Exception as e:
            logger.error(f"LLM模型初始化失败: {str(e)}")
            # 确保至少有一个工作的embedding方法
            from src.storage.embedding_manager import EmbeddingManager
            self.embedding_model = EmbeddingManager("text-hash")
            logger.info("✅ 回退到文本哈希embedding模型")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档生成embedding

        Args:
            texts: 文本列表

        Returns:
            embedding向量列表
        """
        try:
            # 检查是否使用OpenAI模型
            if hasattr(self.embedding_model, 'embed_documents') and not hasattr(self.embedding_model, 'get_method_info'):
                # OpenAI模型
                embeddings = self.embedding_model.embed_documents(texts)
            else:
                # 本地embedding管理器
                embeddings = self.embedding_model.embed_documents(texts)

            logger.info(f"生成了 {len(embeddings)} 个文档的embedding")
            return embeddings

        except Exception as e:
            logger.error(f"生成embedding失败: {str(e)}")
            # 回退到文本哈希
            from src.storage.embedding_manager import EmbeddingManager
            fallback_manager = EmbeddingManager("text-hash")
            return fallback_manager.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        为查询生成embedding

        Args:
            text: 查询文本

        Returns:
            embedding向量
        """
        try:
            # 检查是否使用OpenAI模型
            if hasattr(self.embedding_model, 'embed_query') and not hasattr(self.embedding_model, 'get_method_info'):
                # OpenAI模型
                embedding = self.embedding_model.embed_query(text)
            else:
                # 本地embedding管理器
                embedding = self.embedding_model.embed_query(text)

            return embedding

        except Exception as e:
            logger.error(f"生成查询embedding失败: {str(e)}")
            # 回退到文本哈希
            from src.storage.embedding_manager import EmbeddingManager
            fallback_manager = EmbeddingManager("text-hash")
            return fallback_manager.embed_query(text)

    def get_embedding_info(self) -> Dict[str, Any]:
        """获取embedding方法信息"""
        # 检查是否使用OpenAI模型
        if hasattr(self.embedding_model, 'get_method_info'):
            return self.embedding_model.get_method_info()
        else:
            # OpenAI模型的信息
            return {
                "method": "openai",
                "model": settings.EMBEDDING_MODEL,
                "description": f"OpenAI {settings.EMBEDDING_MODEL} - 云端API",
                "is_free": False,
                "quality": "高"
            }

class RAGGenerator:
    """RAG问答生成器"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.chat_model = llm_manager.chat_model
        
        # RAG系统提示词
        self.rag_system_prompt = """你是一个专业的知识管理助手。基于提供的上下文内容回答用户问题。

规则：
1. 仅使用上下文中的信息回答问题
2. 每个事实陈述都要引用来源：[来源：文件名]
3. 如果上下文不包含答案，明确说明"我在现有文档中没有找到该信息"
4. 使用Markdown格式组织答案
5. 保持客观、准确、有帮助
6. 优先使用中文回答

上下文：
{context}

问题：{question}

请基于上述规则和上下文回答问题："""
        
        # 文档分类提示词
        self.classification_prompt = """请将以下文档内容分类到预定义的类别中。

类别选项：
1. 工作 - 职业相关的文档、任务、计划、报告
2. 学习 - 教育材料、教程、课程笔记、技能学习
3. 个人 - 日记、想法感悟、生活记录、个人总结
4. 参考 - 手册、指南、规范、标准、参考资料
5. 研究 - 分析报告、实验数据、研究发现、学术内容
6. 想法 - 创意、创新、设计理念、概念构思

文档内容：
{document_content}

请以JSON格式返回分类结果，包含字段：
- category: 类别名称
- priority: 优先级（高/中/低）
- summary: 100字以内摘要
- tags: 3-5个相关标签列表
- confidence: 置信度（0-1）"""
    
    def generate_answer(self, query: str, documents: List[LangChainDocument], 
                       include_sources: bool = True) -> Dict[str, Any]:
        """
        基于检索文档生成回答
        
        Args:
            query: 用户问题
            documents: 检索到的相关文档
            include_sources: 是否包含来源引用
            
        Returns:
            生成的回答信息
        """
        try:
            if not documents:
                return {
                    "answer": "抱歉，我在现有文档中没有找到相关信息。请尝试上传相关文档或重新表述问题。",
                    "confidence": 0.0,
                    "sources": [],
                    "metadata": {"error": "no_documents_found"}
                }
            
            # 构建上下文
            context = self._build_context(documents, include_sources)
            
            # 生成回答
            messages = [
                SystemMessage(content=self.rag_system_prompt.format(
                    context=context,
                    question=query
                )),
                HumanMessage(content="请基于上述上下文回答问题。")
            ]
            
            response = self.chat_model.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # 提取置信度
            confidence = self._estimate_confidence(query, documents, answer)
            
            # 构建结果
            result = {
                "answer": answer,
                "confidence": confidence,
                "sources": self._extract_sources(documents),
                "metadata": {
                    "query": query,
                    "retrieved_docs": len(documents),
                    "model_used": settings.LLM_MODEL,
                    "tokens_used": getattr(response, 'usage_metadata', {}).get('total_tokens', 0) if hasattr(response, 'usage_metadata') else 0
                }
            }
            
            logger.info(f"RAG回答生成成功: 查询='{query[:30]}...', 置信度={confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"RAG回答生成失败: {str(e)}")
            return {
                "answer": "抱歉，回答生成过程中出现错误。请稍后重试。",
                "confidence": 0.0,
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def classify_document(self, content: str) -> Dict[str, Any]:
        """
        分类文档
        
        Args:
            content: 文档内容
            
        Returns:
            分类结果
        """
        try:
            messages = [
                SystemMessage(content=self.classification_prompt),
                HumanMessage(content=f"文档内容：\n\n{content[:2000]}")  # 限制长度避免token过多
            ]
            
            response = self.chat_model.invoke(messages)
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析JSON结果
            import json
            try:
                # 提取JSON部分
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_text = result_text[json_start:json_end]
                    classification = json.loads(json_text)
                else:
                    raise ValueError("未找到有效的JSON格式")
            except json.JSONDecodeError:
                # 如果JSON解析失败，使用默认分类
                classification = {
                    "category": "参考",
                    "priority": "中",
                    "summary": content[:100] + "...",
                    "tags": ["未分类"],
                    "confidence": 0.5
                }
            
            # 验证和标准化分类结果
            if classification.get("category") not in settings.DOCUMENT_CATEGORIES:
                classification["category"] = "参考"
            if classification.get("priority") not in settings.PRIORITY_LEVELS:
                classification["priority"] = "中"
            
            return classification
            
        except Exception as e:
            logger.error(f"文档分类失败: {str(e)}")
            return {
                "category": "参考",
                "priority": "中", 
                "summary": content[:100] + "...",
                "tags": ["分类失败"],
                "confidence": 0.0
            }
    
    def generate_summary(self, content: str, max_length: int = 300) -> str:
        """
        生成文档摘要
        
        Args:
            content: 文档内容
            max_length: 最大长度
            
        Returns:
            摘要文本
        """
        try:
            summary_prompt = f"""请为以下文档生成简洁的摘要，长度不超过{max_length}字：

{content}

摘要要求：
1. 提取关键主题和要点
2. 保持客观准确
3. 使用简洁的中文表达
4. 结构化呈现要点

摘要："""
            
            messages = [
                HumanMessage(content=summary_prompt)
            ]
            
            response = self.chat_model.invoke(messages)
            summary = response.content if hasattr(response, 'content') else str(response)
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"生成摘要失败: {str(e)}")
            return content[:max_length] + "..." if len(content) > max_length else content
    
    def _build_context(self, documents: List[LangChainDocument], include_sources: bool) -> str:
        """构建上下文内容"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source_info = ""
            if include_sources and doc.metadata:
                filename = doc.metadata.get('filename', '未知文件')
                chunk_id = doc.metadata.get('chunk_id', 0)
                source_info = f"\n来源 [{i}]: {filename} (第{chunk_id + 1}部分)"
            
            context_parts.append(f"[{i}] {doc.page_content}{source_info}")
        
        return "\n\n".join(context_parts)
    
    def _estimate_confidence(self, query: str, documents: List[LangChainDocument], answer: str) -> float:
        """估算回答置信度"""
        try:
            # 基于多个因素估算置信度
            factors = []
            
            # 1. 检索文档数量
            doc_count_score = min(len(documents) / 3, 1.0)
            factors.append(doc_count_score)
            
            # 2. 文档相关性分数
            if documents and documents[0].metadata:
                avg_relevance = sum(
                    doc.metadata.get('relevance_score', 0.5) for doc in documents
                ) / len(documents)
                factors.append(avg_relevance)
            
            # 3. 回答长度
            if answer and "没有找到" not in answer and "不清楚" not in answer:
                length_score = min(len(answer) / 200, 1.0)
                factors.append(length_score)
            else:
                factors.append(0.3)
            
            # 计算综合置信度
            confidence = sum(factors) / len(factors) if factors else 0.5
            
            # 限制在0-1范围内
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"置信度估算失败: {str(e)}")
            return 0.5
    
    def _extract_sources(self, documents: List[LangChainDocument]) -> List[Dict[str, Any]]:
        """提取信息来源"""
        sources = []
        seen_files = set()
        
        for doc in documents:
            if doc.metadata:
                filename = doc.metadata.get('filename', '未知文件')
                if filename not in seen_files:
                    seen_files.add(filename)
                    sources.append({
                        "filename": filename,
                        "source": doc.metadata.get('source', ''),
                        "chunk_id": doc.metadata.get('chunk_id', 0),
                        "relevance_score": doc.metadata.get('relevance_score', 0.0)
                    })
        
        # 按相关性分数排序
        sources.sort(key=lambda x: x['relevance_score'], reverse=True)
        return sources

class ModelRouter:
    """模型路由器"""
    
    def __init__(self):
        self.primary_model = None
        self.fallback_model = None
        self._initialize_routing()
    
    def _initialize_routing(self):
        """初始化模型路由"""
        try:
            # 主要模型（DeepSeek）
            self.primary_model = ChatDeepSeek(
                model=settings.LLM_MODEL,
                api_key=settings.DEEPSEEK_API_KEY,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            # 备用模型（GPT-4o-mini）
            self.fallback_model = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            logger.info("模型路由初始化成功")
            
        except Exception as e:
            logger.error(f"模型路由初始化失败: {str(e)}")
            raise
    
    def route_request(self, prompt: str, complexity: str = "normal") -> Any:
        """
        路由请求到合适的模型
        
        Args:
            prompt: 提示词
            complexity: 复杂度（简单/正常/复杂）
            
        Returns:
            模型响应
        """
        try:
            # 根据复杂度选择模型
            if complexity == "complex":
                # 复杂任务使用更强大的模型
                model = self.primary_model
            elif complexity == "simple":
                # 简单任务可以使用较小模型
                model = self.primary_model
            else:
                # 正常任务
                model = self.primary_model
            
            # 尝试主要模型
            try:
                response = model.invoke([HumanMessage(content=prompt)])
                return response
            except Exception as e:
                logger.warning(f"主要模型失败，切换到备用模型: {str(e)}")
                
                # 降级到备用模型
                response = self.fallback_model.invoke([HumanMessage(content=prompt)])
                return response
                
        except Exception as e:
            logger.error(f"所有模型都失败了: {str(e)}")
            raise
