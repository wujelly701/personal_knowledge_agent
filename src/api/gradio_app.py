"""
Gradio用户界面
个人知识管理助手的Web界面
"""

import os
import logging
import gradio as gr
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import settings
from src.ingestion.document_loader_simple import DocumentLoader, DocumentClassifier
from src.storage.vector_store_simple import VectorStore, HybridRetriever
from src.generation.llm_manager import LLMManager, RAGGenerator, ModelRouter
from src.utils.search_history import SearchHistoryManager

# 获取日志器（日志已在main.py中统一配置）
logger = logging.getLogger(__name__)

class KnowledgeManagerApp:
    """知识管理应用主类"""

    def __init__(self):
        # 初始化组件
        self.document_loader = DocumentLoader()
        self.document_classifier = DocumentClassifier()
        self.vector_store = VectorStore()
        self.hybrid_retriever = HybridRetriever(self.vector_store)
        # 智能初始化LLM组件
        self.llm_manager = None
        self.rag_generator = None
        self.model_router = None
        self.llm_enabled = False

        # 尝试初始化LLM组件
        try:
            self.llm_manager = LLMManager()
            if self.llm_manager.chat_model:  # 只有在有DeepSeek API的情况下才启用完整功能
                self.rag_generator = RAGGenerator(self.llm_manager)
                self.model_router = ModelRouter()
                self.llm_enabled = True
                logger.info("✅ LLM功能已启用")
            else:
                logger.info("ℹ️ DeepSeek API未配置，使用简化回答模式")
        except Exception as e:
            logger.warning(f"LLM功能初始化失败，将使用简化模式: {e}")

        # 搜索历史管理器
        self.search_history_manager = SearchHistoryManager()

        # 状态管理
        self.conversation_history = []
        self.current_session_id = None

        # 验证配置
        settings.validate()

        logger.info("知识管理应用初始化完成")

    def load_and_process_files(self, files: List[str], progress=gr.Progress()) -> str:
        """
        加载和处理上传的文件

        Args:
            files: 上传的文件路径列表
            progress: Gradio进度跟踪器

        Returns:
            处理结果消息
        """
        from src.utils.recovery import RecoveryManager
        import os
        
        # 初始化恢复管理器
        recovery_manager = RecoveryManager()
        
        try:
            if not files:
                return "请选择要上传的文件。"

            # 文件大小限制 (50MB)
            MAX_FILE_SIZE = 50 * 1024 * 1024
            
            # 支持的文件类型
            SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.doc', '.docx'}

            logger.info(f"开始处理 {len(files)} 个文件")
            progress(0, desc="开始处理文件...")

            # 批量处理文件
            all_documents = []
            processed_files = []
            skipped_files = []
            updated_files = []
            failed_files = []
            
            total_files = len(files)
            for idx, file_path in enumerate(files):
                try:
                    # 更新进度
                    current_progress = idx / total_files
                    file_name = Path(file_path).name
                    progress(current_progress, desc=f"处理文件 {idx+1}/{total_files}: {file_name}")
                    
                    # 保存哈希值变量
                    new_content_hash = None
                    
                    # === 文件验证 ===
                    # 1. 检查文件是否存在
                    if not os.path.exists(file_path):
                        logger.warning(f"文件不存在: {file_name}")
                        failed_files.append((file_name, "文件不存在"))
                        continue
                    
                    # 2. 检查文件类型
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext not in SUPPORTED_EXTENSIONS:
                        logger.warning(f"不支持的文件类型 {file_ext}: {file_name}")
                        failed_files.append((file_name, f"不支持的文件类型 {file_ext}"))
                        continue
                    
                    # 3. 检查文件大小
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > MAX_FILE_SIZE:
                            size_mb = file_size / (1024 * 1024)
                            logger.warning(f"文件过大 ({size_mb:.1f}MB): {file_name}")
                            failed_files.append((file_name, f"文件过大 ({size_mb:.1f}MB，限制50MB)"))
                            continue
                        if file_size == 0:
                            logger.warning(f"文件为空: {file_name}")
                            failed_files.append((file_name, "文件为空"))
                            continue
                    except OSError as e:
                        logger.error(f"无法读取文件大小 {file_name}: {str(e)}")
                        failed_files.append((file_name, "无法读取文件大小"))
                        continue
                    
                    # 4. 检查文件读取权限
                    if not os.access(file_path, os.R_OK):
                        logger.warning(f"没有文件读取权限: {file_name}")
                        failed_files.append((file_name, "没有读取权限"))
                        continue
                    
                    # === 保存处理检查点 ===
                    recovery_manager.save_checkpoint('file_upload', {
                        'file_name': file_name,
                        'file_path': file_path,
                        'stage': 'validation_passed'
                    })
                    
                    # 计算新文件内容哈希（统一在此处计算）
                    try:
                        with open(file_path, 'rb') as f:
                            import hashlib
                            new_content_hash = hashlib.md5(f.read()).hexdigest()
                    except PermissionError:
                        logger.error(f"读取文件权限被拒绝: {file_name}")
                        failed_files.append((file_name, "读取权限被拒绝"))
                        continue
                    except IOError as e:
                        logger.error(f"文件读取错误 {file_name}: {str(e)}")
                        failed_files.append((file_name, f"文件读取错误: {str(e)}"))
                        continue
                    
                    # 检查是否已存在同名文档
                    existing_docs = self.vector_store.collection.get()
                    file_exists = False
                    if existing_docs['metadatas']:
                        for metadata in existing_docs['metadatas']:
                            if metadata.get('filename') == file_name:
                                file_exists = True
                                break
                    
                    # 如果文件已存在，判断是否相同
                    if file_exists:
                        # 获取旧文件的哈希（从该文件的任意一个块的metadata中读取file_hash）
                        old_file_hash = None
                        for metadata in existing_docs['metadatas']:
                            if metadata.get('filename') == file_name:
                                # 从metadata中读取文件级别的hash
                                old_file_hash = metadata.get('file_hash')
                                break
                        
                        # 比较哈希值
                        if new_content_hash == old_file_hash:
                            # 内容完全相同，跳过处理
                            logger.info(f"文件内容未变化，跳过: {file_name}")
                            skipped_files.append(file_name)
                            continue
                        else:
                            # 内容不同，删除旧版本，添加新版本
                            logger.info(f"检测到文件内容变化，更新: {file_name}")
                            self.vector_store.delete_documents({"filename": file_name})
                            updated_files.append(file_name)
                    
                    # === 加载文档 ===
                    recovery_manager.save_checkpoint('file_upload', {
                        'file_name': file_name,
                        'stage': 'loading_document'
                    })
                    
                    progress(current_progress + 0.3/total_files, desc=f"加载文档: {file_name}")
                    documents = self.document_loader.load_file(file_path)

                    # 为每个文档添加分类信息和文件哈希
                    progress(current_progress + 0.5/total_files, desc=f"分类文档: {file_name}")
                    import time
                    upload_timestamp = time.time()  # 获取上传时间戳
                    
                    for doc in documents:
                        classification = self.document_classifier.classify_document(doc)
                        # 清理None值（ChromaDB不接受None类型的metadata）
                        classification = {k: v for k, v in classification.items() if v is not None}
                        doc.metadata.update(classification)
                        # 添加文件级别的hash到每个chunk
                        doc.metadata['file_hash'] = new_content_hash
                        # 添加上传时间戳
                        doc.metadata['upload_time'] = upload_timestamp

                    all_documents.extend(documents)
                    processed_files.append(file_name)
                    
                    # 清除成功的检查点
                    recovery_manager.clear_checkpoint()

                except PermissionError as e:
                    logger.error(f"文件权限错误 {file_path}: {str(e)}")
                    failed_files.append((Path(file_path).name, "权限不足"))
                    continue
                except IOError as e:
                    logger.error(f"文件IO错误 {file_path}: {str(e)}")
                    failed_files.append((Path(file_path).name, f"IO错误: {str(e)}"))
                    continue
                except Exception as e:
                    logger.warning(f"文件处理失败 {file_path}: {str(e)}")
                    failed_files.append((Path(file_path).name, str(e)))
                    continue

            # 添加到向量存储
            if all_documents:
                progress(0.9, desc=f"生成Embedding ({len(all_documents)} 个文档块)...")
                success = self.vector_store.add_documents(all_documents)
                if success:
                    progress(1.0, desc="处理完成!")
                    result = f"✅ 成功处理 {len(all_documents)} 个文档块\n\n"
                    
                    # 显示处理统计
                    if processed_files:
                        result += f"📄 **新增文件**: {len(processed_files)} 个\n"
                        for fname in processed_files[:5]:  # 只显示前5个
                            result += f"  • {fname}\n"
                        if len(processed_files) > 5:
                            result += f"  • ... 还有 {len(processed_files)-5} 个\n"
                    
                    if updated_files:
                        result += f"\n🔄 **更新文件**: {len(updated_files)} 个（内容已变化）\n"
                        for fname in updated_files:
                            result += f"  • {fname}\n"
                    
                    if skipped_files:
                        result += f"\n⏭️ **跳过文件**: {len(skipped_files)} 个（内容未变化）\n"
                        for fname in skipped_files:
                            result += f"  • {fname}\n"
                    
                    if failed_files:
                        result += f"\n❌ **失败文件**: {len(failed_files)} 个\n"
                        for fname, reason in failed_files:
                            result += f"  • {fname}: {reason}\n"
                    
                    result += f"\n📊 **文档分类统计**：\n"

                    # 统计分类信息
                    categories = {}
                    for doc in all_documents:
                        category = doc.metadata.get('category', '未知')
                        categories[category] = categories.get(category, 0) + 1

                    for category, count in categories.items():
                        category_desc = {
                            "工作": "与工作相关的文档内容",
                            "学习": "学习笔记或教程",
                            "研究": "研究或学术相关的内容",
                            "参考": "参考资料或引用内容",
                            "想法": "个人见解或创意想法",
                            "个人": "个人日常或生活记录",
                            "未知": "未分类的内容"
                        }.get(category, "其他分类内容")

                        result += f"  • {category}: {count} 个块 ({category_desc})\n"

                    result += f"\n💡 这些文档块现在已经存储在知识库中，可以通过智能问答功能进行查询和对话。"
                    return result
                else:
                    return "❌ 文件处理失败，请重试。"
            elif skipped_files or failed_files:
                # 只有跳过和失败的情况
                result = ""
                if skipped_files:
                    result += f"⏭️ **跳过文件**: {len(skipped_files)} 个（内容未变化）\n"
                    for fname in skipped_files:
                        result += f"  • {fname}\n"
                if failed_files:
                    if skipped_files:
                        result += "\n"
                    result += f"❌ **失败文件**: {len(failed_files)} 个\n"
                    for fname, reason in failed_files:
                        result += f"  • {fname}: {reason}\n"
                return result if result else "⚠️ 没有成功处理任何文件。"
            else:
                return "⚠️ 没有成功处理任何文件。"

        except Exception as e:
            logger.error(f"文件处理异常: {str(e)}")
            # 尝试恢复上次的检查点
            checkpoint = recovery_manager.load_last_checkpoint()
            if checkpoint:
                logger.info(f"检测到检查点: {checkpoint}")
                return f"❌ 处理失败: {str(e)}\n💡 上次处理到: {checkpoint.get('file_name', '未知')} - {checkpoint.get('stage', '未知阶段')}"
            return f"❌ 处理失败: {str(e)}"

    def chat_with_knowledge(self, message: str, history: List[Dict[str, str]]) -> str:
        """
        基于知识库的对话（智能模式：LLM优先，简化模式备用）

        Args:
            message: 用户消息
            history: 对话历史

        Returns:
            AI回答
        """
        try:
            if not message.strip():
                return "请输入您的问题。"

            logger.info(f"收到用户问题: {message[:50]}..., 历史轮数: {len(history)//2 if history else 0}")

            # 构建上下文查询（结合最近3轮对话）
            context_query = message
            if history and len(history) >= 2:
                # 获取最近3轮对话（6条消息）
                recent_history = history[-6:] if len(history) > 6 else history
                context_parts = []
                for msg in recent_history:
                    if msg['role'] == 'user':
                        context_parts.append(f"用户: {msg['content']}")
                    elif msg['role'] == 'assistant':
                        # 只保留简短摘要，不包含完整回答
                        content = msg['content'][:100] if len(msg['content']) > 100 else msg['content']
                        context_parts.append(f"助手: {content}")
                
                # 组合查询（当前问题 + 上下文）
                context_query = f"{' '.join(context_parts[-2:])} {message}"
                logger.info(f"使用上下文查询: {context_query[:100]}...")

            # 检索相关文档
            retrieved_docs = self.vector_store.search(context_query, k=settings.TOP_K)

            if not retrieved_docs:
                return "我在知识库中没有找到相关信息。请尝试：\n1. 检查问题表述\n2. 上传相关文档\n3. 使用不同的关键词"

            # 如果LLM功能可用且有DeepSeek API，使用RAG生成器
            if self.llm_enabled and self.rag_generator:
                try:
                    # 构建对话历史（传递给LLM）
                    conversation_context = []
                    if history:
                        # 只保留最近5轮对话
                        recent = history[-10:] if len(history) > 10 else history
                        for msg in recent:
                            conversation_context.append({
                                'role': msg['role'],
                                'content': msg['content'][:500]  # 限制长度
                            })
                    
                    # 使用RAG生成器生成智能回答（传递对话历史）
                    result = self.rag_generator.generate_answer(
                        query=message,
                        documents=retrieved_docs,
                        include_sources=True,
                        conversation_history=conversation_context  # 传递上下文
                    )

                    # 构建回答
                    answer_parts = []
                    answer_parts.append("🤖 **AI 智能回答：**\n")
                    answer_parts.append(f"{result['answer']}\n")

                    # 添加置信度信息
                    if result['confidence'] > 0:
                        confidence_level = "高" if result['confidence'] > 0.8 else "中" if result['confidence'] > 0.5 else "低"
                        answer_parts.append(f"\n📊 置信度：{confidence_level} ({result['confidence']:.2f})")

                    # 添加来源信息（带引用编号）
                    if result['sources']:
                        answer_parts.append("\n\n📚 **引用来源：**\n")
                        for i, source in enumerate(result['sources'][:3], 1):
                            filename = source['filename']
                            relevance = source['relevance_score']
                            chunk_id = source.get('chunk_id', '?')
                            # 添加可点击的引用格式
                            answer_parts.append(f"[{i}] 📄 **{filename}** (chunk {chunk_id}, 相关性: {relevance:.2f})\n")
                            # 显示引用内容片段
                            if 'content' in source:
                                preview = source['content'][:150] if len(source['content']) > 150 else source['content']
                                answer_parts.append(f"   _{preview}..._\n\n")

                    # 添加token使用量
                    if result['metadata']['tokens_used']:
                        answer_parts.append(f"\n💬 Token使用量：{result['metadata']['tokens_used']}")
                    
                    # 生成相关问题推荐
                    try:
                        suggested_questions = self._generate_suggested_questions(message, result['answer'], retrieved_docs)
                        if suggested_questions:
                            answer_parts.append("\n\n💡 **您可能还想了解：**\n")
                            for i, q in enumerate(suggested_questions, 1):
                                answer_parts.append(f"{i}. {q}\n")
                    except Exception as sq_error:
                        logger.warning(f"推荐问题生成失败: {sq_error}")

                    answer = "".join(answer_parts)

                    logger.info(f"AI 回答生成完成: 查询='{message[:30]}...', 置信度={result['confidence']:.2f}, 使用上下文={len(conversation_context)}条")
                    return answer

                except Exception as e:
                    logger.warning(f"RAG生成失败，回退到简化模式: {e}")
                    # 回退到简化模式
                    # 回退到简化模式

            # 简化模式回答（不依赖LLM）
            answer_parts = []
            answer_parts.append("🔍 **基于知识库的回答：**\n")

            # 添加最相关的文档内容
            for i, doc in enumerate(retrieved_docs[:3], 1):
                filename = doc.metadata.get('filename', '未知文件')
                relevance = doc.metadata.get('relevance_score', 0.0)
                content = doc.page_content

                # 截取最相关的部分
                if len(content) > 300:
                    content = content[:300] + "..."

                answer_parts.append(f"**{i}. {filename}** (相关性: {relevance:.2f})\n")
                answer_parts.append(f"{content}\n")

            # 添加来源信息
            answer_parts.append("\n📚 **相关文档来源：**\n")
            for i, doc in enumerate(retrieved_docs[:3], 1):
                filename = doc.metadata.get('filename', '未知文件')
                answer_parts.append(f"{i}. {filename}\n")

            answer_parts.append(f"\n💡 *提示：找到 {len(retrieved_docs)} 个相关文档片段*")

            # 如果有embedding信息，添加embedding方法信息
            if hasattr(self, 'vector_store') and hasattr(self.vector_store, 'embedding_manager'):
                method_info = self.vector_store.embedding_manager.get_method_info()
                answer_parts.append(f"\n🔧 *Embedding方法：{method_info['description']}*")

            answer = "".join(answer_parts)

            logger.info(f"搜索完成，找到 {len(retrieved_docs)} 个相关文档")
            return answer

        except Exception as e:
            logger.error(f"对话生成失败: {str(e)}")
            return f"抱歉，回答生成过程中出现错误：{str(e)}"
    
    def _generate_suggested_questions(self, original_query: str, answer: str, documents: List) -> List[str]:
        """
        基于当前问题和答案生成相关问题推荐
        
        Args:
            original_query: 原始问题
            answer: AI回答
            documents: 检索到的文档
            
        Returns:
            推荐问题列表（3-5个）
        """
        try:
            if not self.llm_enabled or not self.llm_manager:
                # 没有LLM，使用简单的基于关键词的推荐
                return self._generate_simple_suggestions(documents)
            
            # 使用LLM生成推荐问题
            prompt = f"""基于以下对话，生成3-5个用户可能感兴趣的后续问题。

原始问题: {original_query}
回答: {answer[:300]}...

请生成简洁、具体的后续问题，每行一个问题，不要编号。直接输出问题即可。"""
            
            messages = [
                SystemMessage(content="你是一个助手，帮助用户发现相关问题。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm_manager.chat_model.invoke(messages)
            suggestions_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析建议问题
            suggestions = []
            for line in suggestions_text.strip().split('\n'):
                line = line.strip()
                # 移除编号（如 "1. "）
                line = line.lstrip('0123456789.、- ')
                if line and len(line) > 5:  # 过滤太短的行
                    suggestions.append(line)
            
            return suggestions[:5]  # 最多返回5个
            
        except Exception as e:
            logger.warning(f"推荐问题生成失败: {e}")
            return self._generate_simple_suggestions(documents)
    
    def _generate_simple_suggestions(self, documents: List) -> List[str]:
        """
        基于文档内容生成简单的问题推荐（不使用LLM）
        
        Args:
            documents: 检索到的文档
            
        Returns:
            推荐问题列表
        """
        suggestions = []
        categories = set()
        filenames = set()
        
        # 收集文档信息
        for doc in documents[:3]:
            if 'category' in doc.metadata:
                categories.add(doc.metadata['category'])
            if 'filename' in doc.metadata:
                filenames.add(doc.metadata['filename'])
        
        # 生成基于分类的问题
        if categories:
            cat = list(categories)[0]
            suggestions.append(f"还有哪些关于{cat}的内容？")
        
        # 生成基于文件的问题
        if len(filenames) > 1:
            suggestions.append(f"这些文档之间有什么联系？")
        
        # 通用问题
        suggestions.append("能否提供更详细的说明？")
        suggestions.append("有相关的示例吗？")
        
        return suggestions[:3]

    def search_knowledge(self, query: str, mode: str = "混合检索", top_k: int = 5, progress=gr.Progress()) -> str:
        """
        搜索知识库

        Args:
            query: 搜索查询
            mode: 搜索模式
            top_k: 返回结果数量
            progress: Gradio进度跟踪器

        Returns:
            搜索结果
        """
        try:
            if not query.strip():
                return "⚠️ 请输入搜索关键词。"

            logger.info(f"[搜索] 开始搜索: query='{query}', mode='{mode}', top_k={top_k}")
            progress(0, desc="正在搜索...")

            # 根据模式选择检索方法
            if mode == "混合检索":
                progress(0.3, desc="执行混合检索...")
                documents = self.hybrid_retriever.hybrid_search(query, k=top_k)
                logger.info(f"[搜索] 混合检索完成，找到{len(documents)}个结果")
            else:
                progress(0.3, desc="执行语义检索...")
                documents = self.vector_store.search(query, k=top_k)
                logger.info(f"[搜索] 语义检索完成，找到{len(documents)}个结果")

            progress(0.7, desc="记录搜索历史...")
            # 记录搜索历史
            self.search_history_manager.add_search(
                query=query,
                mode=mode,
                top_k=top_k,
                results_count=len(documents)
            )

            if not documents:
                progress(1.0, desc="搜索完成")
                return f"❌ 未找到与 '{query}' 相关的文档。\n\n💡 提示：请确保已上传文档到知识库。"

            progress(0.9, desc="格式化结果...")
            # 构建搜索结果
            result = f"🔍 **搜索结果** (共找到 {len(documents)} 个相关文档块):\n\n"

            for i, doc in enumerate(documents, 1):
                filename = doc.metadata.get('filename', '未知文件')
                category = doc.metadata.get('category', '未分类')
                chunk_id = doc.metadata.get('chunk_id', 0)
                total_chunks = doc.metadata.get('total_chunks', '?')
                relevance = doc.metadata.get('relevance_score', 0.0)
                
                content_preview = doc.page_content[:200].strip()
                if len(doc.page_content) > 200:
                    content_preview += "..."

                result += f"**{i}. {filename}** (第{chunk_id + 1}/{total_chunks}块 | {category})\n"
                result += f"   📊 相关性: {relevance:.2f}\n"
                result += f"   📝 内容预览: {content_preview}\n\n"

            result += f"\n💡 *搜索模式: {mode} | 返回{len(documents)}个结果*"

            progress(1.0, desc="搜索完成!")
            logger.info(f"[搜索] 返回格式化结果，长度={len(result)}")
            return result

        except Exception as e:
            logger.error(f"[搜索] 搜索失败: {str(e)}", exc_info=True)
            return f"❌ 搜索失败: {str(e)}\n\n💡 提示：请查看日志文件获取详细信息。"

    def get_document_list(self) -> list:
        """
        获取所有文档列表
        
        Returns:
            文档列表 [[filename, chunks, category, file_type, file_size, last_updated], ...]
        """
        try:
            from datetime import datetime
            import time
            
            # 从collection获取所有metadata和IDs
            all_docs = self.vector_store.collection.get(include=['metadatas'])
            
            # 按文件名分组统计
            file_stats = {}
            for i, metadata in enumerate(all_docs['metadatas']):
                filename = metadata.get('filename', '未知')
                if filename not in file_stats:
                    # 获取上传时间，如果不存在则使用None标记为旧文档
                    upload_time = metadata.get('upload_time')
                    
                    file_stats[filename] = {
                        'filename': filename,
                        'chunks': 0,
                        'category': metadata.get('category', '未分类'),
                        'file_type': metadata.get('file_type', '未知'),
                        'file_size': metadata.get('file_size', 0),
                        'upload_time': upload_time  # 从metadata读取，可能为None
                    }
                file_stats[filename]['chunks'] += 1
            
            # 转换为列表格式
            result = []
            for f in file_stats.values():
                # 格式化文件大小
                size_mb = f['file_size']
                if size_mb < 0.01:
                    size_str = f"{size_mb * 1024:.1f} KB"
                else:
                    size_str = f"{size_mb:.2f} MB"
                
                # 格式化上传时间
                if f['upload_time'] is not None:
                    # 有时间戳，格式化显示
                    last_updated = datetime.fromtimestamp(f['upload_time']).strftime('%Y-%m-%d %H:%M')
                else:
                    # 旧文档，显示"未记录"
                    last_updated = "未记录"
                
                result.append([
                    f['filename'], 
                    f['chunks'], 
                    f['category'], 
                    f['file_type'],
                    size_str,
                    last_updated
                ])
            
            logger.info(f"获取文档列表成功，共{len(result)}个文件")
            return result
            
        except Exception as e:
            logger.error(f"获取文档列表失败: {str(e)}")
            return []
    
    def preview_document(self, filename: str, preview_chunks: int = 3) -> str:
        """
        预览文档内容
        
        Args:
            filename: 文件名
            preview_chunks: 预览的chunk数量（默认3个）
            
        Returns:
            预览内容（Markdown格式）
        """
        try:
            if not filename or not filename.strip():
                return "⚠️ 请输入要预览的文件名"
            
            logger.info(f"预览文档: {filename}")
            
            # 获取该文件的所有chunks
            all_docs = self.vector_store.collection.get(
                where={"filename": filename},
                include=['metadatas', 'documents']
            )
            
            if not all_docs['ids']:
                return f"⚠️ 未找到文件: {filename}"
            
            # 按chunk_id排序
            chunks_data = list(zip(
                all_docs['metadatas'],
                all_docs['documents']
            ))
            chunks_data.sort(key=lambda x: x[0].get('chunk_id', 0))
            
            # 构建预览内容
            total_chunks = len(chunks_data)
            preview_count = min(preview_chunks, total_chunks)
            
            result = f"# 📄 文档预览: {filename}\n\n"
            result += f"**总块数**: {total_chunks} | **预览块数**: {preview_count}\n\n"
            result += "---\n\n"
            
            for i in range(preview_count):
                metadata, content = chunks_data[i]
                chunk_id = metadata.get('chunk_id', i)
                result += f"### 📌 Chunk {chunk_id + 1}/{total_chunks}\n\n"
                
                # 显示metadata信息
                category = metadata.get('category', '未分类')
                file_type = metadata.get('file_type', '未知')
                result += f"**分类**: {category} | **类型**: {file_type}\n\n"
                
                # 显示内容（限制长度）
                preview_text = content[:500] if len(content) > 500 else content
                if len(content) > 500:
                    preview_text += "\n\n*（内容过长，已截断...）*"
                
                result += f"```\n{preview_text}\n```\n\n"
                result += "---\n\n"
            
            if total_chunks > preview_count:
                result += f"\n💡 *还有 {total_chunks - preview_count} 个chunk未显示*"
            
            logger.info(f"文档预览成功: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"文档预览失败: {str(e)}")
            return f"❌ 预览失败: {str(e)}"
    
    def delete_document_by_filename(self, filename: str) -> str:
        """
        删除指定文件名的所有文档块
        
        Args:
            filename: 要删除的文件名
            
        Returns:
            删除结果消息
        """
        try:
            if not filename.strip():
                return "⚠️ 请输入要删除的文件名"
            
            logger.info(f"准备删除文档: {filename}")
            
            # 调用底层删除方法
            success = self.vector_store.delete_documents({"filename": filename})
            
            if success:
                logger.info(f"文档删除成功: {filename}")
                return f"✅ 已成功删除文件: {filename}\n\n💡 提示：请刷新文档列表查看更新"
            else:
                logger.warning(f"文档未找到: {filename}")
                return f"⚠️ 未找到文件: {filename}\n\n💡 提示：请检查文件名是否正确（区分大小写）"
                
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return f"❌ 删除失败: {str(e)}"
    
    def update_document(self, old_filename: str, file) -> str:
        """
        更新文档（先删除旧版本，再上传新版本）
        
        Args:
            old_filename: 要删除的旧文件名
            file: 新版本文件对象
            
        Returns:
            更新结果消息
        """
        try:
            if not old_filename or not old_filename.strip():
                return "⚠️ 请输入要更新的文件名"
            
            if file is None:
                return "⚠️ 请选择新文件"
            
            logger.info(f"开始更新文档: {old_filename}")
            
            # 1. 删除旧版本
            logger.info(f"删除旧版本: {old_filename}")
            delete_success = self.vector_store.delete_documents({"filename": old_filename})
            
            if not delete_success:
                logger.warning(f"旧文档未找到: {old_filename}，继续上传新文档")
            
            # 2. 上传新版本
            file_path = file.name if hasattr(file, 'name') else str(file)
            logger.info(f"上传新版本: {file_path}")
            result = self.load_and_process_files([file_path])
            
            return f"✅ 文档更新成功！\n\n旧文档: {old_filename}\n新文档: {Path(file_path).name}\n\n{result}"
            
        except Exception as e:
            logger.error(f"文档更新失败: {str(e)}", exc_info=True)
            return f"❌ 更新失败: {str(e)}"

    def get_statistics(self) -> str:
        """
        获取知识库统计信息

        Returns:
            统计信息
        """
        try:
            stats = self.vector_store.get_stats()

            if not stats:
                return "知识库为空或统计数据获取失败。"

            # 获取文档列表
            file_list = self.get_document_list()

            result = "📊 **知识库统计信息**\n\n"
            
            # 文件和块统计
            result += f"• 📁 **文件数**: {len(file_list)}\n"
            result += f"• 📄 **文档块总数**: {stats.get('total_documents', 0)}\n"
            
            if len(file_list) > 0:
                avg_chunks = stats.get('total_documents', 0) / len(file_list)
                result += f"• 📊 **平均分块数**: {avg_chunks:.1f}\n"
            
            # 按分类统计
            category_counts = {}
            for file_data in file_list:
                cat = file_data[2] if len(file_data) > 2 else '未分类'  # category在索引2
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            if category_counts:
                result += f"\n**📂 分类统计**:\n"
                for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                    result += f"  • {cat}: {count}个文件\n"
            
            # 数据库信息
            result += f"\n• 🗄️ **数据库路径**: {stats.get('vector_db_path', 'N/A')}\n"
            result += f"• 📦 **集合名称**: {stats.get('collection_name', 'N/A')}\n"
            result += f"• 🔢 **向量维度**: {stats.get('embeddings_dimension', 'N/A')}\n"

            # Embedding方法信息
            if hasattr(self.vector_store, 'embedding_manager'):
                method_info = self.vector_store.embedding_manager.get_method_info()
                result += f"\n**🔧 Embedding配置**:\n"
                result += f"  • 方法: {method_info['description']}\n"
                result += f"  • 维度: {method_info['dimension']}\n"
                result += f"  • 免费: {'是 ✅' if method_info['is_free'] else '否'}\n"

            # 显示LLM功能状态
            if self.llm_enabled:
                result += f"\n**🤖 AI功能**: ✅ 已启用 (DeepSeek)\n"
                # 获取embedding信息
                if self.llm_manager and hasattr(self.llm_manager, 'get_embedding_info'):
                    embedding_info = self.llm_manager.get_embedding_info()
                    result += f"  • 🔧 Embedding: {embedding_info['description']}\n"
            else:
                result += f"\n**🤖 AI功能**: ⚠️ 简化模式\n"
                result += f"  • 💡 提示: 配置DEEPSEEK_API_KEY以启用智能问答\n"

            result += "\n💡 *提示：上传更多文档以获得更详细的统计信息*"

            return result

        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return f"获取统计信息失败: {str(e)}"

    def create_interface(self) -> gr.Interface:
        """创建Gradio界面"""

        # 设置Gradio主题
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        )

        with gr.Blocks(theme=theme, title=settings.APP_NAME) as demo:
            # 标题
            gr.Markdown(
                f"""
                # 🤖 {settings.APP_NAME} v{settings.APP_VERSION}
                
                基于LangChain和RAG技术的个人知识管理助手
                
                支持文档上传、智能检索、问答交互等功能
                """
            )

            with gr.Tabs():
                # Tab 1: 文档上传
                with gr.TabItem("📤 文档上传"):
                    with gr.Row():
                        file_input = gr.File(
                            label="选择文档文件",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".docx"]
                        )

                    with gr.Row():
                        upload_btn = gr.Button("🚀 处理文档", variant="primary")

                    upload_status = gr.Textbox(
                        label="处理状态",
                        lines=5,
                        max_lines=10
                    )

                    upload_btn.click(
                        self.load_and_process_files,
                        inputs=[file_input],
                        outputs=[upload_status]
                    ).then(
                        lambda: None,  # 上传成功后清空文件选择器
                        outputs=[file_input]
                    )

                # Tab 2: 智能问答
                with gr.TabItem("💬 智能问答"):
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=500,
                        type='messages'
                    )

                    msg_input = gr.Textbox(
                        label="您的问题",
                        placeholder="请输入您想了解的问题...",
                        scale=4
                    )

                    with gr.Row():
                        send_btn = gr.Button("💭 提问", variant="primary")
                        clear_btn = gr.Button("🗑️ 清空对话")

                    def process_user_message(message, history):
                        """处理用户消息并生成AI回答"""
                        print(f"[DEBUG] process_user_message called! Message: {message}")
                        print(f"[DEBUG] Current history length: {len(history) if history else 0}")

                        if not message or not message.strip():
                            return "", history or []

                        try:
                            # 调用chat_with_knowledge函数生成回答
                            ai_response = self.chat_with_knowledge(message, history or [])

                            # 构建新的对话历史（添加用户问题和AI回答）
                            new_history = (history or []) + [
                                {"role": "user", "content": message},
                                {"role": "assistant", "content": ai_response}
                            ]

                            # 限制历史长度（保留最近10轮对话 = 20条消息）
                            if len(new_history) > 20:
                                new_history = new_history[-20:]

                            print(f"[DEBUG] Generated response, total history entries: {len(new_history)}")

                            # 返回清空的输入框和完整的对话历史
                            return "", new_history

                        except Exception as e:
                            print(f"[ERROR] Failed to process message: {str(e)}")
                            error_msg = f"抱歉，处理您的问题时出现错误: {str(e)}"

                            # 构建错误情况下的对话历史
                            new_history = (history or []) + [
                                {"role": "user", "content": message},
                                {"role": "assistant", "content": error_msg}
                            ]

                            return "", new_history

                    def clear_chat():
                        """清空对话历史"""
                        print(f"[DEBUG] Chat history cleared")
                        return []

                    # 绑定事件处理器
                    msg_input.submit(
                        process_user_message,
                        inputs=[msg_input, chatbot],
                        outputs=[msg_input, chatbot]
                    )

                    send_btn.click(
                        process_user_message,
                        inputs=[msg_input, chatbot],
                        outputs=[msg_input, chatbot]
                    )

                    clear_btn.click(
                        clear_chat,
                        outputs=[chatbot]
                    )

                # Tab 3: 搜索功能
                with gr.TabItem("🔍 文档搜索"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            search_input = gr.Textbox(
                                label="搜索关键词",
                                placeholder="输入关键词进行搜索..."
                            )
                        with gr.Column(scale=1):
                            history_dropdown = gr.Dropdown(
                                label="📝 搜索历史",
                                choices=[],
                                interactive=True,
                                allow_custom_value=False
                            )

                    with gr.Row():
                        search_mode = gr.Dropdown(
                            choices=["混合检索", "语义检索"],
                            value="混合检索",
                            label="搜索模式"
                        )

                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="返回结果数量",
                            elem_id="top_k_slider"
                        )

                    with gr.Row():
                        search_btn = gr.Button("🔍 搜索", variant="primary")
                        refresh_history_btn = gr.Button("🔄 刷新历史")
                        clear_history_btn = gr.Button("🗑️ 清空历史")

                    search_results = gr.Textbox(
                        label="搜索结果",
                        lines=15,
                        max_lines=20
                    )

                    # 搜索历史显示
                    with gr.Accordion("📋 搜索历史记录", open=False):
                        history_display = gr.Markdown(value="暂无搜索历史")

                    # 定义辅助函数
                    def refresh_history_dropdown():
                        """刷新历史下拉菜单"""
                        choices = self.search_history_manager.get_history_dropdown_choices(20)
                        return gr.Dropdown(choices=choices)

                    def refresh_history_display():
                        """刷新历史显示"""
                        return self.search_history_manager.format_history_for_display(15)

                    def clear_search_history():
                        """清空搜索历史"""
                        self.search_history_manager.clear_history()
                        return gr.Dropdown(choices=[]), "✅ 搜索历史已清空"

                    def use_history_query(selected_query):
                        """使用历史查询"""
                        if selected_query:
                            return selected_query
                        return ""

                    # 绑定事件
                    search_btn.click(
                        self.search_knowledge,
                        inputs=[search_input, search_mode, top_k_slider],
                        outputs=[search_results]
                    ).then(
                        refresh_history_dropdown,
                        outputs=[history_dropdown]
                    ).then(
                        refresh_history_display,
                        outputs=[history_display]
                    )
                    
                    # 支持Enter键搜索
                    search_input.submit(
                        self.search_knowledge,
                        inputs=[search_input, search_mode, top_k_slider],
                        outputs=[search_results]
                    ).then(
                        refresh_history_dropdown,
                        outputs=[history_dropdown]
                    ).then(
                        refresh_history_display,
                        outputs=[history_display]
                    )

                    # 点击历史记录填充到搜索框
                    history_dropdown.change(
                        use_history_query,
                        inputs=[history_dropdown],
                        outputs=[search_input]
                    )

                    # 刷新历史按钮
                    refresh_history_btn.click(
                        refresh_history_dropdown,
                        outputs=[history_dropdown]
                    ).then(
                        refresh_history_display,
                        outputs=[history_display]
                    )

                    # 清空历史按钮
                    clear_history_btn.click(
                        clear_search_history,
                        outputs=[history_dropdown, search_results]
                    ).then(
                        refresh_history_display,
                        outputs=[history_display]
                    )

                # Tab 4: 统计信息
                with gr.TabItem("📊 统计信息"):
                    stats_btn = gr.Button("📈 获取统计信息", variant="primary")
                    stats_display = gr.Textbox(
                        label="统计信息",
                        lines=15,
                        max_lines=20
                    )

                    stats_btn.click(
                        self.get_statistics,
                        outputs=[stats_display]
                    )

                # Tab 5: 文档管理
                with gr.TabItem("🗂️ 文档管理"):
                    gr.Markdown(
                        """
                        ### 📁 知识库文档管理
                        
                        在这里您可以查看、删除和更新知识库中的文档。
                        
                        💡 **提示**: 点击文件名可以自动填充到下方的删除/更新输入框
                        """
                    )
                    
                    # 刷新按钮
                    with gr.Row():
                        refresh_list_btn = gr.Button("🔄 刷新文档列表", variant="primary")
                    
                    # 文档列表展示
                    file_list_display = gr.Dataframe(
                        headers=["文件名", "分块数", "分类", "类型", "文件大小", "最后更新"],
                        label="知识库文档列表",
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown("---")
                    
                    # 删除文档功能
                    with gr.Row():
                        with gr.Column(scale=3):
                            delete_filename_input = gr.Textbox(
                                label="📝 要删除的文件名",
                                placeholder="输入完整的文件名（如：python_learning_notes.md）",
                                info="请从上方列表中复制文件名"
                            )
                        with gr.Column(scale=1):
                            delete_btn = gr.Button("🗑️ 删除文档", variant="stop")
                    
                    delete_status = gr.Textbox(
                        label="删除状态",
                        lines=3,
                        max_lines=5
                    )
                    
                    gr.Markdown("---")
                    
                    # 更新文档功能
                    gr.Markdown("### 🔄 更新文档")
                    
                    with gr.Row():
                        update_filename_input = gr.Textbox(
                            label="📝 要更新的文件名",
                            placeholder="输入要替换的文档名称（如：python_learning_notes.md）",
                            info="旧文档将被删除，新文档将被上传"
                        )
                    
                    with gr.Row():
                        update_file_input = gr.File(
                            label="📤 选择新文档",
                            file_count="single",
                            file_types=[".pdf", ".txt", ".md", ".docx"]
                        )
                    
                    with gr.Row():
                        update_btn = gr.Button("🔄 更新文档", variant="primary")
                    
                    update_status = gr.Textbox(
                        label="更新状态",
                        lines=5,
                        max_lines=10
                    )
                    
                    gr.Markdown("---")
                    
                    # 文档预览功能
                    gr.Markdown("### 👀 文档预览")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            preview_filename_input = gr.Textbox(
                                label="📝 要预览的文件名",
                                placeholder="输入完整的文件名（如：python_learning_notes.md）",
                                info="请从上方列表中复制文件名"
                            )
                        with gr.Column(scale=1):
                            preview_chunks_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="预览块数",
                                info="选择要预览的chunk数量"
                            )
                        with gr.Column(scale=1):
                            preview_btn = gr.Button("👀 预览文档", variant="primary")
                    
                    preview_display = gr.Markdown(
                        label="文档预览",
                        value="点击'预览文档'按钮查看文档内容"
                    )
                    
                    # 绑定事件处理器
                    def refresh_file_list():
                        """刷新文件列表并返回格式化的数据"""
                        try:
                            file_list = self.get_document_list()
                            # get_document_list() 已经返回列表格式，直接返回即可
                            return file_list if file_list else [["暂无文档", "0", "-", "-", "-", "-"]]
                        
                        except Exception as e:
                            logger.error(f"刷新文件列表失败: {str(e)}")
                            return [["错误", str(e), "-", "-", "-", "-"]]
                    
                    def select_file_from_list(evt: gr.SelectData):
                        """从列表中选择文件，自动填充文件名"""
                        if evt.value:
                            # evt.value 是选中单元格的值
                            # evt.index 是 [row, col]
                            row, col = evt.index
                            file_list = self.get_document_list()
                            if row < len(file_list):
                                filename = file_list[row][0]  # 第一列是文件名
                                return filename, filename, filename
                        return "", "", ""
                    
                    # 绑定按钮事件
                    refresh_list_btn.click(
                        refresh_file_list,
                        outputs=[file_list_display]
                    )
                    
                    # 点击表格自动填充文件名
                    file_list_display.select(
                        select_file_from_list,
                        outputs=[delete_filename_input, update_filename_input, preview_filename_input]
                    )
                    
                    delete_btn.click(
                        self.delete_document_by_filename,
                        inputs=[delete_filename_input],
                        outputs=[delete_status]
                    ).then(
                        refresh_file_list,  # 删除后自动刷新列表
                        outputs=[file_list_display]
                    ).then(
                        lambda: "",  # 删除后清空输入框
                        outputs=[delete_filename_input]
                    )
                    
                    update_btn.click(
                        self.update_document,
                        inputs=[update_filename_input, update_file_input],
                        outputs=[update_status]
                    ).then(
                        refresh_file_list,  # 更新后自动刷新列表
                        outputs=[file_list_display]
                    ).then(
                        lambda: ("", None),  # 更新后清空输入框和文件选择器
                        outputs=[update_filename_input, update_file_input]
                    )
                    
                    # 绑定预览事件
                    preview_btn.click(
                        self.preview_document,
                        inputs=[preview_filename_input, preview_chunks_slider],
                        outputs=[preview_display]
                    )

            # 页脚
            gr.Markdown(
                """
                ---
                
                ## 🚀 使用说明
                
                1. **上传文档**: 选择文档文件并点击"处理文档"
                2. **智能问答**: 在对话中输入问题，系统将基于知识库回答
                3. **文档搜索**: 使用关键词搜索相关文档内容
                4. **查看统计**: 了解知识库的整体状况
                
                ## ⚙️ 技术特性
                
                - 🧠 **智能架构**: 自动选择最佳embedding方法 (OpenAI → Sentence Transformers → text-hash)
                - 📚 **RAG技术**: 检索增强生成，确保答案准确
                - 🔍 **混合搜索**: 向量搜索 + 关键词搜索
                - 💡 **智能分类**: 自动文档分类和标签
                - 📊 **成本优化**: 无API费用 + 智能降级策略
                - 🔄 **自动回退**: API不可用时自动切换到免费方案
                """
            )

        return demo

def create_app() -> gr.Interface:
    """创建应用实例"""
    app = KnowledgeManagerApp()
    return app.create_interface()

if __name__ == "__main__":
    # 创建并启动应用
    interface = create_app()

    # 启动服务器
    interface.launch(
        server_name=settings.GRADIO_SERVER_HOST,
        server_port=settings.GRADIO_SERVER_PORT,
        share=settings.GRADIO_SHARE,
        debug=settings.GRADIO_DEBUG,
        show_error=True,
        quiet=False
    )
