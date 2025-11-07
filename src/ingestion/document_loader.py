"""
文档加载和处理模块
支持多种文件格式的文档处理和文本分割
"""

import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """文档加载器"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_file(self, file_path: str) -> List[LangChainDocument]:
        """
        加载单个文档文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            分割后的文档块列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = file_path.suffix.lower()
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise ValueError(f"文件过大: {file_size_mb:.1f}MB > {settings.MAX_FILE_SIZE_MB}MB")
        
        if file_extension not in settings.SUPPORTED_FILE_TYPES:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        try:
            # 提取文本内容
            if file_extension == ".pdf":
                content = self._extract_pdf_text(file_path)
            elif file_extension == ".docx":
                content = self._extract_docx_text(file_path)
            elif file_extension == ".md":
                content = self._extract_markdown_text(file_path)
            elif file_extension == ".txt":
                content = self._extract_text_text(file_path)
            else:
                raise ValueError(f"未实现处理逻辑: {file_extension}")
            
            # 分割文档
            chunks = self.text_splitter.split_text(content)
            
            # 转换为LangChain文档格式
            documents = []
            for i, chunk in enumerate(chunks):
                doc = LangChainDocument(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "file_type": file_extension,
                        "file_size": file_size_mb
                    }
                )
                documents.append(doc)
            
            logger.info(f"处理文件 {file_path.name}: {len(documents)} 个文档块")
            return documents
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")
            raise
    
    def load_multiple_files(self, file_paths: List[str]) -> List[LangChainDocument]:
        """
        批量处理多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            所有文档块的列表
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = self.load_file(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"跳过文件 {file_path}: {str(e)}")
                continue
        
        logger.info(f"批量处理完成: {len(file_paths)} 个文件, {len(all_documents)} 个文档块")
        return all_documents
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """提取PDF文本"""
        text_content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)
                except Exception as e:
                    logger.warning(f"PDF页面 {page_num} 处理失败: {str(e)}")
                    continue
        
        return "\n".join(text_content)
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """提取DOCX文本"""
        doc = Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        return "\n".join(text_content)
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """提取Markdown文本"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_text_text(self, file_path: Path) -> str:
        """提取纯文本"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

class DocumentClassifier:
    """文档分类器"""
    
    def __init__(self):
        self.categories = settings.DOCUMENT_CATEGORIES
        self.priorities = settings.PRIORITY_LEVELS
    
    def classify_document(self, document: LangChainDocument) -> Dict[str, Any]:
        """
        对文档进行分类
        
        Args:
            document: 文档对象
            
        Returns:
            分类结果字典
        """
        # 简单的基于关键词的分类规则
        # 在实际应用中，这里可以集成更复杂的ML模型
        content = document.page_content.lower()
        
        category_rules = {
            "工作": ["项目", "会议", "工作", "任务", "计划", "报告", "需求"],
            "学习": ["学习", "教程", "课程", "笔记", "知识", "技能", "练习"],
            "个人": ["日记", "想法", "感悟", "生活", "家庭", "个人", "心情"],
            "参考": ["参考", "文档", "手册", "指南", "规范", "标准", "资料"],
            "研究": ["研究", "分析", "实验", "数据", "结论", "发现", "探索"],
            "想法": ["想法", "创意", "创新", "设计", "概念", "思路", "灵感"]
        }
        
        scores = {}
        for category, keywords in category_rules.items():
            score = sum(1 for keyword in keywords if keyword in content)
            scores[category] = score
        
        # 选择得分最高的类别
        if scores:
            category = max(scores, key=scores.get)
            if scores[category] == 0:
                category = "参考"  # 默认类别
        else:
            category = "参考"
        
        # 根据文件类型和内容长度确定优先级
        if "重要" in content or "urgent" in content:
            priority = "高"
        elif len(content) > 2000:
            priority = "中"
        else:
            priority = "低"
        
        # 生成简短摘要
        summary = content[:100] + "..." if len(content) > 100 else content
        
        return {
            "category": category,
            "priority": priority,
            "summary": summary,
            "tags": self._extract_keywords(content),
            "confidence": scores.get(category, 0) / max(len(content.split()) / 100, 1)
        }
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        # 在实际应用中可以使用更复杂的NLP技术
        common_words = {"的", "了", "是", "在", "有", "和", "与", "或", "但", "而"}
        
        words = content.split()
        word_freq = {}
        
        for word in words:
            word = word.strip('，。！？；：""''()（）[]【】')
            if len(word) > 1 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回频率最高的5个词作为标签
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, freq in top_words]
