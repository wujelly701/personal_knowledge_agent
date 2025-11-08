"""
文档分类器模块
支持LLM智能分类和规则分类两种模式
"""

import logging
from typing import List, Dict, Any
from langchain_core.documents import Document as LangChainDocument
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """文档分类器 - 支持LLM智能分类"""
    
    def __init__(self, llm_manager=None):
        """
        初始化分类器
        
        Args:
            llm_manager: LLM管理器（可选）。如果提供，则使用LLM智能分类；否则使用规则分类
        """
        self.categories = settings.DOCUMENT_CATEGORIES
        self.priorities = settings.PRIORITY_LEVELS
        self.llm_manager = llm_manager
        self.llm_enabled = llm_manager is not None and hasattr(llm_manager, 'chat_model') and llm_manager.chat_model is not None
        
        if self.llm_enabled:
            logger.info("✅ 文档分类器：使用LLM智能分类模式")
        else:
            logger.info("ℹ️ 文档分类器：使用规则分类模式（简化）")
    
    def classify_document(self, document: LangChainDocument) -> Dict[str, Any]:
        """
        对文档进行分类
        
        Args:
            document: 文档对象
            
        Returns:
            分类结果字典
        """
        content = document.page_content
        
        # 如果启用LLM，优先使用智能分类
        if self.llm_enabled:
            try:
                return self._classify_with_llm(content)
            except Exception as e:
                logger.warning(f"LLM分类失败，回退到规则分类: {e}")
                # 回退到规则分类
        
        # 规则分类（简化模式）
        return self._classify_with_rules(content)
    
    def _classify_with_llm(self, content: str) -> Dict[str, Any]:
        """
        使用LLM进行智能分类
        
        Args:
            content: 文档内容
            
        Returns:
            分类结果字典
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # 限制内容长度（避免token过多）
        analysis_content = content[:1000] if len(content) > 1000 else content
        
        prompt = f"""请分析以下文档内容，并提供分类信息。

文档内容：
{analysis_content}

请按以下格式返回（每行一个字段）：
类别: [从以下选择一个: 工作, 学习, 个人, 参考, 研究, 想法]
优先级: [从以下选择一个: 高, 中, 低]
标签: [提取3-5个关键标签，用逗号分隔]
摘要: [一句话总结，不超过50字]

要求：
1. 类别必须是列表中的一个
2. 标签要具体、简洁（2-4字）
3. 摘要要突出核心主题"""

        messages = [
            SystemMessage(content="你是一个文档分类专家，擅长快速准确地分类和总结文档内容。"),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm_manager.chat_model.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析LLM响应
            result = self._parse_llm_response(response_text, content)
            logger.info(f"LLM分类完成: 类别={result['category']}, 标签={result['tags']}")
            return result
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise
    
    def _parse_llm_response(self, response_text: str, original_content: str) -> Dict[str, Any]:
        """
        解析LLM返回的分类结果
        
        Args:
            response_text: LLM响应文本
            original_content: 原始文档内容
            
        Returns:
            标准化的分类结果
        """
        lines = response_text.strip().split('\n')
        
        # 默认值
        category = "参考"
        priority = "中"
        tags = []
        summary = original_content[:100] + "..."
        
        # 解析每一行
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if '类别' in line or 'Category' in line.lower():
                # 提取类别
                for cat in self.categories:
                    if cat in line:
                        category = cat
                        break
            
            elif '优先级' in line or 'Priority' in line.lower():
                # 提取优先级
                if '高' in line or 'high' in line.lower():
                    priority = "高"
                elif '低' in line or 'low' in line.lower():
                    priority = "低"
                else:
                    priority = "中"
            
            elif '标签' in line or 'Tag' in line.lower():
                # 提取标签
                tags_part = line.split(':', 1)[-1].strip()
                # 移除可能的方括号
                tags_part = tags_part.strip('[]')
                tags = [t.strip() for t in tags_part.split(',') if t.strip()]
            
            elif '摘要' in line or 'Summary' in line.lower():
                # 提取摘要
                summary_part = line.split(':', 1)[-1].strip()
                if summary_part and len(summary_part) > 10:
                    summary = summary_part
        
        # 验证和修正
        if category not in self.categories:
            category = "参考"
        
        if not tags:
            # 如果LLM没有返回标签，使用规则提取
            tags = self._extract_keywords(original_content)[:5]
        
        return {
            "category": category,
            "priority": priority,
            "summary": summary,
            "tags": ",".join(tags[:5]),  # 最多5个标签
            "confidence": 0.85,  # LLM分类置信度较高
            "classification_method": "llm"  # 标记使用LLM分类
        }
    
    def _classify_with_rules(self, content: str) -> Dict[str, Any]:
        """
        使用规则进行分类（简化模式）
        
        Args:
            content: 文档内容
            
        Returns:
            分类结果字典
        """
        content_lower = content.lower()
        
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
            score = sum(1 for keyword in keywords if keyword in content_lower)
            scores[category] = score
        
        # 选择得分最高的类别
        if scores:
            category = max(scores, key=scores.get)
            if scores[category] == 0:
                category = "参考"  # 默认类别
        else:
            category = "参考"
        
        # 根据文件类型和内容长度确定优先级
        if "重要" in content_lower or "urgent" in content_lower:
            priority = "高"
        elif len(content) > 2000:
            priority = "中"
        else:
            priority = "低"
        
        # 生成简短摘要
        summary = content[:100] + "..." if len(content) > 100 else content
        
        # 提取关键词并转换为字符串
        keywords = self._extract_keywords(content)
        
        return {
            "category": category,
            "priority": priority,
            "summary": summary,
            "tags": ",".join(keywords),  # 转换为字符串
            "confidence": round(scores.get(category, 0) / max(len(content.split()) / 100, 1), 3),
            "classification_scores": "; ".join([f"{k}:{v}" for k, v in scores.items() if v > 0]),  # 添加scores作为字符串
            "classification_method": "rules"  # 标记使用规则分类
        }
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        # 在实际应用中可以使用更复杂的NLP技术
        common_words = {"的", "了", "是", "在", "有", "和", "与", "或", "但", "而", "这", "那", "个", "等", "及"}
        
        words = content.split()
        word_freq = {}
        
        for word in words:
            word = word.strip('，。！？；：""''()（）[]【】')
            if len(word) > 1 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回频率最高的5个词作为标签
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, freq in top_words]
