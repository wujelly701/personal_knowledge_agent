"""
提示词模板和配置
"""

from typing import Dict, Any

# RAG系统提示词模板
RAG_SYSTEM_PROMPT = """你是一个专业的知识管理助手。基于提供的上下文内容回答用户问题。

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
CLASSIFICATION_PROMPT = """请将以下文档内容分类到预定义的类别中。

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

# 摘要生成提示词
SUMMARY_PROMPT = """请为以下文档生成简洁的摘要：

{content}

摘要要求：
1. 提取关键主题和要点
2. 保持客观准确
3. 使用简洁的中文表达
4. 结构化呈现要点
5. 长度控制在{length}字以内

摘要："""

# 关键词提取提示词
KEYWORD_EXTRACTION_PROMPT = """请从以下文本中提取关键词和标签：

{content}

要求：
1. 提取5-10个关键词
2. 关键词应具有代表性
3. 避免过于宽泛的词汇
4. 优先选择名词和名词短语
5. 以逗号分隔的列表形式返回

关键词："""

# 错误处理提示词
ERROR_HANDLING_PROMPT = """当用户问题与文档内容不匹配时，使用以下回应：

"我在现有文档中没有找到与您问题直接相关的信息。您可以：

1. 检查问题表述是否准确
2. 上传相关文档到知识库
3. 使用更具体的关键词搜索
4. 联系管理员添加相关内容"

请根据具体情况调整回应。"""

# 搜索分析提示词
SEARCH_ANALYSIS_PROMPT = """分析用户的搜索查询，提取以下信息：

查询：{query}

请返回JSON格式的分析结果：
- semantic_query: 去除过滤条件的核心语义查询
- keywords: 提取的关键词列表
- filters: 可能的过滤条件（分类、时间等）
- intent: 搜索意图（查找、比较、分析等）

分析结果："""

# 对话上下文提示词模板
CONVERSATION_CONTEXT_PROMPT = """之前的对话历史：
{conversation_history}

基于之前的对话历史和当前问题，请提供连贯的回答：

当前问题：{question}

要求：
1. 保持对话连贯性
2. 引用之前讨论的内容
3. 避免重复已说明的信息
4. 适当引导对话方向"""

# 性能优化提示词
PERFORMANCE_TIPS = {
    "chunking": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separator": ["\n\n", "\n", " ", ""]
    },
    "retrieval": {
        "top_k": 5,
        "similarity_threshold": 0.7,
        "diversity_factor": 0.3
    },
    "generation": {
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 0.9
    }
}

def format_prompt(template: str, **kwargs) -> str:
    """格式化提示词模板"""
    return template.format(**kwargs)

def get_model_config(model_type: str) -> Dict[str, Any]:
    """获取模型配置"""
    configs = {
        "embedding": {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "max_tokens": 8191
        },
        "chat": {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "fallback": {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 500
        }
    }
    
    return configs.get(model_type, configs["chat"])

def evaluate_answer_quality(answer: str, context: str, query: str) -> Dict[str, Any]:
    """评估回答质量"""
    quality_metrics = {
        "length_score": min(len(answer) / 200, 1.0),
        "keyword_match": 0.0,  # 需要实现关键词匹配逻辑
        "context_relevance": 0.0,  # 需要实现上下文相关性计算
        "fluency_score": 0.8,  # 简化处理
        "accuracy_score": 0.7   # 简化处理
    }
    
    # 计算综合质量分数
    quality_metrics["overall_score"] = sum(quality_metrics.values()) / len(quality_metrics)
    
    return quality_metrics
