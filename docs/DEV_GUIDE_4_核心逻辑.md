# 个人知识管理Agent系统 - 核心业务逻辑文档

## 文档信息
- **版本**: v1.0.0
- **创建日期**: 2025-11-07
- **适用人员**: 后端开发、算法工程师、新入职开发者

---

## 1. 核心业务概述

本系统基于 **RAG (Retrieval-Augmented Generation)** 技术实现知识问答，核心业务逻辑包括:
1. 文档处理与向量化
2. 向量检索与相似度匹配
3. 上下文构建与答案生成
4. 多方案Embedding降级策略

---

## 2. RAG问答流程详解

### 2.1 完整RAG流程图

```
用户提问
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 查询预处理                                                │
│  • 文本清洗                                                   │
│  • 查询优化 (可选)                                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 查询Embedding生成                                         │
│  • 使用与文档相同的embedding方法                              │
│  • 生成查询向量 (384/1536维)                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 向量相似度检索                                            │
│  • ChromaDB HNSW索引搜索                                      │
│  • 计算欧几里得距离                                           │
│  • 返回Top-K候选文档 (k=5)                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 相关性评分与排序                                          │
│  • 距离归一化                                                 │
│  • 动态阈值调整                                               │
│  • relevance_score计算                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 上下文构建                                                │
│  • 整合检索到的文档片段                                       │
│  • 添加来源标识 [来源: 文件名]                                │
│  • 格式化为LLM可读格式                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌────────────────────┐    ┌────────────────────┐
│ 有DeepSeek API     │    │ 无API (降级模式)    │
├────────────────────┤    ├────────────────────┤
│ 6a. LLM生成        │    │ 6b. 简化回答        │
│ • 调用DeepSeek API │    │ • 返回检索结果摘要  │
│ • 提示词工程       │    │ • 展示相关文档片段  │
│ • 引用来源         │    │ • 标注相关性分数    │
│ • 置信度评估       │    │                    │
└────────┬───────────┘    └────────┬───────────┘
         │                         │
         └───────────┬─────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  7. 结果返回                                                  │
│  • Markdown格式化                                             │
│  • 添加来源引用                                               │
│  • 显示置信度/相关性                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心代码实现

#### 2.2.1 chat_with_knowledge() 主流程
位置: `src/api/gradio_app.py`

```python
def chat_with_knowledge(self, message: str, history: List[Dict[str, str]]) -> str:
    # 1. 输入验证
    if not message.strip():
        return "请输入您的问题。"
    
    # 2. 向量检索
    retrieved_docs = self.vector_store.search(message, k=settings.TOP_K)
    
    if not retrieved_docs:
        return "我在知识库中没有找到相关信息..."
    
    # 3. 智能模式: 使用RAG生成器
    if self.llm_enabled and self.rag_generator:
        result = self.rag_generator.generate_answer(
            query=message,
            documents=retrieved_docs,
            include_sources=True
        )
        # 构建答案 (带AI生成内容、置信度、来源)
        answer = self._build_ai_answer(result)
        return answer
    
    # 4. 简化模式: 基于检索结果的摘要
    answer = self._build_simple_answer(retrieved_docs, message)
    return answer
```

#### 2.2.2 RAG生成器核心逻辑
位置: `src/generation/llm_manager.py`

```python
def generate_answer(self, query: str, documents: List[LangChainDocument], 
                   include_sources: bool = True) -> Dict[str, Any]:
    # 1. 构建上下文
    context = self._build_context(documents, include_sources)
    
    # 2. 构建提示词
    messages = [
        SystemMessage(content=self.rag_system_prompt.format(
            context=context,
            question=query
        )),
        HumanMessage(content="请基于上述上下文回答问题。")
    ]
    
    # 3. 调用LLM
    response = self.chat_model.invoke(messages)
    answer = response.content
    
    # 4. 估算置信度
    confidence = self._estimate_confidence(query, documents, answer)
    
    # 5. 返回结果
    return {
        "answer": answer,
        "confidence": confidence,
        "sources": self._extract_sources(documents),
        "metadata": {
            "query": query,
            "retrieved_docs": len(documents),
            "tokens_used": response.usage_metadata.get('total_tokens', 0)
        }
    }
```

---

## 3. Embedding方案详解

### 3.1 多方案架构

系统支持4种Embedding方案，按优先级自动选择:

```
┌────────────────────────────────────────────────────────────┐
│              Embedding方案选择策略                          │
├────────────────────────────────────────────────────────────┤
│  1️⃣ OpenAI text-embedding-3-small (最高质量)              │
│     条件: 有OPENAI_API_KEY                                 │
│     维度: 1536                                             │
│     特点: 云端API、高质量、需付费                          │
│              ↓ (API不可用)                                 │
│                                                            │
│  2️⃣ Sentence Transformers (all-MiniLM-L6-v2)             │
│     条件: 安装了sentence-transformers库                    │
│     维度: 384                                              │
│     特点: 本地运行、免费、高质量                           │
│              ↓ (库未安装/网络问题)                         │
│                                                            │
│  3️⃣ 文本哈希 (Text Hash)                                  │
│     条件: 无条件 (保底方案)                                │
│     维度: 384                                              │
│     特点: 零依赖、极快、中等质量                           │
│                                                            │
│  4️⃣ TF-IDF词袋 (可选)                                     │
│     条件: 安装了scikit-learn                               │
│     维度: 1000                                             │
│     特点: 关键词优化、适合关键词搜索                       │
└────────────────────────────────────────────────────────────┘
```

### 3.2 各方案实现细节

#### 3.2.1 Sentence Transformers (推荐)
位置: `src/storage/embedding_manager.py`

**实现**:
```python
def _try_sentence_transformers(self) -> bool:
    from sentence_transformers import SentenceTransformer
    
    model_name = "all-MiniLM-L6-v2"
    self.model = SentenceTransformer(model_name)
    self.method = "sentence-transformers"
    self.embedding_dim = 384
    return True

def _embed_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
    return self.model.encode(texts, show_progress_bar=False).tolist()
```

**特点**:
- ✅ 质量高: 在多个基准测试中表现优异
- ✅ 速度快: 本地GPU加速 (如果有CUDA)
- ✅ 离线可用: 模型下载后可离线使用
- ⚠️ 首次需联网: 下载模型文件 (~90MB)

#### 3.2.2 文本哈希 (保底方案)
位置: `src/storage/embedding_manager.py`

**实现**:
```python
def _embed_text_hash(self, texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        hash_features = []
        for i in range(self.embedding_dim):  # 384维
            # 使用不同种子生成哈希
            hash_obj = hashlib.md5(f"{text}_{i}".encode())
            hash_value = int(hash_obj.hexdigest(), 16) % 1000
            hash_features.append(hash_value / 1000.0)
        embeddings.append(hash_features)
    return embeddings
```

**特点**:
- ✅ 零依赖: 只需Python标准库
- ✅ 极快: 纯Python实现，无模型加载
- ✅ 稳定性: 相同文本始终生成相同向量
- ⚠️ 语义理解弱: 无法捕捉深层语义

#### 3.2.3 TF-IDF词袋
位置: `src/storage/embedding_manager.py`

**实现**:
```python
def _embed_bow_tfidf(self, texts: List[str]) -> List[List[float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if not hasattr(self, '_tfidf_vectorizer'):
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,  # 1000维
            stop_words='english',
            lowercase=True
        )
    
    tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray().tolist()
```

**特点**:
- ✅ 关键词敏感: 适合关键词搜索
- ✅ 可解释性强: 每个维度对应一个词
- ⚠️ 需要训练: 首次需要在语料上fit
- ⚠️ 无法处理未见词: OOV问题

---

## 4. 检索算法详解

### 4.1 向量相似度检索

#### 4.1.1 HNSW算法原理
ChromaDB使用 **HNSW (Hierarchical Navigable Small World)** 索引:

```
多层图结构:
Layer 2: [Node A] ──── [Node B]
           │              │
Layer 1: [A]──[C]──[D]──[B]──[E]
           │  │  │  │  │  │  │
Layer 0: [A][X][C][Y][D][Z][B][W][E]
         └─ 所有节点 ─┘

查询过程:
1. 从最高层入口节点开始
2. 贪心搜索找到该层最近邻
3. 下降到下一层
4. 重复直到Layer 0
5. 返回Top-K最近邻
```

**时间复杂度**: O(log N)

#### 4.1.2 相似度计算
位置: `src/storage/vector_store_simple.py`

**欧几里得距离**:
```python
# ChromaDB内部计算
distance = sqrt(sum((query[i] - doc[i])^2 for i in range(dim)))
```

**相关性分数转换**:
```python
def calculate_relevance_score(distance, min_dist, max_dist):
    # 1. 归一化到 [0, 1]
    if max_dist > min_dist:
        relative_distance = (distance - min_dist) / (max_dist - min_dist)
        relevance_score = 1.0 - relative_distance
    else:
        relevance_score = 0.5
    
    # 2. 动态阈值调整
    if distance > 2.0:  # 距离很远
        relevance_score = max(0.0, min(0.3, relevance_score))
    elif distance > 1.5:  # 距离较远
        relevance_score = max(0.1, min(0.5, relevance_score))
    elif distance < 0.3:  # 非常相似
        relevance_score = max(0.7, min(1.0, relevance_score))
    else:  # 中等距离
        relevance_score = max(0.2, min(0.8, relevance_score))
    
    return relevance_score
```

### 4.2 混合检索 (Hybrid Retrieval)

#### 4.2.1 混合检索策略
位置: `src/storage/vector_store_simple.py`

```python
def hybrid_search(self, query: str, k: int = 5, 
                 vector_weight: float = 0.7, 
                 keyword_weight: float = 0.3) -> List[LangChainDocument]:
    # 1. 向量语义搜索
    vector_results = self.vector_store.search(query, k=k*2)
    
    # 2. 关键词搜索 (TODO: 当前未完全实现)
    keyword_results = self._keyword_search(query, k=k*2)
    
    # 3. 融合结果
    combined_results = self._fusion_results(
        vector_results,
        keyword_results,
        vector_weight,
        keyword_weight
    )
    
    return combined_results[:k]
```

#### 4.2.2 结果融合算法
```python
def _fusion_results(self, vector_results, keyword_results, 
                   vector_weight, keyword_weight):
    all_results = []
    
    # 添加向量搜索结果
    for doc in vector_results:
        doc.metadata['vector_score'] = doc.metadata.get('relevance_score', 0.5)
        doc.metadata['keyword_score'] = 0
        doc.metadata['combined_score'] = vector_weight * doc.metadata['vector_score']
        all_results.append(doc)
    
    # 添加关键词搜索结果
    for doc in keyword_results:
        doc.metadata['vector_score'] = 0
        doc.metadata['keyword_score'] = 0.5
        doc.metadata['combined_score'] = keyword_weight * doc.metadata['keyword_score']
        all_results.append(doc)
    
    # 按综合得分排序
    all_results.sort(key=lambda x: x.metadata.get('combined_score', 0), reverse=True)
    
    # 去重
    unique_results = remove_duplicates(all_results)
    
    return unique_results
```

**注意**: 当前版本关键词检索部分返回空列表，主要依赖向量检索。

---

## 5. 文档处理逻辑

### 5.1 文档加载与分块

#### 5.1.1 文本分块算法
位置: `src/ingestion/document_loader_simple.py`

**分块策略**:
```python
def simple_text_splitter(text: str, chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> List[str]:
    # 1. 按段落分割
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # 2. 判断是否需要新块
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                # 3. 保持重叠
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + " " + paragraph
            else:
                current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph
    
    # 4. 进一步分割大块 (按句子)
    final_chunks = further_split_by_sentences(chunks, chunk_size, chunk_overlap)
    
    return final_chunks
```

**分块参数**:
- `chunk_size`: 1000字符
- `chunk_overlap`: 200字符
- **目的**: 保持语义完整性，避免截断句子

#### 5.1.2 文档元数据提取
```python
def load_file(self, file_path: str) -> List[LangChainDocument]:
    # 1. 提取文本内容
    content = self._extract_text(file_path)
    
    # 2. 分块
    chunks = simple_text_splitter(content, self.chunk_size, self.chunk_overlap)
    
    # 3. 转换为LangChain文档
    documents = []
    for i, chunk in enumerate(chunks):
        doc = LangChainDocument(
            page_content=chunk,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "file_type": file_path.suffix,
                "file_size": file_path.stat().st_size / (1024*1024),
                "content_hash": str(hash(chunk))
            }
        )
        documents.append(doc)
    
    return documents
```

### 5.2 文档分类算法

#### 5.2.1 基于规则的分类
位置: `src/ingestion/document_loader_simple.py`

```python
def classify_document(self, document: LangChainDocument) -> Dict[str, Any]:
    content = document.page_content.lower()
    
    # 分类规则
    category_rules = {
        "工作": ["项目", "会议", "工作", "任务", "计划", "报告"],
        "学习": ["学习", "教程", "课程", "笔记", "知识", "技能"],
        "个人": ["日记", "想法", "感悟", "生活", "家庭", "个人"],
        "参考": ["参考", "文档", "手册", "指南", "规范", "标准"],
        "研究": ["研究", "分析", "实验", "数据", "结论", "发现"],
        "想法": ["想法", "创意", "创新", "设计", "概念", "思路"]
    }
    
    # 关键词匹配计分
    scores = {}
    for category, keywords in category_rules.items():
        score = sum(1 for keyword in keywords if keyword in content)
        scores[category] = score
    
    # 选择最高分类别
    category = max(scores, key=scores.get) if scores else "参考"
    if scores[category] == 0:
        category = "参考"  # 默认分类
    
    # 优先级判断
    if "重要" in content or "紧急" in content:
        priority = "高"
    elif len(content) > 2000:
        priority = "中"
    else:
        priority = "低"
    
    return {
        "category": category,
        "priority": priority,
        "summary": content[:100] + "...",
        "tags": ",".join(self._extract_keywords(content)),
        "confidence": scores[category] / max(len(content.split()) / 100, 1)
    }
```

#### 5.2.2 关键词提取
```python
def _extract_keywords(self, content: str) -> List[str]:
    common_words = {"的", "了", "是", "在", "有", "和"}
    
    words = content.split()
    word_freq = {}
    
    for word in words:
        word = word.strip('，。！？；：""''()（）')
        if len(word) > 1 and word not in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 返回频率最高的5个词
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return [word for word, freq in top_words]
```

---

## 6. 提示词工程

### 6.1 RAG系统提示词

位置: `src/generation/llm_manager.py`

```python
rag_system_prompt = """你是一个专业的知识管理助手。基于提供的上下文内容回答用户问题。

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
```

**设计原则**:
- ✅ 明确角色定位
- ✅ 清晰的规则约束
- ✅ 强制来源引用
- ✅ 避免幻觉 (Hallucination)

### 6.2 上下文构建

```python
def _build_context(self, documents: List[LangChainDocument], 
                  include_sources: bool = True) -> str:
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        filename = doc.metadata.get('filename', '未知文件')
        content = doc.page_content
        
        if include_sources:
            context_parts.append(f"[文档 {i} - 来源: {filename}]\n{content}\n")
        else:
            context_parts.append(f"{content}\n")
    
    return "\n---\n".join(context_parts)
```

**上下文示例**:
```
[文档 1 - 来源: python_notes.md]
Python装饰器是一种设计模式，允许在不修改原函数代码的情况下...

---

[文档 2 - 来源: python_notes.md]
装饰器的基本语法是使用@符号，例如：@decorator...

---

[文档 3 - 来源: advanced_python.pdf]
装饰器可以接受参数，实现更灵活的功能扩展...
```

---

## 7. 置信度评估

### 7.1 置信度计算逻辑

位置: `src/generation/llm_manager.py`

```python
def _estimate_confidence(self, query: str, documents: List[LangChainDocument], 
                        answer: str) -> float:
    confidence = 0.0
    
    # 1. 基于文档相关性
    relevance_scores = [
        doc.metadata.get('relevance_score', 0.5) 
        for doc in documents
    ]
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
    confidence += avg_relevance * 0.4  # 40%权重
    
    # 2. 基于文档数量
    doc_count_score = min(len(documents) / 5.0, 1.0)
    confidence += doc_count_score * 0.2  # 20%权重
    
    # 3. 基于答案长度
    answer_length_score = min(len(answer) / 500, 1.0)
    confidence += answer_length_score * 0.2  # 20%权重
    
    # 4. 基于关键词匹配
    query_keywords = set(query.lower().split())
    answer_keywords = set(answer.lower().split())
    keyword_overlap = len(query_keywords & answer_keywords) / len(query_keywords) if query_keywords else 0
    confidence += keyword_overlap * 0.2  # 20%权重
    
    return min(confidence, 1.0)
```

**置信度等级**:
- **0.8 - 1.0**: 高置信度 (文档高度相关，答案充分)
- **0.5 - 0.8**: 中等置信度 (文档相关，但可能不完整)
- **0.0 - 0.5**: 低置信度 (文档相关性低，答案不确定)

---

## 8. 降级策略详解

### 8.1 Embedding降级链

```
┌───────────────────────────────────────────────┐
│      Embedding生成降级流程                     │
├───────────────────────────────────────────────┤
│  1. 尝试 OpenAI API                            │
│     if settings.OPENAI_API_KEY:               │
│         return OpenAIEmbeddings(...)          │
│                ↓ (失败)                        │
│                                               │
│  2. 尝试 Sentence Transformers                │
│     try:                                      │
│         import sentence_transformers          │
│         return SentenceTransformer(...)       │
│                ↓ (失败)                        │
│                                               │
│  3. 回退到文本哈希                             │
│     return TextHashEmbedding(...)             │
│     (保底方案，永不失败)                       │
└───────────────────────────────────────────────┘
```

**实现代码**:
```python
def _initialize_embeddings(self):
    # 优先级1: OpenAI
    if settings.OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            self.embeddings = OpenAIEmbeddings(...)
            logger.info("✅ 使用OpenAI Embeddings")
            return
        except Exception as e:
            logger.warning(f"OpenAI初始化失败: {e}")
    
    # 优先级2: Sentence Transformers
    if EMBEDDING_MANAGER_AVAILABLE:
        try:
            optimal_method = settings.get_optimal_embedding_method()
            self.embedding_manager = EmbeddingManager(optimal_method)
            logger.info(f"✅ 使用{optimal_method}")
            return
        except Exception as e:
            logger.warning(f"Embedding管理器失败: {e}")
    
    # 优先级3: 文本哈希 (保底)
    logger.info("🔄 使用文本哈希embedding")
    self.embeddings = None  # 触发文本哈希逻辑
```

### 8.2 LLM生成降级链

```
┌───────────────────────────────────────────────┐
│      LLM生成降级流程                           │
├───────────────────────────────────────────────┤
│  1. 有DeepSeek API密钥                         │
│     if settings.DEEPSEEK_API_KEY:             │
│         return RAGGenerator.generate_answer() │
│                ↓ (无API密钥)                   │
│                                               │
│  2. 简化回答模式                               │
│     return build_simple_answer()              │
│     • 展示检索到的文档摘要                     │
│     • 标注相关性分数                           │
│     • 提供文档来源链接                         │
└───────────────────────────────────────────────┘
```

---

## 9. 性能优化策略

### 9.1 批量处理优化
```python
# ✅ 批量生成embedding
embeddings = embedding_manager.embed_documents(texts)  # 一次性处理多个

# ❌ 避免循环调用
for text in texts:
    embedding = embedding_manager.embed_query(text)  # 效率低
```

### 9.2 缓存机制 (待实现)
```python
# TODO: 实现embedding缓存
embedding_cache = {}

def get_embedding_with_cache(text: str):
    if text in embedding_cache:
        return embedding_cache[text]
    
    embedding = embedding_manager.embed_query(text)
    embedding_cache[text] = embedding
    return embedding
```

### 9.3 Top-K限制
```python
# 默认返回5个结果，避免过度检索
TOP_K = 5
results = vector_store.search(query, k=TOP_K)
```

---

## 10. 未实现功能清单

### 10.1 关键词检索 (BM25)
位置: `src/storage/vector_store_simple.py:461`

```python
def _keyword_search(self, query: str, k: int, filter_dict: Optional[Dict]) -> List[LangChainDocument]:
    """简单的关键词搜索"""
    # TODO: 实现BM25或其他关键词搜索算法
    # 暂时返回空列表
    return []
```

**建议实现**:
```python
from rank_bm25 import BM25Okapi

def _keyword_search(self, query: str, k: int, filter_dict: Optional[Dict]):
    # 1. 获取所有文档
    all_docs = self.vector_store.collection.get()
    
    # 2. 构建BM25索引
    tokenized_docs = [doc.split() for doc in all_docs['documents']]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 3. 搜索
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    
    # 4. 返回Top-K
    top_indices = np.argsort(scores)[::-1][:k]
    # ... 构建返回结果
```

### 10.2 高级文档分类
当前使用简单的关键词匹配，可以升级为LLM分类:

```python
# 当前: 基于规则
category = classify_by_keywords(content)

# 建议: 使用LLM
if llm_manager.chat_model:
    category = llm_manager.classify_document(content)
```

---

## 11. 调试与监控

### 11.1 日志记录
```python
# 关键步骤记录日志
logger.info(f"搜索完成: 查询='{query[:50]}...', 结果数量={len(documents)}")
logger.warning(f"主方案失败: {e}, 降级到备用方案")
logger.error(f"操作失败: {str(e)}")
```

### 11.2 性能监控
```python
import time

start_time = time.time()
results = vector_store.search(query, k=5)
elapsed_time = time.time() - start_time
logger.info(f"检索耗时: {elapsed_time:.3f}秒")
```

---

## 12. 最佳实践

### 12.1 RAG优化建议
- ✅ 合理设置chunk_size (推荐1000)
- ✅ 保持chunk_overlap (推荐200)
- ✅ Top-K不宜过大 (推荐5)
- ✅ 提示词清晰明确
- ✅ 强制来源引用

### 12.2 Embedding选择
- 💰 **无预算**: 文本哈希
- 🎯 **平衡**: Sentence Transformers (推荐)
- 🏆 **高质量**: OpenAI Embeddings

### 12.3 错误处理
- ✅ 每个关键操作都有try-except
- ✅ 降级策略完善
- ✅ 用户友好的错误提示

---

## 13. 参考资料

- [RAG论文: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [HNSW算法](https://arxiv.org/abs/1603.09320)
- [BM25算法详解](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)
- [提示词工程指南](https://www.promptingguide.ai/)

---

**文档维护**: 核心逻辑变更时请及时更新本文档
