# 个人知识管理Agent系统 - 接口设计文档

## 文档信息
- **版本**: v1.0.0
- **创建日期**: 2025-11-07
- **适用人员**: 后端开发、前端开发、API集成开发者

---

## 1. 接口设计概述

本系统采用**内部API**设计，各模块通过Python类方法进行交互。前端使用Gradio框架，通过函数回调与后端通信。

### 1.1 接口分类

| 接口类型 | 说明 | 示例 |
|---------|------|------|
| **内部Python API** | 模块间Python方法调用 | `DocumentLoader.load_file()` |
| **Gradio UI事件** | 用户界面交互事件 | 按钮点击、文件上传 |
| **外部API调用** | 调用第三方服务 | OpenAI API、DeepSeek API |

---

## 2. 核心模块API设计

### 2.1 文档加载模块 (DocumentLoader)

#### 类定义
```python
class DocumentLoader:
    """文档加载和处理器"""
```

#### 2.1.1 load_file()
**功能**: 加载单个文档文件并分块

**方法签名**:
```python
def load_file(self, file_path: str) -> List[LangChainDocument]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| file_path | str | ✅ | 文档文件绝对路径 |

**返回值**:
```python
List[LangChainDocument]  # 文档块列表
```

**返回的Document结构**:
```python
LangChainDocument(
    page_content="文档文本内容",
    metadata={
        "source": "/path/to/file.pdf",
        "filename": "file.pdf",
        "chunk_id": 0,
        "total_chunks": 5,
        "file_type": ".pdf",
        "file_size": 2.5,  # MB
        "content_hash": "hash_value"
    }
)
```

**异常**:
- `FileNotFoundError`: 文件不存在
- `ValueError`: 文件格式不支持或文件过大
- `Exception`: 文档解析失败

**使用示例**:
```python
loader = DocumentLoader()
try:
    documents = loader.load_file("/path/to/document.pdf")
    print(f"成功加载 {len(documents)} 个文档块")
except FileNotFoundError:
    print("文件不存在")
```

#### 2.1.2 load_multiple_files()
**功能**: 批量加载多个文档

**方法签名**:
```python
def load_multiple_files(self, file_paths: List[str]) -> List[LangChainDocument]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| file_paths | List[str] | ✅ | 文件路径列表 |

**返回值**:
```python
List[LangChainDocument]  # 所有文档的块列表
```

**特性**:
- 遇到错误自动跳过，不中断整体流程
- 返回所有成功加载的文档块

---

### 2.2 向量存储模块 (VectorStore)

#### 类定义
```python
class VectorStore:
    """向量数据库管理器"""
```

#### 2.2.1 add_documents()
**功能**: 添加文档到向量数据库

**方法签名**:
```python
def add_documents(self, documents: List[LangChainDocument]) -> bool
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| documents | List[LangChainDocument] | ✅ | 待添加的文档列表 |

**返回值**:
```python
bool  # True: 成功, False: 失败
```

**处理流程**:
1. 提取文档文本内容
2. 生成embedding向量
3. 存储到ChromaDB
4. 返回操作结果

**使用示例**:
```python
vector_store = VectorStore()
success = vector_store.add_documents(documents)
if success:
    print("文档添加成功")
```

#### 2.2.2 search()
**功能**: 搜索相关文档

**方法签名**:
```python
def search(
    self, 
    query: str, 
    k: int = 5, 
    filter_dict: Optional[Dict] = None
) -> List[LangChainDocument]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| query | str | ✅ | - | 搜索查询文本 |
| k | int | ❌ | 5 | 返回结果数量 |
| filter_dict | Dict | ❌ | None | 元数据过滤条件 |

**filter_dict 示例**:
```python
{
    "filename": "python_notes.md",
    "category": "学习"
}
```

**返回值**:
```python
List[LangChainDocument]  # 按相关性排序的文档列表
```

**返回Document包含的评分字段**:
```python
metadata={
    # 原始元数据 ...
    "search_score": 0.85,      # 距离分数
    "relevance_score": 0.75,   # 相关性分数 (0-1)
    "doc_id": "doc_xxx"
}
```

**使用示例**:
```python
results = vector_store.search(
    query="Python装饰器如何使用?",
    k=5,
    filter_dict={"category": "学习"}
)
```

#### 2.2.3 delete_documents()
**功能**: 删除指定文档

**方法签名**:
```python
def delete_documents(self, filter_dict: Dict[str, Any]) -> bool
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| filter_dict | Dict[str, Any] | ✅ | 删除条件 |

**常用删除条件**:
```python
# 按文件名删除
{"filename": "old_document.pdf"}

# 按分类删除
{"category": "临时"}
```

**返回值**:
```python
bool  # True: 成功, False: 失败
```

#### 2.2.4 get_stats()
**功能**: 获取数据库统计信息

**方法签名**:
```python
def get_stats(self) -> Dict[str, Any]
```

**返回值示例**:
```python
{
    "collection_name": "knowledge_base",
    "total_documents": 150,
    "vector_db_path": "./data/vector_db",
    "embedding_method": "sentence-transformers",
    "embeddings_dimension": 384,
    "embedding_description": "Sentence Transformers (all-MiniLM-L6-v2)",
    "is_free_embedding": True,
    "privacy_protected": True
}
```

---

### 2.3 Embedding管理模块 (EmbeddingManager)

#### 类定义
```python
class EmbeddingManager:
    """多方案Embedding管理器"""
```

#### 2.3.1 embed_documents()
**功能**: 批量生成文档embedding

**方法签名**:
```python
def embed_documents(self, texts: List[str]) -> List[List[float]]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| texts | List[str] | ✅ | 文本列表 |

**返回值**:
```python
List[List[float]]  # embedding向量列表，每个向量维度根据方案而定
```

**向量维度说明**:
| 方案 | 维度 |
|------|------|
| Sentence Transformers | 384 |
| 文本哈希 | 384 |
| TF-IDF | 1000 |
| OpenAI | 1536 |

#### 2.3.2 embed_query()
**功能**: 生成单个查询的embedding

**方法签名**:
```python
def embed_query(self, text: str) -> List[float]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| text | str | ✅ | 查询文本 |

**返回值**:
```python
List[float]  # embedding向量
```

#### 2.3.3 get_method_info()
**功能**: 获取当前embedding方法信息

**方法签名**:
```python
def get_method_info(self) -> Dict[str, Any]
```

**返回值示例**:
```python
{
    "method": "sentence-transformers",
    "dimension": 384,
    "description": "Sentence Transformers (all-MiniLM-L6-v2) - 免费高质量",
    "is_free": True,
    "privacy_protected": True
}
```

---

### 2.4 LLM管理模块 (LLMManager)

#### 类定义
```python
class LLMManager:
    """LLM模型管理器"""
```

#### 2.4.1 初始化
**方法签名**:
```python
def __init__(self)
```

**功能**:
- 自动初始化embedding模型
- 自动初始化聊天模型（如果有API密钥）
- 智能降级策略

#### 2.4.2 embed_documents() / embed_query()
与 `EmbeddingManager` 相同接口，统一管理

---

### 2.5 RAG生成模块 (RAGGenerator)

#### 类定义
```python
class RAGGenerator:
    """RAG问答生成器"""
```

#### 2.5.1 generate_answer()
**功能**: 基于检索文档生成智能回答

**方法签名**:
```python
def generate_answer(
    self,
    query: str,
    documents: List[LangChainDocument],
    include_sources: bool = True
) -> Dict[str, Any]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| query | str | ✅ | - | 用户问题 |
| documents | List[LangChainDocument] | ✅ | - | 检索到的文档 |
| include_sources | bool | ❌ | True | 是否包含来源引用 |

**返回值结构**:
```python
{
    "answer": "基于上下文的AI回答...",
    "confidence": 0.85,  # 置信度 0-1
    "sources": [
        {
            "filename": "python_notes.md",
            "relevance_score": 0.92,
            "chunk_id": 3
        }
    ],
    "metadata": {
        "query": "用户原始问题",
        "retrieved_docs": 5,
        "model_used": "deepseek-chat",
        "tokens_used": 450
    }
}
```

**使用示例**:
```python
rag_gen = RAGGenerator(llm_manager)
result = rag_gen.generate_answer(
    query="什么是Python装饰器?",
    documents=retrieved_docs
)
print(result['answer'])
print(f"置信度: {result['confidence']}")
```

#### 2.5.2 classify_document()
**功能**: 使用LLM对文档进行智能分类

**方法签名**:
```python
def classify_document(self, content: str) -> Dict[str, Any]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| content | str | ✅ | 文档内容（最多2000字符） |

**返回值结构**:
```python
{
    "category": "学习",
    "priority": "高",
    "summary": "关于Python装饰器的学习笔记...",
    "tags": ["Python", "装饰器", "高级特性"],
    "confidence": 0.9
}
```

---

### 2.6 混合检索模块 (HybridRetriever)

#### 类定义
```python
class HybridRetriever:
    """混合检索器（向量+关键词）"""
```

#### 2.6.1 hybrid_search()
**功能**: 混合检索（向量搜索 + 关键词搜索）

**方法签名**:
```python
def hybrid_search(
    self,
    query: str,
    k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    filter_dict: Optional[Dict] = None
) -> List[LangChainDocument]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| query | str | ✅ | - | 搜索查询 |
| k | int | ❌ | 5 | 返回结果数 |
| vector_weight | float | ❌ | 0.7 | 向量搜索权重 |
| keyword_weight | float | ❌ | 0.3 | 关键词搜索权重 |
| filter_dict | Dict | ❌ | None | 元数据过滤 |

**返回Document包含的评分**:
```python
metadata={
    # 原始元数据 ...
    "vector_score": 0.85,      # 向量搜索分数
    "keyword_score": 0.60,     # 关键词搜索分数
    "combined_score": 0.78     # 综合分数
}
```

**注意**: 当前版本关键词搜索部分**未完全实现**（TODO），主要依赖向量搜索。

---

### 2.7 文档分类模块 (DocumentClassifier)

#### 类定义
```python
class DocumentClassifier:
    """文档智能分类器"""
```

#### 2.7.1 classify_document()
**功能**: 基于规则对文档分类

**方法签名**:
```python
def classify_document(self, document: LangChainDocument) -> Dict[str, Any]
```

**参数说明**:
| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| document | LangChainDocument | ✅ | 待分类文档 |

**返回值结构**:
```python
{
    "category": "学习",        # 分类: 工作/学习/个人/参考/研究/想法
    "priority": "中",          # 优先级: 高/中/低
    "summary": "文档摘要...",
    "tags": "Python,学习,笔记", # 逗号分隔的标签字符串
    "confidence": 0.75
}
```

**分类规则**:
基于关键词匹配规则，参见 `document_loader_simple.py` 中的 `category_rules`。

---

## 3. Gradio UI事件接口

### 3.1 KnowledgeManagerApp 类

#### 3.1.1 load_and_process_files()
**功能**: 处理用户上传的文件

**方法签名**:
```python
def load_and_process_files(self, files: List[str]) -> str
```

**参数**:
- `files`: Gradio File组件传递的文件路径列表

**返回**:
- `str`: 处理结果消息（显示在UI上）

**UI绑定**:
```python
upload_btn.click(
    self.load_and_process_files,
    inputs=[file_input],
    outputs=[status_output]
)
```

#### 3.1.2 chat_with_knowledge()
**功能**: 智能问答对话

**方法签名**:
```python
def chat_with_knowledge(
    self,
    message: str,
    history: List[Dict[str, str]]
) -> str
```

**参数**:
- `message`: 用户输入的问题
- `history`: 对话历史（Gradio自动管理）

**返回**:
- `str`: AI回答（Markdown格式）

**UI绑定**:
```python
chatbot.submit(
    self.chat_with_knowledge,
    inputs=[msg, chatbot],
    outputs=[chatbot, msg]
)
```

#### 3.1.3 search_knowledge()
**功能**: 搜索知识库

**方法签名**:
```python
def search_knowledge(
    self,
    query: str,
    mode: str = "混合检索",
    top_k: int = 5
) -> str
```

**参数**:
| 参数名 | 类型 | 说明 |
|--------|------|------|
| query | str | 搜索关键词 |
| mode | str | 检索模式（混合检索/语义检索/关键词检索） |
| top_k | int | 返回结果数量 |

**返回**:
- `str`: 格式化的搜索结果（Markdown）

#### 3.1.4 get_statistics()
**功能**: 获取知识库统计信息

**方法签名**:
```python
def get_statistics(self) -> str
```

**返回**:
- `str`: 格式化的统计信息（Markdown）

**返回内容示例**:
```markdown
📊 知识库统计信息

📁 文档统计:
  • 总文档数: 15 个文件
  • 总文档块: 127 个

📂 分类统计:
  • 学习: 8 个文件
  • 工作: 5 个文件
  • 参考: 2 个文件

🧠 Embedding信息:
  • 方法: Sentence Transformers (all-MiniLM-L6-v2)
  • 维度: 384
  • 免费方案: ✅
```

#### 3.1.5 get_document_list()
**功能**: 获取文档列表

**方法签名**:
```python
def get_document_list(self) -> List[List[str]]
```

**返回**:
```python
[
    ["python_notes.md", "12", "学习", ".md"],
    ["project_plan.pdf", "8", "工作", ".pdf"],
    # ...
]
```

**UI显示**: 在Gradio Dataframe组件中展示

#### 3.1.6 delete_document_by_filename()
**功能**: 删除指定文件

**方法签名**:
```python
def delete_document_by_filename(self, filename: str) -> str
```

**参数**:
- `filename`: 要删除的文件名（完整文件名）

**返回**:
- `str`: 删除结果消息

#### 3.1.7 update_document()
**功能**: 更新文档（删除旧版本+上传新版本）

**方法签名**:
```python
def update_document(self, old_filename: str, file) -> str
```

**参数**:
- `old_filename`: 要替换的旧文件名
- `file`: Gradio File对象（新文件）

**返回**:
- `str`: 更新结果消息

---

## 4. 外部API调用接口

### 4.1 OpenAI API

#### 4.1.1 Embeddings API
**用途**: 高质量文本嵌入（可选）

**调用方式**:
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY
)
vectors = embeddings.embed_documents(texts)
```

**配置**:
- 环境变量: `OPENAI_API_KEY`
- 模型: `text-embedding-3-small`
- 维度: 1536

---

### 4.2 DeepSeek API

#### 4.2.1 Chat Completions API
**用途**: LLM对话生成

**调用方式**:
```python
from langchain_deepseek import ChatDeepSeek

chat_model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=settings.DEEPSEEK_API_KEY,
    temperature=0.7,
    max_tokens=500
)
response = chat_model.invoke(messages)
```

**配置**:
- 环境变量: `DEEPSEEK_API_KEY`
- 模型: `deepseek-chat`
- 温度: 0.7
- 最大Token: 500

---

## 5. 数据结构定义

### 5.1 LangChainDocument

**LangChain核心文档对象**:
```python
from langchain_core.documents import Document

doc = Document(
    page_content="文档文本内容",
    metadata={
        # 文件基本信息
        "source": "/absolute/path/to/file.pdf",
        "filename": "file.pdf",
        "file_type": ".pdf",
        "file_size": 2.5,  # MB
        
        # 分块信息
        "chunk_id": 0,
        "total_chunks": 5,
        "content_hash": "hash_value",
        
        # 分类信息
        "category": "学习",
        "priority": "高",
        "tags": "Python,学习",
        "summary": "文档摘要...",
        
        # 检索相关（搜索时添加）
        "search_score": 0.85,
        "relevance_score": 0.75,
        "doc_id": "doc_xxx"
    }
)
```

### 5.2 配置对象 (Settings)

**全局配置类**:
```python
class Settings:
    # API密钥
    OPENAI_API_KEY: Optional[str]
    DEEPSEEK_API_KEY: Optional[str]
    
    # 路径配置
    VECTOR_DB_PATH: str = "./data/vector_db"
    
    # RAG参数
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    
    # Embedding配置
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_METHOD: str = "auto"
    
    # Gradio配置
    GRADIO_SERVER_PORT: int = 8888
    GRADIO_SERVER_HOST: str = "127.0.0.1"
    
    # 支持的文件类型
    SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx"]
    MAX_FILE_SIZE_MB = 50
    
    # 分类标签
    DOCUMENT_CATEGORIES = ["工作", "学习", "个人", "参考", "研究", "想法"]
    PRIORITY_LEVELS = ["高", "中", "低"]
```

---

## 6. 错误码与异常

### 6.1 常见异常

| 异常类型 | 触发场景 | 处理方式 |
|---------|---------|---------|
| `FileNotFoundError` | 文件不存在 | 提示用户检查路径 |
| `ValueError` | 参数无效（文件过大、格式不支持） | 返回错误信息 |
| `ImportError` | 依赖库缺失 | 降级到备用方案 |
| `APIError` | API调用失败 | 重试或降级 |

### 6.2 错误处理示例

```python
try:
    documents = loader.load_file(file_path)
except FileNotFoundError:
    return "❌ 文件不存在，请检查路径"
except ValueError as e:
    return f"❌ 文件验证失败: {str(e)}"
except Exception as e:
    logger.error(f"未知错误: {e}")
    return "❌ 处理失败，请查看日志"
```

---

## 7. API调用流程示例

### 7.1 完整的文档上传流程

```python
# 1. 用户上传文件 (Gradio UI)
files = ["/path/to/doc1.pdf", "/path/to/doc2.md"]

# 2. 调用处理方法
app = KnowledgeManagerApp()
result = app.load_and_process_files(files)

# 内部流程:
# 2.1 DocumentLoader.load_file() - 加载文档
# 2.2 DocumentClassifier.classify_document() - 分类
# 2.3 EmbeddingManager.embed_documents() - 生成embedding
# 2.4 VectorStore.add_documents() - 存储到数据库

# 3. 返回结果给用户
print(result)  # "✅ 成功处理 15 个文档块..."
```

### 7.2 完整的智能问答流程

```python
# 1. 用户提问
query = "Python装饰器如何使用?"

# 2. 调用问答方法
app = KnowledgeManagerApp()
answer = app.chat_with_knowledge(query, history=[])

# 内部流程:
# 2.1 VectorStore.search() - 检索相关文档
# 2.2 RAGGenerator.generate_answer() - 生成答案
#     2.2.1 构建上下文
#     2.2.2 调用DeepSeek API (如果可用)
#     2.2.3 提取来源引用

# 3. 返回格式化答案
print(answer)  # Markdown格式的答案
```

---

## 8. 接口调用最佳实践

### 8.1 性能优化
```python
# ✅ 批量处理
documents = loader.load_multiple_files(file_paths)

# ❌ 避免循环单个处理
for path in file_paths:
    loader.load_file(path)  # 效率低
```

### 8.2 错误处理
```python
# ✅ 优雅降级
try:
    result = primary_method()
except Exception:
    result = fallback_method()

# ❌ 直接崩溃
result = risky_method()  # 可能抛异常
```

### 8.3 资源管理
```python
# ✅ 限制返回数量
results = vector_store.search(query, k=5)

# ❌ 无限制返回
results = vector_store.search(query, k=1000)  # 可能OOM
```

---

## 9. 接口版本管理

当前版本: **v1.0.0**

### 9.1 兼容性保证
- 主版本号变更：不兼容的API变更
- 次版本号变更：向后兼容的功能新增
- 修订号变更：向后兼容的问题修复

### 9.2 废弃策略
- 标记为 `@deprecated` 并在日志中警告
- 至少保留一个大版本周期
- 提供迁移指南

---

## 10. 参考资料

- [LangChain Document API](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [ChromaDB Python Client](https://docs.trychroma.com/reference/python-client)
- [Gradio Event Listeners](https://www.gradio.app/docs/chatinterface)

---

**文档维护**: 接口变更时请及时更新本文档
