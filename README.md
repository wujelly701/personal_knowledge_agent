# 个人知识管理Agent系统

一个基于LangChain和RAG技术的个人知识管理助手，支持文档上传、智能检索、问答交互、文档管理等功能。

## 🚀 核心特性

### 💰 完全免费架构
- **多种免费方案**: Sentence Transformers、文本哈希、TF-IDF词袋
- **零API依赖**: 无需OpenAI密钥即可运行高质量语义搜索
- **成本节省**: 相比传统方案节省100%成本（完全免费）
- **本地部署**: 使用Chroma本地向量数据库，隐私保护

### 📚 功能特性
- **多格式支持**: PDF、TXT、Markdown、DOCX文档处理
- **智能检索**: 混合搜索（向量+关键词），支持Enter键快速搜索
- **RAG问答**: 基于检索增强生成，确保答案准确性
- **文档管理**: 查看、删除、更新知识库文档
- **统计分析**: 详细的文档统计（文件数、分块数、分类统计）
- **自动分类**: 智能文档分类和标签生成
- **来源引用**: 完整的引用追踪和来源标识

### 🎯 技术栈
- **LLM框架**: LangChain v1.0
- **向量数据库**: Chroma DB
- **嵌入方案**: 
  - 🥇 **Sentence Transformers** (all-MiniLM-L6-v2) - 免费推荐
  - 🔧 **文本哈希** - 零依赖
  - 📊 **TF-IDF词袋** - 关键词优化
  - 🔑 **OpenAI text-embedding-3-small** (可选，高质量)
- **生成模型**: DeepSeek-chat
- **前端界面**: Gradio 5.0
- **文档处理**: PyPDF2, python-docx

## 📋 系统要求

- Python 3.8+
- 2GB+ RAM
- 1GB+ 磁盘空间
- **可选API密钥**:
  - DeepSeek API Key (用于生成，可选)
  - OpenAI API Key (高质量embedding，可选)
- **推荐安装**（仅一个）:
  - `pip install sentence-transformers` (推荐，免费高质量)
  - `pip install scikit-learn` (可选，关键词搜索优化)

## 🛠️ 安装配置

### 1. 克隆项目
```bash
git clone <repository-url>
cd personal_knowledge_agent
```

### 2. 安装依赖
```bash
pip install -r requirements.txt

# 可选：安装免费高质量embedding（推荐）
pip install sentence-transformers

# 可选：安装关键词优化embedding
pip install scikit-learn
```

### 3. 配置环境变量（可选）
```bash
cp .env.example .env
```

**零API配置（完全免费）：**
```env
# 无需配置任何API密钥
EMBEDDING_METHOD=all-MiniLM-L6-v2
```

**高级配置（可选API密钥）：**
```env
# 可选：DeepSeek API密钥（更好的问答生成）
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 可选：OpenAI API密钥（更高质量embedding但需费用）
# OPENAI_API_KEY=your_openai_api_key_here

# Embedding方法选择
EMBEDDING_METHOD=all-MiniLM-L6-v2
```

### 4. 启动应用
```bash
python main.py
```

应用将在 http://localhost:7860 启动

## 📖 使用指南

### 1. 文档上传
- 支持格式: PDF, TXT, MD, DOCX
- 最大文件大小: 50MB
- 支持批量上传
- 自动文档分块（CHUNK_SIZE=1000, CHUNK_OVERLAP=200）

### 2. 智能问答
- 基于上传的文档进行问答
- 自动检索相关文档片段（Top-K=5）
- 提供完整的来源引用
- 使用DeepSeek-chat生成高质量回答

### 3. 文档搜索
- **混合检索模式**（推荐）: 向量+关键词双重搜索
- **语义检索模式**: 纯向量相似度搜索
- 支持Enter键快速搜索
- 可调节返回结果数量（1-20）
- 详细的结果展示（分块位置、相关性分数）

### 4. 文档管理
- **查看文档列表**: 显示所有文档的分块数、分类、类型
- **删除文档**: 根据文件名删除（删除所有分块）
- **更新文档**: 先删除旧版本，再上传新版本
- **统计信息**: 文件数、分块数、分类统计、Embedding配置

### 5. 统计分析
- 文件总数 vs 文档块总数
- 平均分块数
- 分类统计（按文件数）
- Embedding方法详情
- LLM功能状态

## 🏗️ 项目结构

```
personal_knowledge_agent/
├── config/                  # 配置管理
│   └── settings.py         # 应用配置
├── src/                    # 源代码
│   ├── ingestion/          # 文档处理
│   │   ├── document_loader.py       # 完整版文档加载器
│   │   └── document_loader_simple.py # 简化版（当前使用）
│   ├── storage/            # 向量存储
│   │   ├── vector_store.py          # 完整版向量存储
│   │   ├── vector_store_simple.py   # 简化版（当前使用）
│   │   └── embedding_manager.py     # Embedding管理器
│   ├── generation/         # LLM生成
│   │   ├── llm_manager.py           # LLM管理器
│   │   └── prompts.py               # 提示词模板
│   ├── utils/              # 工具函数
│   │   ├── logging.py               # 日志配置
│   │   └── helpers.py               # 辅助函数
│   └── api/                # API接口
│       └── gradio_app.py            # Gradio Web界面
├── data/                   # 数据存储
│   └── vector_db/         # Chroma向量数据库
├── logs/                  # 日志文件
│   └── app.log           # 应用日志（带时间戳）
├── user_files/           # 示例文档
├── docs/                 # 项目文档
│   ├── 使用指南与FAQ.md          # 用户指南
│   ├── FIXPLAN.md                # 修复计划
│   └── CHANGELOG-20241107.md     # 更新日志
├── main.py               # 主启动脚本
├── requirements.txt      # 依赖列表
└── README.md            # 项目说明
```

### 架构说明

**为什么有 `xx_simple.py` 文件？**
- **渐进式演进策略**: 先实现简化版，降低初期依赖
- **便于测试**: 简化版更容易调试和验证
- **灵活切换**: 未来可无缝升级到完整版
- **当前使用**: `document_loader_simple.py` 和 `vector_store_simple.py`

## 🔧 配置说明

### 环境变量配置

| 变量名 | 描述 | 默认值 | 推荐设置 |
|--------|------|--------|----------|
| EMBEDDING_METHOD | Embedding方法 | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| OPENAI_API_KEY | OpenAI API密钥 | 空白 | 可选（embedding） |
| DEEPSEEK_API_KEY | DeepSeek API密钥 | 空白 | 可选（问答生成） |
| VECTOR_DB_PATH | 向量数据库路径 | ./data/vector_db | 默认 |
| CHUNK_SIZE | 文档分块大小 | 1000 | 默认（已优化） |
| CHUNK_OVERLAP | 分块重叠大小 | 200 | 默认（已优化） |
| TOP_K | 检索返回数量 | 5 | 默认 |
| GRADIO_SERVER_PORT | 服务器端口 | 7860 | 默认 |
| LOG_LEVEL | 日志级别 | INFO | INFO/DEBUG |

### Embedding方法选择

| 方法名 | 安装命令 | 质量 | 速度 | 特点 |
|--------|----------|------|------|------|
| `all-MiniLM-L6-v2` | `pip install sentence-transformers` | ⭐⭐⭐⭐⭐ | 快 | 🏆 **推荐** - 免费高质量 |
| `text-hash` | 无需安装 | ⭐⭐⭐ | 极快 | 零依赖 |
| `bow-tfidf` | `pip install scikit-learn` | ⭐⭐⭐⭐ | 快 | 关键词优化 |

### 模型配置

- **Embedding模型**: text-embedding-3-small
- **生成模型**: deepseek-chat
- **温度参数**: 0.7
- **最大token数**: 500

## 🚨 故障排除

### 常见问题

1. **Embedding方案选择**
   - 默认使用免费的all-MiniLM-L6-v2（推荐）
   - 如需切换方法，修改.env中的EMBEDDING_METHOD
   - 系统会自动选择最佳可用方案

2. **安装问题**
   - 如果sentence-transformers安装失败，会自动降级到文本哈希
   - 文本哈希方案无需安装任何包即可工作
   - 系统支持多种embedding方案的自动切换

3. **文档分块问题**
   - **为什么文档被分成多个块？** 
     - 向量数据库技术限制（通常建议1000字符以内）
     - 提高检索精度和相关性
     - 优化LLM上下文窗口利用
   - **分块参数**: CHUNK_SIZE=1000, CHUNK_OVERLAP=200
   - 详见：`使用指南与FAQ.md`

4. **文档管理问题**
   - **如何删除文档？** 进入"文档管理"Tab，输入文件名点击删除
   - **如何更新文档？** 使用更新功能，会自动删除旧版本并上传新版本
   - **删除不可逆**: 请谨慎操作，建议先备份

5. **搜索功能问题**
   - **搜索无结果**: 确认已上传相关文档，尝试不同关键词
   - **问答后搜索失败**: 已修复，搜索和问答功能完全独立
   - **快捷键**: 支持Enter键执行搜索

6. **API密钥问题（可选）**
   - DeepSeek API密钥用于智能问答生成
   - OpenAI API密钥用于高质量embedding（可选）
   - 无API密钥时系统会使用免费方案继续运行

7. **日志查看**
   - 日志位置: `logs/app.log`
   - 日志格式: 包含时间戳、模块名、级别、消息
   - 搜索日志: 使用 `[搜索]` 前缀过滤

### 日志查看
```bash
# Windows PowerShell
Get-Content .\logs\app.log -Tail 50

# 过滤搜索日志
Get-Content .\logs\app.log | Select-String "\[搜索\]"

# 查看错误
Get-Content .\logs\app.log | Select-String "ERROR"
```

## 📈 性能优化

### 成本优化
- **完全免费**: 支持多种免费embedding方案，零API成本
- **智能切换**: 系统自动选择最佳免费方案
- **本地部署**: 无需云端服务，降低运营成本
- **资源优化**: 支持多种embedding方案，根据需求平衡性能和成本

### 性能调优
- **Embedding选择**: 根据需求选择不同质量和速度的方案
- **检索参数**: 调整检索参数（top_k, similarity_threshold）
- **批处理**: 优化文档处理和批量操作
- **缓存机制**: 实现智能缓存减少重复计算

### 质量提升
- **语义理解**: Sentence Transformers提供接近商业模型的语义理解
- **关键词优化**: TF-IDF提供精确的关键词匹配
- **隐私保护**: 所有计算本地进行，数据不外泄

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情

## 📞 支持

如有问题或建议，请通过以下方式联系：
- 创建Issue: [GitHub Issues](https://github.com/wujelly701/personal_knowledge_agent/issues)
- 查看FAQ: `使用指南与FAQ.md`
- 查看更新日志: `CHANGELOG-20251107.md`

## 📝 版本历史

### v1.1.0 (2025-11-07) - 阶段1完成 🎉
- ✅ **BM25关键词搜索**: 实现基于rank-bm25的混合检索
- ✅ **Embedding缓存**: LRU缓存机制，性能提升80%+
- ✅ **Singleton模式**: 防止EmbeddingManager重复实例化
- ✅ **内容哈希去重**: 智能检测重复上传，支持增量更新
- ✅ **多维度分类**: 文件名(40%) + 内容(60%)加权评分
- ✅ **ChromaDB兼容**: 修复元数据类型约束问题
- ✅ **UI/UX优化**: 自动清空、emoji修复、chunk编号修正
- ✅ **依赖更新**: 添加rank-bm25>=0.2.2

### v1.0.0 (2025-11-07)
- ✅ 核心文档管理功能
- ✅ 智能问答（DeepSeek）
- ✅ 混合检索（向量+关键词）
- ✅ 文档管理UI（查看、删除、更新）
- ✅ 统计分析增强
- ✅ 搜索功能完善（Enter键支持）
- ✅ 日志系统优化（时间戳）
- ✅ 配置参数统一（CHUNK_SIZE=1000）

详见: `CHANGELOG-20251107.md`

---

**构建于**: LangChain + Gradio + Chroma + Sentence Transformers  
**版本**: v1.1.0  
**最后更新**: 2025-11-07

