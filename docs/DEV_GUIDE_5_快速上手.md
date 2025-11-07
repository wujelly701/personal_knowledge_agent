# 个人知识管理Agent系统 - 快速上手指南

## 文档信息
- **版本**: v1.0.0
- **创建日期**: 2025-11-07
- **适用人员**: 新入职开发者、运维人员、系统管理员

---

## 1. 环境准备

### 1.1 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+) |
| **Python** | 3.8 或更高版本 |
| **内存** | 最低 2GB, 推荐 4GB+ |
| **磁盘空间** | 最低 1GB (用于代码+数据+模型) |
| **网络** | 可选 (无网络可使用文本哈希embedding) |

### 1.2 检查Python版本

```bash
python --version
# 或
python3 --version

# 期望输出: Python 3.8.x 或更高
```

如果版本不符，请访问 [python.org](https://www.python.org/downloads/) 下载最新版本。

---

## 2. 项目安装

### 2.1 克隆项目

```bash
# 如果使用Git
git clone <repository-url>
cd personal_knowledge_agent

# 或直接下载ZIP并解压
```

### 2.2 创建虚拟环境 (推荐)

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3 安装依赖

#### 2.3.1 核心依赖 (必须)
```bash
pip install -r requirements.txt
```

**核心库说明**:
- `gradio`: Web界面框架
- `langchain`: LLM应用框架
- `chromadb`: 向量数据库
- `pypdf2`, `python-docx`: 文档解析

#### 2.3.2 可选依赖 (推荐)

**Sentence Transformers** (免费高质量embedding):
```bash
pip install sentence-transformers
```

**TF-IDF支持**:
```bash
pip install scikit-learn
```

#### 2.3.3 验证安装

```bash
python -c "import gradio; import chromadb; print('✅ 核心依赖安装成功')"
python -c "import sentence_transformers; print('✅ Sentence Transformers已安装')"
```

---

## 3. 配置系统

### 3.1 创建配置文件

在项目根目录创建 `.env` 文件:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

如果没有 `.env.example`，手动创建 `.env`:

```env
# ========== API配置 (可选) ==========
# DeepSeek API密钥 (用于智能问答，可选)
# DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here

# OpenAI API密钥 (用于高质量embedding，可选)
# OPENAI_API_KEY=sk-your-openai-api-key-here

# ========== Embedding配置 ==========
# 选择embedding方法:
# - auto: 自动选择 (推荐)
# - all-MiniLM-L6-v2: Sentence Transformers (免费)
# - text-hash: 文本哈希 (零依赖)
# - openai: OpenAI Embeddings (需API密钥)
EMBEDDING_METHOD=auto

# ========== 数据库配置 ==========
VECTOR_DB_PATH=./data/vector_db

# ========== RAG参数 ==========
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
MAX_TOKENS=500
TEMPERATURE=0.7

# ========== Gradio配置 ==========
GRADIO_SERVER_PORT=8888
GRADIO_SERVER_HOST=127.0.0.1
GRADIO_SHARE=false
GRADIO_DEBUG=0

# ========== 日志配置 ==========
LOG_LEVEL=INFO
DEBUG=false
```

### 3.2 配置说明

#### 3.2.1 零配置运行 (完全免费)
如果不配置任何API密钥，系统将:
- ✅ 使用 Sentence Transformers (如已安装) 或文本哈希
- ✅ 使用简化回答模式 (基于检索结果)
- ✅ 所有功能正常运行，只是无AI生成功能

#### 3.2.2 最佳配置 (推荐)
```env
# 配置DeepSeek API (获取智能回答)
DEEPSEEK_API_KEY=sk-your-key

# 使用免费高质量embedding
EMBEDDING_METHOD=all-MiniLM-L6-v2
```

**获取DeepSeek API密钥**:
1. 访问 [platform.deepseek.com](https://platform.deepseek.com/)
2. 注册账号
3. 创建API密钥

---

## 4. 启动系统

### 4.1 首次启动

```bash
python main.py
```

**启动日志示例**:
```
INFO:__main__:启动 个人知识管理助手 v1.0.0
INFO:__main__:配置: DEBUG=False, LOG_LEVEL=INFO
INFO:__main__:验证配置...
INFO:config.settings:✅ 使用免费的Sentence Transformers
INFO:__main__:初始化应用...
INFO:src.storage.vector_store_simple:使用现有集合: knowledge_base
INFO:src.storage.vector_store_simple:✅ 使用Sentence Transformers embedding
INFO:__main__:启动服务器: http://127.0.0.1:8888
INFO:__main__:按 Ctrl+C 停止服务器
Running on local URL:  http://127.0.0.1:8888
```

### 4.2 访问Web界面

启动成功后:
1. 浏览器会自动打开 `http://127.0.0.1:8888`
2. 如未自动打开，手动访问该地址

### 4.3 停止系统

在终端按 `Ctrl+C` 停止服务器。

---

## 5. 基本使用流程

### 5.1 上传文档

1. 切换到 **📤 文档上传** 标签页
2. 点击"选择文件"上传文档 (.pdf, .txt, .md, .docx)
3. 点击"处理文档"
4. 等待处理完成

**示例输出**:
```
✅ 成功处理 12 个文档块

📊 处理详情：
  您的文档被自动分割成 12 个可管理的文本块

  • 学习: 8 个 (教育材料或学习内容)
  • 参考: 4 个 (参考资料或引用内容)

💡 这些文档块现在已经存储在知识库中...
```

### 5.2 智能问答

1. 切换到 **💬 智能问答** 标签页
2. 在输入框输入问题
3. 按Enter或点击"发送"
4. 查看AI回答

**示例问答**:
```
用户: Python装饰器如何使用?

AI: 🤖 AI 智能回答：
Python装饰器是一种设计模式，允许在不修改原函数代码的情况下扩展功能。
[来源: python_notes.md]

基本语法如下:
@decorator
def function():
    pass

📊 置信度：高 (0.92)

📚 信息来源：
1. python_notes.md (相关性: 0.95)
2. advanced_python.pdf (相关性: 0.78)
```

### 5.3 文档搜索

1. 切换到 **🔍 文档搜索** 标签页
2. 输入搜索关键词
3. 选择检索模式 (混合检索/语义检索)
4. 设置返回结果数量
5. 点击"搜索"或按Enter

### 5.4 查看统计

1. 切换到 **📊 统计信息** 标签页
2. 点击"获取统计信息"
3. 查看知识库整体状况

### 5.5 文档管理

1. 切换到 **🗂️ 文档管理** 标签页
2. 点击"刷新文档列表"查看所有文档
3. 删除文档: 输入文件名 → 点击"删除文档"
4. 更新文档: 输入旧文件名 + 选择新文件 → 点击"更新文档"

---

## 6. 开发环境配置

### 6.1 IDE推荐

**推荐IDE**:
- **VS Code** (推荐)
  - 安装插件: Python, Pylance
- **PyCharm** (专业版/社区版)

### 6.2 VS Code配置

**创建 `.vscode/settings.json`**:
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

**创建 `.vscode/launch.json`** (调试配置):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Main",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
```

### 6.3 代码风格

**安装代码格式化工具**:
```bash
pip install black pylint
```

**格式化代码**:
```bash
black src/
```

**检查代码质量**:
```bash
pylint src/
```

---

## 7. 调试技巧

### 7.1 开启DEBUG模式

**修改 `.env`**:
```env
DEBUG=true
LOG_LEVEL=DEBUG
GRADIO_DEBUG=1
```

**重启应用**后，日志将更详细:
```
DEBUG:src.storage.vector_store_simple:使用简单文本哈希查询embedding
DEBUG:src.storage.vector_store_simple:生成了 1 个查询的embedding
INFO:src.storage.vector_store_simple:搜索完成: 查询='Python装饰器...', 结果数量=5
```

### 7.2 查看日志文件

日志文件位置: `logs/app.log`

```bash
# 实时查看日志 (Linux/macOS)
tail -f logs/app.log

# Windows (使用PowerShell)
Get-Content logs\app.log -Wait -Tail 50
```

### 7.3 常见问题排查

#### 问题1: 端口被占用
```
ERROR: Address already in use
```

**解决方案**:
```env
# 修改端口
GRADIO_SERVER_PORT=7860
```

#### 问题2: Embedding模型下载失败
```
WARNING: Sentence Transformers不可用，将回退到文本哈希方法
```

**解决方案**:
```bash
# 手动下载模型
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 或使用文本哈希 (无需下载)
EMBEDDING_METHOD=text-hash
```

#### 问题3: 文档上传失败
```
ERROR: 文件处理失败
```

**检查步骤**:
1. 文件大小 < 50MB
2. 文件格式: .pdf, .txt, .md, .docx
3. 文件未损坏
4. 查看详细错误日志

---

## 8. 测试系统

### 8.1 快速测试

**测试脚本**: `quick_check.py`
```bash
python quick_check.py
```

**输出示例**:
```
✅ ChromaDB连接成功
✅ Embedding生成成功
✅ 文档加载成功
✅ 向量存储成功
✅ 检索功能正常
```

### 8.2 单元测试

**运行测试**:
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_embedding.py -v
```

### 8.3 手动测试用例

#### 测试用例1: 文档上传
1. 上传 `user_files/python_learning_notes.md`
2. 验证处理成功
3. 检查统计信息是否更新

#### 测试用例2: 智能问答
1. 提问: "Python装饰器如何使用?"
2. 验证返回相关答案
3. 检查来源引用是否正确

#### 测试用例3: 文档搜索
1. 搜索关键词: "装饰器"
2. 验证返回相关文档
3. 检查相关性分数

---

## 9. 常见问题 (FAQ)

### 9.1 环境相关

**Q: 需要GPU吗?**
A: 不需要。Sentence Transformers可以使用CPU运行，速度稍慢但完全可用。

**Q: 可以在离线环境运行吗?**
A: 可以。使用文本哈希embedding (`EMBEDDING_METHOD=text-hash`)，无需任何网络连接。

**Q: 支持Docker部署吗?**
A: 支持。参考 `Dockerfile` 和 `docker-compose.yml`。

### 9.2 功能相关

**Q: 没有DeepSeek API能用吗?**
A: 可以。系统会自动降级到简化回答模式，展示检索到的文档摘要。

**Q: 支持哪些文档格式?**
A: 目前支持 PDF、TXT、Markdown、DOCX。

**Q: 一次可以上传多少文件?**
A: 默认最多20个文件，每个文件最大50MB。

**Q: 知识库数据存在哪里?**
A: 本地 `./data/vector_db/` 目录，所有数据本地存储。

### 9.3 性能相关

**Q: 处理大文档很慢?**
A: 
- 使用Sentence Transformers会更快
- 考虑减小chunk_size
- 使用GPU加速 (如果有)

**Q: 搜索速度慢?**
A:
- 减小TOP_K值
- 使用元数据过滤缩小范围
- ChromaDB自动优化索引

---

## 10. 进阶操作

### 10.1 自定义embedding方案

**修改 `config/settings.py`**:
```python
EMBEDDING_METHOD = "all-MiniLM-L6-v2"  # 或其他方案
```

**支持的方案**:
- `auto`: 智能选择
- `all-MiniLM-L6-v2`: Sentence Transformers
- `text-hash`: 文本哈希
- `bow-tfidf`: TF-IDF词袋
- `openai`: OpenAI Embeddings

### 10.2 调整RAG参数

**修改 `.env`**:
```env
CHUNK_SIZE=1500      # 增大分块大小
CHUNK_OVERLAP=300    # 增大重叠
TOP_K=10             # 返回更多结果
TEMPERATURE=0.5      # 降低LLM创造性
```

### 10.3 数据备份

**备份向量数据库**:
```bash
# 停止应用
# 复制数据库目录
cp -r ./data/vector_db ./backup/vector_db_$(date +%Y%m%d)
```

**恢复备份**:
```bash
# 停止应用
rm -rf ./data/vector_db
cp -r ./backup/vector_db_20250107 ./data/vector_db
```

---

## 11. 开发工作流

### 11.1 添加新功能

1. **创建功能分支**:
   ```bash
   git checkout -b feature/new-feature
   ```

2. **开发功能**:
   - 遵循现有代码结构
   - 添加注释和文档字符串
   - 编写单元测试

3. **测试**:
   ```bash
   python -m pytest tests/
   ```

4. **提交代码**:
   ```bash
   git add .
   git commit -m "feat: 添加新功能"
   git push origin feature/new-feature
   ```

### 11.2 修复Bug

1. **定位问题**:
   - 查看日志 `logs/app.log`
   - 开启DEBUG模式
   - 使用断点调试

2. **修复并测试**:
   - 修改代码
   - 运行相关测试
   - 验证修复效果

3. **提交修复**:
   ```bash
   git commit -m "fix: 修复XX问题"
   ```

### 11.3 代码审查

**审查清单**:
- [ ] 代码符合PEP 8规范
- [ ] 添加了必要的注释
- [ ] 更新了相关文档
- [ ] 通过所有测试
- [ ] 无明显性能问题
- [ ] 错误处理完善

---

## 12. 部署指南

### 12.1 本地部署 (已完成)
按照上述步骤即可。

### 12.2 服务器部署

#### 12.2.1 Linux服务器

**安装依赖**:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

**部署项目**:
```bash
git clone <repo-url>
cd personal_knowledge_agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**使用systemd管理** (创建 `/etc/systemd/system/knowledge-agent.service`):
```ini
[Unit]
Description=Personal Knowledge Agent
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/personal_knowledge_agent
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

**启动服务**:
```bash
sudo systemctl enable knowledge-agent
sudo systemctl start knowledge-agent
sudo systemctl status knowledge-agent
```

#### 12.2.2 Docker部署

**构建镜像**:
```bash
docker build -t knowledge-agent:latest .
```

**运行容器**:
```bash
docker run -d \
  -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/.env:/app/.env \
  --name knowledge-agent \
  knowledge-agent:latest
```

**使用docker-compose**:
```bash
docker-compose up -d
```

### 12.3 访问控制

**启用外部访问** (修改 `.env`):
```env
GRADIO_SERVER_HOST=0.0.0.0
GRADIO_SHARE=false
```

**使用Nginx反向代理** (可选):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8888;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 13. 参考资源

### 13.1 项目文档
- [架构设计文档](./DEV_GUIDE_1_架构设计.md)
- [接口设计文档](./DEV_GUIDE_2_接口设计.md)
- [数据库设计文档](./DEV_GUIDE_3_数据库设计.md)
- [核心逻辑文档](./DEV_GUIDE_4_核心逻辑.md)

### 13.2 外部资源
- [Gradio文档](https://www.gradio.app/docs)
- [LangChain教程](https://python.langchain.com/)
- [ChromaDB文档](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### 13.3 社区支持
- GitHub Issues: 报告Bug和功能请求
- 项目Wiki: 更多教程和示例

---

## 14. 快速命令参考

### 14.1 常用命令

```bash
# 启动应用
python main.py

# 运行测试
python quick_check.py
python -m pytest tests/

# 查看日志
tail -f logs/app.log

# 格式化代码
black src/

# 备份数据库
cp -r ./data/vector_db ./backup/
```

### 14.2 环境管理

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 退出虚拟环境
deactivate

# 更新依赖
pip install --upgrade -r requirements.txt

# 冻结依赖
pip freeze > requirements.txt
```

---

## 15. 下一步

恭喜! 你已经掌握了系统的基本使用。

**建议接下来**:
1. 📖 阅读[核心逻辑文档](./DEV_GUIDE_4_核心逻辑.md)了解RAG原理
2. 🔧 尝试修改embedding方案和RAG参数
3. 💡 开发自己的功能扩展
4. 🤝 参与项目贡献

**祝你使用愉快!** 🎉

---

**文档维护**: 使用流程变更时请及时更新本文档
