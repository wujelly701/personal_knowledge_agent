# Docker部署指南

## 📋 前置要求

在云服务器上安装：
- Docker Engine 20.10+
- Docker Compose 2.0+

### 安装Docker（Ubuntu/Debian）

```bash
# 更新包索引
sudo apt update

# 安装必要的包
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# 添加Docker官方GPG密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 添加Docker仓库
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 安装Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到docker组（避免每次使用sudo）
sudo usermod -aG docker $USER

# 重新登录或运行
newgrp docker
```

### 安装Docker Compose

```bash
# Docker Compose v2已包含在docker-ce中
# 验证安装
docker compose version
```

---

## 🚀 快速部署

### 1. 克隆代码到云服务器

```bash
# SSH登录到云服务器
ssh user@your-server-ip

# 克隆仓库
git clone https://github.com/wujelly701/personal_knowledge_agent.git
cd personal_knowledge_agent
```

### 2. 配置环境变量（可选）

```bash
# 如果需要使用API密钥，编辑docker-compose.yml
nano docker-compose.yml

# 取消注释并填入API密钥：
# - OPENAI_API_KEY=sk-xxx
# - DEEPSEEK_API_KEY=sk-xxx
```

### 3. 启动服务

```bash
# 构建并启动容器
docker compose up -d

# 查看日志
docker compose logs -f

# 检查容器状态
docker compose ps
```

### 4. 访问应用

在浏览器中访问：
```
http://your-server-ip:7860
```

---

## 📝 常用命令

### 容器管理

```bash
# 启动服务
docker compose up -d

# 停止服务
docker compose down

# 重启服务
docker compose restart

# 查看运行状态
docker compose ps

# 查看实时日志
docker compose logs -f

# 查看最近100行日志
docker compose logs --tail=100

# 进入容器
docker compose exec knowledge-agent bash

# 重新构建镜像
docker compose build

# 重新构建并启动
docker compose up -d --build
```

### 数据管理

```bash
# 备份数据目录
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/

# 查看数据目录大小
du -sh data/ logs/

# 清理日志
docker compose exec knowledge-agent bash -c "echo '' > /app/logs/app.log"
```

### 更新应用

```bash
# 拉取最新代码
git pull origin main

# 重新构建并启动
docker compose up -d --build

# 或者不停机更新
docker compose build
docker compose up -d --no-deps --build knowledge-agent
```

---

## 🔧 高级配置

### 1. 使用Nginx反向代理

创建 `nginx.conf`：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket支持（如果需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

启动Nginx：

```bash
# 安装Nginx
sudo apt install nginx

# 复制配置
sudo cp nginx.conf /etc/nginx/sites-available/knowledge-agent
sudo ln -s /etc/nginx/sites-available/knowledge-agent /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启Nginx
sudo systemctl restart nginx
```

### 2. 配置HTTPS（使用Let's Encrypt）

```bash
# 安装Certbot
sudo apt install certbot python3-certbot-nginx

# 获取SSL证书
sudo certbot --nginx -d your-domain.com

# 自动续期已配置，测试续期
sudo certbot renew --dry-run
```

### 3. 资源限制调整

编辑 `docker-compose.yml`：

```yaml
deploy:
  resources:
    limits:
      cpus: '4'        # 增加CPU限制
      memory: 8G       # 增加内存限制
    reservations:
      cpus: '2'
      memory: 4G
```

### 4. 使用外部数据卷

```yaml
volumes:
  knowledge_data:
    driver: local

services:
  knowledge-agent:
    volumes:
      - knowledge_data:/app/data
      - ./logs:/app/logs
```

---

## 🔒 安全建议

### 1. 配置防火墙

```bash
# 安装UFW
sudo apt install ufw

# 允许SSH
sudo ufw allow 22/tcp

# 允许HTTP/HTTPS（如果使用Nginx）
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 或者只允许应用端口（不推荐直接暴露）
# sudo ufw allow 7860/tcp

# 启用防火墙
sudo ufw enable

# 查看状态
sudo ufw status
```

### 2. 修改默认端口

编辑 `docker-compose.yml`：

```yaml
ports:
  - "8080:7860"  # 使用8080代替7860
```

### 3. 使用密钥认证

在Gradio中添加认证（修改 `main.py`）：

```python
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=("admin", "your_password_here"),  # 添加认证
    share=False
)
```

### 4. 环境变量加密

使用 `.env` 文件：

```bash
# 创建.env文件
cat > .env << EOF
OPENAI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=sk-xxx
EOF

# 修改docker-compose.yml
env_file:
  - .env

# 设置文件权限
chmod 600 .env
```

---

## 📊 监控和日志

### 1. 查看系统资源使用

```bash
# 实时监控容器资源
docker stats knowledge_agent

# 查看磁盘使用
df -h

# 查看容器详情
docker inspect knowledge_agent
```

### 2. 配置日志轮转

创建 `/etc/docker/daemon.json`：

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

重启Docker：

```bash
sudo systemctl restart docker
```

### 3. 使用监控工具

```bash
# 安装ctop（Docker容器监控）
sudo wget https://github.com/bcicen/ctop/releases/download/v0.7.7/ctop-0.7.7-linux-amd64 -O /usr/local/bin/ctop
sudo chmod +x /usr/local/bin/ctop

# 运行
ctop
```

---

## 🛠️ 故障排除

### 容器无法启动

```bash
# 查看详细日志
docker compose logs knowledge-agent

# 检查容器状态
docker compose ps

# 检查端口占用
sudo netstat -tulpn | grep 7860

# 重新构建
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 内存不足

```bash
# 查看内存使用
free -h

# 增加swap空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 网络问题

```bash
# 检查Docker网络
docker network ls
docker network inspect bridge

# 重启网络
docker compose down
docker network prune
docker compose up -d
```

### 数据持久化问题

```bash
# 检查挂载点
docker compose exec knowledge-agent df -h

# 检查权限
ls -la data/ logs/

# 修复权限
sudo chown -R 1000:1000 data/ logs/
```

---

## 🔄 备份和恢复

### 自动备份脚本

创建 `backup.sh`：

```bash
#!/bin/bash

BACKUP_DIR="/backup"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/path/to/personal_knowledge_agent"

# 创建备份目录
mkdir -p $BACKUP_DIR

# 停止容器
cd $APP_DIR
docker compose stop

# 备份数据
tar -czf $BACKUP_DIR/knowledge_agent_$DATE.tar.gz data/ logs/

# 启动容器
docker compose start

# 删除7天前的备份
find $BACKUP_DIR -name "knowledge_agent_*.tar.gz" -mtime +7 -delete

echo "Backup completed: knowledge_agent_$DATE.tar.gz"
```

设置定时任务：

```bash
# 编辑crontab
crontab -e

# 每天凌晨2点备份
0 2 * * * /path/to/backup.sh >> /var/log/knowledge_agent_backup.log 2>&1
```

### 恢复备份

```bash
# 停止服务
docker compose down

# 恢复数据
tar -xzf /backup/knowledge_agent_20251107.tar.gz

# 启动服务
docker compose up -d
```

---

## 📈 性能优化

### 1. 使用更快的Embedding模型

编辑 `docker-compose.yml`：

```yaml
environment:
  - EMBEDDING_METHOD=all-MiniLM-L6-v2  # 推荐，平衡性能和质量
```

### 2. 调整Worker数量

如果使用Gunicorn（修改main.py）：

```python
# 添加Gunicorn配置
workers = 4  # CPU核心数 x 2 + 1
```

### 3. 使用SSD存储

确保数据目录在SSD上：

```yaml
volumes:
  - /mnt/ssd/knowledge_agent/data:/app/data
```

---

## 📞 获取帮助

- GitHub Issues: https://github.com/wujelly701/personal_knowledge_agent/issues
- 文档: README.md
- FAQ: 使用指南与FAQ.md

---

**部署日期**: 2025-11-07  
**Docker版本**: 20.10+  
**Docker Compose版本**: 2.0+
