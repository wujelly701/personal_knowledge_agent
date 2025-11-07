#!/bin/bash

# 个人知识管理Agent - Docker快速部署脚本
# 使用方法: ./deploy.sh [start|stop|restart|logs|status|update|backup]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印消息
print_message() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查Docker和Docker Compose
check_requirements() {
    print_message "检查环境要求..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose未安装"
        exit 1
    fi
    
    print_message "环境检查通过 ✓"
}

# 启动服务
start_service() {
    print_message "启动服务..."
    docker compose up -d
    print_message "服务已启动 ✓"
    print_message "访问地址: http://localhost:7860"
}

# 停止服务
stop_service() {
    print_message "停止服务..."
    docker compose down
    print_message "服务已停止 ✓"
}

# 重启服务
restart_service() {
    print_message "重启服务..."
    docker compose restart
    print_message "服务已重启 ✓"
}

# 查看日志
view_logs() {
    print_message "查看实时日志 (Ctrl+C退出)..."
    docker compose logs -f
}

# 查看状态
check_status() {
    print_message "服务状态:"
    docker compose ps
    echo ""
    print_message "资源使用:"
    docker stats --no-stream knowledge_agent
}

# 更新应用
update_app() {
    print_message "更新应用..."
    
    # 拉取最新代码
    if [ -d .git ]; then
        print_message "拉取最新代码..."
        git pull origin main
    fi
    
    # 重新构建
    print_message "重新构建镜像..."
    docker compose build --no-cache
    
    # 重启服务
    print_message "重启服务..."
    docker compose up -d --force-recreate
    
    print_message "更新完成 ✓"
}

# 备份数据
backup_data() {
    BACKUP_DIR="./backups"
    mkdir -p $BACKUP_DIR
    
    BACKUP_FILE="$BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    print_message "开始备份..."
    print_message "停止服务..."
    docker compose stop
    
    print_message "打包数据..."
    tar -czf $BACKUP_FILE data/ logs/
    
    print_message "启动服务..."
    docker compose start
    
    print_message "备份完成: $BACKUP_FILE ✓"
    
    # 清理7天前的备份
    find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +7 -delete
    print_message "旧备份已清理"
}

# 主函数
main() {
    check_requirements
    
    case "$1" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        logs)
            view_logs
            ;;
        status)
            check_status
            ;;
        update)
            update_app
            ;;
        backup)
            backup_data
            ;;
        *)
            echo "使用方法: $0 {start|stop|restart|logs|status|update|backup}"
            echo ""
            echo "命令说明:"
            echo "  start   - 启动服务"
            echo "  stop    - 停止服务"
            echo "  restart - 重启服务"
            echo "  logs    - 查看实时日志"
            echo "  status  - 查看服务状态"
            echo "  update  - 更新应用"
            echo "  backup  - 备份数据"
            exit 1
            ;;
    esac
}

main "$@"
