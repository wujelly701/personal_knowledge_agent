"""
对话历史数据库管理模块
使用SQLite存储会话和对话消息
"""

import sqlite3
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class ConversationDatabase:
    """对话历史数据库管理器"""
    
    def __init__(self, db_path: str = "data/conversations.db"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        logger.info(f"✅ 对话数据库已初始化: {self.db_path}")
    
    def _init_database(self):
        """创建数据库表结构"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # 创建会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    title TEXT,
                    total_turns INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # 创建消息表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_messages 
                ON messages(session_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_active 
                ON sessions(last_active DESC)
            """)
            
            conn.commit()
            logger.info("数据库表结构创建成功")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def create_session(self, title: Optional[str] = None) -> str:
        """
        创建新会话
        
        Args:
            title: 会话标题（可选，默认为"新对话"）
            
        Returns:
            会话ID
        """
        session_id = str(uuid.uuid4())
        
        if not title:
            title = f"新对话 {datetime.now().strftime('%m-%d %H:%M')}"
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO sessions (session_id, title)
                VALUES (?, ?)
            """, (session_id, title))
            
            conn.commit()
            logger.info(f"创建新会话: {session_id} - {title}")
            return session_id
            
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """
        添加消息到会话
        
        Args:
            session_id: 会话ID
            role: 角色（'user' 或 'assistant'）
            content: 消息内容
            metadata: 额外元数据（可选）
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # 插入消息
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, json.dumps(metadata) if metadata else None))
            
            # 更新会话的最后活跃时间和轮次
            cursor.execute("""
                UPDATE sessions 
                SET last_active = CURRENT_TIMESTAMP,
                    total_turns = total_turns + 1
                WHERE session_id = ?
            """, (session_id,))
            
            # 如果是第一条用户消息，更新会话标题
            cursor.execute("""
                SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'
            """, (session_id,))
            user_msg_count = cursor.fetchone()[0]
            
            if user_msg_count == 1 and role == 'user':
                # 使用第一个问题的前30字作为标题
                title = content[:30] + "..." if len(content) > 30 else content
                cursor.execute("""
                    UPDATE sessions SET title = ? WHERE session_id = ?
                """, (title, session_id))
            
            conn.commit()
            logger.debug(f"消息已添加: {session_id} - {role}")
            
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_session_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        获取会话的所有消息
        
        Args:
            session_id: 会话ID
            limit: 最多返回多少条消息（可选）
            
        Returns:
            消息列表
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            query = """
                SELECT message_id, role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (session_id,))
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                message = {
                    'message_id': row[0],
                    'role': row[1],
                    'content': row[2],
                    'timestamp': row[3],
                    'metadata': json.loads(row[4]) if row[4] else None
                }
                messages.append(message)
            
            logger.debug(f"获取会话消息: {session_id} - {len(messages)}条")
            return messages
            
        except Exception as e:
            logger.error(f"获取会话消息失败: {e}")
            return []
        finally:
            conn.close()
    
    def get_all_sessions(self, limit: int = 100) -> List[Dict]:
        """
        获取所有会话列表（按最后活跃时间倒序）
        
        Args:
            limit: 最多返回多少个会话
            
        Returns:
            会话列表
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT session_id, title, created_at, last_active, total_turns
                FROM sessions
                ORDER BY last_active DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                session = {
                    'session_id': row[0],
                    'title': row[1],
                    'created_at': row[2],
                    'last_active': row[3],
                    'total_turns': row[4]
                }
                sessions.append(session)
            
            logger.debug(f"获取会话列表: {len(sessions)}个")
            return sessions
            
        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            return []
        finally:
            conn.close()
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        获取会话详细信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息字典
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT session_id, title, created_at, last_active, total_turns, metadata
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'session_id': row[0],
                    'title': row[1],
                    'created_at': row[2],
                    'last_active': row[3],
                    'total_turns': row[4],
                    'metadata': json.loads(row[5]) if row[5] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"获取会话信息失败: {e}")
            return None
        finally:
            conn.close()
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其所有消息
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功删除
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # 删除消息
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            
            # 删除会话
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            
            conn.commit()
            logger.info(f"删除会话: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def update_session_title(self, session_id: str, new_title: str) -> bool:
        """
        更新会话标题
        
        Args:
            session_id: 会话ID
            new_title: 新标题
            
        Returns:
            是否成功更新
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE sessions SET title = ? WHERE session_id = ?
            """, (new_title, session_id))
            
            conn.commit()
            logger.info(f"更新会话标题: {session_id} -> {new_title}")
            return True
            
        except Exception as e:
            logger.error(f"更新会话标题失败: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def clear_old_sessions(self, days: int = 30) -> int:
        """
        清理旧会话（超过指定天数未活跃）
        
        Args:
            days: 天数阈值
            
        Returns:
            删除的会话数量
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # 查找旧会话
            cursor.execute("""
                SELECT session_id FROM sessions
                WHERE julianday('now') - julianday(last_active) > ?
            """, (days,))
            
            old_sessions = [row[0] for row in cursor.fetchall()]
            
            # 删除消息
            cursor.execute("""
                DELETE FROM messages WHERE session_id IN (
                    SELECT session_id FROM sessions
                    WHERE julianday('now') - julianday(last_active) > ?
                )
            """, (days,))
            
            # 删除会话
            cursor.execute("""
                DELETE FROM sessions
                WHERE julianday('now') - julianday(last_active) > ?
            """, (days,))
            
            conn.commit()
            logger.info(f"清理了 {len(old_sessions)} 个旧会话（>{days}天未活跃）")
            return len(old_sessions)
            
        except Exception as e:
            logger.error(f"清理旧会话失败: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # 会话总数
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0]
            
            # 消息总数
            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]
            
            # 平均每会话轮次
            cursor.execute("SELECT AVG(total_turns) FROM sessions")
            avg_turns = cursor.fetchone()[0] or 0
            
            # 最近7天活跃会话
            cursor.execute("""
                SELECT COUNT(*) FROM sessions
                WHERE julianday('now') - julianday(last_active) <= 7
            """)
            recent_active = cursor.fetchone()[0]
            
            return {
                'total_sessions': total_sessions,
                'total_messages': total_messages,
                'avg_turns_per_session': round(avg_turns, 1),
                'recent_active_sessions': recent_active
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
        finally:
            conn.close()
