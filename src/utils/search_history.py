"""
搜索历史管理器
保存和管理用户的搜索历史记录
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)


class SearchHistoryManager:
    """搜索历史管理器"""

    def __init__(self, history_file: str = "data/search_history.json", max_records: int = 50):
        """
        初始化搜索历史管理器

        Args:
            history_file: 历史记录文件路径
            max_records: 最大保存记录数
        """
        self.history_file = Path(history_file)
        self.max_records = max_records
        self.history: deque = deque(maxlen=max_records)

        # 确保数据目录存在
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载历史记录
        self._load_history()

    def _load_history(self) -> None:
        """从文件加载历史记录"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 转换为deque
                    self.history = deque(data, maxlen=self.max_records)
                    logger.info(f"成功加载 {len(self.history)} 条搜索历史")
            else:
                logger.info("历史记录文件不存在，创建新的记录")
        except Exception as e:
            logger.error(f"加载搜索历史失败: {e}")
            self.history = deque(maxlen=self.max_records)

    def _save_history(self) -> None:
        """保存历史记录到文件"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                # 转换deque为list以便JSON序列化
                json.dump(list(self.history), f, ensure_ascii=False, indent=2)
            logger.debug(f"成功保存 {len(self.history)} 条搜索历史")
        except Exception as e:
            logger.error(f"保存搜索历史失败: {e}")

    def add_search(self, query: str, mode: str = "混合检索", top_k: int = 5, 
                   results_count: int = 0) -> None:
        """
        添加一条搜索记录

        Args:
            query: 搜索查询
            mode: 搜索模式
            top_k: 返回结果数
            results_count: 实际返回的结果数量
        """
        if not query or not query.strip():
            return

        # 创建搜索记录
        record = {
            "query": query.strip(),
            "mode": mode,
            "top_k": top_k,
            "results_count": results_count,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 检查是否与最近一条记录重复（避免连续重复）
        if self.history and len(self.history) > 0:
            last_record = self.history[-1]
            if (last_record.get("query") == record["query"] and 
                last_record.get("mode") == record["mode"]):
                logger.debug(f"跳过重复的搜索记录: {query}")
                return

        # 添加到历史记录（deque会自动处理max_records限制）
        self.history.append(record)

        # 保存到文件
        self._save_history()

        logger.info(f"添加搜索历史: '{query}' ({mode})")

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的搜索记录

        Args:
            limit: 返回记录数量

        Returns:
            搜索记录列表（从新到旧）
        """
        # 从deque中提取最近的记录（倒序）
        recent = list(self.history)[-limit:]
        recent.reverse()  # 最新的在前
        return recent

    def search_history(self, keyword: str) -> List[Dict[str, Any]]:
        """
        在历史记录中搜索包含关键词的查询

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的搜索记录列表
        """
        if not keyword:
            return self.get_recent()

        keyword_lower = keyword.lower()
        matches = [
            record for record in self.history
            if keyword_lower in record.get("query", "").lower()
        ]

        # 倒序排列（最新的在前）
        matches.reverse()
        return matches

    def clear_history(self) -> None:
        """清空所有历史记录"""
        self.history.clear()
        self._save_history()
        logger.info("已清空搜索历史")

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取搜索历史统计信息

        Returns:
            统计信息字典
        """
        if not self.history:
            return {
                "total_searches": 0,
                "unique_queries": 0,
                "most_common_queries": [],
                "modes_distribution": {}
            }

        # 统计查询频率
        from collections import Counter

        queries = [record["query"] for record in self.history]
        modes = [record["mode"] for record in self.history]

        query_counter = Counter(queries)
        mode_counter = Counter(modes)

        return {
            "total_searches": len(self.history),
            "unique_queries": len(query_counter),
            "most_common_queries": query_counter.most_common(5),
            "modes_distribution": dict(mode_counter)
        }

    def format_history_for_display(self, limit: int = 10) -> str:
        """
        格式化历史记录用于UI显示

        Args:
            limit: 显示记录数量

        Returns:
            格式化的历史记录字符串
        """
        recent = self.get_recent(limit)

        if not recent:
            return "暂无搜索历史"

        lines = ["## 最近搜索记录\n"]
        for i, record in enumerate(recent, 1):
            query = record.get("query", "")
            mode = record.get("mode", "未知")
            date = record.get("date", "")
            results_count = record.get("results_count", 0)

            lines.append(
                f"{i}. **{query}** "
                f"({mode}, {results_count}个结果) "
                f"- {date}"
            )

        return "\n".join(lines)

    def get_history_dropdown_choices(self, limit: int = 20) -> List[str]:
        """
        获取历史记录的下拉选择列表

        Args:
            limit: 返回记录数量

        Returns:
            历史查询列表
        """
        recent = self.get_recent(limit)
        # 返回唯一的查询文本
        seen = set()
        choices = []
        for record in recent:
            query = record.get("query", "")
            if query and query not in seen:
                seen.add(query)
                choices.append(query)
        return choices
