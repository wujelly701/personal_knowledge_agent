"""
个人知识管理Agent系统配置
"""
import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings:
    """应用配置类"""
    
    # API配置
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    
    # 数据库路径
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    METADATA_DB_PATH: str = os.getenv("METADATA_DB_PATH", "./data/metadata.db")
    
    # 应用设置
    APP_NAME: str = os.getenv("APP_NAME", "个人知识管理助手")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # RAG配置
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "500"))
    
    # Gradio设置
    GRADIO_SERVER_PORT: int = int(os.getenv("GRADIO_SERVER_PORT", "8888"))
    GRADIO_SERVER_HOST: str = os.getenv("GRADIO_SERVER_HOST", "127.0.0.1 ")
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "False").lower() == "true"
    GRADIO_DEBUG: int = 1 if os.getenv("GRADIO_DEBUG", "0") else 0
    
    # 模型配置
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI模型名（可选）
    EMBEDDING_METHOD: str = os.getenv("EMBEDDING_METHOD", "auto")  # 智能选择方法
    # 可选方案:
    # - "auto": 智能选择（优先OpenAI，然后Sentence Transformers，最后text-hash）
    # - "openai": 使用OpenAI API (需要OPENAI_API_KEY)
    # - "all-MiniLM-L6-v2": Sentence Transformers (免费，高质量，推荐)
    # - "text-hash": 文本哈希 (零依赖，最快，兼容性好)
    # - "bow-tfidf": 词袋模型 (需要sklearn，关键词优化)
    LLM_MODEL: str = "deepseek-chat"
    TEMPERATURE: float = 0.7

    # 文档处理配置
    SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx"]
    MAX_FILE_SIZE_MB = 50
    MAX_FILES_PER_BATCH = 20

    # 检索配置
    RETRIEVAL_MODES = {
        "混合检索": "hybrid",
        "语义检索": "semantic",
        "关键词检索": "keyword"
    }

    # 分类标签
    DOCUMENT_CATEGORIES = ["工作", "学习", "个人", "参考", "研究", "想法"]
    PRIORITY_LEVELS = ["高", "中", "低"]

    @classmethod
    def get_optimal_embedding_method(cls) -> str:
        """
        智能选择最佳embedding方法
        优先级：OpenAI > Sentence Transformers > text-hash
        """
        # 如果明确指定了方法，就使用指定的
        if cls.EMBEDDING_METHOD not in ["auto", ""]:
            return cls.EMBEDDING_METHOD

        # 智能选择逻辑
        if cls.OPENAI_API_KEY:
            # 优先级1: OpenAI (如果提供了API密钥)
            return "openai"
        else:
            # 优先级2: Sentence Transformers (免费高质量)
            try:
                import sentence_transformers
                return "all-MiniLM-L6-v2"
            except ImportError:
                # 优先级3: text-hash (零依赖回退)
                return "text-hash"

    @classmethod
    def validate(cls):
        """验证配置（简化版本：警告但不阻止）"""
        warnings = []

        # 选择最佳embedding方法
        optimal_method = cls.get_optimal_embedding_method()

        if optimal_method == "openai" and not cls.OPENAI_API_KEY:
            warnings.append("无法使用OpenAI embedding：缺少OPENAI_API_KEY")
        elif optimal_method == "all-MiniLM-L6-v2":
            try:
                import sentence_transformers
                warnings.append("✅ 使用免费的Sentence Transformers (all-MiniLM-L6-v2)进行embedding")
            except ImportError:
                warnings.append("Sentence Transformers不可用，将回退到文本哈希方法")
        elif optimal_method == "text-hash":
            warnings.append("✅ 使用文本哈希方法进行embedding（零成本）")

        if not cls.DEEPSEEK_API_KEY:
            warnings.append("缺少DEEPSEEK_API_KEY环境变量：系统将提供基于文档内容的简化回答")

        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                if warning.startswith("✅"):
                    logger.info(warning)
                else:
                    logger.warning(warning)
            logger.info("继续运行系统...")
        
        return True

# 创建配置实例
settings = Settings()
