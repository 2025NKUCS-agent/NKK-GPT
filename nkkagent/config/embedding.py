from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable
from nkkagent.config.llmtypes import YamlModel

class EmbeddingType(Enum):
    """Embedding type"""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    ZHIPUAI = "zhipuai"
    ONNX = "onnx"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    DASHSCOPE = "dashscope"
    QIANFAN = "qianfan"
    CUSTOM = "custom"

    def __missing__(self, key):
        return self.OPENAI


class EmbeddingConfig(YamlModel):
    """Embedding config"""

    # API Configuration
    api_key: str = ""
    api_type: EmbeddingType = EmbeddingType.OPENAI
    base_url: str = "https://api.openai.com/v1"
    api_version: Optional[str] = None

    # Model Configuration
    model: str = "text-embedding-ada-002"

    # For Cloud Service Provider like Baidu/Alibaba
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    endpoint: Optional[str] = None  # for self-deployed model on the cloud

    # For Network
    proxy: Optional[str] = None

    # For Custom Embedding
    custom_embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None

    # For Amazon Bedrock
    region_name: Optional[str] = None


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Get embedding for text"""
    # This is a placeholder function
    # In a real implementation, this would call the appropriate API based on the model
    return [0.0] * 1536  # Return a dummy embedding vector