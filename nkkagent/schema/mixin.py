
# 先添加项目根目录到Python路径
import sys
import os
# 获取项目根目录的绝对路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 将项目根目录添加到模块搜索路径
sys.path.append(root_dir)
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from uuid import uuid4
from datetime import datetime
from typing import Optional, Dict, Any
from nkkagent.config.exceptions import handle_exception
# from pathlib import Path  # 未使用

class MixinModel(BaseModel):
    """
    统一基类，整合以下核心功能：
    1. 公共字段 (ID/时间戳)
    2. 多态序列化
    3. 异常处理
    4. 消息路由基础能力
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    
    # 公共字段
    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=datetime.now)
    
    # 消息路由字段（来自Message类）
    sent_from: str = ""
    send_to: set[str] = {"ALL"}
    cause_by: str = ""
    
    # 多态序列化支持（来自SerializationMixin）
    __subclasses_map__: Dict[str, Any] = {}
    
    @model_serializer(mode="wrap")
    def _serialize_with_class_type(self, serializer):
        data = serializer(self)
        data["__class__"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return data
    
    @model_validator(mode="wrap")
    @classmethod
    def _deserialize_with_class_type(cls, value, handler):
        if isinstance(value, dict) and "__class__" in value:
            class_path = value.pop("__class__")
            target_cls = cls.__subclasses_map__.get(class_path, cls)
            return target_cls(**value)
        return handler(value)
    
    @classmethod
    @handle_exception
    def loads(cls, data: str) -> Optional["MixinModel"]:
        return cls.model_validate_json(data)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__subclasses_map__[f"{cls.__module__}.{cls.__name__}"] = cls