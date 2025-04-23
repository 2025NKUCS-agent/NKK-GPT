from mixin import MixinModel
from typing import Callable, Dict  # 这两个导入都被使用了

class ParseFunction(MixinModel):
    """
    类似ParseFunction的扩展能力集成
    功能：统一参数解析模板
    """
    _parser_registry: Dict[str, Callable] = {}
    
    @classmethod
    def register_parser(cls, name: str):
        def decorator(func: Callable):
            cls._parser_registry[name] = func
            return func
        return decorator
    
    def parse(self, data_type: str, raw_data: str):
        parser = self._parser_registry.get(data_type)
        if not parser:
            raise ValueError(f"No parser registered for {data_type}")
        return parser(raw_data)
    