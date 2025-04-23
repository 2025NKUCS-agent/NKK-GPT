from enum import Enum
# from typing import Any, List, Literal, Optional, Union  # 只使用了Enum，其他类型未使用

from pydantic import BaseModel, Field

class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

