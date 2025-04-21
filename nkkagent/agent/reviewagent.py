# import json
# import logging
# import time
# from pathlib import Path
from typing import Annotated, Any, Literal, List, Optional
# import yaml
# from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, model_validator
# from simple_parsing.helpers.fields import field
# from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
# from tenacity import RetryError
# from typing_extensions import Self
# from unidiff import UnidiffParseError

# from config.types import AgentConfig, AgentRunResult, StepOutput
# from config.exceptions import (
#     ContentPolicyViolationError,
#     ContextWindowExceededError,
#     CostLimitExceededError,
#     FormatError,
#     TotalCostLimitExceededError,
#     _BlockedActionError,
#     _RetryWithOutput,
#     _RetryWithoutOutput,
#     _TotalExecutionTimeExceeded,
# )

from nkkagent.config.llmtypes import AgentInfo

from nkkagent.llm.llm import LLM

from nkkagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax
from nkkagent.logs import get_logger
from nkkagent.agent.pae import PlanAndAct
from nkkagent.agent.toolcallagent import ToolCallAgent
from nkkagent.schema.message import Memory
from nkkagent.schema.agentstate import AgentState

class ReviewAgent(PlanAndAct, ToolCallAgent):
    """A code review agent that inherits from PlanAndAct and ToolCallAgent"""
    
    name: str = "review_agent"
    description: str = "A code review agent that can perform code reviews and provide feedback"
    
    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None
    
    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    
    max_steps: int = 10
    current_step: int = 0
    
    async def plan(self) -> bool:
        """Process current state and decide next action"""
        # Implement review planning logic
        return True
        
    async def act(self) -> str:
        """Execute decided actions"""
        # Implement review action logic
        return "Review completed"
        
    async def review_code(self, code: str) -> List[str]:
        """Review code and return review comments"""
        # Implement code review logic
        return ["Example code review comment"]
        
    async def provide_feedback(self, review_results: List[str]) -> str:
        """Provide feedback based on review results"""
        # Implement feedback logic
        return "Feedback content"


