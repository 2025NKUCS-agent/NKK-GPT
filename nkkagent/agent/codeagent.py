from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List 
from nkkagent.config.llmtypes import AgentInfo
from nkkagent.agent.toolcallagent import ToolCallAgent
from nkkagent.prompt.swe import SYSTEM_PROMPT
from nkkagent.tools.bash import Bash
from nkkagent.tools.str_replace_editor import StrReplaceEditor
from nkkagent.tools.terminate import Terminate
from nkkagent.tools.tool_collection import ToolCollection

class CodeAgent(ToolCallAgent):

    """An agent that implements the SWEAgent paradigm for executing code and natural conversations."""
    name: str = "swe"
    description: str = "an autonomous AI programmer that interacts directly with the computer to solve tasks."
    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = ""
    available_tools: ToolCollection = ToolCollection(
        Bash(), StrReplaceEditor(), Terminate()
    )
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])
    max_steps: int = 20
