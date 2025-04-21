from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator
from nkkagent.agent.browseragent import BrowserContextHelper
from nkkagent.agent.toolcallagent import ToolCallAgent
from nkkagent.tools.terminate import Terminate
from nkkagent.tools.tool_collection import ToolCollection
from nkkagent.prompt.user import SYSTEM_PROMPT,NEXT_STEP_PROMPT
from nkkagent.tools.actiongraph_tool import ActionNodeGraphTool
from nkkagent.tools.python_execute import PythonExecute
from nkkagent.tools.browser_use_tool import BrowserUseTool
from nkkagent.tools.knowledge_graph_tool import KnowledgeGraphTool
from nkkagent.tools.file_operators import FileOperator
from nkkagent.tools.str_replace_editor import StrReplaceEditor
from nkkagent.config.config import config
# 导入日志记录器
from nkkagent.logs import get_logger
logger = get_logger(__name__)



class UserAgent(ToolCallAgent):
    """A versatile general-purpose agent."""

    name: str = "UserAgent"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), BrowserUseTool(), StrReplaceEditor(), Terminate(),KnowledgeGraphTool(),ActionNodeGraphTool()
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "UserAgent":
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """Process current state and decide next actions with appropriate context."""
        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # Restore original prompt
        self.next_step_prompt = original_prompt

        return result

    async def cleanup(self):
        """Clean up Manus agent resources."""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
