
import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from nkkagent.agent.pae import PlanAndAct
from nkkagent.agent.toolcallagent import ToolCallAgent, TOOL_CALL_REQUIRED
from nkkagent.schema.agentstate import AgentState
from nkkagent.schema.message import Message
from nkkagent.tools.planning import PlanningTool
from nkkagent.tools.terminate import Terminate
from nkkagent.tools.tool_collection import ToolCollection
from nkkagent.prompt.planning import PLANNING_SYSTEM_PROMPT, NEXT_STEP_PROMPT

# 导入日志记录器
from nkkagent.logs import get_logger
logger = get_logger(__name__)

class PlanAgent(PlanAndAct, ToolCallAgent):
    """
    PlanAgent combines planning capabilities with tool execution.
    
    This agent first creates a structured plan using the PlanningTool,
    then executes each step of the plan using available tools.
    It tracks progress and can adapt the plan as needed during execution.
    """
    
    name: str = "planner"
    description: str = "An agent that plans and executes tasks using available tools."
    
    # Override prompts with planning-specific ones
    system_prompt: str = PLANNING_SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT
    
    # Add planning tool to available tools
    available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(
        PlanningTool(),
        Terminate()
    ))
    
    # Track the current plan and its state
    current_plan_id: Optional[str] = None
    plan_created: bool = False
    current_plan_step: int = 0
    
    async def plan(self) -> bool:
        """
        Create or update a plan based on the current state.
        
        Returns:
            bool: True if planning was successful and execution should proceed,
                 False otherwise.
        """
        # Use the think method from ToolCallAgent to generate tool calls
        should_act = await super().think()
        
        # Check if a plan was created or updated
        if self.tool_calls:
            for call in self.tool_calls:
                if call.function and call.function.name == "planning":
                    try:
                        args = json.loads(call.function.arguments or "{}")
                        if "command" in args and args["command"] in ["create", "update"]:
                            if "plan_id" in args:
                                self.current_plan_id = args["plan_id"]
                                self.plan_created = True
                    except Exception as e:
                        logger.error(f"Error processing planning tool call: {e}")
        
        return should_act
    
    async def act(self) -> str:
        """
        Execute the current step of the plan using available tools.
        
        Returns:
            str: Result of the execution.
        """
        # If no plan exists yet, create one
        if not self.plan_created and self.current_step == 1:
            # Add a message prompting to create a plan first
            self.memory.add_message(
                Message.assistant_message(
                    "I need to create a plan first before taking any actions."
                )
            )
            return "Creating initial plan..."
        
        # Execute tool calls using ToolCallAgent's act method
        result = await super().act()
        
        # Update plan step tracking if needed
        if self.tool_calls:
            for call in self.tool_calls:
                if call.function and call.function.name == "planning":
                    try:
                        args = json.loads(call.function.arguments or "{}")
                        if args.get("command") == "mark_step" and "step_status" in args:
                            if args["step_status"] == "completed":
                                self.current_plan_step += 1
                    except Exception as e:
                        logger.error(f"Error updating plan step: {e}")
        
        return result
    
    async def think(self) -> bool:
        """
        Implementation of ReActAgent's think method.
        This delegates to the plan method to maintain the planning workflow.
        
        Returns:
            bool: True if thinking was successful and execution should proceed,
                 False otherwise.
        """
        return await self.plan()
    
    async def step(self) -> str:
        """
        Execute a single step: plan and act.
        
        Returns:
            str: Result of the step execution.
        """
        # First plan
        should_act = await self.plan()
        
        # Then act if planning indicates we should
        if not should_act:
            return "Planning complete - no action needed"
        
        return await self.act()
    
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """
        Handle special tool execution and state changes.
        
        Args:
            name: The name of the tool being executed
            result: The result of the tool execution
            **kwargs: Additional arguments
        """
        await super()._handle_special_tool(name=name, result=result, **kwargs)
        
        # Check if all plan steps are completed
        if self.plan_created and name == "planning":
            try:
                args = kwargs.get("args", {})
                if args.get("command") == "get" and self.current_plan_id:
                    # Check if all steps are marked as completed
                    if isinstance(result, dict) and "step_statuses" in result:
                        plan_complete = all(
                            status == "completed" 
                            for status in result["step_statuses"]
                        )
                        
                        if plan_complete and result["step_statuses"]:
                            logger.info(f"🎉 Plan '{self.current_plan_id}' has been completed!")
                            self.state = AgentState.FINISHED
            except Exception as e:
                logger.error(f"Error checking plan completion: {e}")
    
    async def cleanup(self):
        """
        Clean up resources used by the agent's tools.
        """
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")
    
    async def run(self, request: Optional[str] = None) -> str:
        """
        Run the agent with cleanup when done.
        """
        try:
            return await super().run(request)
        finally:
            await self.cleanup()