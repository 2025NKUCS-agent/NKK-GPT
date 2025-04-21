from __future__ import annotations

import asyncio
# import copy
import json
# import logging
# import time
# from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nkkagent.agent.pae import PlanAndAct
from nkkagent.agent.toolcallagent import ToolCallAgent, TOOL_CALL_REQUIRED

from nkkagent.schema.agentstate import AgentState
from nkkagent.schema.message import Message
from nkkagent.tools.planning import PlanningTool
from nkkagent.tools.terminate import Terminate
from nkkagent.tools.tool_collection import ToolCollection
from nkkagent.tools.config import ToolConfig,ToolHandler
# 导入日志记录器
from nkkagent.logs import get_logger
logger = get_logger(__name__)

class ManageAgentConfig(BaseModel):
    """
    ManageAgent的配置类，定义了管理代理所需的配置参数
    """
    name: str = "manager"
    description: str = "An agent that manages and assigns tasks based on plan results."
    
    # 模型配置
    #model: ModelConfig = Field(default_factory=ModelConfig)
    
    # 工具配置
    tools: ToolConfig = Field(default_factory=ToolConfig)
    
    # 最大执行步骤数
    max_steps: int = 30
    
    # 系统提示词
    system_prompt: Optional[str] = None
    
    # 下一步提示词
    next_step_prompt: Optional[str] = None
    
    # 子代理配置
    #sub_agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    # pydantic配置
    model_config = ConfigDict(extra="forbid")

class ManageAgent(PlanAndAct, ToolCallAgent):
    """
    ManageAgent负责根据计划结果分配和管理任务。
    
    该代理首先获取或创建一个结构化计划，然后根据计划步骤分配任务给不同的子代理执行。
    它跟踪任务进度，管理子代理的状态，并根据执行情况调整计划。
    """
    
    name: str = "manager"
    description: str = "An agent that manages and assigns tasks based on plan results."
    
    # 可用工具集合
    available_tools: ToolCollection = Field(default_factory=lambda: ToolCollection(
        PlanningTool(),
        Terminate()
    ))
    
    # 跟踪当前计划和状态
    current_plan_id: Optional[str] = None
    plan_created: bool = False
    current_plan_step: int = 0
    
    # 子代理管理
    #sub_agents: Dict[str, AbstractAgent] = Field(default_factory=dict)
    #active_agents: Dict[str, bool] = Field(default_factory=dict)
    #task_assignments: Dict[int, str] = Field(default_factory=dict)  # 步骤索引到代理名称的映射
    
    def __init__(self, config: ManageAgentConfig):
        """
        初始化ManageAgent
        
        Args:
            config: ManageAgent的配置对象
        """
        super().__init__()
        self.config = config
        self.name = config.name
        self.description = config.description
        self.system_prompt = config.system_prompt
        self.next_step_prompt = config.next_step_prompt
        self.max_steps = config.max_steps
        
        # 初始化工具和模型
        self.tools = ToolHandler(config.tools)
        #self.model = get_model(config.model, self.tools)
        
        # 初始化子代理
        #self._initialize_sub_agents()
    
    # def _initialize_sub_agents(self):
    #     """
    #     初始化所有配置的子代理
    #     """
    #     for agent_name, agent_config in self.config.sub_agents.items():
    #         try:
    #             # 根据代理类型创建相应的代理实例
    #             if agent_config.type == "default":
    #                 self.sub_agents[agent_name] = DefaultAgent.from_config(agent_config)
    #             # 可以添加其他类型的代理初始化
                
    #             self.active_agents[agent_name] = False  # 初始状态为非活跃
    #             logger.info(f"✅ 成功初始化子代理: {agent_name}")
    #         except Exception as e:
    #             logger.error(f"❌ 初始化子代理 '{agent_name}' 失败: {e}")
    
    async def plan(self) -> bool:
        """
        创建或更新基于当前状态的计划。
        
        Returns:
            bool: 如果计划成功且应该继续执行，则为True，否则为False。
        """
        # 使用ToolCallAgent的think方法生成工具调用
        should_act = await super().think()
        
        # 检查是否创建或更新了计划
        if self.tool_calls:
            for call in self.tool_calls:
                if call.function and call.function.name == "planning":
                    try:
                        args = json.loads(call.function.arguments or "{}")
                        if "command" in args and args["command"] in ["create", "update"]:
                            if "plan_id" in args:
                                self.current_plan_id = args["plan_id"]
                                self.plan_created = True
                                logger.info(f"📝 计划已创建/更新: {self.current_plan_id}")
                    except Exception as e:
                        logger.error(f"处理计划工具调用时出错: {e}")
        
        return should_act
    
    # async def assign_tasks(self):
    #     """
    #     根据当前计划分配任务给子代理
    #     
    #     Returns:
    #         bool: 如果任务分配成功，则为True，否则为False
    #     """
    #     if not self.current_plan_id:
    #         logger.warning("⚠️ 无法分配任务: 没有活动计划")
    #         return False
        
    #     try:
    #         # 获取当前计划详情
    #         plan_result = await self.available_tools.execute(
    #             name="planning", 
    #             tool_input={"command": "get", "plan_id": self.current_plan_id}
    #         )
            
    #         if not isinstance(plan_result, dict) or "steps" not in plan_result:
    #             logger.error(f"❌ 获取计划详情失败: {plan_result}")
    #             return False
            
    #         # 分析计划步骤并分配任务
    #         steps = plan_result.get("steps", [])
    #         step_statuses = plan_result.get("step_statuses", [])
            
    #         # 找出未分配或需要重新分配的任务
    #         for i, (step, status) in enumerate(zip(steps, step_statuses)):
    #             # 跳过已完成的步骤
    #             if status == "completed":
    #                 continue
                
    #             # 如果步骤未分配或分配的代理不活跃，则分配任务
    #             if i not in self.task_assignments or not self.active_agents.get(self.task_assignments[i], False):
    #                 # 根据步骤内容选择最合适的代理
    #                 agent_name = self._select_agent_for_task(step)
    #                 if agent_name:
    #                     self.task_assignments[i] = agent_name
    #                     self.active_agents[agent_name] = True
    #                     logger.info(f"📋 步骤 {i+1} 分配给代理: {agent_name}")
                        
    #                     # 更新步骤状态为进行中
    #                     await self.available_tools.execute(
    #                         name="planning",
    #                         tool_input={
    #                             "command": "mark_step",
    #                             "plan_id": self.current_plan_id,
    #                             "step_index": i,
    #                             "step_status": "in_progress",
    #                             "step_notes": f"Assigned to {agent_name}"
    #                         }
    #                     )
            
    #         return True
    #     except Exception as e:
    #         logger.error(f"❌ 分配任务时出错: {e}")
    #         return False
    
    # def _select_agent_for_task(self, task_description: str) -> Optional[str]:
    #     """
    #     根据任务描述选择最合适的代理
    #     
    #     Args:
    #         task_description: 任务描述
            
    #     Returns:
    #         str: 选中的代理名称，如果没有合适的代理则返回None
    #     """
    #     # 这里可以实现更复杂的代理选择逻辑，例如基于任务类型、关键词匹配等
    #     # 简单实现：选择第一个可用的代理
    #     for agent_name, is_active in self.active_agents.items():
    #         if not is_active:
    #             return agent_name
        
    #     logger.warning("⚠️ 没有可用的代理来执行任务")
    #     return None
    
    # async def monitor_tasks(self):
    #     """
    #     监控正在执行的任务，更新其状态
    #     
    #     Returns:
    #         bool: 如果所有任务都已完成，则为True，否则为False
    #     """
    #     all_completed = True
        
    #     for step_index, agent_name in self.task_assignments.items():
    #         if not self.active_agents.get(agent_name, False):
    #             continue
                
    #         agent = self.sub_agents.get(agent_name)
    #         if not agent:
    #             continue
                
    #         # 检查代理状态
    #         if agent.state == AgentState.FINISHED:
    #             # 标记步骤为已完成
    #             await self.available_tools.execute(
    #                 name="planning",
    #                 tool_input={
    #                     "command": "mark_step",
    #                     "plan_id": self.current_plan_id,
    #                     "step_index": step_index,
    #                     "step_status": "completed",
    #                     "step_notes": f"Completed by {agent_name}"
    #                 }
    #             )
                
    #             # 重置代理状态为可用
    #             self.active_agents[agent_name] = False
    #             logger.info(f"✅ 代理 {agent_name} 完成了步骤 {step_index+1}")
    #         elif agent.state == AgentState.ERROR:
    #             # 标记步骤为阻塞
    #             await self.available_tools.execute(
    #                 name="planning",
    #                 tool_input={
    #                     "command": "mark_step",
    #                     "plan_id": self.current_plan_id,
    #                     "step_index": step_index,
    #                     "step_status": "blocked",
    #                     "step_notes": f"Error in {agent_name}: {agent.last_error if hasattr(agent, 'last_error') else 'Unknown error'}"
    #                 }
    #             )
                
    #             # 重置代理状态为可用
    #             self.active_agents[agent_name] = False
    #             logger.warning(f"⚠️ 代理 {agent_name} 在执行步骤 {step_index+1} 时出错")
    #         else:
    #             # 代理仍在执行任务
    #             all_completed = False
        
    #     return all_completed
    
    # async def run_sub_agent(self, agent_name: str, task: str) -> str:
    #     """
    #     运行指定的子代理执行任务
    #     
    #     Args:
    #         agent_name: 子代理名称
    #         task: 要执行的任务描述
            
    #     Returns:
    #         str: 执行结果
    #     """
    #     agent = self.sub_agents.get(agent_name)
    #     if not agent:
    #         return f"错误: 找不到代理 '{agent_name}'"
        
    #     try:
    #         # 设置代理状态为运行中
    #         agent.state = AgentState.RUNNING
            
    #         # 运行代理
    #         result = await agent.run(task)
            
    #         return result
    #     except Exception as e:
    #         logger.error(f"运行子代理 '{agent_name}' 时出错: {e}")
    #         if hasattr(agent, 'last_error'):
    #             agent.last_error = str(e)
    #         agent.state = AgentState.ERROR
    #         return f"错误: {str(e)}"
    
    async def act(self) -> str:
        """
        执行当前步骤的操作：分配任务、监控进度、更新计划
        
        Returns:
            str: 执行结果
        """
        # 如果没有计划，先创建一个
        if not self.plan_created and self.current_step == 1:
            # 添加消息提示先创建计划
            self.memory.add_message(
                Message.assistant_message(
                    "我需要先创建一个计划，然后再分配任务。"
                )
            )
            return "正在创建初始计划..."
        
        # 执行工具调用
        result = await super().act()
        
        # 分配任务
        if self.plan_created:
            #await self.assign_tasks()
            
            # 监控任务进度
            #all_completed = await self.monitor_tasks()
            
            #if all_completed:
            #    logger.info(f"🎉 所有任务已完成！")
            #    self.state = AgentState.FINISHED
            pass
        
        return result
    
    async def think(self) -> bool:
        """
        实现ReActAgent的think方法，委托给plan方法以保持计划工作流
        
        Returns:
            bool: 如果思考成功且应该继续执行，则为True，否则为False
        """
        return await self.plan()
    
    async def step(self) -> str:
        """
        执行单个步骤：计划和行动
        
        Returns:
            str: 步骤执行结果
        """
        # 首先计划
        should_act = await self.plan()
        
        # 如果计划指示应该行动，则执行行动
        if not should_act:
            return "计划完成 - 无需行动"
        
        return await self.act()
    
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """
        处理特殊工具执行和状态变更
        
        Args:
            name: 正在执行的工具名称
            result: 工具执行结果
            **kwargs: 额外参数
        """
        await super()._handle_special_tool(name=name, result=result, **kwargs)
        
        # 检查所有计划步骤是否已完成
        if self.plan_created and name == "planning":
            try:
                args = kwargs.get("args", {})
                if args.get("command") == "get" and self.current_plan_id:
                    # 检查所有步骤是否标记为已完成
                    if isinstance(result, dict) and "step_statuses" in result:
                        plan_complete = all(
                            status == "completed" 
                            for status in result["step_statuses"]
                        )
                        
                        if plan_complete and result["step_statuses"]:
                            logger.info(f"🎉 计划 '{self.current_plan_id}' 已完成！")
                            self.state = AgentState.FINISHED
            except Exception as e:
                logger.error(f"检查计划完成状态时出错: {e}")
    
    async def cleanup(self):
        """
        清理代理使用的资源
        """
        logger.info(f"🧹 正在清理代理 '{self.name}' 的资源...")
        
        # 清理所有子代理
        #for agent_name, agent in self.sub_agents.items():
        #    try:
        #        if hasattr(agent, 'cleanup') and asyncio.iscoroutinefunction(agent.cleanup):
        #            logger.debug(f"🧼 清理子代理: {agent_name}")
        #            await agent.cleanup()
        #    except Exception as e:
        #        logger.error(f"🚨 清理子代理 '{agent_name}' 时出错: {e}", exc_info=True)
        
        # 清理工具资源
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 清理工具: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 清理工具 '{tool_name}' 时出错: {e}", exc_info=True
                    )
        
        logger.info(f"✨ 代理 '{self.name}' 的清理完成。")
    
    async def run(self, request: Optional[str] = None) -> str:
        """
        运行代理，完成后进行清理
        """
        try:
            return await super().run(request)
        finally:
            await self.cleanup()