
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

from nkkagent.actions.skill_action import ArgumentsParingAction, SkillAction
from nkkagent.actions.talk_action import TalkAction
from nkkagent.learn.skill_loader import SkillsDeclaration
from nkkagent.logs import logger
from nkkagent.memory.brain_memory import BrainMemory
from nkkagent.roles import Role
from nkkagent.schema import Message


class MessageType(Enum):
    Talk = "TALK"
    Skill = "SKILL"


class Assistant(Role):
    """Assistant for solving common issues."""

    name: str = "Lily"
    profile: str = "An assistant"
    goal: str = "Help to solve problem"
    constraints: str = "Talk in {language}"
    desc: str = ""
    memory: BrainMemory = Field(default_factory=BrainMemory)
    skills: Optional[SkillsDeclaration] = None
