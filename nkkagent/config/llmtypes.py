"""This file has types/dataclass definitions that are used in the SWE agent
for exchanging data between different modules/functions/classes.
They oftentimes cannot be defined in the same file where they are used
because of circular dependencies.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal, Dict, Optional, List
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict
import yaml

class StepOutput(BaseModel):
    """Output of a single step of the agent"""
    role: Literal["default","user", "assistant", "system","manager","code_editor","code_interpreter"]
    thought: str = ""
    action: str = ""
    output: str = ""
    observation: str = ""
    execution_time: float = 0.0
    done: bool = False
    exit_status: int | str | None = None
    submission: str | None = None
    state: dict[str, str] = {}
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_ids: list[str] | None = None
    """State of the environment at the end of the step"""
    extra_info: dict[str, Any] = {}

    def to_template_format_dict(self) -> dict[str, str | int | float | bool | None]:
        """Used for formatting (error) prompt templates"""
        out = {}
        for k, v in self.model_dump().items():
            if k in ("tool_calls", "tool_call_ids", "state"):
                continue
            out[k] = v
        out |= self.state
        return out

class _HistoryItem(TypedDict):
    role:  Literal["default","user", "assistant", "system","manager","code_editor","code_interpreter"]
    content: str | list[dict[str, Any]]
    message_type: Literal["thought", "action", "observation"]

class HistoryItem(TypedDict):
    agent: str
    is_demo: bool
    thought: str
    action: str | None
    tool_calls: list[dict[str, str]] | None
    tool_call_ids: list[str] | None
    tags: list[str]
    cache_control: dict[str, Any] | None
    """HistoryProcessors can add these tags to enable special processing"""

History = list[HistoryItem]

# todo: Make this actually have the dataclasses instead of dict versions
class AgentInfo(TypedDict, total=False):
    # same as `APIStats` from models.py
    model_stats: dict[str, float]
    exit_status: str | None
    submission: str | None
    # same as `ReviewerResult`
    review: dict[str, Any]
    edited_files: str
    edit_files_list: list[str]
    edit_files_lines: list[str]
    edited_files_percentage: float

class AgentRunResult(BaseModel):
    info: AgentInfo



class YamlModel(BaseModel):
    """Base class for yaml model"""

    extra_fields: Optional[Dict[str, str]] = None

    @classmethod
    def read_yaml(cls, file_path: Path, encoding: str = "utf-8") -> Dict:
        """Read yaml file and return a dict"""
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding=encoding) as file:
            return yaml.safe_load(file)

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "YamlModel":
        """Read yaml file and return a YamlModel instance"""
        return cls(**cls.read_yaml(file_path))

    def to_yaml_file(self, file_path: Path, encoding: str = "utf-8") -> None:
        """Dump YamlModel instance to yaml file"""
        with open(file_path, "w", encoding=encoding) as file:
            yaml.dump(self.model_dump(), file)


class YamlModelWithoutDefault(YamlModel):
    """YamlModel without default values"""

    @model_validator(mode="before")
    @classmethod
    def check_not_default_config(cls, values):
        """Check if there is any default config in config2.yaml"""
        if any(["YOUR" in v for v in values]):
            raise ValueError("Please set your config in config2.yaml")
        return values

class Trajectory(BaseModel):
    """轨迹记录，用于记录agent的执行过程"""
    steps: List[StepOutput] = Field(default_factory=list)
    info: AgentInfo = Field(default_factory=dict)
