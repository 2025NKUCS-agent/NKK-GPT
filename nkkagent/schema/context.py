from __future__ import annotations

# import asyncio  # 未使用
import json
# import os.path  # 未使用
# import uuid  # 未使用
from abc import ABC
# from asyncio import Queue, QueueEmpty, wait_for  # 未使用
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union
# import serialization  # 未使用
from nkkagent.config.exceptions import handle_exception
from nkkagent.schema.document import Document
from nkkagent.schema.mixin import MixinModel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from nkkagent.const import (
    MESSAGE_ROUTE_CAUSE_BY,
    MESSAGE_ROUTE_FROM,
    MESSAGE_ROUTE_TO,
    MESSAGE_ROUTE_TO_ALL,
    PRDS_FILE_REPO,
    SYSTEM_DESIGN_FILE_REPO,
    TASK_FILE_REPO,
)
# 定义泛型类型变量
T = TypeVar("T", bound="BaseModel")  #可以操作任意类型的变量
'''
class BaseContext(BaseModel, ABC):
    @classmethod
    @handle_exception
    def loads(cls: Type[T], val: str) -> Optional[T]:
        i = json.loads(val)
        return cls(**i)
'''
class BaseContext(MixinModel):
    pass

class CodingContext(BaseContext):
    filename: str
    design_doc: Optional[Document] = None
    task_doc: Optional[Document] = None
    code_doc: Optional[Document] = None
    code_plan_and_change_doc: Optional[Document] = None


class TestingContext(BaseContext):
    filename: str
    code_doc: Document
    test_doc: Optional[Document] = None


class RunCodeContext(BaseContext):
    mode: str = "script"
    code: Optional[str] = None
    code_filename: str = ""
    test_code: Optional[str] = None
    test_filename: str = ""
    command: List[str] = Field(default_factory=list)
    working_directory: str = ""
    additional_python_paths: List[str] = Field(default_factory=list)
    output_filename: Optional[str] = None
    output: Optional[str] = None


class RunCodeResult(BaseContext):
    summary: str
    stdout: str
    stderr: str


class CodeSummarizeContext(BaseModel):
    design_filename: str = ""
    task_filename: str = ""
    codes_filenames: List[str] = Field(default_factory=list)
    reason: str = ""

    @staticmethod
    def loads(filenames: List) -> CodeSummarizeContext:
        ctx = CodeSummarizeContext()
        for filename in filenames:
            if Path(filename).is_relative_to(SYSTEM_DESIGN_FILE_REPO):
                ctx.design_filename = str(filename)
                continue
            if Path(filename).is_relative_to(TASK_FILE_REPO):
                ctx.task_filename = str(filename)
                continue
        return ctx

    def __hash__(self):
        return hash((self.design_filename, self.task_filename))


class BugFixContext(BaseContext):
    filename: str = ""


class CodePlanAndChangeContext(BaseModel):
    requirement: str = ""
    issue: str = ""
    prd_filename: str = ""
    design_filename: str = ""
    task_filename: str = ""

    @staticmethod
    def loads(filenames: List, **kwargs) -> CodePlanAndChangeContext:
        ctx = CodePlanAndChangeContext(requirement=kwargs.get("requirement", ""), issue=kwargs.get("issue", ""))
        for filename in filenames:
            filename = Path(filename)
            if filename.is_relative_to(PRDS_FILE_REPO):
                ctx.prd_filename = filename.name
                continue
            if filename.is_relative_to(SYSTEM_DESIGN_FILE_REPO):
                ctx.design_filename = filename.name
                continue
            if filename.is_relative_to(TASK_FILE_REPO):
                ctx.task_filename = filename.name
                continue
        return ctx
