# import asyncio  # 未使用
# import json  # 未使用
# import os.path  # 未使用
# import uuid  # 未使用
# from abc import ABC  # 未使用
# from asyncio import Queue, QueueEmpty, wait_for  # 未使用
# from json import JSONDecodeError  # 未使用
# from pathlib import Path  # 未使用
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union
import sys
import os
# 获取项目根目录的绝对路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 将项目根目录添加到模块搜索路径
sys.path.append(root_dir)
from nkkagent.repo_parser import DotClassInfo
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

from nkkagent.logs import logger
from nkkagent.config.exceptions import handle_exception
from nkkagent.schema.mixin import MixinModel

# mermaid class view
class UMLClassMeta(MixinModel):
    name: str = ""
    visibility: str = ""

    @staticmethod
    def name_to_visibility(name: str) -> str:
        if name == "__init__":
            return "+"
        if name.startswith("__"):
            return "-"
        elif name.startswith("_"):
            return "#"
        return "+"


class UMLClassAttribute(UMLClassMeta):
    value_type: str = ""
    default_value: str = ""

    def get_mermaid(self, align=1) -> str:
        content = "".join(["\t" for i in range(align)]) + self.visibility
        if self.value_type:
            content += self.value_type.replace(" ", "") + " "
        name = self.name.split(":", 1)[1] if ":" in self.name else self.name
        content += name
        if self.default_value:
            content += "="
            if self.value_type not in ["str", "string", "String"]:
                content += self.default_value
            else:
                content += '"' + self.default_value.replace('"', "") + '"'
        # if self.abstraction:
        #     content += "*"
        # if self.static:
        #     content += "$"
        return content


class UMLClassMethod(UMLClassMeta):
    args: List[UMLClassAttribute] = Field(default_factory=list)
    return_type: str = ""

    def get_mermaid(self, align=1) -> str:
        content = "".join(["\t" for i in range(align)]) + self.visibility
        name = self.name.split(":", 1)[1] if ":" in self.name else self.name
        content += name + "(" + ",".join([v.get_mermaid(align=0) for v in self.args]) + ")"
        if self.return_type:
            content += " " + self.return_type.replace(" ", "")
        # if self.abstraction:
        #     content += "*"
        # if self.static:
        #     content += "$"
        return content


class UMLClassView(UMLClassMeta):
    attributes: List[UMLClassAttribute] = Field(default_factory=list)
    methods: List[UMLClassMethod] = Field(default_factory=list)

    def get_mermaid(self, align=1) -> str:
        content = "".join(["\t" for i in range(align)]) + "class " + self.name + "{\n"
        for v in self.attributes:
            content += v.get_mermaid(align=align + 1) + "\n"
        for v in self.methods:
            content += v.get_mermaid(align=align + 1) + "\n"
        content += "".join(["\t" for i in range(align)]) + "}\n"
        return content

    @classmethod
    def load_dot_class_info(cls, dot_class_info: DotClassInfo) -> "UMLClassView":
        visibility = UMLClassView.name_to_visibility(dot_class_info.name)
        class_view = cls(name=dot_class_info.name, visibility=visibility)
        for i in dot_class_info.attributes.values():
            visibility = UMLClassAttribute.name_to_visibility(i.name)
            attr = UMLClassAttribute(name=i.name, visibility=visibility, value_type=i.type_, default_value=i.default_)
            class_view.attributes.append(attr)
        for i in dot_class_info.methods.values():
            visibility = UMLClassMethod.name_to_visibility(i.name)
            method = UMLClassMethod(name=i.name, visibility=visibility, return_type=i.return_args.type_)
            for j in i.args:
                arg = UMLClassAttribute(name=j.name, value_type=j.type_, default_value=j.default_)
                method.args.append(arg)
            method.return_type = i.return_args.type_
            class_view.methods.append(method)
        return class_view
