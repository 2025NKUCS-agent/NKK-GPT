# import asyncio  # 未使用
# import json  # 未使用
import os.path
# import uuid  # 未使用
# from abc import ABC  # 未使用
# from asyncio import Queue, QueueEmpty, wait_for  # 未使用
# from json import JSONDecodeError  # 未使用
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, Union
# import serialization  # 未使用
from nkkagent.actions.action_output import ActionOutput
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
class Document(MixinModel):
    """
    Represents a document.
    """

    root_path: str = ""
    filename: str = ""
    content: str = ""

    def get_meta(self) -> "Document":
        """Get metadata of the document.

        :return: A new Document instance with the same root path and filename.
        """

        return Document(root_path=self.root_path, filename=self.filename)
    # 生成一个仅包含路径和文件名（不包含内容）的新的Document实例，用于轻量级元数据传输
    
    
    @property
    def root_relative_path(self):
        """Get relative path from root of git repository.

        :return: relative path from root of git repository.
        """
        return os.path.join(self.root_path, self.filename)
    # 生成文档在Git仓库中的相对路径
    

    def __str__(self):
        return self.content

    def __repr__(self):
        return self.content
    # 直接返回文档内容

class Documents(MixinModel):
    """A class representing a collection of documents.

    Attributes:
        docs (Dict[str, Document]): A dictionary mapping document names to Document instances.
    """

    docs: Dict[str, Document] = Field(default_factory=dict)

    @classmethod
    def from_iterable(cls, documents: Iterable[Document]) -> "Documents":
        """Create a Documents instance from a list of Document instances.

        :param documents: A list of Document instances.
        :return: A Documents instance.
        """

        docs = {doc.filename: doc for doc in documents}
        return Documents(docs=docs)
    # 将文档列表转换为字典结构，按filename归类，便于通过文件名快速检索文档


    def to_action_output(self) -> ActionOutput:
        """Convert to action output string.

        :return: A string representing action output.
        """

        return ActionOutput(content=self.model_dump_json(), instruct_content=self)
    #将文档集合装换为标准化的任务输出格式，用于集成到任务执行流程
