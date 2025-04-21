import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from nkkagent.default_config import Config

class AttrDict(BaseModel):
    """A dict-like object that allows access to keys as attributes, compatible with Pydantic."""

    model_config = ConfigDict(extra="allow")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        return self.__dict__.get(key, None)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def set(self, key, val: Any):
        self.__dict__[key] = val

    def get(self, key, default: Any = None):
        return self.__dict__.get(key, default)

    def remove(self, key):
        if key in self.__dict__:
            self.__delattr__(key)

class Context(AttrDict):
    """The context of the agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kwargs: AttrDict = AttrDict()
    config: Config = Config.default()

    
    #repo: Optional[ProjectRepo] = None
    #git_repo: Optional[GitRepository] = None
    src_workspace: Optional[Path] = None
    
    def llm_with_cost_manager_from_llm_config(self, llm_config):
        """Create an LLM instance from LLMConfig"""
        from nkkagent.llm.llm import LLM
        return LLM(llm_config=llm_config)
    #cost_manager: CostManager = CostManager()

    #_llm: Optional[BaseLLM] = None

    def new_environ(self):
        """Return a new os.environ object"""
        env = os.environ.copy()
        # i = self.options
        # env.update({k: v for k, v in i.items() if isinstance(v, str)})
        return env