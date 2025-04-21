

from typing import Optional

from pydantic import Field, model_validator

from nkkagent.actions import SearchAndSummarize
from nkkagent.actions.action_node import ActionNode
from nkkagent.actions.action_output import ActionOutput
from nkkagent.logs import logger
from nkkagent.roles import Role
from nkkagent.schema import Message
from nkkagent.tools.search_engine import SearchEngine


class Searcher(Role):
    """
    Represents a Searcher role responsible for providing search services to users.

    Attributes:
        name (str): Name of the searcher.
        profile (str): Role profile.
        goal (str): Goal of the searcher.
        constraints (str): Constraints or limitations for the searcher.
        search_engine (SearchEngine): The search engine to use.
    """

    name: str = Field(default="Alice")
    profile: str = Field(default="Smart Assistant")
    goal: str = "Provide search services for users"
    constraints: str = "Answer is rich and complete"
    search_engine: Optional[SearchEngine] = None
