

import re

from nkkagent.actions import UserRequirement
from nkkagent.actions.write_teaching_plan import TeachingPlanBlock, WriteTeachingPlanPart
from nkkagent.logs import logger
from nkkagent.roles import Role
from nkkagent.schema import Message
from nkkagent.utils.common import any_to_str, awrite


class Teacher(Role):
    """Support configurable teacher roles,
    with native and teaching languages being replaceable through configurations."""

    name: str = "Lily"
    profile: str = "{teaching_language} Teacher"
    goal: str = "writing a {language} teaching plan part by part"
    constraints: str = "writing in {language}"
    desc: str = ""
