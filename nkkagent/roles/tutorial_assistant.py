

from datetime import datetime
from typing import Dict

from nkkagent.actions.write_tutorial import WriteContent, WriteDirectory
from nkkagent.const import TUTORIAL_PATH
from nkkagent.logs import logger
from nkkagent.roles.role import Role, RoleReactMode
from nkkagent.schema import Message
from nkkagent.utils.file import File


class TutorialAssistant(Role):
    """Tutorial assistant, input one sentence to generate a tutorial document in markup format.

    Args:
        name: The name of the role.
        profile: The role profile description.
        goal: The goal of the role.
        constraints: Constraints or requirements for the role.
        language: The language in which the tutorial documents will be generated.
    """

    name: str = "Stitch"
    profile: str = "Tutorial Assistant"
    goal: str = "Generate tutorial documents"
    constraints: str = "Strictly follow Markdown's syntax, with neat and standardized layout"
    language: str = "Chinese"

    topic: str = ""
    main_title: str = ""
    total_content: str = ""
