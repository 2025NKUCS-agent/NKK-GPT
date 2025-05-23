

from nkkagent.actions import UserRequirement, WritePRD
from nkkagent.actions.prepare_documents import PrepareDocuments
from nkkagent.roles.role import Role, RoleReactMode
from nkkagent.utils.common import any_to_name


class ProductManager(Role):
    """
    Represents a Product Manager role responsible for product development and management.

    Attributes:
        name (str): Name of the product manager.
        profile (str): Role profile, default is 'Product Manager'.
        goal (str): Goal of the product manager.
        constraints (str): Constraints or limitations for the product manager.
    """

    name: str = "Alice"
    profile: str = "Product Manager"
    goal: str = "efficiently create a successful product that meets market demands and user expectations"
    constraints: str = "utilize the same language as the user requirements for seamless communication"
    todo_action: str = ""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.set_actions([PrepareDocuments, WritePRD])
        self._watch([UserRequirement, PrepareDocuments])
        self.rc.react_mode = RoleReactMode.BY_ORDER
        self.todo_action = any_to_name(WritePRD)

    async def _observe(self, ignore_memory=False) -> int:
        return await super()._observe(ignore_memory=True)
