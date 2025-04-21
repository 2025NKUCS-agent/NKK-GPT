
from nkkagent.actions import Action
from nkkagent.const import PROMPT_PATH
from nkkagent.logs import logger
from nkkagent.skill import Skill


class SkillManager:
    """Used to manage all skills"""

    def __init__(self):
        self._skills: dict[str:Skill] = {}

    def add_skill(self, skill: Skill):
        """
        Add a skill, add the skill to the skill pool and searchable storage
        :param skill: Skill
        :return:
        """
        self._skills[skill.name] = skill
        self._store.add(skill.desc, {"name": skill.name, "desc": skill.desc}, skill.name)

    def del_skill(self, skill_name: str):
        """
        Delete a skill, remove the skill from the skill pool and searchable storage
        :param skill_name: Skill name
        :return:
        """
        self._skills.pop(skill_name)
        self._store.delete(skill_name)

    def get_skill(self, skill_name: str) -> Skill:
        """
        Obtain a specific skill by skill name
        :param skill_name: Skill name
        :return: Skill
        """
        return self._skills.get(skill_name)

    def retrieve_skill(self, desc: str, n_results: int = 2) -> list[Skill]:
        """
        Obtain skills through the search engine
        :param desc: Skill description
        :return: Multiple skills
        """
        return self._store.search(desc, n_results=n_results)["ids"][0]

    def retrieve_skill_scored(self, desc: str, n_results: int = 2) -> dict:
        """
        Obtain skills through the search engine
        :param desc: Skill description
        :return: Dictionary consisting of skills and scores
        """
        return self._store.search(desc, n_results=n_results)

    def generate_skill_desc(self, skill: Skill) -> str:
        """
        Generate descriptive text for each skill
        :param skill:
        :return:
        """
        path = PROMPT_PATH / "generate_skill.md"
        text = path.read_text()
        logger.info(text)


if __name__ == "__main__":
    manager = SkillManager()
    manager.generate_skill_desc(Action())
