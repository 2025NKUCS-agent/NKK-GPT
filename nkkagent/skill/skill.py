from nkkagent.actions import Action

class Skill(Action):
    """Skill"""

    def __init__(self, name: str, description: str, action: Action):
        self.name = name
        self.description = description
        self.action = action    
