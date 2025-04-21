

from nkkagent.tools.graphaction_tool import ActionNode

GenerateQuestions = ActionNode(
    key="Questions",
    instruction="Task: Refer to the context to further inquire about the details that interest you, within a word limit"
    " of 150 words. Please provide the specific details you would like to inquire about here",
    example=["1. What ...", "2. How ...", "3. ..."],
)
