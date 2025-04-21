
from nkkagent.tools.graphaction_tool import ActionNode

QUESTIONS = ActionNode(
    key="Questions",
    expected_type=list[str],
    instruction="""Role: You are an interviewer of our company who is well-knonwn in frontend or backend develop;
Requirement: Provide a list of questions for the interviewer to ask the interviewee, by reading the resume of the interviewee in the context.
Attention: Provide as markdown block as the format above, at least 10 questions.""",
    example=["1. What ...", "2. How ..."],
)

PrepareInterview = ActionNode(
    key="PrepareInterview",
    expected_type=list[str],
    instruction="""Role: You are an interviewer of our company who is well-knonwn in frontend or backend develop;
Requirement: Provide a list of questions for the interviewer to ask the interviewee, by reading the resume of the interviewee in the context.
Attention: Provide as markdown block as the format above, at least 10 questions.""",
    example=["1. What ...", "2. How ..."],
)

