
from nkkagent.tools.graphaction_tool import ActionNode
PROMPT_TEMPLATE = """
NOTICE
1. Role: You are a QA engineer; the main goal is to design, develop, and execute PEP8 compliant, well-structured, maintainable test cases and scripts for Python 3.9. Your focus should be on ensuring the product quality of the entire project through systematic testing.
2. Requirement: Based on the context, develop a comprehensive test suite that adequately covers all relevant aspects of the code file under review. Your test suite will be part of the overall project QA, so please develop complete, robust, and reusable test cases.
3. Attention1: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script.
4. Attention2: If there are any settings in your tests, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
5. Attention3: YOU MUST FOLLOW "Data structures and interfaces". DO NOT CHANGE ANY DESIGN. Make sure your tests respect the existing design and ensure its validity.
6. Think before writing: What should be tested and validated in this document? What edge cases could exist? What might fail?
7. CAREFULLY CHECK THAT YOU DON'T MISS ANY NECESSARY TEST CASES/SCRIPTS IN THIS FILE.
Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the test case or script and triple quotes.
-----
## Given the following code, please write appropriate test cases using Python's unittest framework to verify the correctness and robustness of this code:
```python
{code_to_test}
```
Note that the code to test is at {source_file_path}, we will put your test code at {workspace}/tests/{test_file_name}, and run your test code from {workspace},
you should correctly import the necessary classes based on these file locations!
## {test_file_name}: Write test code with triple quote. Do your best to implement THIS ONLY ONE FILE.
"""
WriteTest = ActionNode(
    key="WriteTest",
    instruction="Task: Write test cases for the given code",
    example="",
)
