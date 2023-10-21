GENERAL_PROMPT_TEMPLATE = """
You are a helpful assistant that helps evaluate the quality of two responses to a prompt.

Answer by awarding a score between 0 and 10 to each response, where 0 means the response is completely inappropriate and 10 means the response is very good.
A response that is acceptable should never be awarded less than 6 out of 10.

Answer base on the following criteria:
1. Is the response grammatically correct?
2. Is the response semantically correct?
3. Is the response coherent?
4. Is the response relevant to the prompt?

Output format (csv):
<score1 from 0 to 10>,<score2 from 0 to 10>

Which of the following responses is the most appropriate to the following instruction?

{prompt}

Response 1: 
{response1}


Response 2: 
{response2}

Output:
"""
