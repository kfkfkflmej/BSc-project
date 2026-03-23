import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

name = "pop_qa_subset"


paraphrase_prompt = """
    Paraphrase the following question, without changing its meaning.
    Make sure you only output a single question only.
    Question: {q}
    Paraphrased Question:
"""

clarify_prompt = """
    Clarify the following question by rewriting it in a clearer, more complete form.
    If the question is ambiguous, add missing details to make it understandable.
    Make sure you only output a single question only.    
    Original Question: {q}
    Clarified Question:
"""

sampling_prompt = """
    Please answer the following question. 
    Question : {q} 
    At the end of your solution, indicate your final answer inside a boxed environment, like: $\\boxed{{answer}}$.
"""

check_prompt = """
    Following is your previous response to the question:
    Question : {q} 

    Your previous response: {a}
    Check your previous response carefully and respond the question again.
    At the end of your solution, indicate your final answer inside a boxed environment, like: $\\boxed{{answer}}$.
"""

iterative_prompt = """
    Consider the following question:
    Question: {q}

    Below are other proposed solutions:
    {aseq}
    
    Now, provide your own solution to the question.  
    At the end of your solution, indicate your final answer inside a boxed environment, like: $\\boxed{{answer}}$.
"""

def extract_answer(text):
    pattern = r"boxed\{\s*([A-Ea-e])\s*\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).upper()
    else:
        # Fallback: check if single capital letters A–E appear in the text
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in text:
                return letter
    return ""

def eval_answer(model_output, ground_truth_text, question):
    ground_truth = extract_ground_truth(ground_truth_text)
    model_pred = extract_answer(model_output)
    logging.info(model_output)
    logging.info(model_pred)
    logging.info(ground_truth)
    result = True if model_pred == ground_truth else False
    logging.info(f"eval answer : {result}")
    return result

def extract_ground_truth(text):
    return text