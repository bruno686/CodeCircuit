import json
import re
from openai import OpenAI
from tqdm import tqdm
import time

client = OpenAI(
    base_url="http://35.220.164.252:3888/v1/",
    api_key="sk-Oq94SGsHiOdAe3yMSAKB40UC7fZ6VHfMKuKzV73tHsL7Ukmr"
)

input_file = "gemma_mbpp_responses.jsonl"
output_file = "gemma_mbpp_correctness.jsonl"

def load_responses(path):
    r = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r.append(json.loads(line))
    return r

def build_prompt(code, task, n_lines):
    return (
        "You are a Python code correctness checker. "
        "Review the provided Python code line by line. For each line, output a score of 1 or 0 according to the following rules:\n\n"
        "1. If the line has a syntax error, output 0.\n"
        "2. If the line is syntactically correct but violates the coding requirements described in the task, output 0 for that line and all subsequent lines. Once a line violates the requirements, all following lines must also be 0.\n"
        "3. If the line is syntactically correct and meets the coding requirements so far, output 1.\n\n"
        f"The code has exactly {n_lines} lines. Output only a Python list of exactly {n_lines} integers, e.g., [1, 1, 0, 1]. Do not include any explanation or extra text.\n\n"
        "4. Focus solely on whether the substantive code contains issues, ignoring parameter variable names, function names, and so on."
        f"Task Description:\n---\n{task}\n---\n\n"
        f"Code to Check:\n---\n{code}\n---"
    )

def parse_output(text, n_lines):
    m = re.search(r'\[\s*(\d\s*(?:,\s*\d\s*)*)\s*\]', text)
    lst = eval(m.group(0))
    assert len(lst) == n_lines
    return lst

def process(responses):
    with open(output_file, "w", encoding="utf-8") as w:
        for item in tqdm(responses):
            code = item["generated_code"]
            n_lines = code.count("\n") + 1
            prompt = build_prompt(code, item["text"], n_lines)

            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            raw_output = r.choices[0].message.content
            scores = parse_output(raw_output, n_lines)

            item["line_correctness_scores"] = scores

            w.write(json.dumps(item, ensure_ascii=False) + "\n")
            w.flush()

process(load_responses(input_file))
