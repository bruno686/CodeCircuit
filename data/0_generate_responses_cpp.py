import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

INPUT_FILE = "gemma_mbpp_responses.jsonl"
OUTPUT_FILE = "gemma_mbpp_cpp_responses.jsonl"

SYSTEM_PROMPT_CPP = (
    "You are a professional C++ developer. "
    "You must output ONLY valid C++ code. "
    "Do NOT output Python, comments, explanations, markdown, or any text. "
    "Output only C++ code that can compile."
    "Forbidden: Markdown blocks (```), explanations, and comments."
)

def build_prompt_cpp(text):
    return SYSTEM_PROMPT_CPP + "\n\n" + text

def clean_output(text: str):
    text = text.strip()
    if text.startswith("```cpp"):
        text = text[len("```cpp"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
    lines = text.splitlines()
    lines = [l for l in lines if not l.strip().startswith("//")]
    lines = [l for l in lines if l.strip() != ""]
    return "\n".join(lines).strip()

def generate_cpp(text, max_new_tokens=256):
    prompt = build_prompt_cpp(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return clean_output(generated)

import re

def clean_output(text: str):
    text = re.sub(r'```[a-zA-Z+]*', '', text)

    lines = text.splitlines()
    cleaned_lines = []
    seen_includes = set() # 用于头文件去重

    for line in lines:
        stripped = line.strip()
        
        # 2. 过滤掉注释、空行以及残留的反引号
        if not stripped or stripped.startswith("//") or stripped.startswith("`"):
            continue
            
        # 3. 头文件去重逻辑
        if stripped.startswith("#include"):
            if stripped in seen_includes:
                continue
            seen_includes.add(stripped)
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()

# 实时写入文件
with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for i, line in enumerate(f_in):
        item = json.loads(line)
        text = item["text"]
        text = text.replace("python", "C++").replace("Python", "C++")

        java_code = generate_cpp(text)
        java_code = clean_output(java_code)
        result = {
            "task_id": item.get("task_id", i),
            "text": text,
            "generated_code": java_code,
        }

        # 写入文件并立即 flush
        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        f_out.flush()  # 确保立刻写入磁盘
        print(f"Generated example {i} -> written to file")