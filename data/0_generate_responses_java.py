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
OUTPUT_FILE = "gemma_mbpp_java_responses.jsonl"

SYSTEM_PROMPT_JAVA = (
    "You are a professional Java developer. "
    "You must output ONLY valid Java code. "
    "Do NOT output Python, comments, explanations, markdown, or any text. "
    "Output only Java code that can compile."
)

def build_prompt_java(text):
    return SYSTEM_PROMPT_JAVA + "\n\n" + text

def clean_output(text: str):
    text = text.strip()
    if text.startswith("```java"):
        text = text[len("```java"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
    lines = text.splitlines()
    lines = [l for l in lines if not l.strip().startswith("//")]
    lines = [l for l in lines if l.strip() != ""]
    return "\n".join(lines).strip()

def generate_java(text, max_new_tokens=256):
    prompt = build_prompt_java(text)
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


# 实时写入文件
with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for i, line in enumerate(f_in):
        item = json.loads(line)
        text = item["text"]
        text = text.replace("python", "Java").replace("Python", "Java")

        java_code = generate_java(text)

        result = {
            "task_id": item.get("task_id", i),
            "text": text,
            "generated_code": java_code,
        }

        # 写入文件并立即 flush
        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
        f_out.flush()  # 确保立刻写入磁盘
        print(f"Generated example {i} -> written to file")
