import json

input_file = "gemma_mbpp_correctness.jsonl"
output_file = "gemma_mbpp_incremental.jsonl"

def load_responses(path):
    responses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            responses.append(json.loads(line))
    return responses

def generate_incremental_records(responses):
    records = []
    for item in responses:
        task_id = item.get("task_id")
        task_text = item.get("text", "")
        code = item.get("generated_code", "")
        scores = item.get("line_correctness_scores", [])
        lines = code.split("\n")
        for i in range(1, len(lines)+1):
            # label = correctness of the last line in the current cumulative snippet
            label = scores[i-1] if i <= len(scores) else 1
            new_record = {
                "task_id": task_id,
                "text": task_text + " " + "\n".join(lines[:i]),
                "generated_code": "\n".join(lines[:i]),
                "label": label
            }
            records.append(new_record)
    return records

def save_records(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    responses = load_responses(input_file)
    incremental_records = generate_incremental_records(responses)
    save_records(incremental_records, output_file)
    print(f"Generated {len(incremental_records)} incremental records and saved to {output_file}")


