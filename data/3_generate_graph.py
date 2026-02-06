from pathlib import Path
import torch
import json
from circuit_tracer import ReplacementModel, attribute
from tqdm import tqdm

# --- Configuration and Model Loading ---
MODEL_NAME = 'google/gemma-2-2b-it'
TRANSCODER_NAME = "gemma"
model = ReplacementModel.from_pretrained(MODEL_NAME, TRANSCODER_NAME, dtype=torch.bfloat16).to('cuda')

# --- Graph Generation Parameters ---
MAX_N_LOGITS = 10
DESIRED_LOGIT_PROB = 0.95
MAX_FEATURE_NODES = 8192
BATCH_SIZE = 64
OFFLOAD = None
VERBOSE = False

# --- Input/Output Paths ---
# DATA_FILE_PATH = 'gemma_mbpp_incremental.jsonl'
# OUTPUT_DIR = Path('graph')

# DATA_FILE_PATH = 'gemma_mbpp_cpp_incremental.jsonl'
# OUTPUT_DIR = Path('cpp_graph')

DATA_FILE_PATH = 'gemma_mbpp_java_incremental.jsonl'
OUTPUT_DIR = Path('java_graph')

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Graph files will be saved in: {OUTPUT_DIR.resolve()}")

# Output metadata file (jsonl append mode)
METADATA_FILE_PATH = OUTPUT_DIR / 'graph_metadata.jsonl'


def load_data(file_path: str) -> list:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def append_metadata(metadata: dict):
    """Append one metadata entry to the jsonl file."""
    with open(METADATA_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata) + "\n")


def generate_and_save_graphs(data_list: list, output_dir: Path, model: ReplacementModel):
    task_step_counters = {}

    for i, item in enumerate(tqdm(data_list, desc="Generating Graphs")):
        prompt = item['text']
        task_id = item['task_id']
        current_step = task_step_counters.get(task_id, 0)

        graph_name = f'graph_{i}_{task_id}_{current_step}.pt'
        graph_path = output_dir / graph_name

        # --- Skip if graph exists ---
        if graph_path.exists():
            print(f"\nSkipping generation for existing file: {graph_name}")

            metadata = {
                'expr_id': task_id,
                'step_number': current_step,
                'before_after': 'after',
                'graph_path': str(graph_path.resolve()),
                'step_labels': item['label'],
                'original_expression': item['text'],
            }
            append_metadata(metadata)

            task_step_counters[task_id] = current_step + 1
            continue

        # --- Generation ---
        if len(prompt) > 550:
            continue

        graph = attribute(
            prompt=prompt,
            model=model,
            max_n_logits=MAX_N_LOGITS,
            desired_logit_prob=DESIRED_LOGIT_PROB,
            batch_size=BATCH_SIZE,
            max_feature_nodes=MAX_FEATURE_NODES,
            offload=OFFLOAD,
            verbose=VERBOSE
        )

        graph.to_pt(graph_path)

        metadata = {
            'expr_id': task_id,
            'step_number': current_step,
            'before_after': 'after',
            'graph_path': str(graph_path.resolve()),
            'step_labels': item['label'],
            'original_expression': item['text'],
        }

        append_metadata(metadata)
        task_step_counters[task_id] = current_step + 1


if __name__ == "__main__":
    # Clear metadata file if already exists
    open(METADATA_FILE_PATH, 'w').close()

    data_list = load_data(DATA_FILE_PATH)
    generate_and_save_graphs(data_list, OUTPUT_DIR, model)

    print("\n--- Process Complete ---")
    print(f"Metadata appended to: {METADATA_FILE_PATH.resolve()}")
