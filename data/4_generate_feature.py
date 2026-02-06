import json
from pathlib import Path
from tqdm import tqdm
from graph_dataset import GraphDataset  

# METADATA_PATH = "/home/kaiyu/CodeCircuit/data/graph/graph_metadata.jsonl"
# OUTPUT_PATH = "gemma_mbpp_features.jsonl"

# METADATA_PATH = "/home/kaiyu/CodeCircuit/data/cpp_graph/graph_metadata.jsonl"
# OUTPUT_PATH = "gemma_mbpp_cpp_features.jsonl"

METADATA_PATH = "/home/kaiyu/CodeCircuit/data/java_graph/graph_metadata.jsonl"
OUTPUT_PATH = "gemma_mbpp_java_features.jsonl"


def load_metadata(jsonl_path: str):
    graph_entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                graph_entries.append(entry)
    return graph_entries


def append_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_and_save_features():
    print("Loading metadata...")
    metadata = load_metadata(METADATA_PATH)

    print(f"Loaded {len(metadata)} graph entries.")

    dataset = GraphDataset(
        graph_paths=metadata,
        feature_type='advanced_graph_features',
        use_cpu=False
    )

    print("Start extracting features...")

    for idx in tqdm(range(len(dataset))):
        if idx ==0 or idx ==1:
            continue
        sample = dataset[idx]

        if not sample["success"]:
            print(f" Failed to load graph for expr_id={sample['expr_id']} step={sample['step_number']}")
            continue

        feature = sample.get("features")
        if feature is not None:
            feature = feature.tolist()
        else:
            feature = []

        output_record = {
            "expr_id": sample["expr_id"],
            "step_number": sample["step_number"],
            "before_after": sample["before_after"],
            "graph_path": metadata[idx]["graph_path"],
            "step_labels": sample["step_labels"],
            "original_expression": sample["original_expression"],
            "feature": feature
        }

        append_jsonl(OUTPUT_PATH, output_record)

    print(" Feature extraction complete.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_and_save_features()
