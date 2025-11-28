from pathlib import Path
import torch

from IPython.display import display, IFrame
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
from circuit_tracer.frontend.local_server import serve


model_name = 'google/gemma-2-2b'
transcoder_name = "gemma"
model = ReplacementModel.from_pretrained(model_name, transcoder_name, dtype=torch.bfloat16)

prompt = "The capital of state containing Dallas is"  # What you want to get the graph for
max_n_logits = 10   # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
max_feature_nodes = 8192  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
batch_size=256  # Batch size when attributing
offload=None # Offload various parts of the model during attribution to save memory. Can be 'disk', 'cpu', or None (keep on GPU)
verbose = True  # Whether to display a tqdm progress bar and timing report

graph = attribute(
    prompt=prompt,
    model=model,
    max_n_logits=max_n_logits,
    desired_logit_prob=desired_logit_prob,
    batch_size=batch_size,
    max_feature_nodes=max_feature_nodes,
    offload=offload,
    verbose=verbose
)

graph_dir = 'graphs'
graph_name = 'example_graph.pt'
graph_dir = Path(graph_dir)
graph_dir.mkdir(exist_ok=True)
graph_path = graph_dir / graph_name

graph.to_pt(graph_path)

print(graph)

# slug = "dallas-austin"  # this is the name that you assign to the graph
# graph_file_dir = './graph_files'  # where to write the graph files. no need to make this one; create_graph_files does that for you
# node_threshold=0.8  # keep only the minimum # of nodes whose cumulative influence is >= 0.8
# edge_threshold=0.98  # keep only the minimum # of edges whose cumulative influence is >= 0.98

# create_graph_files(
#     graph_or_path=graph_path,  # the graph to create files for
#     slug=slug,
#     output_path=graph_file_dir,
#     node_threshold=node_threshold,
#     edge_threshold=edge_threshold
# )

# port = 8046
# server = serve(data_dir='./graph_files/', port=port)
# print(f"Use the IFrame below, or open your graph here: f'http://localhost:{port}/index.html'")
# display(IFrame(src=f'http://localhost:{port}/index.html', width='100%', height='800px'))