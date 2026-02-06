"""
Module for feature extraction

Load circuit graphs and step labels, prepare data for visualization.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Import circuit_tracer components for advanced feature extraction
from circuit_tracer.graph import Graph, prune_graph

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# # This must be done before any multiprocessing operations
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

class GraphDataset(Dataset):
    """
    Dataset class for loading circuit graphs in parallel.
    """

    def __init__(self, graph_paths: List[Dict], feature_type: str = 'adjacency_matrix', use_cpu: bool = True):
        """
        Initialize the dataset.

        Args:
            graph_paths: List of dictionaries containing graph metadata
            feature_type: Type of feature to extract
            use_cpu: Whether to load tensors to CPU (True) or GPU (False)
        """
        self.graph_paths = graph_paths
        self.feature_type = feature_type
        self.use_cpu = use_cpu

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        """
        Load a single graph and extract features.

        Args:
            idx: Index of the graph to load

        Returns:
            Dictionary containing features and metadata
        """
        item = self.graph_paths[idx]
        graph_path = item['graph_path']

        try:
            # Load graph on CPU or GPU based on flag
            if self.use_cpu:
                graph = torch.load(graph_path, weights_only=False, map_location='cpu')
            else:
                graph = torch.load(graph_path, weights_only=False)

            # Extract features
            features = self._extract_single_feature(graph, graph_path, self.feature_type)

            # Explicitly delete graph to free memory
            del graph

            if features is not None:
                return {
                    'expr_id': item['expr_id'],
                    'step_number': item['step_number'],
                    'before_after': item['before_after'],
                    'features': features,
                    'step_labels': item['step_labels'],
                    'original_expression': item['original_expression'],
                    'success': True
                }
            else:
                return {
                    'expr_id': item['expr_id'],
                    'step_number': item['step_number'],
                    'before_after': item['before_after'],
                    'success': False
                }

        except Exception as e:
            print(f"Error loading {graph_path}: {e}")
            return {
                'expr_id': item['expr_id'],
                'step_number': item['step_number'],
                'before_after': item['before_after'],
                'success': False
            }

    def _extract_single_feature(self, graph: dict, graph_path: str, feature_type: str) -> Optional[np.ndarray]:
        """
        Extract features from a single graph.

        Args:
            graph: Graph dictionary
            feature_type: Type of feature to extract

        Returns:
            Feature array or None if not found
        """

        if feature_type == 'adjacency_matrix':
            adj_matrix = graph.get('adjacency_matrix', torch.tensor([]))
            if adj_matrix.numel() > 0:
                # Extract to numpy and delete tensor immediately
                features = adj_matrix.flatten().cpu().numpy()
                del adj_matrix # Explicit cleanup
                return features

        elif feature_type == 'active_features':
            active_features = graph.get('active_features', torch.tensor([]))
            if active_features.numel() > 0:
                features = active_features.flatten().cpu().numpy()
                del active_features # Explicit cleanup
                return features

        elif feature_type == 'activation_values':
            activation_values = graph.get('activation_values', torch.tensor([]))
            if activation_values.numel() > 0:
                features = activation_values.flatten().cpu().numpy()
                del activation_values # Explicit cleanup
                return features

        elif feature_type == 'selected_features':
            selected_features = graph.get('selected_features', torch.tensor([]))
            if selected_features.numel() > 0:
                features = selected_features.flatten().cpu().numpy()
                del selected_features
                return features

        elif feature_type == 'advanced_graph_features':
            graph_obj = Graph.from_pt(graph_path)
            # Use the advanced feature extraction function
            return self._extract_advanced_features(graph_obj)

        return None

    def _extract_advanced_features(self, graph: 'Graph', node_threshold: float = 0.8) -> np.ndarray:
            """
            Extracts a flat, fixed-size feature vector from a circuit-tracer Graph object.

            Args:
                graph: Circuit-tracer Graph object
                node_threshold: Threshold for pruning

            Returns:
                Feature vector as numpy array
            """
            # Ensure graph tensors are on the CPU for processing
            graph.to("cpu")

            # Prune the graph to focus on influential components
            node_mask, _, cumulative_scores = prune_graph(graph, node_threshold=node_threshold)

            features = []
            n_layers = graph.cfg.n_layers
            n_pos = graph.n_pos

            # --- Level 1: High-Level Stats ---
            pruned_node_indices = node_mask.nonzero().squeeze().tolist()
            if isinstance(pruned_node_indices, int):
                pruned_node_indices = [pruned_node_indices]

            n_features_total = len(graph.selected_features)
            n_error_nodes_total = n_layers * n_pos

            pruned_feature_nodes = [i for i in pruned_node_indices if i < n_features_total]
            pruned_error_nodes = [i for i in pruned_node_indices if n_features_total <= i < n_features_total + n_error_nodes_total]

            features.append(graph.activation_values.shape[0]) # Total active features
            features.append(len(pruned_feature_nodes))      # Pruned feature node count
            features.append(len(pruned_error_nodes))        # Pruned error node count

            # Logit stats
            if graph.logit_probabilities.numel() > 0:
                probs = graph.logit_probabilities
                features.append(probs[0].item()) # Top logit probability
                features.append(-torch.sum(probs * torch.log(probs + 1e-8)).item()) # Logit entropy (add small epsilon)
            else:
                features.extend([0.0, 0.0])

            # --- Level 2: Aggregated Node Stats ---
            node_influences = cumulative_scores[node_mask]
            features.append(node_influences.mean().item()) # Mean node influence

            if len(pruned_error_nodes) > 0:
                error_influences = cumulative_scores[pruned_error_nodes]
                features.append(error_influences.sum().item()) # Total error influence
                features.append(error_influences.mean().item()) # Mean error influence
            else:
                features.extend([0.0, 0.0])

            # Activation stats for pruned feature nodes
            if len(pruned_feature_nodes) > 0:
                selected_indices_for_pruned_nodes = [graph.selected_features[i] for i in pruned_feature_nodes]
                pruned_activations = graph.activation_values[selected_indices_for_pruned_nodes]
                if pruned_activations.numel() > 0:
                    features.append(pruned_activations.mean().item())
                    features.append(pruned_activations.max().item())
                    features.append(pruned_activations.std().item())
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0])

            # Layer-wise feature counts (histogram)
            layer_counts = [0] * n_layers
            for node_idx in pruned_feature_nodes:
                feature_idx = graph.selected_features[node_idx]
                if feature_idx < len(graph.active_features):
                    layer, _, _ = graph.active_features[feature_idx].tolist()
                    if layer < n_layers:
                        layer_counts[layer] += 1
            features.extend(layer_counts)

            # --- Level 3: New Topological and Edge-Based Features ---
            # Call the helper function
            topo_features_dict = self._extract_topological_and_edge_features(pruned_node_indices, graph)

            # Append the new features in a fixed, reliable order
            feature_keys_ordered = [
                'sum_edge_weights', 'mean_edge_weights', 'std_edge_weights', 'n_edges_pruned',
                'graph_density', 'n_connected_components', 'mean_degree_centrality',
                'max_degree_centrality', 'mean_betweenness_centrality', 'max_betweenness_centrality',
                'avg_shortest_path_length', 'input_to_logit_shortest_path'
            ]
            for key in feature_keys_ordered:
                features.append(topo_features_dict[key])

            return np.array(features)

    def _extract_topological_and_edge_features(self, pruned_indices: List[int], graph: Graph) -> Dict[str, float]:
        """
        Calculates edge-based and topological features for a pruned subgraph.

        Args:
            pruned_indices: A list of integer indices for the nodes in the pruned graph.
            graph: The full Graph object, used to access the adjacency matrix and metadata.

        Returns:
            A dictionary of calculated features.
        """
        features = {}
        num_nodes_pruned = len(pruned_indices)

        # Define default values for all features to ensure a fixed vector size.
        # This is crucial if the pruned graph is too small to meaningful stats.
        default_features = {
            'sum_edge_weights': 0.0, 'mean_edge_weights': 0.0, 'std_edge_weights': 0.0,
            'n_edges_pruned': 0, 'graph_density': 0.0, 'n_connected_components': float(num_nodes_pruned),
            'mean_degree_centrality': 0.0, 'max_degree_centrality': 0.0,
            'mean_betweenness_centrality': 0.0, 'max_betweenness_centrality': 0.0,
            'avg_shortest_path_length': -1.0, # Use -1 to indicate not computed
            'input_to_logit_shortest_path': -1.0 # Use -1 to indicate no path found
        }

        if num_nodes_pruned < 2:
            return default_features

        # --- 1. Create the pruned subgraph from the full adjacency matrix ---
        full_adj_matrix = graph.adjacency_matrix
        pruned_adj_matrix = full_adj_matrix[np.ix_(pruned_indices, pruned_indices)]

        # Create a NetworkX Graph object for topological analysis
        # We create a directed graph as influence is directional.
        G = nx.from_numpy_array(pruned_adj_matrix.numpy(), create_using=nx.DiGraph)

        # --- 2. Calculate Edge-Based Features ---
        # 'get_edge_attributes' is a clean way to get all weights.
        edge_weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))

        if edge_weights.size > 0:
            features['sum_edge_weights'] = np.sum(edge_weights)
            features['mean_edge_weights'] = np.mean(edge_weights)
            features['std_edge_weights'] = np.std(edge_weights)
        else:
            features['sum_edge_weights'] = 0.0
            features['mean_edge_weights'] = 0.0
            features['std_edge_weights'] = 0.0

        # --- 3. Calculate Topological and Structural Features ---
        # Number of paths (interpreted as number of active edges)
        features['n_edges_pruned'] = G.number_of_edges()
        features['graph_density'] = nx.density(G)

        # For directed graphs, we consider weakly connected components.
        features['n_connected_components'] = nx.number_weakly_connected_components(G)

        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight') # Weighted is more meaningful
        features['mean_degree_centrality'] = np.mean(list(degree_centrality.values())) 
        features['max_degree_centrality'] = np.max(list(degree_centrality.values())) if degree_centrality else 0.0
        features['mean_betweenness_centrality'] = np.mean(list(betweenness_centrality.values())) 
        features['max_betweenness_centrality'] = np.max(list(betweenness_centrality.values())) if betweenness_centrality else 0.0

        # --- 4. Calculate Shortest Path Features ---
        # Average shortest path length for the largest connected component
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        if len(largest_cc) > 1:
            subgraph = G.subgraph(largest_cc)
            try:
                features['avg_shortest_path_length'] = nx.average_shortest_path_length(subgraph, weight='weight')
            except nx.NetworkXError:
                features['avg_shortest_path_length'] = -1.0 # Disconnected component
        else:
            features['avg_shortest_path_length'] = -1.0

        # --- 5. Calculate specific 'Input Token to Logit' shortest path ---
        # First, identify the global indices for each node type
        n_features_total = len(graph.selected_features)
        n_error_nodes_total = graph.cfg.n_layers * graph.n_pos
        start_token_nodes = n_features_total + n_error_nodes_total # Start of token nodes
        end_token_nodes = start_token_nodes + graph.n_pos # End of token nodes
        start_logit_nodes = end_token_nodes # Logit nodes start where token nodes end

        # Create a mapping from global node index to the local index in the pruned graph 'G'
        global_to_local_idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(pruned_indices)}

        # Find which input tokens and logit nodes are present in our pruned graph
        pruned_input_nodes_local = [
            global_to_local_idx_map[i] for i in pruned_indices
            if start_token_nodes <= i < end_token_nodes
        ]
        pruned_logit_nodes_local = [
            global_to_local_idx_map[i] for i in pruned_indices
            if i >= start_logit_nodes
        ]

        min_path_len = float('inf')
        if pruned_input_nodes_local and pruned_logit_nodes_local:
            for source_node in pruned_input_nodes_local:
                for target_node in pruned_logit_nodes_local:
                    if nx.has_path(G, source=source_node, target=target_node):
                        path_len = nx.shortest_path_length(G, source=source_node, target=target_node, weight='weight')
                        if path_len < min_path_len:
                            min_path_len = path_len

        features['input_to_logit_shortest_path'] = min_path_len if min_path_len != float('inf') else -1.0

        # Merge calculated features with defaults to ensure all keys are present
        for key in default_features:
            if key not in features:
                features[key] = default_features[key]

        return features

class CircuitAnalyzer:
    """
    Simple class to load circuit graphs and step labels for visualization.
    """

    def __init__(self, graph_dir: str = "/graphs/"):
        self.graph_dir = Path(graph_dir)
        self.data = [] # List to store all loaded data

    def load_all_data(
        self, expressions_json_path: str, graph_name_prefix: str,
        feature_type: str = 'adjacency_matrix', before_after: str = 'both',
        num_workers: int = 4, batch_size: int = 32, use_cpu: bool = True
    ):
        """
        Load all available graphs and their corresponding step labels in parallel.
        Extract features immediately to avoid memory issues.

        Args:
            expressions_json_path: Path to expressions JSON file
            graph_name_prefix: Prefix for graph file names
            feature_type: Type of feature to extract ('adjacency_matrix', 'active_features', etc.)
            before_after: Which graphs to load ('before', 'after', or 'both')
            num_workers: Number of parallel workers for loading
            batch_size: Batch size for DataLoader
            use_cpu: Whether to load tensors to CPU (True) or GPU (False)
        """
        # Load expressions
        with open(expressions_json_path, 'r') as f:
            expressions = json.load(f)
            

if __name__ == "__main__":
    graph_paths = [
    {
        'expr_id': 0,
        'step_number': 0,
        'before_after': 'before',
        'graph_path': '/home/kaiyu/CodeCircuit/data/graph/graph_0_11_0.pt',
        'step_labels': 0,
        'original_expression': ''
    }]

    dataset = GraphDataset(
    graph_paths,
    feature_type='advanced_graph_features',   
    use_cpu=False
    )
    sample = dataset[0]

    print("加载是否成功:", sample['success'])
    print("特征 shape:", sample['features'].shape)
    print("特征内容:", sample['features'])