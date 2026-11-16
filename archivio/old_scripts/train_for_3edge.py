
#CREATE THE TRAIN DATASTRUCTURE FOR 3 EDGES

#%%

import pandas as pd
import networkx as nx
import os

# Specify the features to keep
features_to_keep = [
    'Source IP', 'Destination IP', 'Source Port', 'Destination Port',
    'Protocol', 'Timestamp', 'Flow Duration', 'Label'
]

# Load your dataset
data = pd.read_csv('./data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')  # Replace 'your_data.csv' with your actual file name

data.columns = data.columns.str.strip()

# Keep only the specified features
filtered_data = data[features_to_keep]

# Convert 'Timestamp' to datetime
filtered_data['Timestamp'] = pd.to_datetime(filtered_data['Timestamp'])   # aggiunto dayfirst = True

# Order the data by 'Timestamp'
filtered_data = filtered_data.sort_values(by='Timestamp')

# Save the temporally ordered data to a new file
filtered_data.to_csv('filtered_train_3edge.csv', index=False)

print("Filtered and temporally ordered data saved to 'filtered_train_3edge.csv'.")


#%%

#check for inside of csv (just for test, no need for run)

# Load the CSV file
file_path = "filtered_train_3edge.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Check if the label column contains '1'
label_column = 'Label'  # Replace with the actual label column name if different
if label_column in df.columns:
    label_distribution = df[label_column].value_counts()
    print("Label Distribution:")
    print(label_distribution)

    if 1 in label_distribution.index:
        print("The CSV contains label '1'.")
    else:
        print("The CSV does NOT contain label '1'.")
else:
    print(f"'{label_column}' column not found in the CSV.")





#%% CREATED HOURLY GRAPH WITH 3 EDGES FROM TRAIN DATASET


def create_test_graphs_edge_labels(df, output_dir):
    """
    Split the DataFrame into hourly slices and create graphs for each slice.
    Each edge gets a valid label (e.g., 0 or 1) read from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame with temporal data.
        output_dir (str): Directory to save the graphs.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group the DataFrame into hourly slices using the datetime index.
    # (Assumes the DataFrame index is already a DateTimeIndex)
    time_slices = [g for _, g in df.groupby(pd.Grouper(freq='H'))]

    # Mappa per convertire le etichette testuali in numeri (necessario per la GNN)
    label_mapping = {'BENIGN': 0, 'DDoS': 1}

    for slice_index, slice_df in enumerate(time_slices):
        if slice_df.empty:
            continue

        # Print value counts of the 'Label' column in this time-slice.
        print(f"Hour {slice_index} - Traffic Distribution:")
        print(slice_df['Label'].value_counts())

        # Create a MultiDiGraph for this time-slice.
        G = nx.MultiDiGraph()

        for _, row in slice_df.iterrows():
            src_ip = row['Source IP']
            dst_ip = row['Destination IP']

            # Convert the label to an int (if missing or invalid, you can decide a fallback; here we assume it is valid)
            #try:
            #    label = int(row['Label'])
            #except Exception as e:
            #   print(f"Skipping row due to invalid label: {row['Label']}; error: {e}")
            #    continue

            # Convertiamo la Label testuale in numero
            raw_label = row['Label']
            label = label_mapping.get(raw_label, 0) # Default a 0 se sconosciuto

            if pd.isna(src_ip) or pd.isna(dst_ip):
                continue

            # Add nodes if not already present.
            if not G.has_node(src_ip):
                G.add_node(src_ip)
            if not G.has_node(dst_ip):
                G.add_node(dst_ip)

            # Add edges for different interactions.
            # 1. Network Edge
            G.add_edge(src_ip, dst_ip, key='network',
                       label=label,
                     #selected fetures
                       interaction='network_communication')

            # 2. Context Edge
            G.add_edge(src_ip, dst_ip, key='context',
                       label=label,
                     #selected fetures
                       interaction='context')

            # 3. Knowledge Edge
            G.add_edge(src_ip, dst_ip, key='knowledge',
                       label=label,
                       #selected fetures
                       interaction='knowledge')


        # Save the graph as a .gpickle file.

        # Nota: nx.write_gpickle è rimosso nelle versioni recenti di NetworkX.
        # Usiamo pickle direttamente o salviamo in altro formato.

        #graph_path = os.path.join(output_dir, f"test_graph_hour_{slice_index}.gpickle")
        #nx.write_gpickle(G, graph_path)
        #print(f"Test graph for hour {slice_index} saved to {graph_path}")

        import pickle
        graph_path = os.path.join(output_dir, f"test_graph_hour_{slice_index}.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)

        print(f"Grafo dell'ora {slice_index} salvato in {graph_path}")


# Usage Example for graph creation
if __name__ == "__main__":
    # Read CSV and prepare DataFrame.
    df_test = pd.read_csv('filtered_train_3edge.csv')
    df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'])
    # Set Timestamp as index and sort (required for grouping by hour)
    df_test = df_test.set_index('Timestamp').sort_index()

    output_test_dir = "3ed_trai_h_graphs"
    create_test_graphs_edge_labels(df_test, output_test_dir)



#%%Riplika of previous code:

import pickle

def add_node_features(G):
    """
    Adds additional features to nodes in the graph, including:
    - Node degree
    - Community ID
    - Temporal activity (average edge count per node)
    - Node centrality (betweenness centrality)

    Parameters:
        G (nx.MultiDiGraph): The input graph.

    Returns:
        nx.MultiDiGraph: The graph with added node features.
    """
    # Add degree
    for node in G.nodes:
        G.nodes[node]['degree'] = G.degree[node]

    # Add community detection (Label Propagation)
    undirected_graph = nx.Graph(G)  # Convert to undirected for community detection
    communities = nx.community.label_propagation_communities(undirected_graph)
    community_mapping = {node: community_id for community_id, community in enumerate(communities) for node in community}

    for node in G.nodes:
        G.nodes[node]['community'] = community_mapping.get(node, -1)

    # Add centrality (Betweenness Centrality)
    centrality = nx.betweenness_centrality(G)
    for node, value in centrality.items():
        G.nodes[node]['centrality'] = value

    return G

def create_test_graphs_edge_labels(df, output_dir):
    """
    Split the DataFrame into hourly slices and create graphs for each slice.
    Each edge gets a valid label (e.g., 0 or 1) read from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame with temporal data.
        output_dir (str): Directory to save the graphs.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Mapping per risolvere l'errore 'invalid literal for int(): DDoS'
    label_mapping = {'BENIGN': 0, 'DDoS': 1}

    # Group the DataFrame into hourly slices using the datetime index.
    time_slices = [g for _, g in df.groupby(pd.Grouper(freq='H'))]

    for slice_index, slice_df in enumerate(time_slices):
        if slice_df.empty:
            continue

        # Print value counts of the 'Label' column in this time-slice.
        print(f"Hour {slice_index}:")
        print(slice_df['Label'].value_counts())

        # Create a MultiDiGraph for this time-slice.
        G = nx.MultiDiGraph()

        for _, row in slice_df.iterrows():
            src_ip = row['Source IP']
            dst_ip = row['Destination IP']

            # Convert the label to an int (if missing or invalid, you can decide a fallback; here we assume it is valid)
            #try:
            #    label = int(row['Label'])
            #except Exception as e:
            #    print(f"Skipping row due to invalid label: {row['Label']}; error: {e}")
            #    continue

            # TRADUZIONE LABEL: Trasforma "DDoS" in 1 e "BENIGN" in 0
            label = label_mapping.get(row['Label'], 0)

            if pd.isna(src_ip) or pd.isna(dst_ip):
                continue

            # Add nodes if not already present.
            if not G.has_node(src_ip):
                G.add_node(src_ip)
            if not G.has_node(dst_ip):
                G.add_node(dst_ip)

            # Add edges for different interactions.
            # 1. Network Edge
            G.add_edge(src_ip, dst_ip, key='network',
                       label=label,
                        #selected fetures
                       interaction='network_communication')

            # 2. Context Edge
            G.add_edge(src_ip, dst_ip, key='context',
                       label=label,
                      #selected fetures
                       interaction='context')

            # 3. Knowledge Edge
            G.add_edge(src_ip, dst_ip, key='knowledge',
                       label=label,
                         #selected fetures
                       interaction='knowledge')

        # Add node features
        G = add_node_features(G)

        # Save the graph as a .gpickle file.
        #graph_path = os.path.join(output_dir, f"test_graph_hour_{slice_index}.gpickle")
        #nx.write_gpickle(G, graph_path)
        #print(f"Test graph for hour {slice_index} saved to {graph_path}")

        # Salvataggio: gpickle non è più supportato in NetworkX 3.0+
        graph_path = os.path.join(output_dir, f"test_graph_hour_{slice_index}.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Grafo ora {slice_index} salvato con successo.")

# Usage Example for graph creation
if __name__ == "__main__":
    # Read CSV and prepare DataFrame.
    df_test = pd.read_csv('filtered_train_3edge.csv')

    # Pulisce i nomi delle colonne da spazi extra (comuni nel CIC-IDS2017)
    df_test.columns = df_test.columns.str.strip()

    df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'])
    # Set Timestamp as index and sort (required for grouping by hour)
    df_test = df_test.set_index('Timestamp').sort_index()

    output_test_dir = "3ed_trai_h_graphs"
    create_test_graphs_edge_labels(df_test, output_test_dir)

#%% COMMUNITY DETECTION FOR GRAPHS AND THEN UPDATE THE GRAPH WITH THE LABEL OF COMMUNITY FOR EACH NODE 


def detect_and_label_communities_lpa(graph):
    """
    Perform community detection using the Label Propagation Algorithm (LPA) and label nodes with community IDs.
    Adds 'x' attribute based on the 'community' label.

    Parameters:
        graph (nx.MultiDiGraph): Input graph.

    Returns:
        graph (nx.MultiDiGraph): Updated graph with community labels and 'x' attributes.
    """
    # Convert MultiDiGraph to Graph (undirected graph for LPA)
    undirected_graph = nx.Graph(graph)

    # Perform community detection using LPA
    communities = nx.community.label_propagation_communities(undirected_graph)

    # Assign community labels to nodes and add 'x' attribute
    for community_id, community in enumerate(communities):
        for node in community:
            graph.nodes[node]['community'] = community_id
            graph.nodes[node]['x'] = [community_id]  # 'x' is a feature; wrap in a list for PyTorch Geometric compatibility

    return graph


def process_graphs_with_lpa(input_dir, output_dir):
    """
    Detect communities using LPA, update graphs with community labels, and add 'x' attribute.

    Parameters:
        input_dir (str): Directory containing input graphs.
        output_dir (str): Directory to save updated graphs.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each graph file in the input directory
    for graph_file in os.listdir(input_dir):
        if not graph_file.endswith('.gpickle'):
            continue

        # Load the graph
        graph_path = os.path.join(input_dir, graph_file)
        G = nx.read_gpickle(graph_path)

        # Detect communities using LPA and label nodes
        G = detect_and_label_communities_lpa(G)

        # Save the updated graph
        updated_graph_path = os.path.join(output_dir, graph_file)
        nx.write_gpickle(G, updated_graph_path)
        print(f"Updated graph with LPA communities and 'x' attribute saved to {updated_graph_path}")


# Example usage
if __name__ == "__main__":
    # Input directory containing graphs
    input_graph_dir = "3ed_trai_h_graphs"

    # Output directory for updated graphs
    output_graph_dir = "3ed_trai_h_graphs_commun"

    # Process graphs and add community labels using LPA
    process_graphs_with_lpa(input_graph_dir, output_graph_dir)

#%% CONVERT MULTIPGRAPH TO HETERODATA 

import torch
from torch_geometric.data import HeteroData


def multiDiGraph_to_hetero_with_label(G: nx.MultiDiGraph) -> HeteroData:
    """
    Converts a MultiDiGraph with multiple edge types to a HeteroData object.
    Preserves the 'label' field in data[rel_type].edge_label.
    """
    data = HeteroData()
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    data['ip'].num_nodes = G.number_of_nodes()

    # Add node-level features
    x = []
    community_labels = []

    for node in G.nodes():
        #community = G.nodes[node].get('community', -1)
        #community_labels.append(community)
        #x.append([community])

        community = G.nodes[node].get('community', -1)
        degree = G.nodes[node].get('degree', 0)
        centrality = G.nodes[node].get('centrality', 0)
        # Il paper usa community, in-degree e out-degree come feature dei nodi
        x.append([float(community), float(degree), float(centrality)])

    #data['ip'].community = torch.tensor(community_labels, dtype=torch.long)
    data['ip'].x = torch.tensor(x, dtype=torch.float)

    # Process each edge from G.
    for u, v, key, edge_attrs in G.edges(data=True, keys=True):
        src = node_mapping[u]
        dst = node_mapping[v]
        rel_type = ('ip', key, 'ip')

        if rel_type not in data.edge_types:
            data[rel_type].edge_index = []
            data[rel_type].edge_attr = []
            data[rel_type].edge_label = []  # Container for the label

        data[rel_type].edge_index.append([src, dst])
        feature_vec = []

        #if key == 'network':
        #    for attr_name in [ #selected fetures]:
        #        feature_vec.append(edge_attrs.get(attr_name, 0))
        #elif key == 'context':
        #    for attr_name in [ #selected fetures]:
        #        feature_vec.append(edge_attrs.get(attr_name, 0))
        #elif key == 'knowledge':
        #    for attr_name in [ #selected fetures]:
        #        feature_vec.append(edge_attrs.get(attr_name, 0))

        if key == 'network': # Network Communication Features [cite: 260, 563]
            feature_vec = [edge_attrs.get('Source Port', 0), edge_attrs.get('Destination Port', 0)]
        elif key == 'context': # Contextual Features (IAT, temporal) [cite: 266, 564]
            feature_vec = [edge_attrs.get('Flow Duration', 0)]
        elif key == 'knowledge': # Knowledge-Based (Flags, Packet size) [cite: 271, 565]
            feature_vec = [edge_attrs.get('Protocol', 0)]

        data[rel_type].edge_attr.append(feature_vec)
        # Save the label
        label = edge_attrs.get('label', 0)  # Default to -1 if label is missing
        #if label == -1:
        #    print(f"Warning: Missing or invalid label for edge {u} -> {v} of type {key}")
        data[rel_type].edge_label.append(label)

    # Convert lists to tensors.
    for rel_type in data.edge_types:
        data[rel_type].edge_index = torch.tensor(data[rel_type].edge_index, dtype=torch.long).t().contiguous()
        if data[rel_type].edge_attr:
            data[rel_type].edge_attr = torch.tensor(data[rel_type].edge_attr, dtype=torch.float)
        if data[rel_type].edge_label:
            data[rel_type].edge_label = torch.tensor(data[rel_type].edge_label, dtype=torch.long)
    return data

def process_and_save_hetero_graphs_with_label(input_dir, output_dir):
    """
    Converts all .gpickle graphs in a directory to HeteroData objects and saves them as .pt,
    preserving the 'label' field in data[rel_type].edge_label.
    """
    os.makedirs(output_dir, exist_ok=True)
    for graph_file in os.listdir(input_dir):
        #if not graph_file.endswith('.gpickle'):
        if not graph_file.endswith('.pkl'):
            continue

        graph_path = os.path.join(input_dir, graph_file)
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        #G = nx.read_gpickle(graph_path)
        hetero_data = multiDiGraph_to_hetero_with_label(G)
        hetero_path = os.path.join(output_dir, graph_file.replace('.pkl', '.pt'))
        torch.save(hetero_data, hetero_path)

        #print(f"Saved HeteroData with labels to {hetero_path}")
        print(f"Converted: {graph_file} -> {hetero_path}")

if __name__ == "__main__":
    #input_test_dir = "3ed_trai_h_graphs_commun"         # Input .gpickle files (with communities added)
    #output_test_pt_dir = "3ed_trai_h_graphs_hetero_graphs" # Output .pt files
    #process_and_save_hetero_graphs_with_label(input_test_dir, output_test_pt_dir)
    input_dir_pkl = "3ed_trai_h_graphs"
    output_dir_pt = "/content/drive/MyDrive/GraphIDS/hetero_graphs_pt"
process_and_save_hetero_graphs_with_label(input_dir_pkl, output_dir_pt)

#%% TEST FOR INSIDE OF GRAPH, NO NEED TO RUN IT 

#was test for inside of .pt ( no need to run)
import torch
import os

def inspect_pt_file(file_path):
    """
    Inspects the contents of a .pt file and prints its structure.

    Parameters:
        file_path (str): Path to the .pt file.
    """
    data = torch.load(file_path)
    print(f"Inspecting file: {file_path}")
    print("-" * 40)

    # Check if it's a PyTorch Geometric HeteroData object
    if isinstance(data, dict):
        print("File contains a dictionary. Keys:")
        for key, value in data.items():
            print(f"  {key}: {type(value)}")
            if isinstance(value, torch.Tensor):
                print(f"    Tensor shape: {value.shape}")
    elif hasattr(data, 'keys') and hasattr(data, 'edge_index_dict'):
        print("File contains a HeteroData object.")
        print(f"Node types: {data.node_types}")
        for node_type in data.node_types:
            print(f"  Node type '{node_type}':")
            if 'x' in data[node_type]:
                print(f"    Node features 'x': shape {data[node_type].x.shape}")
            else:
                print("    No node features ('x') found.")
            if 'num_nodes' in data[node_type]:
                print(f"    Number of nodes: {data[node_type].num_nodes}")

        print(f"Edge types: {data.edge_types}")
        for edge_type in data.edge_types:
            print(f"  Edge type {edge_type}:")
            if 'edge_index' in data[edge_type]:
                print(f"    Edge index: shape {data[edge_type].edge_index.shape}")
            if 'edge_attr' in data[edge_type]:
                print(f"    Edge attributes: shape {data[edge_type].edge_attr.shape}")
    else:
        print("Unknown data format.")
    print("-" * 40)

def inspect_all_pt_files(directory):
    """
    Inspects all .pt files in a given directory.

    Parameters:
        directory (str): Path to the directory containing .pt files.
    """
    print(f"Inspecting .pt files in directory: {directory}")
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            inspect_pt_file(os.path.join(directory, file))

# Directory containing your .pt files
input_graph_dir = "3ed_trai_h_graphs_hetero_graphs"

# Inspect all files in the directory
inspect_all_pt_files(input_graph_dir)



#%%was test for inside of graph ( no need to run)


def inspect_community_in_gpickle(file_path):
    """
    Inspects the presence of the 'community' attribute in a .gpickle file.

    Parameters:
        file_path (str): Path to the .gpickle file.
    """
    print(f"Inspecting file: {file_path}")
    print("-" * 40)

    # Load the graph
    G = nx.read_gpickle(file_path)

    # Check for 'community' attribute in nodes
    if all('community' in G.nodes[node] for node in G.nodes()):
        print(f"All nodes have a 'community' attribute.")
        print("Sample 'community' values:")
        sample_communities = {node: G.nodes[node]['community'] for node in list(G.nodes)[:10]}
        print(sample_communities)
    else:
        missing = [node for node in G.nodes() if 'community' not in G.nodes[node]]
        print(f"Some nodes are missing the 'community' attribute. Missing nodes: {missing[:10]} (only showing first 10)")

    print(f"Total nodes: {len(G.nodes())}")
    print("-" * 40)


def inspect_all_gpickle_files(directory):
    """
    Inspects the 'community' attribute in all .gpickle files in a given directory.

    Parameters:
        directory (str): Path to the directory containing .gpickle files.
    """
    print(f"Inspecting .gpickle files in directory: {directory}")
    for file in os.listdir(directory):
        if file.endswith(".gpickle"):
            inspect_community_in_gpickle(os.path.join(directory, file))


# Directory containing your .gpickle files
input_graph_dir = "3ed_trai_h_graphs_commun"

# Inspect all files in the directory for the 'community' attribute
inspect_all_gpickle_files(input_graph_dir)

