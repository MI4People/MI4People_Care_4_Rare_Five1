#%% [markdown]
# This script provides some ideas for the visualization of the outputs from the graph algorithms (gds_exploration_mutate.py)
# -> adapt and expand according to individual needs and interests

#%%
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# Load node properties
# Replace 'node_properties.csv' with the path to your data
nodes_df = pd.read_csv("node_properties.csv")

# %%
# Dimensionality reduction (t-SNE, UMAP) to visualize node embeddings
# here: t-SNE -> PCA or UMAP may also be considered

# Load all embeddings into a dictionary; convert string to list
node_embeddings = {
    #'FastRP_weighted': nodes_df['fastRP_relationship_weight'].apply(eval).to_list(), # visualization of weighted FastRP embeddings was not supported
    'FastRP': nodes_df['fastRP'].apply(eval).to_list(),
    'GraphSAGE': nodes_df['GraphSAGE_relationship_weight'].apply(eval).to_list(),
    'Node2Vec_weighted': nodes_df['Node2Vec_relationship_weight'].apply(eval).to_list(),
    'Node2Vec': nodes_df['Node2Vec'].apply(eval).to_list(),
    'HashGNN': nodes_df['hashgnn'].apply(eval).to_list()
}

# convert list to numpy array for compatibility with dimensionality reduction
node_embeddings = {name: np.array(values) for name, values in node_embeddings.items()}

# Define distinct colors for each label
unique_labels = nodes_df["node_labels"].unique()
color_palette = ['red', 'blue', 'green', 'orange', 'purple']  # Add more colors if needed
color_map = {label: color_palette[i] for i, label in enumerate(unique_labels)}

# Map node colors to the color palette
node_colors = nodes_df["node_labels"].map(color_map)


for name, embeddings in node_embeddings.items():
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Plot t-SNE
    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=node_colors, s=10, alpha=0.8)
    plt.title(f"t-SNE of {name} node embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Add legend
    for label in unique_labels:
        plt.scatter([], [], c=color_map[label], label=label)  # Empty scatter for legend
    plt.legend(title="Node Labels", loc="upper right")

    plt.show()


# %%

import pandas as pd
import matplotlib.pyplot as plt

# Replace 'node_properties.csv' with the path to your data; already loaded above
# nodes_df = pd.read_csv("node_properties.csv")

# get all community information

# Count the number of nodes per community
communities_dict = {
    'Louvain': 'louvain',
    'Louvain_weighted': 'louvain_relationship_weight',
    'Label Propagation': 'label_propagation',
    'Label Propagation_weighted': 'label_propagation_relationship_weight'
}

print(communities_dict)
#%%

# Plot the distribution of communities by size (= number of nodes)

# Bin the communities based on node counts
bins = [0, 1, 20, 100, float('inf')]  # Define the bins
labels = ['1', '1-20', '20-100', '>100']  # Bin labels

for name, community in communities_dict.items():
    # Bin the communities based on node counts
    nodes_per_community = nodes_df[community].value_counts()
    binned_communities = pd.cut(nodes_per_community, bins=bins, labels=labels, include_lowest=True)

    # Count the number of communities in each bin
    bin_counts = binned_communities.value_counts(sort=False)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(bin_counts.index.astype(str), bin_counts.values, color='skyblue', edgecolor='black')
    plt.yscale('log')  # Log scale for better visualization
    plt.title(f"Distribution of Communities by Size ({name})", fontsize=16)
    plt.xlabel("Number of Nodes in Community (Bins)", fontsize=14)
    plt.ylabel("Number of Communities", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%

# Plot the distribution of node labels in large communities (which node types are in the largest communities?)

for name, community in communities_dict.items():
    community_counts = nodes_df[community].value_counts()
    large_communities = community_counts[community_counts >= 10].index
    nodes_in_large = nodes_df[nodes_df[community].isin(large_communities)]
    node_labels_large = nodes_in_large.groupby(community)['node_labels'].apply(list)
    label_distribution_large = nodes_in_large.groupby([community, 'node_labels']).size().reset_index(name='count')

    # Plot the distribution of node labels in large communities
    pivot_data = label_distribution_large.pivot(index=community, columns='node_labels', values='count').fillna(0)

    pivot_data.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='tab10')
    plt.title(f"Node Label Distribution Across Communities for {name}", fontsize=16)
    plt.xlabel("Community ID", fontsize=14)
    plt.legend(title="Node Labels")
    if name == 'Louvain_weighted':
        plt.ylabel("Count of Nodes", fontsize=14)
        plt.tight_layout()
        plt.show()
    else:
        plt.ylabel("Count of Nodes (log-scale)", fontsize=14)
        plt.yscale('log')
        plt.ylim(0.1, None)
        plt.tight_layout()
        plt.show()

# %%
# centrality measures

centrality_dict = {
    'Degree_weighted': nodes_df['Degree_relationship_weight'],
    'Degree': nodes_df['Degree'],
    'PageRank_weighted': nodes_df['PageRank_relatonship_weight'],
    'PageRank': nodes_df['PageRank'],
}
# %%
print(centrality_dict)
# %%
for name, centrality_values in centrality_dict.items():
    # Plot the distribution of centrality values (how many nodes have a certain centrality value?)
    plt.figure(figsize=(10, 6))
    sns.histplot(centrality_values, kde=True, color='skyblue', bins=10)
    plt.title(f"Distribution of {name} Centrality", fontsize=16)
    plt.xlabel("Centrality Value", fontsize=14)
    plt.ylabel("Frequency (number of nodes)", fontsize=14)
    plt.ylim(0, None) # Set y-axis lower limit to 0
    plt.tight_layout()
    plt.show()

# %%

centralities = ['Degree', 'Degree_relationship_weight', 'PageRank', 'PageRank_relatonship_weight']

# Plot the centrality values of the top 10 nodes (node_id, node_type) with the highest centrality for each centrality measure 

for value in centralities:
    top_degree = nodes_df.sort_values(by = value ,ascending=False).head(10)
    # colors
    unique_labels = top_degree['node_labels'].unique()
    color_map = {label: color for label, color in zip(unique_labels, ['red', 'blue', 'green', 'orange', 'purple', 'cyan'])}
    bar_colors = top_degree['node_labels'].map(color_map)
    plt.figure(figsize=(10, 6))
    plt.bar(top_degree['nodeId'], top_degree[f'{value}'], color='skyblue', edgecolor='black')
    plt.title(f"Top 10 Nodes by Centrality - {value}", fontsize=16)
    plt.xlabel("Node ID", fontsize=14)
    plt.ylabel(f"{value} Centrality", fontsize=14)
    plt.xticks(rotation=45)

    for label, color in color_map.items():
        plt.bar([], [], color=color, label=label)  # Add empty bars for legend entries
        plt.legend(title="Node Labels", loc="upper right")

    plt.tight_layout()
    plt.show()

# %%
top_degree = nodes_df.sort_values(by = 'Degree_relationship_weight' ,ascending=False).head(10)
print(top_degree)

# %%
# Similarities
# first impression: Jaccard, Cosine and Overlap similarities are only calculated for nodes defined as source nodes during graph projection (gds_exploration_mutate.py);
#                   also similarities are only computed between nodes of the same type
#                   kNN: as similarities are calculated based on node embeddings, kNN similarity is calculated between all nodes in the graph projection
#                   independent of source or target node and node type

sim_df = pd.read_csv("similarities.csv")
bs_df = sim_df[(sim_df['node1_label'] == 'Biological_sample') & (sim_df['node2_label'] == 'Biological_sample')]

print(sim_df.head())
#%%
#knn = sim_df.drop(columns=['node1_id', 'node2_id', 'Jaccard_similarity', 'Cosine_similarity', 'Overlap_similarity'])
#print(knn.head())
#knn = knn.dropna()
#print(knn.head())

#%%

# Plot similarity heatmaps for Biological_samples

similarities = ['Jaccard_similarity', 'Cosine_similarity', 'Overlap_similarity', 'kNN_similarity']

for value in similarities:
    plt.figure(figsize=(10, 10))
    hmp = bs_df.pivot(index='node1', columns='node2', values=f'{value}')
    sns.heatmap(hmp)
    plt.title(f"Heatmap of {value}", fontsize=20)  # Add title for the heatmap
    plt.xlabel("Node 2", fontsize=14)  # Add x-axis label
    plt.ylabel("Node 1", fontsize=14)  # Add y-axis label
    plt.tight_layout()  # Adjust layout
    plt.show()
# %%

bs2_df = sim_df[(sim_df['node1_label'] == 'Disease') & (sim_df['node2_label'] == 'Biological_sample')]
# -> only yields result for kNN (explanation above)

for value in similarities:
    plt.figure(figsize=(10, 10))
    hmp = bs2_df.pivot(index='node1', columns='node2', values=f'{value}')
    sns.heatmap(hmp)
    plt.title(f"Heatmap of {value}", fontsize=20)  # Add title for the heatmap
    plt.xlabel("Node 2", fontsize=14)  # Add x-axis label
    plt.ylabel("Node 1", fontsize=14)  # Add y-axis label
    plt.tight_layout()  # Adjust layout
    plt.show()

# %%
# Define the columns to check for NaN
columns_to_check = ['Jaccard_similarity', 'Cosine_similarity', 'Overlap_similarity']

# Drop rows where all specified columns are NaN
filtered_df = sim_df.dropna(subset=columns_to_check, how='all')

# Display the filtered DataFrame
#print(filtered_df)

# Filter for rows where node1_label and node2_label are NOT 'Biological_sample'
filtered_rows = filtered_df[(filtered_df['node1_label'] != 'Biological_sample') & (filtered_df['node2_label'] != 'Biological_sample')]

# Print the resulting DataFrame
print(filtered_rows)


# %%
print(bs_df)
# %%
