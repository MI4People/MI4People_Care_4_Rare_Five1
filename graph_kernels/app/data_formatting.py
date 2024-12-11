from grakel import Graph


def transform_A_for_grakel(graphs_isSick):
            grakel_isSick = []
            labels = []
            
            # Iterate over each graph in your `graphs` dictionary
            for graph in graphs_isSick.values():  
                edges = list(graph['edges'])  # Edges are already in the correct format
                node_labels = graph['nodes']  # Nodes are already in the correct format
                edge_labels = graph['edge_labels']
                
                edges_list = [(edge[0], edge[1], edge[2]) for edge in edges]

                # Append the graph to the list in the format grakel expects
                gk_graph = Graph(initialization_object=edges_list, node_labels=node_labels, edge_labels=edge_labels)
                grakel_isSick.append(gk_graph)
                
                # Append the label (isSick) to the labels list
                labels.append(graph['isSick'])
            return grakel_isSick, labels

def transform_B_for_grakel(graphs_icd10):
            grakel_icd10 = []
            labels = []
            
            # Iterate over each graph in your `graphs` dictionary
            for graph in graphs_icd10.values():  
                edges = list(graph['edges'])  # Edges are already in the correct format
                node_labels = graph['nodes']  # Nodes are already in the correct format
                edge_labels = graph['edge_labels']

                edges_list = [(edge[0], edge[1], edge[2]) for edge in edges]

                # Append the graph to the list in the format grakel expects
                grakel_graph = Graph(initialization_object=edges_list, node_labels=node_labels, edge_labels=edge_labels)
                grakel_icd10.append(grakel_graph)
                
                # Append the label (isSick) to the labels list
                labels.append(graph['icd10'])
            return grakel_icd10, labels