import pandas as pd

def neuronal_outputs(neurons, connections, classification, root_ids, synapse_filters, hops=1, all_previous_neurons=None):
    """
    Recursive function to analyze neuronal connections for a specified number of hops.
    
    Parameters:
    neurons (pd.DataFrame): DataFrame of neurons, including 'root_id'.
    connections (pd.DataFrame): DataFrame of synaptic connections including 'pre_root_id', 'post_root_id', etc.
    classification (pd.DataFrame): DataFrame that classifies neurons, includes 'root_id' and other classifications.
    root_ids (list or pd.Series): List of root IDs or DataFrame column to start the analysis from.
    synapse_filters (list): List int filters for each hop.
                            If only one filter is provided, it will be used for all hops.
    hops (int): The number of hops downstream to explore.
    
    Returns:
    connectivity (pd.DataFrame): DataFrame of synaptic connections based on input filters.
    downstream_neurons (pd.DataFrame): DataFrame of downstream neurons after filtering.
    unique_downstream_neurons (pd.DataFrame): DataFrame of unique downstream neurons.
    """
    # Initialize the set of all previous neurons if this is the first hop
    if all_previous_neurons is None:
        all_previous_neurons = set(root_ids)  # Track neurons that have already been visited

    # Return null if hops not valid
    if hops <= 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Check if root_ids is a list or DataFrame column (pd.Series)
    if isinstance(root_ids, list):
        current_root_ids = pd.DataFrame({'root_id': root_ids})
    elif isinstance(root_ids, pd.Series):
        current_root_ids = pd.DataFrame({'root_id': root_ids})
    else:
        raise ValueError("root_ids must be a list or a DataFrame column (pd.Series)")


    # Merge with connections to find downstream neurons
    connectivity = pd.merge(current_root_ids['root_id'], 
                            connections[['pre_root_id', 'post_root_id', 'neuropil', 'syn_count', 'nt_type']], 
                            left_on='root_id', right_on='pre_root_id', 
                            how='inner')

    # Apply synapse filters for this hop
    filters_for_this_hop = synapse_filters[0]
    connectivity = connectivity.query("syn_count >= " + str(filters_for_this_hop))

    # Drop synapse filter already used
    if len(synapse_filters) > 1:
        synapse_filters = synapse_filters[1:]
    
    # Drop 'root_id' column from connectivity
    connectivity = connectivity.drop(columns='root_id')
    
    # Find downstream neurons and their classification
    downstream_neurons = pd.merge(connectivity, classification, left_on='post_root_id', right_on='root_id', how='inner')

    # Remove any neurons that were in the original root_ids
    unique_downstream_neurons = downstream_neurons.copy()
    unique_downstream_neurons = unique_downstream_neurons.loc[~unique_downstream_neurons['root_id'].isin(all_previous_neurons)]

    downstream_neurons = downstream_neurons.drop(columns=['post_root_id','pre_root_id','neuropil','syn_count','nt_type'])
    unique_downstream_neurons = unique_downstream_neurons.drop(columns=['post_root_id','pre_root_id','neuropil','syn_count','nt_type'])

    # Add the new unique downstream neurons to the set of all previous neurons
    all_previous_neurons.update(unique_downstream_neurons['root_id'])

    # If this is the final hop, return the results for this hop
    if hops == 2:
        return connectivity, downstream_neurons, unique_downstream_neurons

    # Recursive call for the next hop using unique downstream neurons as the new root_ids
    return neuronal_outputs(
        neurons, connections, classification, unique_downstream_neurons['root_id'], synapse_filters, hops - 1, all_previous_neurons
    )
