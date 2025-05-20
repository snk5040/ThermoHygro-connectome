import pandas as pd
import navis
import os
import trimesh
import rtree
from joblib import Parallel, delayed
from joblib import parallel_backend
from tqdm import tqdm

# Read 2N IDs from CSV
twoN_ids = pd.read_csv('./input/2N_ids.csv')

# Read previously filtered synapses from CSV
syn2N_df = pd.read_csv('./input/filtered_2N_synapses.csv')
synOlf_df = pd.read_csv('./input/filtered_olfactory_synapses.csv')

# Read ALPN IDs from CSV
alpn_ids = pd.read_csv('./input/alpn_ids.csv')

# Subtract ALPN IDs from 2N IDs (ie non ALPN 2N ids)
other_ids = twoN_ids[~twoN_ids['root_id'].isin(alpn_ids['root_id'])]

# Olfactory Neurons at the 3N level
olf3N_ids = pd.read_csv('./input/3N_olfactory.csv')

# Load all .ply mesh files
meshes = {}
glom_names = []
mesh_dir = './glom_meshes_proc'
for mesh_file in os.listdir(mesh_dir):
    if mesh_file.endswith('.ply'):
        glom_name = mesh_file.replace('.ply', '')
        glom_names.append(glom_name)
        meshes[glom_name] = navis.read_mesh(os.path.join(mesh_dir, mesh_file))
        
# Filter synapses to only include those where post_pt_root_id matches alpn_ids
alpn_syn_df = syn2N_df[syn2N_df['pre_pt_root_id'].isin(alpn_ids['root_id'])]

# Non ALPN 2N synapses
other_syn_df = syn2N_df[syn2N_df['pre_pt_root_id'].isin(other_ids['root_id'])]

olf3N_syn_df = synOlf_df[synOlf_df['pre_pt_root_id'].isin(olf3N_ids['root_id'])]

print('x')
# Initialize new columns for each glomerulus with 0s, for all three modalities
# Create copies once before the loop
alpn_syn_df = alpn_syn_df.copy()
other_syn_df = other_syn_df.copy()
olf3N_syn_df = olf3N_syn_df.copy()

for glom in glom_names:
    alpn_syn_df[glom] = 0
    other_syn_df[glom] = 0
    olf3N_syn_df[glom] = 0

# Check if all dataframes have the same columns
all_same = (alpn_syn_df.columns == other_syn_df.columns).all() and (alpn_syn_df.columns == olf3N_syn_df.columns).all()
print("\nAll dataframes have same columns:", all_same)

# Function to process a single row
def process_row(idx, row, meshes):
    result = {glom_name: 0 for glom_name in meshes.keys()}  # Initialize all glomerulus columns to 0
    try:
        point = row[['pre_pt_position_x', 'pre_pt_position_y', 'pre_pt_position_z']].values
        for glom_name, mesh in meshes.items():
            trimesh_mesh = mesh.trimesh
            if trimesh_mesh.contains([point])[0]:
                result[glom_name] = 1
    except IndexError:
        print(f"Index error occurred at idx {idx}")
    return idx, result

def process_synapse_df(syn_df, meshes):

    # Parallel processing with progress bar
    with parallel_backend("loky", inner_max_num_threads=1):  # Use loky backend for better thread management
        results = Parallel(n_jobs=-1)(
            delayed(process_row)(idx, row, meshes)
            for idx, row in tqdm(syn_df.iterrows(), total=len(syn_df), desc="Processing rows")
        )

    # Update the DataFrame
    for idx, result in results:
        for glom_name, value in result.items():
            syn_df.loc[idx, glom_name] = value
            
    return syn_df

    
###  ALPNs  ###

# Process ALPN synapses
alpn_syn_df = process_synapse_df(alpn_syn_df, meshes)

# Save the processed dataframes to CSV
alpn_syn_df.to_csv('./output/alpn_glomeruli_output.csv', index=False)

# Group by root_id and sum glomeruli columns for all three dataframes
alpn_summed_df = alpn_syn_df.groupby('pre_pt_root_id').sum(numeric_only=True).reset_index()

# Save the summed dataframes
alpn_summed_df.to_csv('./output/alpn_glomeruli_output_sum.csv', index=False)



### Other 2N Neurons ###

# Process non ALPN 2N synapses
other_syn_df = process_synapse_df(other_syn_df, meshes)

# Save the processed dataframes to CSV
other_syn_df.to_csv('./output/other2N_glomeruli_output.csv', index=False)

# Group by root_id and sum glomeruli columns for all three dataframes
other_summed_df = other_syn_df.groupby('pre_pt_root_id').sum(numeric_only=True).reset_index()

# Save the summed dataframes
other_summed_df.to_csv('./output/other2N_glomeruli_output_sum.csv', index=False)



###  Olfactory 3N Neurons  ###

# Process olfactory synapses
olf3N_syn_df = process_synapse_df(olf3N_syn_df, meshes)

# Save the processed dataframes to CSV
olf3N_syn_df.to_csv('./output/olfactory3N_glomeruli_output.csv', index=False)

# Group by root_id and sum glomeruli columns for all three dataframes
olf3N_summed_df = olf3N_syn_df.groupby('pre_pt_root_id').sum(numeric_only=True).reset_index()

# Display the resulting dataframes
olf3N_summed_df.head()

# Save the summed dataframes
olf3N_summed_df.to_csv('./output/olfactory3N_glomeruli_output_sum.csv', index=False)



