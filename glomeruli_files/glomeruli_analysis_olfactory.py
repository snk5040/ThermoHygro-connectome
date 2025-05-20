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
olf3N_syn_df = synOlf_df[synOlf_df['post_pt_root_id'].isin(olf3N_ids['root_id'])]

# Initialize new columns for each glomerulus with 0s, for all three modalities
# Create copies once before the loop
olf3N_syn_df = olf3N_syn_df.copy()

for glom in glom_names:
    olf3N_syn_df[glom] = 0


# Function to process a single row
def process_row(idx, row, meshes):
    result = {glom_name: 0 for glom_name in meshes.keys()}  # Initialize all glomerulus columns to 0
    try:
        point = row[['post_pt_position_x', 'post_pt_position_y', 'post_pt_position_z']].values
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

    
###  Olfactory 3N Neurons  ###

# Process olfactory synapses
olf3N_syn_df = process_synapse_df(olf3N_syn_df, meshes)

# Save the processed dataframes to CSV
olf3N_syn_df.to_csv('./output/olfactory3N_glomeruli_input.csv', index=False)

# Group by root_id and sum glomeruli columns for all three dataframes
olf3N_summed_df = olf3N_syn_df.groupby('pre_pt_root_id').sum(numeric_only=True).reset_index()

# Display the resulting dataframes
olf3N_summed_df.head()

# Save the summed dataframes
olf3N_summed_df.to_csv('./output/olfactory3N_glomeruli_input_sum.csv', index=False)



