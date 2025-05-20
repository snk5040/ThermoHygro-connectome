import gdown
import os

def fetch_from_google_drive(file_id, save_path):
    """
    Downloads a file from Google Drive using gdown.

    Parameters:
        file_id (str): The Google Drive file ID.
        save_path (str): Path where to save the downloaded file.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gdown.download(url, save_path, quiet=False)
    print(f"✅ Downloaded to: {save_path}")

def main():
    cxntome_files_to_download = {
        "neurons_783.csv.gz": "16F8eOKsXFUpqidJqYmvBYZtuep7xXAVi",
        "neuropil_synapse_table_783.csv.gz": "1nhG2g9bGIQMOEPgE_2QXn3YlJBb_RcvE",
        "connections_783.csv.gz": "1tq7bxcTbDNRWHCfD02Dt_z6iBHBDftpR",
        "classification_783.csv.gz": "1L0AmHOqbvXWrX7aX45b5Fg3GOfsU6_rO",
    }

    cxntome_dir = "./"

    for filename, file_id in cxntome_files_to_download.items():
        save_path = os.path.join(cxntome_dir, filename)
        fetch_from_google_drive(file_id, save_path)

    glomeruli_files_to_download = {
        "filtered_2N_synapses.csv": "1qkJJw7z2mOOTFqqNGo8WUdxEA_vZiF9j",
        "filtered_olfactory_synapses.csv": "1bXJQDJiSEmxKEl-0e8E58WShqoIr8Yir",
    }

    glomeruli_dir = "./glomeruli_files/input/"

    for filename, file_id in glomeruli_files_to_download.items():
        save_path = os.path.join(glomeruli_dir, filename)
        fetch_from_google_drive(file_id, save_path)

if __name__ == "__main__":
    main()