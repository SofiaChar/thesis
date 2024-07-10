import csv
import json
import zipfile

def unzip_dataset(zip_path, extract_to, verbose=True):
    """Unzip the dataset to a specified directory with verbose output."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if verbose:
            print(f"Extracting {zip_path} to {extract_to}...")
        for file in zip_ref.namelist():
            if verbose:
                print(f"Extracting {file}...")
            zip_ref.extract(file, extract_to)
    return extract_to

def save_to_csv(flattened_data, filename='/valohai/outputs/hcc_data.csv'):
    # Get all the keys from the first entry for the CSV header
    keys = flattened_data[0].keys()

    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(flattened_data)

    metadata = {"valohai.alias": "hcc_data"}

    metadata_path = '/valohai/outputs/hcc_data.csv.metadata.json'
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile)
