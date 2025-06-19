# This script downloads CSV files from specified URLs and saves them to a local directory.
# It checks if the file already exists before downloading to avoid unnecessary downloads.
# The script handles both regular CSV files and gzipped CSV files.
# It creates the destination folder if it does not exist.
# The script prints messages to indicate the status of each download.
# The script is designed to be run as a standalone program.
# It uses the requests library to handle HTTP requests and file downloads.
import os
import requests

def download_file(url, dest_folder, filename):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    local_filename = os.path.join(dest_folder, filename)
    if os.path.exists(local_filename):
        print(f"File already exists: {local_filename}")
        return local_filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {local_filename}")
    return local_filename

if __name__ == "__main__":
    urls = {
        "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_ventes_68.csv": "ech_annonces_ventes_68.csv",
        "https://raw.githubusercontent.com/klopstock-dviz/immo_vis/master/data/ech_annonces_locations_68.csv": "ech_annonces_locations_68.csv",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2020/full.csv.gz": "full_2020.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2021/full.csv.gz": "full_2021.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2022/full.csv.gz": "full_2022.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/full.csv.gz": "full_2023.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/full.csv.gz": "full_2024.csv.gz"
    }

    dest_folder = "./data/raw"
    for url, filename in urls.items():
        if url.endswith('.csv') or url.endswith('.gz'):
            download_file(url, dest_folder, filename)
        else:
            print(f"Skipping unsupported file type: {url}")