import os
import requests
import pandas as pd
from pathlib import Path

def download_file(url, dest_folder, filename):
    os.makedirs(dest_folder, exist_ok=True)
    local_path = os.path.join(dest_folder, filename)
    if os.path.exists(local_path):
        print(f"✅ File already exists: {local_path}")
        return local_path
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"⬇️ Downloaded: {local_path}")
    return local_path

def convert_to_parquet(input_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    parquet_name = Path(input_path).stem + ".parquet"
    output_path = os.path.join(output_folder, parquet_name)

    if os.path.exists(output_path):
        print(f"⚠️ Parquet already exists: {output_path} — skipping conversion.")
        return

    try:
        print(f"🧩 Converting: {input_path}")
        df = pd.read_csv(input_path, sep=',', low_memory=False)
        df.to_parquet(output_path, index=False)
        print(f"✅ Converted to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to convert {input_path}: {e}")

if __name__ == "__main__":
    urls = {
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2020/full.csv.gz": "full_2020.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2021/full.csv.gz": "full_2021.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2022/full.csv.gz": "full_2022.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/full.csv.gz": "full_2023.csv.gz",
        "https://files.data.gouv.fr/geo-dvf/latest/csv/2024/full.csv.gz": "full_2024.csv.gz"
    }

    raw_dir = "./data/raw"
    parquet_dir = "./data/parquet"

    for url, filename in urls.items():
        if url.endswith(('.csv', '.gz')):
            local_file = download_file(url, raw_dir, filename)
            convert_to_parquet(local_file, parquet_dir)
        else:
            print(f"⚠️ Skipping unsupported file type: {url}")
