"""
Download EUPDCorp from Zenodo.
DOI: 10.5281/zenodo.15056399
Output: data/raw/EUPDCorp.rds
"""

import requests
from pathlib import Path
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/15056399/files/EUPDCorp_1999-2024_v1.RDS"
OUTPUT_PATH = Path("data/raw/EUPDCorp.rds")


def download(url: str, dest: Path, chunk_size: int = 8192):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved -> {dest}")


def main():
    if OUTPUT_PATH.exists():
        print(f"File already exists at {OUTPUT_PATH}, skipping download.")
        return
    download(ZENODO_URL, OUTPUT_PATH)


if __name__ == "__main__":
    main()
