"""Download the public 2,200-row Crop Recommendation dataset."""
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "Crop_recommendation.csv"
URL = "https://huggingface.co/datasets/liaad/agricultural-data/resolve/main/Crop_recommendation.csv"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 10_000:
        print(f"Dataset already exists: {DEST}")
        return
    print("Downloading dataset...")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    if b"N,P,K" not in response.content[:500]:
        raise ValueError("Downloaded file does not look like the expected CSV.")
    DEST.write_bytes(response.content)
    print(f"Saved {len(response.content):,} bytes to {DEST}")

if __name__ == "__main__":
    main()
