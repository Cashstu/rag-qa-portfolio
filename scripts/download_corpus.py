"""Downloads a small, thematically-relevant corpus of arXiv papers into data/raw/.

Run once before local ingestion: `python scripts/download_corpus.py`.
"""

import pathlib
import sys

import requests

PAPERS = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "retrieval_augmented_generation.pdf": "https://arxiv.org/pdf/2005.11401.pdf",
    "bert.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "dense_passage_retrieval.pdf": "https://arxiv.org/pdf/2004.04906.pdf",
    "gpt3_few_shot_learners.pdf": "https://arxiv.org/pdf/2005.14165.pdf",
}

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"


def download_corpus() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in PAPERS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"skip (already present): {filename}")
            continue
        print(f"downloading {filename} ...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
    print(f"done — {len(PAPERS)} papers in {RAW_DIR}")


if __name__ == "__main__":
    try:
        download_corpus()
    except requests.RequestException as exc:
        print(f"failed to download corpus: {exc}", file=sys.stderr)
        sys.exit(1)
