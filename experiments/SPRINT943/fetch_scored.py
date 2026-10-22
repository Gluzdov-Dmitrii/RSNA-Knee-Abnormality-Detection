"""Download the immutable scored V34 export (the CLI historical pull failed)."""
import hashlib
import json
from pathlib import Path
import requests

VERSION = 351665011
URL = f"https://www.kaggle.com/kernels/scriptcontent/{VERSION}/download"
EXPECTED_SHA = "a025fcd5aa25373ac380f60746ba5ca17e70e97f714fdc17e01ce414ec2163b5"

def receipt(content):
    digest = hashlib.sha256(content).hexdigest()
    assert digest == EXPECTED_SHA, f"Pinned source changed: {digest}"
    return {"url": URL, "script_version_id": VERSION, "version": 34,
            "sha256": digest, "bytes": len(content), "author_public_score": 0.943,
            "score_verified_on_page_utc": "2026-09-22", "score_is_ours": False,
            "author_visible_submission_sha256": "8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a"}

if __name__ == "__main__":
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    assert len(response.json()["cells"]) == 24
    r = receipt(response.content)
    root = Path(__file__).parent / "source_v34"
    root.mkdir(parents=True, exist_ok=True)
    (root / "bend-the-knee-to-speedy-raptors-the-original.ipynb").write_bytes(response.content)
    (root / "fetch_receipt.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(json.dumps(r))
