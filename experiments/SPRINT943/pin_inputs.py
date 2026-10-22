"""Read-only input provenance discovery for scored Speedy V34 (no model downloads)."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

ROOT = Path(__file__).resolve().parent
META = ROOT.parents[1] / "tmp/audit_2026-09-22/speedy_raptors/kernel-metadata.json"
SOURCE = ROOT / "source_v34/bend-the-knee-to-speedy-raptors-the-original.ipynb"


def dataset_info(item):
    slug, source_id = item
    api = KaggleApi()
    api.authenticate()
    req = ApiGetDatasetRequest()
    req.owner_slug, req.dataset_slug = slug.split("/")
    with api.build_kaggle_client() as client:
        result = client.datasets.dataset_api_client.get_dataset(req)
    row = {"slug": slug, "dataset_version_id": source_id,
           "dataset_id": result.id, "current_version_number": result.current_version_number,
           "last_updated": str(result.last_updated),
           "versions": [{"number": v.version_number, "date": str(v.creation_date), "notes": v.version_notes}
                        for v in result.versions]}
    response = requests.get(f"https://www.kaggle.com/datasets/{slug}?datasetVersionNumber={result.current_version_number}", timeout=40)
    row["html_status"] = response.status_code
    snippets = []
    for match in re.finditer(r'datasetVersionId|datasetVersionNumber|currentVersionNumber|"versionId"', response.text):
        snippets.append(response.text[max(0, match.start()-60):match.end()+100])
    row["html_version_evidence"] = snippets[:20]
    return row


def main():
    meta = json.loads(META.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    embedded = source["metadata"]["kaggle"]["dataSources"]
    ids = [d["sourceId"] for d in embedded if d["sourceType"] == "datasetVersion"]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(dataset_info, zip(meta["dataset_sources"], ids)):
            rows.append(row)
            print(row["slug"], row["current_version_number"], row["dataset_version_id"], flush=True)
    evidence = {"queried_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": rows}
    (ROOT / "input_metadata_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("Evidence saved")


if __name__ == "__main__":
    main()
