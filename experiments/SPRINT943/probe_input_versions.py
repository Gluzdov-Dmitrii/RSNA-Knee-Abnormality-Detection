"""Small read-only API provenance probes; never fetch dataset archives."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

ROOT = Path(__file__).resolve().parent

def dataset_probe(row):
    url = f"https://www.kaggle.com/api/v1/datasets/download/{row['slug']}?datasetVersionNumber={row['current_version_number']}"
    r = requests.get(url, stream=True, allow_redirects=False, timeout=25)
    result = {"slug": row["slug"], "version": row["current_version_number"], "status": r.status_code,
              "location_without_query": r.headers.get("Location", "").split("?")[0]}
    r.close()
    return result

def kernel_probe(item):
    slug, version_id = item
    api = KaggleApi(); api.authenticate()
    req = ApiGetKernelRequest(); req.user_name, req.kernel_slug = slug.split("/")
    with api.build_kaggle_client() as client:
        result = client.kernels.kernels_api_client.get_kernel(req)
    current = result.blob.source
    url = f"https://www.kaggle.com/kernels/scriptcontent/{version_id}/download"
    r = requests.get(url, timeout=35); r.raise_for_status()
    exact = r.content.decode("utf-8")
    a=json.loads(current); b=json.loads(exact)
    cells_a=[(c['cell_type'], ''.join(c['source'])) for c in a.get('cells',[])]
    cells_b=[(c['cell_type'], ''.join(c['source'])) for c in b.get('cells',[])]
    return {"slug": slug, "kernel_version_id": version_id,
            "current_version_number": result.metadata.current_version_number,
            "source_content_exact_match": current == exact,
            "source_content_json_match": json.loads(current)==json.loads(exact) if current.strip().startswith('{') and exact.strip().startswith('{') else None,
            "cell_source_match": cells_a == cells_b,
            "exact_metadata": {k:v for k,v in b.get('metadata',{}).items() if k in ('kaggle','papermill')},
            "exact_sha256": hashlib.sha256(r.content).hexdigest(),
            "sdk_source_sha256": hashlib.sha256(current.encode()).hexdigest(),
            "last_run_time": str(result.metadata.last_run_time), "source_download": url}

if __name__ == "__main__":
    rows=json.loads((ROOT/'input_metadata_evidence.json').read_text())['datasets']
    with ThreadPoolExecutor(max_workers=4) as pool:
        datasets=list(pool.map(dataset_probe, rows))
    with ThreadPoolExecutor(max_workers=2) as pool:
        kernels=list(pool.map(kernel_probe, [("sofiaanjenje/rsna-knee-e11-train",342671664),("sofiaanjenje/rsna-knee-e13-train",342849430)]))
    (ROOT/'input_version_evidence.json').write_text(json.dumps({'datasets': datasets, 'kernels': kernels},indent=2))
    print(json.dumps({'datasets': len(datasets),'kernels': [{k:v for k,v in x.items() if k!='exact_metadata'} for x in kernels]},indent=2))
