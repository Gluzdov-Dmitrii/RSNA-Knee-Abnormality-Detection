"""Private CPU-only Kaggle environment probe; no training or submission."""
import importlib.metadata as meta
import json
from pathlib import Path
import platform
import sys
import pandas as pd

versions={}
for name in ['torch','torchvision','timm','numpy','pandas','scipy','pydicom','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg','pyjpegls','huggingface-hub','safetensors']:
    try: versions[name]=meta.version(name)
    except meta.PackageNotFoundError: versions[name]=None
root=next(p for p in Path('/kaggle/input').rglob('rsna-knee-abnormality-detection') if (p/'test.csv').exists())
test=pd.read_csv(root/'test.csv'); series=pd.read_csv(root/'test_series.csv')
report=dict(python=sys.version,platform=platform.platform(),versions=versions,
    test_count=len(test),test_columns=test.columns.tolist(),series_columns=series.columns.tolist(),
    sample_columns=pd.read_csv(root/'sample_submission.csv').columns.tolist(),dicom=[])
try:
    import pydicom
    for row in series.head(3).itertuples(index=False):
        path=next((root/'test_series'/row.StudyInstanceUID/row.SeriesInstanceUID).glob('*.dcm'))
        ds=pydicom.dcmread(path)
        item=dict(transfer_syntax=str(ds.file_meta.TransferSyntaxUID))
        try: item.update(shape=list(ds.pixel_array.shape),decode='PASS')
        except Exception as e: item.update(decode='FAIL',error=str(e))
        report['dicom'].append(item)
except Exception as e: report['dicom_error']=str(e)
Path('/kaggle/working/runtime.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
