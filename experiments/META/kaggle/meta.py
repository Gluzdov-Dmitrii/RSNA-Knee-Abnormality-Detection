# Collect a few allowlisted DICOM header fields: one file per series, train + visible test.
import os, glob, json, time
import pandas as pd, pydicom
from concurrent.futures import ThreadPoolExecutor
ROOT = next(p for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                        '/kaggle/input/rsna-knee-abnormality-detection') if os.path.exists(p + '/train.csv'))
KEYS = ['PatientID', 'PatientSex', 'PatientAge', 'PatientWeight', 'Laterality', 'Manufacturer', 'ManufacturerModelName',
        'MagneticFieldStrength', 'SoftwareVersions', 'ReceiveCoilName', 'SeriesDescription', 'BodyPartExamined',
        'StudyDate', 'SeriesDate', 'AcquisitionDate', 'StudyTime', 'InstitutionName', 'StationName', 'FrameOfReferenceUID']
def one(args):
    split, st, se = args
    d = os.path.join(ROOT, split + '_series', st, se)
    try:
        f = next(os.scandir(d)).path
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        r = {k: str(getattr(ds, k, '')) for k in KEYS}
    except Exception as e:
        r = {'error': repr(e)[:200]}
    r.update(split=split, StudyInstanceUID=st, SeriesInstanceUID=se)
    return r
jobs = []
for split in ('train', 'test'):
    s = pd.read_csv(f'{ROOT}/{split}_series.csv', dtype=str)
    jobs += [(split, a, b) for a, b in zip(s.StudyInstanceUID, s.SeriesInstanceUID)]
t = time.time()
with ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(one, jobs))
df = pd.DataFrame(rows)
df.to_csv('/kaggle/working/series_meta.csv', index=False)
print('rows', len(df), 'seconds', round(time.time() - t), 'errors', df.get('error', pd.Series(dtype=str)).notna().sum())
st = df.groupby(['split', 'StudyInstanceUID']).agg(pid=('PatientID', 'first'), npid=('PatientID', 'nunique')).reset_index()
print('studies', len(st), 'studies with >1 PatientID', (st.npid > 1).sum())
tr = st[st.split == 'train']; te = st[st.split == 'test']
c = tr.pid.value_counts()
print('train patients', len(c), 'with >=2 studies', (c >= 2).sum(), 'studies in multi-study patients', c[c >= 2].sum())
print('visible test pids in train:', [p in set(tr.pid) for p in te.pid])
for k in KEYS:
    print(k, df[k].replace('', pd.NA).notna().mean().round(3), df[k].nunique())
