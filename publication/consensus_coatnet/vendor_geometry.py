"""Freeze only report-free geometry functions from the verified local pipeline."""
import ast
from pathlib import Path

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    original=here.parents[1]/'experiments/cache_budget/recheck/pipeline.py'
    source=original.read_text(encoding='utf-8'); tree=ast.parse(source)
    names={'finite_vector','order_files','sample_indices','load_series','render_slot'}
    parts=[ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    text='''"""Frozen report-free geometry; credit Steven Lee, Apache2.0, see NOTICE.md.

Adapted from experiments/cache_budget/recheck/pipeline.py. Only directory selection
is generalized for live test data; numeric cache preprocessing is unchanged.
"""
import numpy as np
import pydicom
from scipy.ndimage import zoom
SLOTS=[("SAG_FLUID","Sagittal",1),("COR_FLUID","Coronal",1),("AX_FLUID","Axial",1),
       ("SAG_STRUCT","Sagittal",0),("COR_STRUCT","Coronal",0),("AX_STRUCT","Axial",0)]

'''+ '\n\n\n'.join(parts)+'\n'
    text=text.replace('def load_series(root, frame, audit, configs):','def load_series(root, frame, audit, configs, series_directory="train_series"):')
    text=text.replace('root / "train_series" / row.StudyInstanceUID','root / series_directory / row.StudyInstanceUID')
    (here/'geometry.py').write_text(text,encoding='utf-8')
