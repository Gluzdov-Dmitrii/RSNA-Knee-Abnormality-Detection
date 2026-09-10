"""Keep only the UID of the58 officially labeled studies; never export reports."""
import argparse
from pathlib import Path
import pandas as pd
from common import TARGETS,UID

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); frame=pd.read_csv(a.train,usecols=[UID]+TARGETS,dtype={UID:str})
    gold=frame.loc[frame[TARGETS].notna().any(axis=1),[UID]]
    if len(gold)!=58 or gold[UID].duplicated().any(): raise ValueError('Gold study contract differs')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); gold.to_csv(a.out,index=False)
    print('Exported58 UIDs only; no reports or diagnosis values.')
