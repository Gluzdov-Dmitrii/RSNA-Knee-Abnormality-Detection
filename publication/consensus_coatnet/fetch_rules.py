"""Read official RSNA pages through explicit SDK arguments (CLI parser workaround)."""
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__=='__main__':
    api=KaggleApi(); api.authenticate()
    pages=api.competition_list_pages('rsna-knee-abnormality-detection')
    result=[{'name':p.name,'content':p.content} for p in pages]
    dest=Path(__file__).resolve().parents[2]/'artifacts/consensus_coatnet/official_pages.json'
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([{'name':p['name'],'characters':len(p['content'])} for p in result]))
