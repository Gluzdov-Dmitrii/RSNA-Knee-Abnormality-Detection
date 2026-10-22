"""Readable HTML preview of the exact notebook; no code is executed."""
import html
import json
from pathlib import Path
from markdown_it import MarkdownIt

ROOT=Path(__file__).resolve().parent
nb=json.loads((ROOT/'kaggle/rsna-knee-v34-documented-strict.ipynb').read_text(encoding='utf-8'))
md=MarkdownIt('commonmark').enable('table')
body=[]
toc=[]
for c in nb['cells']:
    source=''.join(c['source'])
    anchor=c['id']
    if c['cell_type']=='markdown':
        for line in source.splitlines():
            if line.startswith('## '):
                toc.append(f'<a href="#{anchor}">{html.escape(line[3:])}</a>')
                break
        body.append(f'<section id="{anchor}">{md.render(source)}</section>')
    else:
        title='Python · '+anchor+' · '+str(len(source.splitlines()))+' строк'
        body.append(f'<details><summary>{title}</summary><pre><code>{html.escape(source)}</code></pre></details>')
page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>RSNA Knee — V34</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f6f8fa;color:#18212c;font:16px/1.65 "Segoe UI",Arial,sans-serif}
nav{position:fixed;inset:0 auto 0 0;width:285px;overflow:auto;background:#122d3c;color:#eef8fb;padding:24px 18px}
nav strong{display:block;font-size:20px;margin-bottom:18px}nav a{display:block;color:#cde1e8;text-decoration:none;font-size:13px;padding:6px 8px;border-radius:5px}nav a:hover{background:#284758}
main{max-width:1230px;margin-left:285px;padding:36px 54px 100px}h1{font-size:32px;line-height:1.25;color:#102d3e}h2{font-size:23px;line-height:1.3;margin-top:36px}a{color:#14677d}p{max-width:900px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{border:1px solid #d9e3e9;padding:10px 13px;text-align:left}th{background:#e8f1f5}code{font:13px/1.6 Consolas,monospace;background:#eaf0f3;padding:2px 4px}pre{overflow:auto;background:#f0f4f6;padding:18px;border-radius:7px}pre code{padding:0;background:none}details{border:1px solid #ccdce5;border-radius:8px;background:white;margin:14px 0 28px}summary{padding:12px 18px;cursor:pointer;color:#24586b;font-weight:600}details pre{margin:0;max-height:620px;border-radius:0 0 8px 8px}.badge{font-size:13px;color:#527182;border-bottom:1px solid #ccdce5;padding-bottom:16px}section{scroll-margin-top:24px}@media(max-width:900px){nav{display:none}main{margin:0;padding:24px}}
</style><nav><strong>RSNA Knee · V34</strong>'''+''.join(toc)+'''</nav><main><div class="badge">Предпросмотр notebook · Код раскрывается по клику · GPU-проверка и score: см. RESULT.json</div>'''+''.join(body)+'''</main></html>'''
(ROOT/'preview.html').write_text(page,encoding='utf-8')
print('Preview saved:',ROOT/'preview.html')
