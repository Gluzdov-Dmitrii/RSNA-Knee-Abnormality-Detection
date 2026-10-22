"""Package the existing global-weight control as the primary Kaggle output."""
import json
from pathlib import Path

source = Path('tmp/public941_push')
destination = Path('tmp/public939_ablation_push')
destination.mkdir(parents=True, exist_ok=True)
metadata = json.loads((source / 'kernel-metadata.json').read_text())
notebook = json.loads((source / metadata['code_file']).read_text())
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
notebook['cells'].append({
    'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
    'source': "from pathlib import Path\nimport shutil\nimport pandas as pd\nimport numpy as np\np = Path('/kaggle/working/submission_0939_parent_exact.csv')\nx = pd.read_csv(p)\ny = pd.read_csv('/kaggle/working/submission.csv')\nassert x.columns.tolist() == y.columns.tolist()\nassert x.iloc[:, 0].tolist() == y.iloc[:, 0].tolist()\nassert np.isfinite(x.iloc[:, 1:].to_numpy()).all()\nshutil.copyfile('/kaggle/working/submission.csv', '/kaggle/working/submission_probe22_reference.csv')\nshutil.copyfile(p, '/kaggle/working/submission.csv')\nprint('PUB939A: global outer weight 0.60; probe22 reference preserved')\n"
})
metadata['id'] = 'dmitriigluzdov/rsna-pub939-global060-ablation'
metadata['title'] = 'rsna-pub939-global060-ablation'
(destination / metadata['code_file']).write_text(json.dumps(notebook), encoding='utf-8')
(destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
