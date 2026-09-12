"""Run the established contracts against the isolated P04 source, not P03."""
import sys
import types
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/wide_window/source'
BASE=ROOT/'publication/consensus_coatnet'
sys.path.insert(0,str(ROOT/'tmp/cache_budget_deps'))
sys.path.insert(0,str(SOURCE))
suite=unittest.TestSuite()
for filename in ['test_contract.py','test_inference.py']:
    text=(BASE/filename).read_text(encoding='utf-8')
    text=text.replace("HERE.parents[1] / 'experiments/cache_budget/recheck/pipeline.py'",
                      f"Path({str(ROOT/'experiments/cache_budget/recheck/pipeline.py')!r})")
    module=types.ModuleType('p04_'+Path(filename).stem)
    module.__file__=str(SOURCE/filename)
    sys.modules[module.__name__]=module
    exec(compile(text,module.__file__,'exec'),module.__dict__)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
import common
assert Path(common.__file__).resolve().parent==SOURCE
assert common.GEOMETRY['window']==[.1,.9]
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
