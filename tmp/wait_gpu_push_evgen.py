import subprocess
import time
from datetime import datetime

kaggle = r"C:\Users\Dmitry\.venvs\kg\Scripts\kaggle.exe"
folder = r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection\tmp\ready_d2\d2-evgen"

for i in range(40):
    print(datetime.now().strftime("%H:%M:%S"), "try", i + 1, flush=True)
    proc = subprocess.run(
        [kaggle, "kernels", "push", "-p", folder],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.strip(), flush=True)
    if "successfully pushed" in out.lower():
        raise SystemExit(0)
    if "Maximum batch GPU" not in out:
        raise SystemExit(proc.returncode or 1)
    time.sleep(120)
print("still blocked")
raise SystemExit(2)
