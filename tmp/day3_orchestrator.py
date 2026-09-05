"""Day-3 open-code pipeline: infer all five, then submit once each."""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(r"C:\Users\Dmitry\Desktop\Kaggle\RSNA Knee Abnormality Detection")
KAGGLE = Path(r"C:\Users\Dmitry\.venvs\kg\Scripts\kaggle.exe")
STATE_PATH = ROOT / "tmp" / "day3_state.json"
LOG_PATH = ROOT / "tmp" / "day3_orchestrator.log"
SLUG = "rsna-knee-abnormality-detection"
SLEEP_S = 60
MAX_HOURS = 10

JOBS = [
    {
        "name": "prvsiyan",
        "folder": str(ROOT / "tmp" / "ready_d3" / "d3-prvsiyan"),
        "kernel": "dmitriigluzdov/rsna-knee-open-rsna-base-0936",
        "message": "open fork prvsiyan/head-and-shoulders-knees-and-toes public 0.936",
    },
    {
        "name": "lynn",
        "folder": str(ROOT / "tmp" / "ready_d3" / "d3-lynn"),
        "kernel": "dmitriigluzdov/rsna-knee-take-care-of-your-knee",
        "message": "open fork lynnsakurai/rsna-knee-dinosaur-v4-5-validated-repro public 0.936",
    },
    {
        "name": "maverick",
        "folder": str(ROOT / "tmp" / "ready_d3" / "d3-maverick"),
        "kernel": "dmitriigluzdov/rsna-open-sol-v6",
        "message": "open fork maverickss26/rsna-knee-restructured-version-2 public 0.936",
        "backup_folder": str(ROOT / "tmp" / "ready_d3" / "d3-dino-v5"),
        "backup_message": "open fork hyakumanben2025/rsna-knee-dinosaur-v5-repro public 0.936",
    },
    {
        "name": "junaid",
        "folder": str(ROOT / "tmp" / "ready_d3" / "d3-junaid"),
        "kernel": "dmitriigluzdov/rsna-open-sol-fusion",
        "message": "open fork junaid512/rsna-base-enc public 0.936",
    },
    {
        "name": "nishant_v55",
        "folder": str(ROOT / "tmp" / "ready_d3" / "d3-nishant-v55"),
        "kernel": "dmitriigluzdov/rsna-open-sol-raptor",
        "message": "open fork nishantkharga/rsna-knee-full-4-arm-ensemble-v55 public 0.935",
        "backup_folder": str(ROOT / "tmp" / "ready_d3" / "d3-sankurero"),
        "backup_message": "open fork sankurero/rsna-knee-coatnet-raptor-only public 0.935",
    },
]


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state = {
        "jobs": {
            j["name"]: {
                "pushed": False,
                "used_backup": False,
                "version": None,
                "status": None,
                "has_csv": False,
                "submitted": False,
                "submit_ref": None,
                "error": None,
                "origin_message": j["message"],
            }
            for j in JOBS
        }
    }
    save_state(state)
    return state


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_kaggle(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(KAGGLE), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def kernel_status(kernel: str) -> str:
    proc = run_kaggle(["kernels", "status", kernel])
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r'has status "([^"]+)"', out)
    if m:
        return m.group(1)
    return out.strip() or f"exit={proc.returncode}"


def kernel_has_submission_csv(kernel: str) -> bool:
    proc = run_kaggle(["kernels", "files", kernel, "--format", "json"])
    if proc.returncode != 0:
        return False
    try:
        files = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return False
    names = {item.get("name") for item in files if isinstance(item, dict)}
    return "submission.csv" in names


def remaining_today() -> int | None:
    proc = run_kaggle(["competitions", "submission-limits", "-c", SLUG])
    text = proc.stdout or ""
    m = re.search(r"Remaining today:\s*(\d+)", text)
    if not m:
        log("limits parse failed: " + text.strip()[:400])
        return None
    return int(m.group(1))


def list_submissions() -> str:
    proc = run_kaggle(["competitions", "submissions", "-c", SLUG], timeout=180)
    return (proc.stdout or "") + (proc.stderr or "")


def push_folder(api: KaggleApi, folder: str) -> tuple[bool, int | None, str]:
    result = api.kernels_push(folder)
    if result is None:
        return False, None, "empty push result"
    err = getattr(result, "error", None)
    if err:
        return False, None, str(err)
    version = getattr(result, "versionNumber", None)
    if version is None:
        return False, None, "push succeeded but version missing"
    return True, int(version), f"version {version}"


def all_inferences_terminal(state: dict) -> bool:
    for job in JOBS:
        st = state["jobs"][job["name"]]
        if st["submitted"]:
            continue
        if st["has_csv"] and st["version"]:
            continue
        if st.get("error") == "permanent":
            continue
        return False
    return True


def any_ready_unsubmitted(state: dict) -> bool:
    return any(
        st["has_csv"] and st["version"] and not st["submitted"]
        for st in state["jobs"].values()
    )


def submit_job(state: dict, job: dict) -> None:
    st = state["jobs"][job["name"]]
    if st["submitted"]:
        return
    if not st["has_csv"] or not st["version"]:
        return
    left = remaining_today()
    if left is None:
        log(f"{job['name']}: cannot read quota, skip submit this cycle")
        return
    if left <= 0:
        log(f"{job['name']}: no quota remaining")
        return
    msg = st.get("origin_message") or job["message"]
    log(f"SUBMIT {job['name']} kernel={job['kernel']} v={st['version']} left={left}")
    proc = run_kaggle(
        [
            "competitions",
            "submit",
            SLUG,
            "-f",
            "submission.csv",
            "-k",
            job["kernel"],
            "-v",
            str(st["version"]),
            "-m",
            msg,
        ],
        timeout=180,
    )
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    log(f"submit stdout: {out[:800]}")
    st["submitted"] = True
    st["submit_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    listed = list_submissions()
    log("submissions after submit:\n" + listed)
    m = re.search(r"^\s*(\d+)\s+submission\.csv", listed, re.M)
    if m:
        st["submit_ref"] = m.group(1)
        save_state(state)


def maybe_push(api: KaggleApi, state: dict) -> None:
    for job in JOBS:
        st = state["jobs"][job["name"]]
        if st["pushed"] or st["submitted"] or st.get("error") == "permanent":
            continue
        folder = st.get("folder_override") or job["folder"]
        ok, version, detail = push_folder(api, folder)
        log(f"push {job['name']} -> ok={ok} {detail}")
        if ok:
            st["pushed"] = True
            st["version"] = version
            st["status"] = "KernelWorkerStatus.QUEUED"
            st["push_at"] = datetime.now(timezone.utc).isoformat()
            save_state(state)
            return
        if "Maximum batch GPU" in detail:
            return
        st["last_push_error"] = detail
        save_state(state)
        return


def refresh_statuses(state: dict) -> None:
    for job in JOBS:
        st = state["jobs"][job["name"]]
        if not st["pushed"] or st["submitted"] or st.get("error") == "permanent":
            continue
        status = kernel_status(job["kernel"])
        st["status"] = status
        log(f"status {job['name']} v={st['version']} {status}")
        if "COMPLETE" in status:
            if not st.get("saw_running"):
                log(f"{job['name']} ignoring stale COMPLETE until new session starts")
            else:
                has = kernel_has_submission_csv(job["kernel"])
                st["has_csv"] = has
                log(f"files {job['name']} submission.csv={has}")
                if not has:
                    st["error"] = "complete_without_csv"
        elif "ERROR" in status:
            if st.get("saw_running"):
                st["error"] = "run_error"
                log(f"{job['name']} ERROR after running")
                if job.get("backup_folder") and not st.get("used_backup"):
                    log(f"{job['name']} switching to backup")
                    st["used_backup"] = True
                    st["pushed"] = False
                    st["version"] = None
                    st["has_csv"] = False
                    st["folder_override"] = job["backup_folder"]
                    st["origin_message"] = job["backup_message"]
                    st["error"] = None
                    st["saw_running"] = False
                else:
                    st["error"] = "permanent"
            else:
                log(f"{job['name']} still showing previous ERROR, waiting for new session")
        elif any(x in status for x in ("RUNNING", "QUEUED")):
            st["saw_running"] = True
        save_state(state)


def done(state: dict) -> bool:
    for job in JOBS:
        st = state["jobs"][job["name"]]
        if st["submitted"]:
            continue
        if st.get("error") == "permanent":
            continue
        return False
    return True


def main() -> int:
    log("orchestrator start")
    api = KaggleApi()
    api.authenticate()
    state = load_state()
    deadline = time.time() + MAX_HOURS * 3600
    while time.time() < deadline:
        refresh_statuses(state)
        if done(state):
            log("all remaining jobs submitted or permanent")
            return 0
        if all_inferences_terminal(state) and any_ready_unsubmitted(state):
            for job in JOBS:
                submit_job(state, job)
            continue
        maybe_push(api, state)
        if done(state):
            log("all remaining jobs submitted or permanent")
            return 0
        time.sleep(SLEEP_S)
    log("timeout; submitting any completed artifacts")
    for job in JOBS:
        submit_job(state, job)
    return 2 if not done(state) else 0


if __name__ == "__main__":
    raise SystemExit(main())
