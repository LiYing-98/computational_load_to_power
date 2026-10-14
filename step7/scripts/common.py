"""Phase5 file, hash and boundary contracts."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
STEP7 = ROOT / "step7"
SEED = 20261013
TASKBOOK = Path(r"D:\my_work\cpecc\files\日常事务\科技项目\2025年\算电协同\课题一\阶段性研究进展\plan\first_step\Codex_Phase5_Taskbook.docx")
TARGET = "y_gpu_avg_power_watts"


def sha256(path):
    path = Path(path)
    if path.suffix.lower() in {".json", ".csv", ".py", ".md", ".txt", ".html", ".ipynb"}:
        data = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        return hashlib.sha256(data).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def verify_sources():
    freeze = read_json(STEP7 / "config/source_freeze_phase5.json")
    failures = [x["path"] for x in freeze["inputs"] if not (ROOT / x["path"]).exists() or sha256(ROOT / x["path"]) != x["sha256"]]
    if sha256(TASKBOOK) != freeze["taskbook_sha256"]:
        failures.append("taskbook")
    if failures:
        raise ValueError("frozen source integrity mismatch: " + ", ".join(failures))
    return freeze


def assert_feature_boundary(features, *, task_sensitivity=False):
    forbidden = ("throughput", "latency", "duration", "utilization", "temperature", "clock", "measured_", "actual_", "output_tokens_actual", "outer_fold")
    for column in features:
        if column == "diag_architecture_class":
            continue
        if not column.startswith("x_") or any(x in column.lower() for x in forbidden):
            raise ValueError(f"forbidden formal X: {column}")
        if column == "x_protocol_task" and not task_sensitivity:
            raise ValueError("task label belongs to sensitivity only")
