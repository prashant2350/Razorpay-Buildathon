import json
from pathlib import Path
from datetime import datetime, timezone


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "audit_trail.jsonl"


class AuditLog:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")  # fresh run

    def write(self, record: dict):
        record = {"logged_at": datetime.now(timezone.utc).isoformat(), **record}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
