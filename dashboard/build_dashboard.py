"""Injects outputs/results.json into template.html -> outputs/dashboard.html"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def build():
    tpl = (ROOT / "dashboard" / "template.html").read_text()
    data = (ROOT / "outputs" / "results.json").read_text()
    html = tpl.replace("__DATA_JSON__", data)
    out_path = ROOT / "outputs" / "dashboard.html"
    out_path.write_text(html)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    build()
