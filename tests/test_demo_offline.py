"""The demo must run offline apart from the one CDN script that loads the model
runtime. This fails if demo/index.html starts fetching anything else."""
import re
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "demo" / "index.html"

LOADS = re.compile(
    r"""(?:\b(?:src|href)\s*=\s*|\bimport\s*\(\s*|\bfrom\s+|\burl\(\s*|@import\s+)["']?(https?://[^"')\s>]+)""",
    re.I,
)


def test_demo_loads_only_the_model_runtime():
    html = DEMO.read_text(encoding="utf-8")
    hosts = {re.match(r"https?://([^/]+)", url).group(1) for url in LOADS.findall(html)}
    assert hosts == {"cdn.jsdelivr.net"}
