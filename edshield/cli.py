"""Command-line interface.

    edshield extract essay.txt
    edshield redact essay.txt --policy coppa --method mask
    edshield serve --port 8080
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="edshield", description="Local-first student-privacy layer")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="Detect PII and print entities as JSON")
    ex.add_argument("path", help="file path or - for stdin")
    ex.add_argument("--model", default=None, help="model short name, HF id, local dir, or 'rules'")
    ex.add_argument("--device", default="cpu")

    rd = sub.add_parser("redact", help="De-identify a document under a policy")
    rd.add_argument("path")
    rd.add_argument("--policy", default="ferpa")
    rd.add_argument("--method", default="mask", choices=["mask", "replace", "hash", "shift_dates"])
    rd.add_argument("--model", default=None)
    rd.add_argument("--device", default="cpu")
    rd.add_argument("--json", action="store_true", help="emit JSON instead of text")

    sv = sub.add_parser("serve", help="Run the REST service")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)

    pl = sub.add_parser("policies", help="List built-in policies")

    a = p.parse_args(argv)

    if a.cmd == "extract":
        from edshield import extract_pii
        r = extract_pii(_read(a.path), model_name=a.model, device=a.device)
        print(json.dumps(r.to_dict(), indent=2))
    elif a.cmd == "redact":
        from edshield import deidentify
        r = deidentify(_read(a.path), method=a.method, policy=a.policy, model_name=a.model, device=a.device)
        print(json.dumps(r.to_dict(), indent=2) if a.json else r.deidentified_text)
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("edshield.service:app", host=a.host, port=a.port)
    elif a.cmd == "policies":
        from edshield import available_policies
        print("\n".join(available_policies()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
