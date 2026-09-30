"""Upload the trained model and its ONNX export to the Hugging Face Hub.

    hf auth login                       # once, with a token that can write to the organisation
    python training/publish_hub.py --dry-run
    python training/publish_hub.py

Each repository gets the model files and the model card from hub/<name>/README.md.
Only the files listed below are sent: no training arguments, checkpoints or backups.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "spm.model", "added_tokens.json"]
REPOS = {
    "piilo-deberta-v3-small": {
        "folder": "models/piilo-deberta-v3-small-v2",
        "files": TOKENIZER + ["model.safetensors"],
    },
    "piilo-deberta-v3-small-onnx": {
        "folder": "demo/models/piilo-deberta-v3-small-onnx",
        "files": TOKENIZER + ["onnx/model.onnx", "onnx/model_quantized.onnx"],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="edshield")
    ap.add_argument("--root", default=str(ROOT), help="where models/ and demo/models/ are")
    ap.add_argument("--only", choices=list(REPOS), default=None)
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded and stop")
    a = ap.parse_args()

    plan = []
    for name, spec in REPOS.items():
        if a.only and name != a.only:
            continue
        folder = Path(a.root) / spec["folder"]
        card = ROOT / "hub" / name / "README.md"
        files = [f for f in spec["files"] if (folder / f).is_file()]
        missing = [f for f in ("config.json", "tokenizer.json", spec["files"][-1]) if f not in files]
        if missing or not card.is_file():
            raise SystemExit(f"{name}: missing {missing or card} under {folder}")
        plan.append((f"{a.org}/{name}", folder, files, card))
        print(f"{a.org}/{name}")
        print(f"  README.md  <- {card.relative_to(ROOT)}")
        for f in files:
            print(f"  {f:28} {(folder / f).stat().st_size / 1e6:8.1f} MB")
    if a.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print("logged in as", api.whoami()["name"])
    for repo_id, folder, files, card in plan:
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id)
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, allow_patterns=files,
                          commit_message="Upload model files")
        print(f"uploaded https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
