"""Build the static demo from tracked docs; never include runtime receipts or keys."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_SUFFIXES = {".html", ".css", ".js", ".jsx", ".json", ".zip", ".svg", ".png", ".webp", ".ico"}


def build(output: Path) -> dict:
    source = ROOT / "docs"
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Demo output must be outside docs")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "deployment/demo/.openai/hosting.json").read_text())
    if manifest.get("static") != {"directory": "dist"}:
        raise ValueError("Expected the tracked static Site manifest")
    stage = Path(tempfile.mkdtemp(prefix=".demo-build-", dir=output))
    files = {}
    try:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if path.is_symlink():
                raise ValueError("Demo assets cannot be symlinks")
            if not path.is_file() or path.suffix not in ASSET_SUFFIXES or any(p.startswith(".") for p in relative.parts):
                continue
            raw = path.read_bytes()
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
        if "index.html" not in files:
            raise ValueError("Demo has no index.html")
        (stage / "asset-manifest.json").write_text(json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n")
        destination = output / "dist"
        if destination.is_symlink():
            raise ValueError("Demo dist cannot be a symlink")
        if destination.exists():
            shutil.rmtree(destination)
        stage.rename(destination)
        hosting = output / ".openai/hosting.json"
        hosting.parent.mkdir(parents=True, exist_ok=True)
        hosting.write_text(json.dumps(manifest, indent=2) + "\n")
        return {"assets": len(files), "output": str(destination), "project_id": manifest["project_id"]}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/demo")
    print(json.dumps(build(parser.parse_args().output)))
