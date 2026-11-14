"""Build pinned local demo bundles with exact source and artifact provenance.

Install dependencies with ``npm ci``. Release builds require a clean Git checkout;
development builds disclose changes in their provenance. Node 20+ is required;
set PAPER2BENCH_NODE when the default node executable is older.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_SUFFIXES = {".html", ".css", ".js", ".jsx", ".json", ".zip", ".svg", ".png", ".webp", ".ico", ".txt", ".yaml", ".yml"}
BUILD_INPUTS = ("package.json", "package-lock.json", "scripts/build_demo.py", "scripts/build_demo.mjs",
                "deployment/demo/.openai/hosting.json")
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024


class DemoBuildError(ValueError):
    pass


def _git(*arguments):
    result = subprocess.run(["git", "-C", str(ROOT), *arguments], check=True,
                            capture_output=True, timeout=20)
    return result.stdout


def _identity(*, exclude=None):
    if Path(_git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT.resolve():
        raise DemoBuildError("Demo source must have its own Git repository")
    commit = _git("rev-parse", "HEAD").decode().strip()
    tree = _git("rev-parse", "HEAD^{tree}").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise DemoBuildError("Demo requires full GitHub commit and tree identities")
    status_args = ["status", "--porcelain=v1", "--untracked-files=normal"]
    if exclude is not None and exclude.is_relative_to(ROOT):
        # This invocation's private build scratch is an output, never source.
        # Exclude only its exact name; unrelated concurrent edits remain visible.
        status_args += ["--", ".", ":(top,literal,exclude)" + exclude.relative_to(ROOT).as_posix()]
    return {"repository": "https://github.com/cal2020/Paper2Bench", "commit": commit, "tree": tree,
            "dirty": bool(_git(*status_args).strip())}


def _read(relative: str):
    if not isinstance(relative, str) or Path(relative).is_absolute() or "\\" in relative:
        raise DemoBuildError("Demo source paths must be normalized repository-relative names")
    if Path(relative).as_posix() != relative:
        raise DemoBuildError("Demo source paths must be normalized repository-relative names")
    path = ROOT
    for part in Path(relative).parts:
        if part in {".", ".."} or part.startswith(".") and relative != BUILD_INPUTS[-1]:
            raise DemoBuildError("Hidden or escaping demo source path")
        path = path / part
        if path.is_symlink():
            raise DemoBuildError("Demo sources cannot be symlinks")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise DemoBuildError("Demo sources must be regular files")
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise DemoBuildError("Demo source exceeds its byte limit")
    return raw


def _sources(*, release):
    names = set(_git("ls-files", "-z", "--", "docs").decode().split("\0")) - {""}
    if not release:
        names.update(set(_git("ls-files", "--others", "--exclude-standard", "-z", "--", "docs").decode().split("\0")) - {""})
    names = {name for name in names if Path(name).suffix in ASSET_SUFFIXES and
             not any(part.startswith(".") for part in Path(name).parts)}
    files = {name: _read(name) for name in sorted(names | set(BUILD_INPUTS))}
    if sum(map(len, files.values())) > MAX_TOTAL_BYTES:
        raise DemoBuildError("Demo sources exceed their aggregate byte limit")
    if not {"docs/index.html", "docs/example.html"}.issubset(files):
        raise DemoBuildError("Demo needs both application HTML pages")
    return files


def _node():
    executable = os.environ.get("PAPER2BENCH_NODE") or shutil.which("node")
    if not executable:
        raise DemoBuildError("Node 20+ is required; install pinned dependencies with npm ci")
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True, timeout=10)
    version = result.stdout.strip()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version) or int(version.split(".")[0][1:]) < 20:
        raise DemoBuildError("Node 20+ is required; select it with PAPER2BENCH_NODE")
    return executable, version


def _admit_output(output):
    if output.is_symlink():
        raise DemoBuildError("Demo output cannot be a symlink")
    output = output.resolve()
    source = ROOT / "docs"
    if output == source or output.is_relative_to(source):
        raise DemoBuildError("Demo output must be outside docs")
    for target in (output / "dist", output / ".openai", output / ".openai/hosting.json"):
        if target.is_symlink():
            raise DemoBuildError("Demo output components cannot be symlinks")
    if (output / "dist").exists() and not (output / "dist").is_dir():
        raise DemoBuildError("Demo dist must be a directory")
    if (output / ".openai").exists() and not (output / ".openai").is_dir():
        raise DemoBuildError("Demo hosting directory must be a directory")
    if (output / ".openai/hosting.json").exists() and not (output / ".openai/hosting.json").is_file():
        raise DemoBuildError("Demo hosting configuration must be a regular file")
    return output


def build(output: Path, *, release=False) -> dict:
    output = _admit_output(Path(output))
    identity = _identity()
    if release and identity["dirty"]:
        raise DemoBuildError("Release demo builds require a clean Git checkout; commit changes first")
    files = _sources(release=release)
    manifest_raw = files[BUILD_INPUTS[-1]]
    manifest = json.loads(manifest_raw)
    if (manifest.get("static") != {"directory": "dist"} or
            not isinstance(manifest.get("project_id"), str) or not manifest["project_id"].startswith("appgprj_")):
        raise DemoBuildError("Expected the tracked static Site identity and dist configuration")
    existing_hosting = output / ".openai/hosting.json"
    if existing_hosting.exists() and json.loads(existing_hosting.read_text()).get("project_id") != manifest["project_id"]:
        raise DemoBuildError("Refusing to replace a different Site identity")
    if (output / "dist").exists() and not existing_hosting.exists():
        raise DemoBuildError("Refusing to replace an existing dist without this Site's hosting identity")
    package = json.loads(files["package.json"])
    lock = json.loads(files["package-lock.json"])
    dependencies = package.get("devDependencies")
    if (not isinstance(dependencies, dict) or set(dependencies) != {"esbuild", "react", "react-dom"} or
            dependencies != lock.get("packages", {}).get("", {}).get("devDependencies")):
        raise DemoBuildError("Demo build dependencies differ from their lockfile")
    for name, version in dependencies.items():
        if (not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) or
                lock["packages"].get("node_modules/" + name, {}).get("version") != version):
            raise DemoBuildError("Demo build dependencies must be exact locked versions")
        installed = ROOT / "node_modules" / name / "package.json"
        if not installed.is_file() or json.loads(installed.read_text()).get("version") != version:
            raise DemoBuildError("Install the exact demo build dependencies with npm ci")
    node, node_version = _node()
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".demo-build-", dir=output))
    backup = None
    try:
        snapshot, stage = staging / "source", staging / "dist"
        stage.mkdir()
        for name, raw in files.items():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            # Application source is compiled below. Other public docs remain inspectable.
            if (name.startswith("docs/") and Path(name).suffix != ".jsx" and
                    name not in {"docs/data.js", "docs/release-data.js"}):
                target = stage / Path(name).relative_to("docs")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        subprocess.run([node, str(ROOT / "scripts/build_demo.mjs"), "--source", str(snapshot), "--output", str(stage)],
                       check=True, timeout=120, cwd=ROOT)
        source_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        provenance = {"schema_version": 1, "kind": "paper2bench-static-demo",
                      "source": dict(identity, files=source_hashes),
                      "build": {"mode": "release" if release else "development", "node": node_version,
                                "esbuild": dependencies["esbuild"], "react": dependencies["react"],
                                "react_dom": dependencies["react-dom"],
                                "package_lock_sha256": source_hashes["package-lock.json"]},
                      "hosting": {"project_id": manifest["project_id"],
                                  "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest()}}
        (stage / "build-provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
        payloads = {path.relative_to(stage).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(stage.rglob("*")) if path.is_file()}
        (stage / "asset-manifest.json").write_text(json.dumps({"schema_version": 1, "files": payloads}, sort_keys=True, indent=2) + "\n")
        # No source may change while this supposedly exact artifact is assembled.
        if any(_read(name) != raw for name, raw in files.items()) or _identity(exclude=staging) != identity:
            raise DemoBuildError("Demo source changed during build; rebuild from a stable checkout")
        if release and _sources(release=True).keys() != files.keys():
            raise DemoBuildError("Tracked demo source set changed during build")
        destination = output / "dist"
        if destination.exists():
            backup = staging / "previous-dist"
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup:
                backup.rename(destination)
            raise
        hosting = output / ".openai/hosting.json"
        hosting.parent.mkdir(parents=True, exist_ok=True)
        temp_hosting = staging / "hosting.json"
        temp_hosting.write_bytes(manifest_raw)
        temp_hosting.replace(hosting)
        return {"assets": len(payloads), "output": str(destination), "project_id": manifest["project_id"],
                "source_commit": identity["commit"], "source_tree": identity["tree"], "source_dirty": identity["dirty"]}
    finally:
        shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/demo")
    parser.add_argument("--release", action="store_true", help="Require exact clean committed source")
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.output, release=args.release)))
    except (DemoBuildError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Demo build failed: {error}\n")
