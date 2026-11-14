"""Validate a committed static demo and make a deterministic, credential-free tar."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/cal2020/Paper2Bench"
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILES = 512
SOURCE_SUFFIXES = {".html", ".css", ".js", ".jsx", ".json", ".zip", ".svg", ".png", ".webp", ".ico", ".txt", ".yaml", ".yml"}
ESSENTIAL_INPUTS = {"package.json", "package-lock.json", "scripts/build_demo.py", "scripts/build_demo.mjs", "deployment/demo/.openai/hosting.json"}
FORBIDDEN_PARTS = {"node_modules", "runs", "runtime", "credentials", "private", "__pycache__", ".env", ".git"}
SECRET = re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bgh[opsur]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b|\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b")


class ArtifactError(ValueError):
    pass


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _relative(name):
    if not isinstance(name, str) or not name or len(name) > 240 or "\\" in name or any(ord(c) < 32 for c in name):
        raise ArtifactError("Invalid artifact path")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(p.startswith(".") for p in path.parts):
        raise ArtifactError("Hidden or escaping artifact path")
    if any(p.lower() in FORBIDDEN_PARTS for p in path.parts):
        raise ArtifactError("Private or runtime artifact is forbidden")
    return name


def _read(root, name):
    """Bind every artifact component to nonsymlink open descriptors."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(file_descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactError("Artifact must be a regular file")
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ArtifactError("Artifact exceeds its byte limit")
        return raw
    except OSError as error:
        raise ArtifactError("Artifact is missing, inaccessible or a symlink") from error
    finally:
        os.close(descriptor)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ArtifactError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ArtifactError("Nonfinite JSON value")))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ArtifactError("Invalid bounded JSON artifact") from error


def _git(*args):
    try:
        return subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), *args],
                              check=True, capture_output=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactError("Unable to resolve the expected committed source") from error


def _committed(commit, name):
    size = int(_git("cat-file", "-s", f"{commit}:{name}").strip())
    if size > MAX_FILE_BYTES:
        raise ArtifactError("Committed source exceeds its byte limit")
    raw = _git("show", f"{commit}:{name}")
    if len(raw) != size:
        raise ArtifactError("Committed source changed during validation")
    return raw


def _source_inputs(commit):
    result = set(ESSENTIAL_INPUTS)
    for entry in _git("ls-tree", "-r", "-z", commit, "docs").split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        name = path.decode("utf-8")
        if Path(name).suffix in SOURCE_SUFFIXES and not any(part.startswith(".") for part in Path(name).parts):
            _relative(name)
            if metadata.split()[0] not in {b"100644", b"100755"}:
                raise ArtifactError("Committed demo inputs must be regular files")
            result.add(name)
    if len(result) > MAX_FILES:
        raise ArtifactError("Too many committed demo inputs")
    return result


def _inventory(root):
    names, directories = set(), 0
    for current, folders, files in os.walk(root, followlinks=False):
        directories += 1
        if directories > MAX_FILES:
            raise ArtifactError("Artifact directory limit exceeded")
        for name in folders + files:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            _relative(relative)
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ArtifactError("Artifact symlinks are forbidden")
            if name in files:
                if not stat.S_ISREG(info.st_mode):
                    raise ArtifactError("Artifact must be a regular file")
                names.add(relative)
                if len(names) > MAX_FILES:
                    raise ArtifactError("Artifact file limit exceeded")
    return names


def validate(source, expected_commit):
    if not isinstance(expected_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ArtifactError("Expected commit must be a full lowercase GitHub SHA")
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise ArtifactError("Deployment source must be a nonsymlink directory")
    source = source.resolve()
    if _git("cat-file", "-t", expected_commit).strip() != b"commit":
        raise ArtifactError("Expected source identity must resolve to a Git commit")
    extras = {p.name for p in source.iterdir()} - {"dist", ".openai", ".git", ".gitignore", "README.md"}
    if extras:
        raise ArtifactError("Deployment root contains unexpected private or source files")
    if (source / ".openai").is_symlink() or set(p.name for p in (source / ".openai").iterdir()) != {"hosting.json"}:
        raise ArtifactError("Deployment metadata must contain only hosting.json")
    hosting_raw = _read(source, ".openai/hosting.json")
    hosting = _json(hosting_raw)
    tracked_raw = _committed(expected_commit, "deployment/demo/.openai/hosting.json")
    if hosting_raw != tracked_raw or hosting.get("static") != {"directory": "dist"}:
        raise ArtifactError("Hosting identity or configuration differs from committed source")
    dist = source / "dist"
    if dist.is_symlink() or not dist.is_dir():
        raise ArtifactError("Missing regular dist directory")
    inventory = _inventory(dist)
    manifest_raw = _read(dist, "asset-manifest.json")
    manifest = _json(manifest_raw)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or not isinstance(manifest["files"], dict)):
        raise ArtifactError("Invalid asset manifest")
    listed = manifest["files"]
    if ("asset-manifest.json" in listed or inventory != set(listed) | {"asset-manifest.json"}
            or not {"index.html", "example.html", "build-provenance.json"}.issubset(listed)):
        raise ArtifactError("Asset manifest does not describe the exact artifact closure")
    payloads = {".openai/hosting.json": hosting_raw, "dist/asset-manifest.json": manifest_raw}
    total = sum(len(raw) for raw in payloads.values())
    for name, expected in listed.items():
        _relative(name)
        if Path(name).suffix not in SOURCE_SUFFIXES - {".jsx"} or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ArtifactError("Unsupported asset path or hash")
        raw = _read(dist, name)
        if sha256(raw) != expected:
            raise ArtifactError("Asset hash mismatch")
        if Path(name).suffix in {".html", ".css", ".js", ".json", ".svg", ".txt", ".yaml", ".yml"} and SECRET.search(raw):
            raise ArtifactError("Credential material cannot be packaged")
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise ArtifactError("Artifact exceeds its total byte limit")
        payloads["dist/" + name] = raw
    provenance = _json(payloads["dist/build-provenance.json"])
    if (not isinstance(provenance, dict) or type(provenance.get("schema_version")) is not int
            or provenance["schema_version"] != 1 or provenance.get("kind") != "paper2bench-static-demo"):
        raise ArtifactError("Invalid build provenance")
    origin, build, declared_hosting = (provenance.get(name) for name in ("source", "build", "hosting"))
    if (not all(isinstance(value, dict) for value in (origin, build, declared_hosting))
            or origin.get("commit") != expected_commit or origin.get("repository") != REPOSITORY
            or origin.get("dirty") is not False or build.get("mode") != "release"):
        raise ArtifactError("Artifact requires clean release provenance for the expected commit")
    tree = _git("rev-parse", "--verify", f"{expected_commit}^{{tree}}").decode().strip()
    if origin.get("tree") != tree:
        raise ArtifactError("Source tree differs from the expected commit")
    if declared_hosting != {"project_id": hosting.get("project_id"), "manifest_sha256": sha256(hosting_raw)}:
        raise ArtifactError("Hosting provenance mismatch")
    inputs = origin.get("files")
    if not isinstance(inputs, dict) or set(inputs) != _source_inputs(expected_commit):
        raise ArtifactError("Source provenance does not enumerate the exact committed inputs")
    source_total = 0
    for name, expected in inputs.items():
        raw = _committed(expected_commit, name)
        source_total += len(raw)
        if source_total > MAX_TOTAL_BYTES or sha256(raw) != expected:
            raise ArtifactError("Committed source input hash mismatch or byte limit")
    if build.get("package_lock_sha256") != inputs["package-lock.json"]:
        raise ArtifactError("Dependency lock provenance mismatch")
    if _inventory(dist) != inventory or _read(source, ".openai/hosting.json") != hosting_raw:
        raise ArtifactError("Deployment files changed during validation")
    return payloads, {"verified": True, "project_id": hosting["project_id"],
                      "source_commit": expected_commit, "source_tree": tree,
                      "files": len(payloads), "bytes": total}


def package(source, output, expected_commit, *, verify_only=False):
    payloads, report = validate(source, expected_commit)
    if verify_only:
        return report
    if output is None:
        raise ArtifactError("Archive output is required unless --verify-only is used")
    output = Path(output)
    if output.is_symlink() or output.resolve().is_relative_to(Path(source).resolve()):
        raise ArtifactError("Archive must be outside deployment source and cannot be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".demo-archive-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name in sorted(payloads):
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = len(payloads[name]), 0o644, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        archive.addfile(info, io.BytesIO(payloads[name]))
            stream.flush()
            os.fsync(stream.fileno())
        archive_hash = sha256(Path(temporary).read_bytes())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dict(report, archive=str(output.resolve()), sha256=archive_hash)


def extract_archive(archive, destination, expected_commit):
    """Extract only bounded regular payloads, then admit their committed identity."""
    archive, destination = Path(archive), Path(destination)
    if archive.is_symlink():
        raise ArtifactError("Archive cannot be a symlink")
    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ArtifactError("Archive must be a regular file")
        compressed = stream.read(MAX_TOTAL_BYTES + 1)
    if len(compressed) > MAX_TOTAL_BYTES:
        raise ArtifactError("Compressed archive exceeds its byte limit")
    # Bound decompression before tarfile interprets extension headers or names.
    tar_limit = MAX_TOTAL_BYTES + MAX_FILES * 2048 + 10240
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
            raw_tar = stream.read(tar_limit + 1)
    except (OSError, EOFError) as error:
        raise ArtifactError("Invalid gzip archive") from error
    if len(raw_tar) > tar_limit:
        raise ArtifactError("Decompressed archive exceeds its byte limit")
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ArtifactError("Extraction destination must be a new or empty nonsymlink directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".demo-extract-", dir=destination.parent))
    try:
        names, total = set(), 0
        try:
            with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive_file:
                for member in archive_file:
                    if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE} or member.name in names or len(names) >= MAX_FILES + 2:
                        raise ArtifactError("Archive allows only unique bounded regular files")
                    name = member.name
                    if name != ".openai/hosting.json":
                        _relative(name)
                        if not name.startswith("dist/"):
                            raise ArtifactError("Unexpected archive root entry")
                    if member.size < 0 or member.size > MAX_FILE_BYTES:
                        raise ArtifactError("Archive member exceeds its byte limit")
                    total += member.size
                    if total > MAX_TOTAL_BYTES:
                        raise ArtifactError("Archive payload exceeds its total byte limit")
                    contents = archive_file.extractfile(member)
                    if contents is None:
                        raise ArtifactError("Archive member is missing")
                    raw = contents.read(MAX_FILE_BYTES + 1)
                    if len(raw) != member.size:
                        raise ArtifactError("Archive member has incomplete contents")
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    names.add(name)
        except tarfile.TarError as error:
            raise ArtifactError("Invalid tar archive") from error
        _, report = validate(stage, expected_commit)
        if destination.exists():
            destination.rmdir()  # Refuses concurrent nonempty changes; never recursively deletes.
        stage.rename(destination)
        return dict(report, extracted_to=str(destination.resolve()), archive_sha256=sha256(compressed))
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--source", type=Path)
    operation.add_argument("--extract", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.extract:
            if args.extract_to is None or args.output or args.verify_only:
                raise ArtifactError("Extraction requires --extract-to and cannot use packaging options")
            report = extract_archive(args.extract, args.extract_to, args.expected_commit)
        else:
            if args.extract_to:
                raise ArtifactError("--extract-to requires --extract")
            report = package(args.source, args.output, args.expected_commit, verify_only=args.verify_only)
        print(json.dumps(report, sort_keys=True))
    except (ArtifactError, OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Demo artifact rejected: {error}\n")


if __name__ == "__main__":
    main()
