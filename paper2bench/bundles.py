"""Bounded data-only manifests and field-allowlisted candidate exports."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
MAX_CONFIG_NODES = 10000
PUBLIC_FIELDS = {"schema_version", "task_id", "revision", "research_question", "paper_type",
                 "method_kind", "task_mode", "readiness", "files", "entrypoint", "inputs",
                 "predictions", "limits", "source", "required_outputs"}
CONFIG_FIELDS = {"task_id", "research_question", "paper_type", "method_kind", "constraints",
                 "budget_per_model", "datasets", "data_sources", "proposed_method", "baselines"}
NESTED_FIELDS = {"name", "description", "source", "revision", "units", "split", "features",
                 "target", "components", "kind", "id", "metric", "direction"}


class BundleError(ValueError):
    pass


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise BundleError("Nonfinite JSON number")
    return value


def _read_open_file(fd, limit):
    """Read from the checked descriptor, rather than re-opening a checked path."""
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise BundleError("Input must be a regular nonsymlink file")
        if info.st_size > limit:
            raise BundleError("Input exceeds size limit")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise BundleError("Input exceeds size limit")
    return raw


def _read_regular_file(path, limit):
    if type(limit) is not int or limit < 1:
        raise BundleError("File size limit must be a positive integer")
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        return _read_open_file(os.open(path, flags), limit)
    except OSError as error:
        raise BundleError("Input must be an accessible regular nonsymlink file") from error


def read_json(path, limit=MAX_JSON_BYTES):
    raw = _read_regular_file(path, limit)
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_float=_finite_float,
                          parse_constant=lambda x: (_ for _ in ()).throw(BundleError("Nonfinite JSON value")))
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise BundleError("Invalid bounded JSON input") from error


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    fd, temp = tempfile.mkstemp(prefix=".p2b-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_bytes(path, raw):
    """Private atomic byte artifact, including an fsync before publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".p2b-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _artifact_parts(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise BundleError("Invalid artifact path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(x.startswith(".") for x in relative.parts):
        raise BundleError("Hidden or escaping artifact path")
    if relative.as_posix() != name or len(name) > 240:
        raise BundleError("Artifact path must be normalized relative POSIX text")
    return relative.parts


def read_artifact(root, name, *, limit=MAX_FILE_BYTES):
    """Read bounded public bytes through nonsymlink directory/file descriptors."""
    parts = _artifact_parts(name)
    if type(limit) is not int or not 1 <= limit <= MAX_FILE_BYTES:
        raise BundleError("Invalid artifact size limit")
    directory = None
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory = os.open(Path(root).resolve(), directory_flags)
        for part in parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        return _read_open_file(os.open(parts[-1], file_flags, dir_fd=directory), limit)
    except OSError as error:
        raise BundleError("Artifact is missing, inaccessible, or contains a symlink") from error
    finally:
        if directory is not None:
            os.close(directory)


def safe_file(root, name):
    """Forbid absolute/traversing/hidden paths and every symlink component."""
    parts = _artifact_parts(name)
    root = Path(root).resolve()
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise BundleError("Symlink artifacts are forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise BundleError("Artifact is missing or outside bundle")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise BundleError("Artifact exceeds size limit")
    return path


def validate_task(manifest_path, *, require_executable=False):
    manifest_path = Path(manifest_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "task_manifest.json"
    data = read_json(manifest_path)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise BundleError("Unsupported task schema version")
    if not isinstance(data.get("task_id"), str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", data["task_id"]):
        raise BundleError("Invalid task ID")
    if type(data.get("revision")) is not int or data["revision"] < 1:
        raise BundleError("Task revision must be a positive integer")
    if not isinstance(data.get("task_mode"), str) or data["task_mode"] not in {"from_scratch", "author_artifact_reexecution", "adaptation"}:
        raise BundleError("Task mode must be explicit")
    if not isinstance(data.get("readiness"), str) or data["readiness"] not in {"draft", "needs_review", "ready", "blocked", "retired"}:
        raise BundleError("Invalid readiness state")
    files = data.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 128:
        raise BundleError("Task must enumerate 1 to 128 public files")
    total = 0
    for name, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise BundleError("Invalid file digest")
        raw = read_artifact(manifest_path.parent, name)
        total += len(raw)
        if total > MAX_BUNDLE_BYTES or digest(raw) != expected:
            raise BundleError("Bundle exceeds bounds or file digest mismatch")
        if any(x.lower() in {"curator", "evaluator", "curator_manifest.json", "metadata.json", "instances.json", "instruction_gt.txt"}
               for x in PurePosixPath(name).parts):
            raise BundleError("Curator artifact cannot be a candidate file")
    limits = data.get("limits", {})
    allowed_limits = {"timeout_seconds", "memory_bytes", "cpus", "pids_limit", "workspace_bytes", "output_limit_bytes"}
    if not isinstance(limits, dict) or set(limits) - allowed_limits:
        raise BundleError("Unknown execution limit")
    for key, value in limits.items():
        try:
            valid_number = type(value) in {int, float} and math.isfinite(value) and value > 0
        except OverflowError:
            valid_number = False
        if not valid_number:
            raise BundleError("Execution limits must be positive finite numbers")
        if key != "cpus" and type(value) is not int:
            raise BundleError("Integral execution limit required")
    for field in ("entrypoint", "inputs", "predictions"):
        if field in data and not isinstance(data[field], str):
            raise BundleError(f"Task {field} must be text")
    if "predictions" in data:
        _artifact_parts(data["predictions"])
    if require_executable:
        if data.get("readiness") != "ready":
            raise BundleError("Task is not ready for execution")
        for field in ("entrypoint", "inputs", "predictions"):
            if not isinstance(data.get(field), str):
                raise BundleError(f"Executable task missing {field}")
        if data["entrypoint"] not in files or not data["entrypoint"].endswith(".py"):
            raise BundleError("Entrypoint must be a pinned Python source file")
        if data["inputs"] not in files:
            raise BundleError("Inputs must be pinned in the task file map")
        if (data["predictions"] in files or not data["predictions"].endswith(".json")):
            raise BundleError("Predictions must be a new relative JSON artifact")
    return data


def public_config(config):
    nodes = 0
    text_bytes = 0

    def project(value, depth=0):
        nonlocal nodes, text_bytes
        nodes += 1
        if nodes > MAX_CONFIG_NODES:
            raise BundleError("Configuration exceeds expanded node limit")
        if depth > 8:
            raise BundleError("Configuration exceeds nesting limit")
        if isinstance(value, dict):
            return {k: project(v, depth + 1) for k, v in value.items() if k in NESTED_FIELDS}
        if isinstance(value, list):
            if len(value) > 256:
                raise BundleError("Configuration list exceeds limit")
            return [project(v, depth + 1) for v in value]
        if value is None or type(value) in {str, bool, int, float}:
            if type(value) is float and not math.isfinite(value):
                raise BundleError("Nonfinite configuration number")
            if isinstance(value, str) and len(value) > 32000:
                raise BundleError("Configuration text exceeds limit")
            if isinstance(value, str):
                text_bytes += len(value.encode("utf-8"))
                if text_bytes > MAX_FILE_BYTES:
                    raise BundleError("Configuration exceeds expanded text limit")
            return value
        raise BundleError("Unsupported configuration value")
    if not isinstance(config, dict):
        raise BundleError("Configuration must be a mapping")
    return {k: project(v) for k, v in config.items() if k in CONFIG_FIELDS}


def write_draft_manifest(task_dir, config, *, source=None):
    """Create safe draft exports without copying reference instructions or arbitrary fields."""
    import yaml
    task_dir = Path(task_dir)
    projected = public_config(config)
    for field in ("research_question", "paper_type", "method_kind"):
        if field in projected and not isinstance(projected[field], str):
            raise BundleError(f"Draft {field} must be text")
    task_id = config.get("task_id", task_dir.name)
    if not isinstance(task_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", task_id):
        raise BundleError("Invalid task ID")
    config_bytes = yaml.safe_dump(projected, sort_keys=False).encode("utf-8")
    if len(config_bytes) > MAX_FILE_BYTES:
        raise BundleError("Draft configuration exceeds size limit")
    candidate = task_dir / "candidate"
    if candidate.is_symlink():
        raise BundleError("Candidate output directory cannot be a symlink")
    candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
    config_file = candidate / "task_config.yaml"
    atomic_bytes(config_file, config_bytes)
    question = projected.get("research_question", "Research question requires curator review")
    instruction = candidate / "instruction.txt"
    atomic_bytes(instruction, (f"Research question: {question}\n\n"
                              "This draft requires source, resource and evaluator review before execution.\n"
                              "Use the public task_config.yaml for the proposed method and resources.\n").encode("utf-8"))
    manifest = {"schema_version": 1, "task_id": task_id, "revision": 1,
                "research_question": question, "paper_type": config.get("paper_type", "empirical_study"),
                "method_kind": config.get("method_kind", "unspecified"), "task_mode": "from_scratch",
                "readiness": "needs_review", "source": source or {},
                "files": {p.name: digest(p.read_bytes()) for p in (config_file, instruction)}}
    atomic_json(candidate / "task_manifest.json", manifest)
    # Root manifest is a locator for compatibility. Execution/export resolves the candidate projection.
    atomic_json(task_dir / "task_manifest.json", dict(manifest, files={"candidate/" + k: v for k, v in manifest["files"].items()}))
    return candidate / "task_manifest.json"


def _candidate_projection(task):
    """Derive the exact public identity and bytes shared by both export audiences."""
    task = Path(task)
    root = task.parent if task.is_file() else task
    manifest_path = task if task.is_file() else root / "task_manifest.json"
    if (root / "candidate" / "task_manifest.json").exists():
        manifest_path = root / "candidate" / "task_manifest.json"
    manifest = validate_task(manifest_path)
    projected = {k: v for k, v in manifest.items() if k in PUBLIC_FIELDS}
    projected["files"] = dict(manifest["files"])
    if isinstance(projected.get("source"), dict):
        projected["source"] = {k: v for k, v in projected["source"].items()
                               if k in {"paper_sha256", "paper_id", "version", "url", "fixture"}
                               and isinstance(v, str)}
    payloads = {}
    total_bytes = 0
    for name in projected["files"]:
        raw = read_artifact(manifest_path.parent, name)
        # validate_task checked earlier; this also rejects ordinary source mutation
        # between validation and deriving the projection's new public identity.
        if digest(raw) != manifest["files"][name]:
            raise BundleError("Source artifact changed during export")
        if name.endswith((".yaml", ".yml")):
            import yaml
            try:
                data = public_config(yaml.safe_load(raw.decode("utf-8")))
            except (yaml.YAMLError, UnicodeDecodeError, RecursionError) as error:
                raise BundleError("Invalid bounded YAML configuration") from error
            raw = yaml.safe_dump(data, sort_keys=False).encode("utf-8")
        total_bytes += len(raw)
        if len(raw) > MAX_FILE_BYTES or total_bytes > MAX_BUNDLE_BYTES:
            raise BundleError("Projected bundle exceeds size bounds")
        payloads[name] = raw
        projected["files"][name] = digest(raw)
    return root, manifest, projected, payloads


def export_bundle(task, output, *, audience="candidate"):
    if audience not in {"candidate", "curator"}:
        raise BundleError("Export audience must be candidate or curator")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise BundleError("Export destination must not already exist")
    root, source_manifest, projected, payloads = _candidate_projection(task)
    if audience == "curator":
        # A projection changes task identity. Rebind only an explicitly supplied
        # evaluator already bound to this exact source task, never an arbitrary one.
        data = read_json(root / "curator_manifest.json")
        source_identity = digest(canonical_json(source_manifest))
        if not isinstance(data, dict) or data.get("task_manifest_sha256") != source_identity:
            raise BundleError("Protected evaluator does not match the source task")
        projected_identity = digest(canonical_json(projected))
        if "protocol" in data:
            # Scientific reviews bind the entire task, submission and dataset
            # identities. A public projection is a new task when any metadata or
            # file byte changes; an export cannot silently renew those reviews.
            if projected_identity != source_identity:
                raise BundleError("Export changes the reviewed scientific task identity; "
                                  "create and review a protocol for the public projection")
            from .evaluation.science import freeze_protocol
            from .evaluation.validation import EvaluationError
            try:
                entrypoint = source_manifest.get("entrypoint")
                submission = source_manifest["files"].get(entrypoint)
                freeze_protocol(data["protocol"], task_manifest=source_manifest,
                                submission_sha256=submission)
            except (EvaluationError, TypeError, KeyError) as error:
                raise BundleError("Scientific protocol is invalid for the source task") from error
        rebound = dict(data, task_manifest_sha256=projected_identity,
                       export_source_task_manifest_sha256=source_identity)
        output.mkdir(parents=True, mode=0o700)
        atomic_json(output / "curator_manifest.json", rebound)
        return output / "curator_manifest.json"
    output.mkdir(parents=True, mode=0o700)
    for name, raw in payloads.items():
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    atomic_json(output / "task_manifest.json", projected)
    validate_task(output / "task_manifest.json")
    return output / "task_manifest.json"
