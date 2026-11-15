"""Export selected, sanitized evidence from the archived local v0.2 core.

Generation requires the original local dist directory. Verification of already
exported public assets needs no private state or optional runtime dependencies.
This is an evidence export, not a runnable Paper2Bench replay bundle or a wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import random
import re
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
INPUTS = (
    "verification-v0.2.json", "verification-v0.2.xml", "optimization-v0.2.zip",
    "seeded-study-v0.2.zip", "seeded-study-v0.2.json",
)
PUBLIC_NAMES = {
    "verification-v0.2.json", "optimization-v0.2.json", "seeded-study-v0.2.json",
    "README.txt", "evidence-v0.2.zip", "manifest.json",
}
MAX_FILE = 8 * 1024 * 1024
MAX_ARCHIVE = 24 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SECRET = re.compile(
    rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|\bgh[opsur]_[A-Za-z0-9]{20,}\b|"
    rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b|\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b"
)


class ReproExportError(ValueError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReproExportError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ReproExportError("nonfinite JSON value")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ReproExportError("invalid bounded JSON evidence") from error


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _pretty(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def _read(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ReproExportError("evidence must be a regular file")
            raw = stream.read(MAX_FILE + 1)
    except OSError as error:
        raise ReproExportError("evidence is missing, inaccessible, or a symlink") from error
    if len(raw) > MAX_FILE:
        raise ReproExportError("evidence exceeds byte limit")
    return raw


def _expect(condition, message):
    if not condition:
        raise ReproExportError(message)


def _integer(value, *, minimum=0, maximum=10**9):
    _expect(type(value) is int and minimum <= value <= maximum, "invalid bounded integer")
    return value


def _number(value):
    _expect(type(value) in (int, float) and math.isfinite(value), "invalid finite number")
    return value


def _close(actual, expected):
    _expect(math.isclose(_number(actual), _number(expected), rel_tol=1e-12, abs_tol=1e-12),
            "recorded metric differs from independent recomputation")


def _safe_id(value):
    _expect(isinstance(value, str) and SAFE_ID.fullmatch(value), "invalid public fixture ID")
    return value


def _archive(raw, *, study=False):
    _expect(len(raw) <= MAX_FILE, "archive exceeds compressed byte limit")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            _expect(0 < len(entries) <= 512, "invalid archive member count")
            _expect(len({item.filename for item in entries}) == len(entries), "duplicate archive member")
            _expect(sum(item.file_size for item in entries) <= MAX_ARCHIVE,
                    "archive exceeds uncompressed byte limit")
            files = {}
            for item in entries:
                name = item.filename
                allowed = name == "manifest.json" or re.fullmatch(r"artifacts/[0-9a-f]{64}", name)
                allowed = allowed or (study and re.fullmatch(r"trials/[0-9]{3}\.zip", name))
                mode = item.external_attr >> 16
                _expect(allowed and not item.is_dir() and not item.flag_bits & 1,
                        "unexpected or encrypted archive member")
                _expect(stat.S_IFMT(mode) in (0, stat.S_IFREG), "nonregular archive member")
                _expect(item.file_size <= MAX_FILE and item.compress_type in (0, 8),
                        "unsupported or oversized archive member")
                with archive.open(item) as stream:
                    payload = stream.read(MAX_FILE + 1)
                _expect(len(payload) == item.file_size <= MAX_FILE, "archive size mismatch")
                if name.startswith("artifacts/"):
                    _expect(_sha(payload) == name.split("/")[1], "artifact hash mismatch")
                files[name] = payload
    except (zipfile.BadZipFile, OSError, RuntimeError, EOFError) as error:
        raise ReproExportError("invalid evidence archive") from error
    manifest = _json(files["manifest.json"])
    expected = manifest.get("artifacts")
    _expect(isinstance(expected, list) and all(isinstance(x, str) and SHA.fullmatch(x) for x in expected),
            "invalid artifact manifest")
    _expect(len(set(expected)) == len(expected), "duplicate artifact identity")
    _expect({"artifacts/" + x for x in expected} == {x for x in files if x.startswith("artifacts/")},
            "artifact file-set mismatch")
    return files, manifest


def _artifact(files, identity):
    return _json(_raw_artifact(files, identity))


def _raw_artifact(files, identity):
    _expect(isinstance(identity, str) and SHA.fullmatch(identity), "invalid artifact reference")
    _expect("artifacts/" + identity in files, "referenced artifact missing")
    return files["artifacts/" + identity]


def _run(raw):
    files, manifest = _archive(raw)
    record = manifest.get("record", {})
    run = record.get("run", {})
    _expect(run.get("state") == "completed", "only completed recorded runs may be exported")
    result = _artifact(files, run.get("result_hash"))
    _expect(result == record.get("result") and result.get("execution_state") == "completed",
            "result identity or execution state mismatch")
    spec = _artifact(files, run.get("spec_hash"))
    _expect(result.get("spec_hash") == run["spec_hash"], "result specification mismatch")
    _expect(spec.get("policy", {}).get("local_only") is True and not result.get("provider_calls"),
            "only local synthetic fixtures without provider calls may be exported")
    _expect(any("synthetic" in x.lower() and "no published" in x.lower()
                for x in spec.get("assumptions", []) if isinstance(x, str)),
            "source must explicitly declare a synthetic non-paper fixture")
    source = _raw_artifact(files, spec.get("source_sha256")).lower()
    _expect(b"synthetic" in source and (b"no published" in source or b"not a published" in source),
            "source text must declare a synthetic non-paper fixture")
    _expect(spec.get("workload", {}).get("kind") == "python", "expected isolated Python fixture")
    _raw_artifact(files, spec["workload"].get("implementation_sha256"))
    engine = _artifact(files, spec.get("engine_identity_sha256"))
    _expect(isinstance(engine.get("modules"), dict) and engine["modules"], "engine source identity missing")
    for module_hash in engine["modules"].values():
        _raw_artifact(files, module_hash)
    dataset = _artifact(files, spec.get("dataset_sha256"))
    measurements = result.get("measurements", {})
    _expect(measurements.get("runtime") == "docker" and
            measurements.get("network_used") is False and
            measurements.get("cleanup_confirmed") is True, "recorded isolation evidence missing")
    return files, run, result, spec, dataset


def _origin(raw, run, spec):
    return {"kind": "recorded_local_synthetic_fixture", "component_version": "0.2.0",
            "source_bundle_sha256": _sha(raw), "spec_sha256": run["spec_hash"],
            "result_sha256": run["result_hash"], "dataset_sha256": spec["dataset_sha256"]}


def _optimization(raw):
    _files, run, result, spec, dataset = _run(raw)
    _expect(dataset.get("task") == "knapsack", "expected recorded optimization fixture")
    predictions = result["predictions"]
    instances = dataset["instances"]
    _expect(0 < len(instances) <= 1000 and len(predictions["ids"]) == len(instances), "invalid fixture size")
    _expect(len({x["id"] for x in instances}) == len(instances), "duplicate problem IDs")
    _expect(len(set(predictions["ids"])) == len(instances), "duplicate prediction IDs")
    by_id = dict(zip(predictions["ids"], predictions["selections"], strict=True))
    public, checked = [], []
    for instance in instances:
        identity = _safe_id(instance["id"])
        weights = [_integer(x, minimum=1, maximum=10000) for x in instance["weights"]]
        values = [_integer(x) for x in instance["values"]]
        capacity = _integer(instance["capacity"], maximum=10000)
        _expect(0 < len(weights) == len(values) <= 16 and identity in by_id,
                "fixture exceeds independent exhaustive-check bound")
        selected = by_id[identity]
        _expect(isinstance(selected, list) and len(set(selected)) == len(selected), "invalid selection")
        selected = [_integer(x, maximum=len(weights) - 1) for x in selected]
        total_weight = sum(weights[x] for x in selected)
        _expect(total_weight <= capacity, "recorded candidate is infeasible")
        objective = sum(values[x] for x in selected)
        optimum = max(
            sum(values[x] for x in subset)
            for size in range(len(weights) + 1)
            for subset in itertools.combinations(range(len(weights)), size)
            if sum(weights[x] for x in subset) <= capacity
        )
        score = objective / optimum if optimum else 1.0
        checked.append({"id": identity, "selection": selected, "feasible": True,
                        "objective": objective, "optimal_objective": optimum,
                        "total_weight": total_weight, "score": score})
        public.append({"id": identity, "weights": weights, "values": values, "capacity": capacity})
    assessment = result["assessment"]
    mean = sum(x["score"] for x in checked) / len(checked)
    _close(assessment["mean_score"], mean)
    _expect(assessment["feasible_count"] == len(checked), "feasibility count mismatch")
    for saved, recomputed in zip(assessment["per_item"], checked, strict=True):
        _expect(all(saved[key] == value for key, value in recomputed.items()), "per-item score mismatch")
    confidence = _number(assessment["confidence"])
    _expect(0 < confidence < 1, "invalid confidence level")
    radius = math.sqrt(math.log(2 / (1 - confidence)) / (2 * len(checked)))
    interval = {"lower": max(0, mean - radius), "upper": min(1, mean + radius), "method": "hoeffding"}
    for bound in ("lower", "upper"):
        _close(assessment["confidence_interval"][bound], interval[bound])
    band = assessment["criterion_band"]
    verdict = "supported" if interval["lower"] >= band["lower"] and interval["upper"] <= band["upper"] else "inconclusive"
    _expect(assessment["verdict"] == verdict, "optimization verdict mismatch")
    return {"schema_version": 1, "kind": "paper2bench_recorded_repro_optimization",
            "origin": _origin(raw, run, spec),
            "scope": "Synthetic software fixture; no published paper reproduced. Static recorded evidence, not a live solver.",
            "summary": {"task": "knapsack", "sample_count": len(checked), "feasible_count": len(checked),
                        "mean_score": mean, "verdict": verdict},
            "assessment": {"metric": "optimality_ratio", "criterion": "optimality_equivalence",
                           "confidence": confidence, "confidence_interval": interval,
                           "criterion_band": band, "implementation_fidelity": "unverified",
                           "protocol_fidelity": "unverified"},
            "data": {"instances": public}, "recomputed": {"method": "independent_exhaustive_enumeration", "per_item": checked}}


def _percentile(values, probability):
    position = (len(values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[min(lower + 1, len(values) - 1)] * fraction


def _study(raw, report_raw):
    files, manifest = _archive(raw, study=True)
    report = _json(report_raw)
    stored = _artifact(files, manifest.get("study_result_hash"))
    _expect({k: v for k, v in report.items() if k != "result_hash"} == stored and
            report.get("result_hash") == manifest["study_result_hash"], "study result identity mismatch")
    plan = _artifact(files, manifest.get("plan_hash"))
    seeds = plan["study"]["seeds"]
    _expect(seeds == [11, 22, 33] and len(manifest.get("trials", [])) == 6,
            "expected prespecified three-seed synthetic fixture")
    _expect({x["member"] for x in manifest["trials"]} == {x for x in files if x.startswith("trials/")},
            "study trial file-set mismatch")
    _expect(report["plan_hash"] == manifest["plan_hash"] and
            report["dataset_sha256"] == plan["dataset_sha256"], "study plan identity mismatch")
    public_trials, scores, public_data = [], {}, None
    for index, trial in enumerate(manifest["trials"]):
        role, seed = ("candidate", "baseline")[index % 2], seeds[index // 2]
        _expect((trial["role"], trial["seed"]) == (role, seed), "trial schedule mismatch")
        _expect(plan["trials"][index] == {k: trial[k] for k in ("role", "seed", "spec_hash")},
                "trial differs from frozen plan")
        member = trial["member"]
        _expect(member == f"trials/{index:03}.zip" and _sha(files[member]) == trial["bundle_sha256"],
                "nested trial identity mismatch")
        _trial_files, run, result, spec, data = _run(files[member])
        _expect(run["spec_hash"] == trial["spec_hash"] and spec["seed"] == seed and
                spec["dataset_sha256"] == plan["dataset_sha256"], "trial protocol mismatch")
        original = _artifact(files, plan["study"][role + "_spec_hash"])
        _expect(spec == {**original, "seed": seed}, "trial altered fields beyond the planned seed")
        clean_data = {split: [{"id": _safe_id(item["id"]),
                               "x": [_number(x) for x in item["x"]], "y": item["y"]}
                              for item in data[split]] for split in ("train", "test")}
        _expect(all(x["y"] in {"a", "b"} for split in clean_data.values() for x in split),
                "unexpected synthetic label")
        if public_data is None:
            public_data = clean_data
        _expect(public_data == clean_data, "study dataset differs across trials")
        predictions = result["predictions"]
        _expect(len(predictions["ids"]) == len(predictions["labels"]) == len(data["test"]) and
                len(set(predictions["ids"])) == len(predictions["ids"]), "invalid classification predictions")
        by_id = dict(zip(predictions["ids"], predictions["labels"], strict=True))
        _expect(set(by_id) == {x["id"] for x in data["test"]} and set(by_id.values()) <= {"a", "b"},
                "classification output mismatch")
        correctness = [int(by_id[item["id"]] == item["y"]) for item in data["test"]]
        accuracy = sum(correctness) / len(correctness)
        _close(result["assessment"]["accuracy"], accuracy)
        scores[(role, seed)] = correctness
        public_trials.append({"role": role, "seed": seed, "state": "completed",
                              "spec_sha256": run["spec_hash"], "result_sha256": run["result_hash"],
                              "accuracy": accuracy,
                              "predictions": {"ids": [x["id"] for x in data["test"]],
                                              "labels": [by_id[x["id"]] for x in data["test"]]}})
    count = len(public_data["test"])
    matrix = [[a - b for a, b in zip(scores[("candidate", seed)], scores[("baseline", seed)], strict=True)] for seed in seeds]
    bootstrap_count = _integer(plan["study"]["bootstrap_samples"], minimum=200, maximum=20000)
    rng = random.Random(plan["bootstrap_seed"])
    distribution = []
    for _ in range(bootstrap_count):
        seed_indices = [rng.randrange(len(seeds)) for _ in seeds]
        case_indices = [rng.randrange(count) for _ in range(count)]
        distribution.append(sum(matrix[s][c] for s in seed_indices for c in case_indices) / (count * len(seeds)))
    distribution.sort()
    confidence = plan["study"]["confidence"]
    tail = (1 - confidence) / 2
    interval = {"lower": _percentile(distribution, tail), "upper": _percentile(distribution, 1 - tail),
                "method": "two_way_crossed_percentile_bootstrap"}
    candidate = sum(x["accuracy"] for x in public_trials if x["role"] == "candidate") / len(seeds)
    baseline = sum(x["accuracy"] for x in public_trials if x["role"] == "baseline") / len(seeds)
    effect = sum(map(sum, matrix)) / (count * len(seeds))
    target = plan["study"]["minimum_effect"]
    verdict = "supported" if interval["lower"] > target else "not_supported" if interval["upper"] <= target else "inconclusive"
    assessment = report["assessment"]
    for key, value in {"candidate_accuracy": candidate, "baseline_accuracy": baseline, "effect": effect}.items():
        _close(assessment[key], value)
    for bound in ("lower", "upper"):
        _close(assessment["confidence_interval"][bound], interval[bound])
    _expect(assessment["verdict"] == verdict and assessment["paired_correctness_differences"] == matrix,
            "study assessment differs from independent recomputation")
    _expect(count == 80, "expected fixed 80-case synthetic study")
    return {"schema_version": 1, "kind": "paper2bench_recorded_repro_seed_study",
            "origin": {"kind": "recorded_local_synthetic_fixture", "component_version": "0.2.0",
                       "source_bundle_sha256": _sha(raw), "source_report_sha256": _sha(report_raw),
                       "plan_sha256": manifest["plan_hash"], "result_sha256": manifest["study_result_hash"]},
            "scope": "Six recorded synthetic Docker trials; no published paper reproduced. Static evidence; no model calls or new executions.",
            "summary": {"seeds": seeds, "trial_count": 6, "sample_count": count,
                        "candidate_accuracy": candidate, "baseline_accuracy": baseline,
                        "effect": effect, "verdict": verdict},
            "assessment": {"metric": "accuracy_difference", "minimum_effect": target,
                           "confidence": confidence, "confidence_interval": interval,
                           "bootstrap_samples": bootstrap_count, "bootstrap_seed": plan["bootstrap_seed"],
                           "implementation_fidelity": "unverified", "protocol_fidelity": "unverified"},
            "data": public_data, "trials": public_trials,
            "recomputed": {"method": "paired_correctness_and_crossed_bootstrap", "differences": matrix},
            "limitations": ["Only three supplied seeds; actual seed consumption and independent fits are unverified.",
                            "Interval assumes exchangeable seed pairs and independent test cases.",
                            "Positive observed effect remains inconclusive; this does not establish paper reproduction."]}


def _verification(raw, xml_raw):
    report = _json(raw)
    _expect(b"<!DOCTYPE" not in xml_raw.upper() and b"<!ENTITY" not in xml_raw.upper(), "XML declarations forbidden")
    try:
        xml = ET.fromstring(xml_raw)
    except ET.ParseError as error:
        raise ReproExportError("invalid verification XML") from error
    suites = list(xml.iter("testsuite"))
    _expect(len(suites) == 1 and report.get("version") == "0.2.0", "unexpected verification receipt")
    suite = suites[0]
    cases = len(list(xml.iter("testcase")))
    _expect(cases == report["pytest_passed"] == 642 and report["subtests_passed"] == 93 and
            int(suite.attrib["tests"]) == cases + report["subtests_passed"], "recorded test counts mismatch")
    for field in ("errors", "failures", "skipped"):
        _expect(int(suite.attrib[field]) == report[field] == 0, "recorded verification was not fully successful")
    _expect(report.get("live_paid_model_calls") is False and report.get("cloud_deployment_created") is False,
            "unexpected live execution scope")
    return {"schema_version": 1, "kind": "paper2bench_recorded_repro_verification",
            "origin": {"kind": "archived_local_verification", "component_version": "0.2.0",
                       "receipt_sha256": _sha(raw), "junit_sha256": _sha(xml_raw)},
            "recorded_at": suite.attrib["timestamp"],
            "scope": "Historical local core verification only. These are not current canonical repository test counts or published-paper evidence.",
            "counts": {"pytest_passed": cases, "subtests_passed": 93, "junit_tests": cases + 93,
                       "failures": 0, "errors": 0, "skipped": 0},
            "canonical_repository_verification": {"status": "separate_run_required", "test_count": None},
            "live_paid_model_calls": False, "cloud_deployment_created": False,
            "checks": ["Docker classification and optimization", "Authenticated remote-worker protocol",
                       "Host-worker SIGKILL deadline", "Isolated PDF parsing", "Multi-seed score recomputation"],
            "published_paper_demo": {"separate_asset": "../results/adam-counterexample.json",
                                     "relationship": "separate finite published-paper case study"}}


GUIDE = """Paper2Bench optional reproduction core: recorded public evidence

These files select evidence from the archived local v0.2 core. The historical
verification receipt records 642 pytest cases plus 93 subtests; those counts
are not the current canonical repository suite. Optimization and seed-study
data are synthetic software fixtures, not a published-paper reproduction.
The separate Adam case study remains at ../results/adam-counterexample.json.

evidence-v0.2.zip contains selected reports, public synthetic data/predictions,
and their checksums. It is NOT a full runtime replay bundle, a wheel, a live
execution service, or proof that the current namespaced code ran these jobs.
No credentials, worker identities, state databases, or private runtime logs are
included. Hashes establish integrity; scientific fidelity remains unverified.

Install the canonical repository, Python 3.11+:
  git clone https://github.com/cal2020/Paper2Bench.git
  cd Paper2Bench
  python -m pip install -e '.[repro]'
  paper2bench-repro --help
  paper2bench-repro --state .paper2bench-repro demo
  paper2bench-repro --state .paper2bench-repro demo --task knapsack

These trusted builtin demos require neither Docker nor model credentials.
Python candidate runs with --python require a local Docker daemon and staged
Python image; see docs/REPRODUCIBILITY.md for placement, images, limits, and
provider setup. Fresh demos create NEW runs distinct from this recorded evidence.
The optional CLI is paper2bench-repro. The original paper2bench CLI and
canonical distribution identity remain unchanged. Archived standalone v0.2
paper2bench wheels are deliberately not redistributed by this export.

Regenerate from the original local artifacts, then commit the public outputs:
  python scripts/export_repro_demo.py --source-dist /path/to/repro-core/dist
Verify all generated files and ZIP members without the original local state:
  python scripts/export_repro_demo.py --verify-only
Build the committed static demo using the existing provenance-preserving flow:
  python scripts/build_demo.py --release

The static build recursively includes these JSON/TXT/ZIP assets and binds them
to build-provenance.json and asset-manifest.json. It makes no provider calls.
"""


def _checksums(files):
    return {name: {"sha256": _sha(raw), "bytes": len(raw)} for name, raw in sorted(files.items())}


def _zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(files.items()):
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(item, raw)
    return buffer.getvalue()


def verify_public_assets(output):
    output = Path(output)
    _expect(output.is_dir() and not output.is_symlink(), "public evidence directory missing or symlinked")
    _expect(not any(parent.is_symlink() for parent in output.absolute().parents),
            "public evidence parents cannot be symlinks")
    _expect({path.name for path in output.iterdir()} == PUBLIC_NAMES, "public file-set mismatch")
    files = {name: _read(output / name) for name in sorted(PUBLIC_NAMES)}
    manifest = _json(files["manifest.json"])
    _expect(manifest.get("kind") == "paper2bench_selected_repro_evidence" and
            manifest.get("generator") == "scripts/export_repro_demo.py" and
            manifest.get("generator_sha256") == _sha(_read(SCRIPT)), "public generator identity mismatch")
    _expect(set(manifest.get("source_inputs", {})) == set(INPUTS), "public input provenance mismatch")
    _expect(manifest.get("files") == _checksums({k: v for k, v in files.items() if k != "manifest.json"}),
            "public hash manifest mismatch")
    expected = {k: v for k, v in files.items() if k not in {"manifest.json", "evidence-v0.2.zip"}}
    expected["sha256-manifest.json"] = _pretty({"schema_version": 1, "files": _checksums(expected)})
    # Rebuild the small deterministic STORE archive instead of decompressing
    # attacker-controlled ZIP entries during readback verification.
    _expect(_zip(expected) == files["evidence-v0.2.zip"], "public ZIP content or layout mismatch")
    for name, raw in files.items():
        if name.endswith((".json", ".txt")):
            _expect(not SECRET.search(raw) and b"/workspace/" not in raw and b'"hostname"' not in raw and
                    b'"worker"' not in raw and b'"run_id"' not in raw, "private runtime field in public evidence")
    return {"verified": True, "files": len(files), "manifest_sha256": _sha(files["manifest.json"])}


def export(source_dist, output=ROOT / "docs/repro"):
    source_dist, output = Path(source_dist), Path(output)
    _expect(output.resolve() != source_dist.resolve(), "output cannot replace source evidence")
    _expect(not output.is_symlink(), "public output cannot be a symlink")
    _expect(not any(parent.is_symlink() for parent in output.absolute().parents),
            "public output parents cannot be symlinks")
    if output.exists():
        _expect(output.is_dir() and {p.name for p in output.iterdir()} <= PUBLIC_NAMES,
                "refusing to overwrite unrelated public files")
    inputs = {name: _read(source_dist / name) for name in INPUTS}
    generator_raw = _read(SCRIPT)
    files = {
        "verification-v0.2.json": _pretty(_verification(inputs[INPUTS[0]], inputs[INPUTS[1]])),
        "optimization-v0.2.json": _pretty(_optimization(inputs[INPUTS[2]])),
        "seeded-study-v0.2.json": _pretty(_study(inputs[INPUTS[3]], inputs[INPUTS[4]])),
        "README.txt": GUIDE.encode(),
    }
    inner = {**files, "sha256-manifest.json": _pretty({"schema_version": 1, "files": _checksums(files)})}
    files["evidence-v0.2.zip"] = _zip(inner)
    files["manifest.json"] = _pretty({"schema_version": 1, "kind": "paper2bench_selected_repro_evidence",
                                     "generator": "scripts/export_repro_demo.py",
                                     "generator_sha256": _sha(generator_raw),
                                     "source_inputs": _checksums(inputs), "files": _checksums(files)})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".repro-export-", dir=output.parent))
    backup = None
    try:
        for name, raw in files.items():
            (stage / name).write_bytes(raw)
        report = verify_public_assets(stage)
        _expect(all(_read(source_dist / name) == raw for name, raw in inputs.items()) and
                _read(SCRIPT) == generator_raw, "source evidence changed during export")
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=".repro-backup-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup:
                backup.rename(output)
            raise
        if backup:
            shutil.rmtree(backup)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dist", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/repro")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            report = verify_public_assets(args.output)
        else:
            if args.source_dist is None:
                parser.error("--source-dist is required to regenerate recorded evidence")
            report = export(args.source_dist, args.output)
        print(json.dumps(report, sort_keys=True))
    except (ReproExportError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Reproduction evidence export failed: {error}\n")
