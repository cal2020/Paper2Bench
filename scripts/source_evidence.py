"""Measure the existing source-admission checks; never equate them to deployment."""
import argparse,datetime,hashlib,json,os,pathlib,re,subprocess,sys,time,xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parents[1]
FILES=("scripts/source_evidence.py","scripts/build_demo.py","scripts/package_demo.py","tests/test_demo_build.py","tests/test_release_artifact.py","tests/test_source_evidence.py","docs/SOURCE_EVIDENCE.md",".github/workflows/demo.yml")
COMMAND=("-m","pytest","-q","tests/test_demo_build.py","tests/test_release_artifact.py","tests/test_source_evidence.py")
def public_files(root):
    rows=[]
    for name in FILES:
        p=root
        for part in pathlib.PurePosixPath(name).parts:
            p=p/part
            if p.is_symlink():raise ValueError("Public allowlist contains symlink")
        if not p.is_file() or not p.resolve().is_relative_to(root.resolve()):raise ValueError("Missing public source file")
        b=p.read_bytes()
        if len(b)>2000000:raise ValueError("Oversized public source")
        rows.append({"path":name,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()})
    return rows
# Pytest JUnit identities, independent of report order or human skip messages.
REQUIRED = frozenset({
    ("tests.test_demo_build", "test_release_build_is_reproducible_and_has_complete_local_script_closure"),
    ("tests.test_demo_build", "test_source_change_during_build_does_not_replace_prior_artifact"),
    ("tests.test_demo_build", "test_release_output_inside_repository_does_not_dirty_its_source"),
    ("tests.test_release_artifact", "test_archive_is_deterministic_and_contains_only_verified_payloads"),
    ("tests.test_release_artifact", "test_safe_extract_round_trip_admits_expected_commit"),
})
OPTIONAL_SKIPS = frozenset({
    ("tests.test_demo_build", "test_fifo_source_is_rejected_without_blocking"),
})
def supports_fifo():
    return hasattr(os, "mkfifo")


MAX_JUNIT_BYTES = 2 * 1024 * 1024
OUTCOMES = {"failure": "failed", "error": "errors", "skipped": "skipped"}


def counts(xml):
    """Return only validated aggregate counts; never expose JUnit text or IDs."""
    try:
        with xml.open("rb") as handle:
            raw = handle.read(MAX_JUNIT_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(raw) > MAX_JUNIT_BYTES or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("invalid JUnit")
    decoded = raw.decode("utf-8")
    if "\x00" in decoded:
        raise ValueError("invalid JUnit")
    root = ET.fromstring(decoded)
    if root.tag == "testsuites" and all(node.tag == "testsuite" for node in root):
        suites = list(root)
    elif root.tag == "testsuite":
        suites = [root]
    else:
        raise ValueError("invalid JUnit")
    if not suites:
        raise ValueError("invalid JUnit")
    cases = set()
    category = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    required_passed = required_skipped = optional_skipped = unexpected_skipped = 0
    for suite in suites:
        suite_stats = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
        for case in suite.findall("testcase"):
            identity = case.get("classname"), case.get("name")
            if any(not s or len(s) > 512 or any(ord(c) < 32 for c in s) for s in identity) or identity in cases:
                raise ValueError("invalid JUnit")
            cases.add(identity)
            seen = [OUTCOMES[child.tag] for child in case if child.tag in OUTCOMES]
            if len(seen) > 1:
                raise ValueError("invalid JUnit")
            outcome = seen[0] if seen else "passed"
            category[outcome] += 1
            suite_stats["tests"] += 1
            if outcome != "passed":
                suite_stats["failures" if outcome == "failed" else outcome] += 1
            if identity in REQUIRED:
                required_passed += outcome == "passed"
                required_skipped += outcome == "skipped"
            elif outcome == "skipped":
                if identity in OPTIONAL_SKIPS and not supports_fifo():
                    optional_skipped += 1
                else:
                    unexpected_skipped += 1
        if any(suite.get(key) != str(value) for key, value in suite_stats.items()):
            raise ValueError("invalid JUnit")
    return {
        "collectedCases": len(cases),
        "executedCases": category["passed"] + category["failed"],
        **category,
        "requiredCases": len(REQUIRED),
        "requiredPassed": required_passed,
        "requiredSkipped": required_skipped,
        "requiredMissing": len(REQUIRED - cases),
        "optionalSkipped": optional_skipped,
        "unexpectedSkipped": unexpected_skipped,
    }


def result_evidence(xml, exitcode, *, timed_out=False, start_failed=False):
    reasons = []
    if timed_out:
        reasons.append("test_process_timed_out")
    elif start_failed:
        reasons.append("test_process_not_started")
    elif exitcode != 0:
        reasons.append("test_process_failed")
    try:
        summary = counts(xml)
    except (ET.ParseError, OSError, ValueError):
        return None, reasons + ["invalid_junit_report"]
    if summary is None:
        return None, reasons + ["missing_junit_report"]
    for condition, reason in (
        (not summary["executedCases"], "no_executed_tests"),
        (summary["failed"] or summary["errors"], "test_failures_or_errors"),
        (summary["requiredMissing"], "required_tests_missing"),
        (summary["requiredSkipped"], "required_tests_skipped"),
        (summary["unexpectedSkipped"], "unexpected_tests_skipped"),
    ):
        if condition:
            reasons.append(reason)
    return summary, reasons

def provenance(env,head):
    if env.get("GITHUB_ACTIONS")!="true":return {"kind":"local","workflowRun":None}
    if env.get("GITHUB_REPOSITORY")!="cal2020/Paper2Bench" or env.get("GITHUB_SHA")!=head:raise ValueError("Workflow source mismatch")
    run=env.get("GITHUB_RUN_ID","");attempt=env.get("GITHUB_RUN_ATTEMPT","")
    if not run.isdigit() or not attempt.isdigit():raise ValueError("Invalid workflow identity")
    return {"kind":"github-actions","workflowRun":"https://github.com/cal2020/Paper2Bench/actions/runs/"+run,"attempt":int(attempt),"event":env.get("GITHUB_EVENT_NAME")}
def measure(output):
    output=pathlib.Path(output).resolve()
    if output.exists():raise ValueError("Fresh output directory required")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if not re.fullmatch("[a-f0-9]{40}",head):raise ValueError("Invalid source revision")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=normal"],cwd=ROOT,text=True).strip():raise ValueError("Commit reviewed source before measuring")
    files=public_files(ROOT);origin=provenance(os.environ,head)
    output.mkdir(parents=True,mode=0o700);xml=output/"source-tests.xml"
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
    timed_out=start_failed=False
    try:
        result=subprocess.run([sys.executable,*COMMAND,"--junitxml="+str(xml)],cwd=ROOT,timeout=180,capture_output=True)
        exitcode=result.returncode
    except subprocess.TimeoutExpired:
        timed_out=True;exitcode=None
    except OSError:
        start_failed=True;exitcode=None
    elapsed=time.monotonic()-clock
    summary,reasons=result_evidence(xml,exitcode,timed_out=timed_out,start_failed=start_failed)
    passed=not reasons
    receipt={"schemaVersion":1,"projectId":"paper2bench","repository":"cal2020/Paper2Bench","sourceCommit":head,"sourceSnapshotId":"cal2020/Paper2Bench@"+head,"sourceURL":"https://github.com/cal2020/Paper2Bench/commit/"+head,"observedAt":start,"execution":origin,"check":{"id":"demo-source-admission","command":["python",*COMMAND],"status":"passed" if passed else "failed","exitCode":exitcode,"timedOut":timed_out,"failureReasons":reasons,"durationSeconds":elapsed,"durationMeaning":"Measured wall-clock duration of this check process only; not human work hours.","tests":summary,"scope":"Build admission and release artifact regression tests only. Tests include explicit fixtures; this receipt records actual execution, not model or scientific results."},"publicFiles":files,"hours":None,"tokens":None,"deployment":None,"boundary":"No inferred hours, model tokens, deployment, full-suite success or scientific correctness. Aggregate test metadata only; no logs, environment, paths or test bodies exported."}
    (output/"source-evidence.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"sourceCommit":head,"status":receipt["check"]["status"],"durationSeconds":elapsed,"tests":summary}))
    return 0 if passed else 1
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);raise SystemExit(measure(p.parse_args().output))
