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
def counts(xml):
    if not xml.exists():return None
    tree=ET.parse(xml).getroot()
    cases=tree.findall(".//testcase")
    return {"collectedCases":len(cases),"failed":sum(c.find("failure") is not None for c in cases),"errors":sum(c.find("error") is not None for c in cases),"skipped":sum(c.find("skipped") is not None for c in cases)}
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
    timed_out=False
    try:
        result=subprocess.run([sys.executable,*COMMAND,"--junitxml="+str(xml)],cwd=ROOT,timeout=180,capture_output=True)
        exitcode=result.returncode
    except subprocess.TimeoutExpired:
        timed_out=True;exitcode=None
    elapsed=time.monotonic()-clock
    summary=counts(xml)
    passed=exitcode==0 and summary is not None and summary["collectedCases"]>0 and summary["failed"]==summary["errors"]==0
    receipt={"schemaVersion":1,"projectId":"paper2bench","repository":"cal2020/Paper2Bench","sourceCommit":head,"sourceSnapshotId":"cal2020/Paper2Bench@"+head,"sourceURL":"https://github.com/cal2020/Paper2Bench/commit/"+head,"observedAt":start,"execution":origin,"check":{"id":"demo-source-admission","command":["python",*COMMAND],"status":"passed" if passed else "failed","exitCode":exitcode,"timedOut":timed_out,"durationSeconds":elapsed,"durationMeaning":"Measured wall-clock duration of this check process only; not human work hours.","tests":summary,"scope":"Build admission and release artifact regression tests only. Tests include explicit fixtures; this receipt records actual execution, not model or scientific results."},"publicFiles":files,"hours":None,"tokens":None,"deployment":None,"boundary":"No inferred hours, model tokens, deployment, full-suite success or scientific correctness. Aggregate test metadata only; no logs, environment, paths or test bodies exported."}
    (output/"source-evidence.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"sourceCommit":head,"status":receipt["check"]["status"],"durationSeconds":elapsed,"tests":summary}))
    return 0 if passed else 1
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);raise SystemExit(measure(p.parse_args().output))
