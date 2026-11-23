"""Receipt safety tests; fixtures never become reported runtime evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('source_evidence',Path(__file__).resolve().parents[1]/'scripts/source_evidence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class EvidenceTests(unittest.TestCase):
 def test_local_origin_not_github(self):self.assertEqual(m.provenance({},'a'*40),{'kind':'local','workflowRun':None})
 def test_source_mismatch(self):
  with self.assertRaises(ValueError):m.provenance({'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'other/repo','GITHUB_SHA':'a'*40},'a'*40)
 def test_run_identity(self):
  e={'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'cal2020/Paper2Bench','GITHUB_SHA':'a'*40,'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'2','GITHUB_EVENT_NAME':'push'}
  self.assertEqual(m.provenance(e,'a'*40)['attempt'],2)
  e['GITHUB_RUN_ID']='../secret'
  with self.assertRaises(ValueError):m.provenance(e,'a'*40)
 def test_missing_counts_unknown(self):self.assertIsNone(m.counts(Path('/nonexistent-evidence-file')))
 def test_aggregate_only(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'unit.xml';p.write_text('<testsuites><testsuite tests="2" failures="1" errors="0" skipped="1"><testcase classname="private" name="private"><failure>private log</failure></testcase><testcase classname="private" name="different"><skipped/></testcase></testsuite></testsuites>')
   s=m.counts(p)
   self.assertEqual({k:s[k] for k in ('collectedCases','executedCases','passed','failed','errors','skipped')},{'collectedCases':2,'executedCases':1,'passed':0,'failed':1,'errors':0,'skipped':1})
   self.assertNotIn('private',str(s))
 def test_allowlist_rejects_symlink(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for n in m.FILES:
    p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
   p=root/m.FILES[0];p.unlink();p.symlink_to(root/m.FILES[1])
   with self.assertRaises(ValueError):m.public_files(root)



class MemoryXML:
 def __init__(self,data):self.data=data
 def open(self,mode):
  from io import BytesIO
  if self.data is None:raise FileNotFoundError
  return BytesIO(self.data)


def junit(cases):
 from xml.etree.ElementTree import Element, SubElement, tostring
 root=Element("testsuites")
 statuses=("failure","error","skipped")
 suite=SubElement(root,"testsuite",{"tests":str(len(cases)),
  **{("failures" if status=="failure" else "errors" if status=="error" else status):
  str(sum(outcome==status for _,outcome in cases)) for status in statuses}})
 for (module,name),outcome in cases:
  case=SubElement(suite,"testcase",{"classname":module,"name":name})
  if outcome:SubElement(case,outcome).text="fixture details must not be exposed"
 return MemoryXML(tostring(root,encoding="utf-8"))


class EvidenceCompletenessTests(unittest.TestCase):
 def test_complete_required_and_one_optional_skip_pass(self):
  cases=[(identity,None) for identity in sorted(m.REQUIRED)]
  cases.extend((identity,"skipped") for identity in m.OPTIONAL_SKIPS)
  from unittest.mock import patch
  with patch.object(m,'supports_fifo',return_value=False):
   summary,reasons=m.result_evidence(junit(cases),0)
  self.assertEqual(reasons,[])
  self.assertEqual(summary["requiredPassed"],len(m.REQUIRED))
  self.assertEqual(summary["executedCases"],len(m.REQUIRED))
  self.assertEqual(summary["optionalSkipped"],1)

 def test_fifo_skip_is_not_optional_when_platform_supports_it(self):
  from unittest.mock import patch
  cases=[(identity,None) for identity in sorted(m.REQUIRED)]
  cases.extend((identity,"skipped") for identity in m.OPTIONAL_SKIPS)
  with patch.object(m,'supports_fifo',return_value=True):
   summary,reasons=m.result_evidence(junit(cases),0)
  self.assertEqual(summary["unexpectedSkipped"],1)
  self.assertEqual(reasons,["unexpected_tests_skipped"])

 def test_all_skips_cannot_pass_on_zero_exit(self):
  summary,reasons=m.result_evidence(junit([(identity,"skipped") for identity in sorted(m.REQUIRED)]),0)
  self.assertEqual(summary["executedCases"],0)
  self.assertEqual(summary["passed"],0)
  self.assertEqual(reasons,["no_executed_tests","required_tests_skipped"])

 def test_missing_skipped_required_and_unexpected_skip_fail(self):
  required=sorted(m.REQUIRED)
  cases=[(identity,None) for identity in required[:-2]]
  cases.extend([(required[-2],"skipped"),(("fixture","other"),"skipped")])
  summary,reasons=m.result_evidence(junit(cases),0)
  self.assertEqual((summary["requiredMissing"],summary["requiredSkipped"],summary["unexpectedSkipped"]),(1,1,1))
  self.assertEqual(reasons,["required_tests_missing","required_tests_skipped","unexpected_tests_skipped"])

 def test_only_executed_reports_are_not_enough(self):
  summary,reasons=m.result_evidence(junit([(("fixture","some_passing_test"),None)]),0)
  self.assertEqual(summary["executedCases"],1)
  self.assertEqual(summary["requiredMissing"],len(m.REQUIRED))
  self.assertEqual(reasons,["required_tests_missing"])

 def test_missing_malformed_or_duplicate_identity_is_invalid(self):
  repeated=next(iter(sorted(m.REQUIRED)))
  for report in (MemoryXML(b"<testsuites>"),junit([(repeated,None),(repeated,None)]),
                 junit([(("","broken"),None)]),
                 MemoryXML(b'<testsuite tests="5" failures="0" errors="0" skipped="0"/>'),
                 MemoryXML(b"<!DOCTYPE testsuite><testsuite/>")):
   self.assertEqual(m.result_evidence(report,0),(None,["invalid_junit_report"]))
  self.assertEqual(m.result_evidence(MemoryXML(None),0),(None,["missing_junit_report"]))

 def test_failed_errored_and_process_flags_are_reported_without_private_text(self):
  cases=[(identity,None) for identity in sorted(m.REQUIRED)]
  cases.extend([(("fixture","fail"),"failure"),(("fixture","setup_error"),"error")])
  summary,reasons=m.result_evidence(junit(cases),1)
  self.assertEqual(reasons,["test_process_failed","test_failures_or_errors"])
  self.assertEqual((summary["failed"],summary["errors"],summary["executedCases"]),(1,1,len(m.REQUIRED)+1))
  self.assertNotIn("fixture",str(summary))
  self.assertEqual(m.result_evidence(MemoryXML(None),None,timed_out=True)[1],
                   ["test_process_timed_out","missing_junit_report"])
  self.assertEqual(m.result_evidence(MemoryXML(None),None,start_failed=True)[1],
                   ["test_process_not_started","missing_junit_report"])

if __name__=='__main__':unittest.main()
