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
   p=Path(d)/'unit.xml';p.write_text('<testsuites><testsuite><testcase name="private"><failure>private log</failure></testcase><testcase><skipped/></testcase></testsuite></testsuites>');s=m.counts(p);self.assertEqual(s,{'collectedCases':2,'failed':1,'errors':0,'skipped':1});self.assertNotIn('private',str(s))
 def test_allowlist_rejects_symlink(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for n in m.FILES:
    p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
   p=root/m.FILES[0];p.unlink();p.symlink_to(root/m.FILES[1])
   with self.assertRaises(ValueError):m.public_files(root)
if __name__=='__main__':unittest.main()
