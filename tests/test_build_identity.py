"""Independent source identity and stale-build rejection; standard library only."""
import hashlib
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from build_identity import source_fingerprint,require_source_build

class Identity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name,value in [('CMakeLists.txt',b'project(test)\n'),('pyproject.toml',b'[project]\n'),('tools/build_identity.py',b'# identity\n'),('src/a.cpp',b'int a;\n'),('include/x/a.hpp',b'int b;\n')]:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
    def test_independent_fingerprint_and_irrelevant_files(self):
        names=['CMakeLists.txt','include/x/a.hpp','pyproject.toml','src/a.cpp','tools/build_identity.py']
        material=''.join(name+':'+hashlib.sha256((self.root/name).read_bytes()).hexdigest()+'\n' for name in names)
        expected=hashlib.sha256(material.encode()).hexdigest()
        self.assertEqual(source_fingerprint(self.root),expected)
        (self.root/'notes.md').write_text('notes')
        self.assertEqual(source_fingerprint(self.root),expected)
    def test_changes_additions_and_removals_change_identity(self):
        before=source_fingerprint(self.root)
        p=self.root/'src/a.cpp';p.write_text('int c;\n')
        self.assertNotEqual(source_fingerprint(self.root),before)
        before=source_fingerprint(self.root);p.unlink()
        self.assertNotEqual(source_fingerprint(self.root),before)
        before=source_fingerprint(self.root);(self.root/'src/new.hpp').write_text('int d;')
        self.assertNotEqual(source_fingerprint(self.root),before)
    def test_worker_rejects_stale_extension_before_loading_or_timing(self):
        # Stub only the imports reached before the gate; no NumPy install needed.
        (self.root/'numpy.py').write_text("__version__='stub'\n")
        for body in ("def build_info(): return {'configuration':'Release','source_fingerprint':'"+'0'*64+"'}\n", "# old extension without build metadata\n"):
            (self.root/'stealth_cpu.py').write_text(body)
            result=subprocess.run([sys.executable,ROOT/'tools/runtime_worker.py','--engine','native_python','--model','missing-model','--input','missing-input','--expected','missing-reference'],
                                  env={**os.environ,'PYTHONPATH':str(self.root),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(result.stdout,'')
            self.assertIn('Rebuild and reinstall',result.stderr)
            self.assertNotIn('FileNotFoundError',result.stderr)

    def test_missing_required_files_rejected(self):
        (self.root/'CMakeLists.txt').unlink()
        with self.assertRaises(FileNotFoundError):source_fingerprint(self.root)
    def test_stale_missing_and_wrong_configuration_rejected(self):
        info=dict(configuration='Release',source_fingerprint=source_fingerprint(self.root))
        require_source_build(info,self.root)
        for bad in ({},dict(info,source_fingerprint='0'*64),dict(info,configuration='Debug')):
            with self.assertRaisesRegex(ValueError,'Rebuild'):require_source_build(bad,self.root)
        (self.root/'src/a.cpp').write_text('int c;')
        with self.assertRaisesRegex(ValueError,'Rebuild'):require_source_build(info,self.root)

if __name__=='__main__':unittest.main()
