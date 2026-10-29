"""Bind a benchmark to its built source; no Git or runtime dependency for hashing."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=('CMakeLists.txt','pyproject.toml','tools/build_identity.py')
PATTERNS={'src':('*.cpp','*.hpp'),'include':('*.hpp',),'bindings':('*.cpp',),
          'python':('*.py',),'cmake':('*.in','*.cmake'),'app':('*.cpp',),'bench':('*.cpp',)}


def source_fingerprint(root=ROOT):
    root=Path(root)
    paths={root/name for name in REQUIRED}
    for directory,patterns in PATTERNS.items():
        for pattern in patterns: paths.update((root/directory).rglob(pattern))
    material=''.join(p.relative_to(root).as_posix()+':'+hashlib.sha256(p.read_bytes()).hexdigest()+'\n'
                     for p in sorted(paths,key=lambda p:p.relative_to(root).as_posix()))
    return hashlib.sha256(material.encode('utf-8')).hexdigest()


def require_source_build(info,root=ROOT):
    if (info.get('configuration')!='Release' or
            info.get('source_fingerprint')!=source_fingerprint(root)):
        raise ValueError('Rebuild and reinstall Stealth from this source tree in Release mode before benchmarking')


def extension_identity(root=ROOT):
    import stealth_cpu as native
    info=getattr(native,'build_info',lambda:{})()
    require_source_build(info,root)
    path=Path(importlib.import_module('stealth_cpu._core').__file__)
    return dict(build_info=info,extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-extension',action='store_true')
    args=parser.parse_args()
    result=extension_identity() if args.check_extension else dict(source_fingerprint=source_fingerprint())
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
