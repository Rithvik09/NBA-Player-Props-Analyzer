"""Verify an experiment and replay/refit using its preserved source, offline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def check_inventory(output):
    output=Path(output).resolve();manifest=json.loads((output/'manifest.json').read_text())
    if manifest['python']!=sys.version:raise ValueError('Python runtime drift')
    for relative,expected in manifest['files'].items():
        path=(output/relative).resolve()
        if not path.is_relative_to(output):raise ValueError('Invalid inventory path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Experiment file drift: '+relative)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--refit',action='store_true');a=p.parse_args()
    output=check_inventory(a.output)
    # A new interpreter ensures no live-worktree modules precede the snapshot.
    code='''import sys,socket,json
from pathlib import Path
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Network forbidden'))
sys.path.insert(0,sys.argv[1])
from scripts.train_traceable_points import verify
print(json.dumps(verify(Path(sys.argv[2]),sys.argv[3]=='1'),indent=2))
'''
    subprocess.run([sys.executable,'-c',code,str(output/'source'),str(output),'1' if a.refit else '0'],cwd=output,check=True)

if __name__=='__main__':main()
