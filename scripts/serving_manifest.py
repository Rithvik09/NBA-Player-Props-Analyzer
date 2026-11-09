"""Export/check the observed serving environment without claiming qualification."""
import argparse,json
from pathlib import Path
from src.serving_provenance import ROOT,build_manifest,verify_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,default=ROOT/'models');p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:
        manifest=json.loads(a.verify.read_text());verify_manifest(manifest,a.model_dir);print(manifest['manifest_id']);return
    if not a.output:p.error('--output or --verify required')
    manifest=build_manifest(a.model_dir);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as h:json.dump(manifest,h,indent=2,allow_nan=False)
    lock=a.output.with_suffix('.requirements.txt')
    with lock.open('x') as h:h.write('\n'.join(f'{k}=={v}' for k,v in manifest['runtime'].items() if k!='python' and v)+'\n')
    print(manifest['manifest_id'])

if __name__=='__main__':main()
