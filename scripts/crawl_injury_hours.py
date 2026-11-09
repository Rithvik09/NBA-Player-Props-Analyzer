"""Recover multiple official publication hours per date; missing hours are not missing days."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,subprocess,time
import pandas as pd
import requests
from scripts.crawl_injury_archive import ROOT,OUT,PDF,ROWS,BUNDLED


def fetch(task,python):
    day,clock=task;output=ROWS/(day+'_'+clock+'.json');path=PDF/(day+'_'+clock+'.pdf')
    if output.exists():return {'day':day,'clock':clock,'status':'cached'}
    # In 2026 the official archive added minute tokens to publication names.
    minute_clock=clock[:-2]+'_00'+clock[-2:]
    remote_clocks=[minute_clock,clock] if day>='2026-01-01' else [clock,minute_clock]
    url=f'https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{remote_clocks[0]}.pdf'
    try:
        if not path.exists():
            for remote_clock in remote_clocks:
                url=f'https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{remote_clock}.pdf'
                r=requests.get(url,timeout=10)
                if r.ok and r.content.startswith(b'%PDF'):break
            else:return {'day':day,'clock':clock,'status':'unavailable','http':r.status_code}
            path.write_bytes(r.content)
        result=subprocess.run([python,'-c',"import json,sys;from src.historical_injuries import parse_pdf;print(json.dumps(parse_pdf(sys.argv[1])))",str(path)],cwd=ROOT,capture_output=True,text=True,timeout=60)
        if result.returncode:return {'day':day,'clock':clock,'status':'parse_failed','error':result.stderr[-250:]}
        report=json.loads(result.stdout)
        if report['published_at'][:10]!=day:raise ValueError('Wrong publication day')
        report['url']=url;output.write_text(json.dumps(report))
        return {'day':day,'clock':clock,'status':'parsed','rows':len(report['rows']),'published_at':report['published_at']}
    except Exception as exc:return {'day':day,'clock':clock,'status':'failed','error':type(exc).__name__}


def run(python,workers=4):
    days=set()
    for path in (ROOT/'logs/points_pilot_cache').glob('*.json'):
        if not (path.name.startswith('players_') or path.name.startswith('playoffs_') or path.name.startswith('playin_')):continue
        frame=pd.read_json(path)
        for day in pd.to_datetime(frame.GAME_DATE).dt.normalize().unique():
            for offset in (0,1):days.add((pd.Timestamp(day)-pd.Timedelta(days=offset)).strftime('%Y-%m-%d'))
    tasks=[(d,c) for d in sorted(days) for c in ('12PM','04PM','07PM')];results=[];start=time.time()
    PDF.mkdir(parents=True,exist_ok=True);ROWS.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(fetch,t,python) for t in tasks]):
            results.append(f.result())
            if len(results)%50==0 or len(results)==len(tasks):
                counts={s:sum(r['status']==s for r in results) for s in ('parsed','cached','unavailable','parse_failed','failed')}
                report={'requested_snapshots':len(tasks),'processed':len(results),'counts':counts,'seconds':time.time()-start,'snapshots':results}
                (OUT/'injury_hour_crawl.json').write_text(json.dumps(report,indent=2));print(len(results),len(tasks),counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--python',default=BUNDLED);p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.python,a.workers)
