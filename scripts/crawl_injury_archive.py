"""Cache official injury PDFs and parse with a configured PDF-capable Python.

Every date/format attempt is logged. HTTP errors are coverage gaps, never healthy
players. Two-day report overlap is preserved and actual header time is used.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,subprocess,sys,time
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/feature_recovery';PDF=OUT/'injury_pdfs';ROWS=OUT/'injury_reports'
BUNDLED=sys.executable # Override --python when using a separate PDF runtime.


def fetch(day,python):
    output=ROWS/(day+'.json')
    if output.exists():return {'day':day,'status':'cached'}
    attempts=[]
    for clock in ('05PM','05_00PM','05_30PM'):
        path=PDF/(day+'_'+clock+'.pdf')
        url=f'https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{clock}.pdf'
        try:
            if not path.exists():
                r=requests.get(url,timeout=15)
                attempts.append({'clock':clock,'http':r.status_code})
                if not r.ok or not r.content.startswith(b'%PDF'):continue
                path.write_bytes(r.content)
            result=subprocess.run([python,'-c',"import json,sys;from src.historical_injuries import parse_pdf;print(json.dumps(parse_pdf(sys.argv[1])))",str(path)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            if result.returncode:
                attempts.append({'clock':clock,'parse_error':result.stderr[-300:]});continue
            report=json.loads(result.stdout)
            if report['published_at'][:10]!=day:
                attempts.append({'clock':clock,'wrong_header_day':report['published_at']});continue
            report['url']=url
            output.write_text(json.dumps(report))
            return {'day':day,'status':'parsed','rows':len(report['rows']),'published_at':report['published_at'],'attempts':attempts}
        except (requests.RequestException,subprocess.SubprocessError,ValueError) as exc:attempts.append({'clock':clock,'error':type(exc).__name__})
    return {'day':day,'status':'unavailable','attempts':attempts}


def run(python=BUNDLED,workers=4,limit=0):
    PDF.mkdir(parents=True,exist_ok=True);ROWS.mkdir(parents=True,exist_ok=True)
    days=set()
    for path in (ROOT/'logs/points_pilot_cache').glob('players_*.json'):
        dates=pd.read_json(path).GAME_DATE
        for date in pd.to_datetime(dates).dt.normalize().unique():
            for offset in (0,1):days.add((pd.Timestamp(date)-pd.Timedelta(days=offset)).strftime('%Y-%m-%d'))
    days=sorted(days)
    if limit:days=days[:limit]
    results=[];start=time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,day,python) for day in days]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%20==0 or len(results)==len(days):
                counts={s:sum(r['status']==s for r in results) for s in ('parsed','cached','unavailable')}
                report={'requested_days':len(days),'completed_days':len(results),'counts':counts,'seconds':time.time()-start,'days':results}
                (OUT/'injury_crawl.json').write_text(json.dumps(report,indent=2));print(len(results),len(days),counts,flush=True)
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--python',default=BUNDLED);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();run(a.python,a.workers,a.limit)
