"""Complete the prior-season source prerequisites without overwriting team snapshots."""
import json
from pathlib import Path
import time
from src import db,precompute_jobs as jobs
from scripts.rebuild_precomputed import PLAN,run_stage,configure_http

OUT=Path('logs/full_feature_audit')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    configure_http()
    with db.connect() as conn:
        jobs.ensure_tables(conn)
    checkpoint=OUT/'prerequisite_crawl.json'
    report=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    season='2021-22';stages=report.setdefault(season,{})
    for spec in PLAN:
        if not spec[3]:continue
        result=run_stage(spec,season,stages,attempts=2)
        stages[spec[0]]=result;checkpoint.write_text(json.dumps(report,indent=2))
        print(spec[0],result,flush=True)
    # Recover exact historical scoring shares from already stored disjoint
    # scoring methods. Save pre-repair values first for reproducibility.
    with db.connect(row_factory=db.dict_row) as conn:
        prior=[dict(r) for r in conn.execute('SELECT * FROM player_scoring_breakdown').fetchall()]
        if not (OUT/'scoring_before_repair.json').exists():
            (OUT/'scoring_before_repair.json').write_text(json.dumps(prior,default=str))
    for season in ('2021-22','2022-23','2023-24','2024-25','2025-26'):
        rows=jobs.compute_player_scoring_breakdown(season)
        if not rows:raise ValueError('Scoring recollection failed: '+season)
        with db.connect() as conn:
            jobs.upsert_player_scoring_breakdown(conn,rows,int(time.time()),season=season)
        print('recollected scoring',season,len(rows),flush=True)
    with db.connect(row_factory=db.dict_row) as conn:
        summary=[dict(r) for r in conn.execute('''SELECT season,COUNT(*) AS rows,
            COUNT(*) FILTER(WHERE pct_pts_midrange>0) AS nonzero_midrange,
            MAX(ABS(pct_uast_fgm+pct_ast_fgm-1)) FILTER (WHERE pct_uast_fgm+pct_ast_fgm>0) AS assisted_sum_error
            FROM player_scoring_breakdown GROUP BY season ORDER BY season''').fetchall()]
    (OUT/'scoring_repair.json').write_text(json.dumps(summary,indent=2,default=str))
    print('scoring repair',summary,flush=True)

if __name__=='__main__':run()
