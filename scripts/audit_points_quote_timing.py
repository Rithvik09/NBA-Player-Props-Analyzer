"""Validate archived quote joins/cutoffs and replay a fixed feature sample."""
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from src.prospective_points import digest,save_new
from src.historical_inputs import HistoricalInputs,load_season_logs
from src.audited_features import build_audited_features,SPECS
from src.points_v5 import enrich_injuries,prepare
from scripts.build_real_points_training import validate_record
from scripts.qualify_real_points_v2 import verified_start
from scripts.train_models import _parse_matchup,_team_id

ROOT=Path('logs/points_v7_research')

def run(sample_size=32, output=None):
    bundle=joblib.load(ROOT/'inputs.joblib');d=bundle['data'];quotes=bundle['quotes'];provenance=json.loads(Path('logs/real_points_merged_05/provenance.json').read_text());sources={};margins=[]
    reference=joblib.load('logs/real_points_merged_05/reference_inputs.joblib')
    for source in provenance['sources']:
        folder=Path(source).parent;report=json.loads((folder/'report.json').read_text());q=report['qualification_path']
        sources[source]=(folder,None if q is None else json.loads(Path(q).read_text()))
    verified={}
    for i,(q,source,gid) in enumerate(zip(quotes,provenance['row_sources'],d['games'])):
        folder,qualified=sources[source];summary=json.loads((folder/f'{gid}_summary.json').read_text())
        if qualified:
            e=qualified['evidence'][str(gid)];key=(str(gid),source)
            if key not in verified:
                assert digest(e['espn_path'])==e['espn_sha256'];assert digest(e['nba_pbp_path'])==e['nba_pbp_sha256']
                bound=verified_start(summary,json.loads(Path(e['nba_pbp_path']).read_text()),json.loads(Path(e['espn_path']).read_text())['payload']);assert bound==pd.Timestamp(e['actual_start_lower_bound']);verified[key]=bound
            summary=dict(summary,gameTimeUTC=verified[key].isoformat())
        cutoff=validate_record(q,summary,set());assert cutoff==pd.Timestamp(reference['quote_cutoffs'][i]);assert float(q['line'])==float(d['lines'][i]);margins.append((pd.Timestamp(summary['gameTimeUTC'])-cutoff).total_seconds())
    # Deterministic spread through all source blocks, independent of outcomes/errors.
    rows=list(range(len(quotes))) if sample_size is None else sorted(set(np.linspace(0,len(quotes)-1,sample_size,dtype=int).tolist()));contexts={};results=[]
    cols=joblib.load('logs/points_v6_real_01/repaired.joblib').columns
    expected=prepare(d['X'])
    for i in rows:
        day=pd.Timestamp(d['dates'][i]);year=day.year if day.month>=10 else day.year-1;season=f'{year}-{str(year+1)[2:]}'
        if season not in contexts:
            logs=load_season_logs(season,include_postseason=True);contexts[season]=(HistoricalInputs(season,joblib.load('logs/expanded_points_v4/source_snapshot.joblib'),logs,audited=True),{(str(r.GAME_ID),int(r.PLAYER_ID)):r for r in logs.itertuples()})
        inputs,lookup=contexts[season];gid=str(d['games'][i]);pid=int(d['players'][i]);row=lookup[(gid,pid)];own,opp,home=_parse_matchup(row.MATCHUP)
        prior=inputs.player_log(pid);prior=prior[pd.to_datetime(prior.GAME_DATE)<day];assert pd.to_datetime(prior.GAME_DATE).max()<day
        inputs.prediction_game_id=gid;inputs.prediction_cutoff=pd.Timestamp(quotes[i]['captured_at']);inputs.recovery._injury_cache={};inputs.recovery._cutoff_cache={}
        f,gaps=build_audited_features(prior,'PTS',pid,_team_id(own),_team_id(opp),day,home,inputs)
        for k in gaps:
            if SPECS[k].status=='active':f[k]=np.nan
        f=enrich_injuries(f,inputs.recovery,_team_id(own),_team_id(opp),day)
        actual=prepare(pd.DataFrame([f]))[cols].iloc[0].to_numpy(dtype=float);want=expected[cols].iloc[i].to_numpy(dtype=float)
        bad=[k for k,ok in zip(cols,np.isclose(actual,want,equal_nan=True)) if not ok]
        results.append({'row':int(i),'game':gid,'player':pid,'different_columns':bad})
        assert not bad,(gid,pid,bad)
    save_new(output or ROOT/'quote_timing_audit.json',{'all_quote_rows_validated':len(quotes),'line_and_cutoff_alignment':True,'independently_rechecked_game_source_pairs':len(verified),'feature_replay_rows':results,'feature_count':len(cols),'minimum_seconds_before_verified_start':min(margins),'seconds_before_start_quantiles':np.quantile(margins,[0,.25,.5,.75,1]).tolist(),'limitation':'Verifies archived metadata and point-in-time construction, not independent proof that provider-recorded capture times are accurate. Different injury availability between synthetic one-hour forecasts and real quote cutoffs is expected.'})
    print('PASS',len(quotes),'quotes;',len(rows),'full174-feature replays')

if __name__=='__main__':run()
