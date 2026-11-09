"""Completed-season recovery sources, cached by season with explicit failures."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,time
import numpy as np
from src import precompute_jobs as jobs
from nba_api.stats.endpoints import leaguedashteamstats,leaguedashptstats,leaguedashplayerstats,leaguedashplayerclutch,leaguedashplayerptshot,leaguedashlineups

OUT=Path('logs/feature_recovery');OUT.mkdir(exist_ok=True)


def records(frame,key):
    if frame.empty:raise ValueError('Empty provider response')
    return {str(int(r[key])):{str(k):None if not np.isfinite(float(v)) else float(v) for k,v in r.items() if isinstance(v,(int,float,np.integer,np.floating))} for r in frame.to_dict('records')}


def fetch(season):
    path=OUT/f'snapshots_{season}.json'
    state=json.loads(path.read_text()) if path.exists() else {'source_season':season,'sources':{},'failures':{}}
    tasks={
        'opponent_zones':lambda:{str(r['team_id']):r for r in jobs.compute_opp_shot_zones(season)},
        'opponent_synergy':lambda:{str(r['team_id']):r for r in jobs.compute_synergy_team_defense(season)},
        'team_misc':lambda:records(leaguedashteamstats.LeagueDashTeamStats(season=season,measure_type_detailed_defense='Misc',per_mode_detailed='PerGame',timeout=30).get_data_frames()[0],'TEAM_ID'),
        'team_base':lambda:records(leaguedashteamstats.LeagueDashTeamStats(season=season,per_mode_detailed='PerGame',timeout=30).get_data_frames()[0],'TEAM_ID'),
        'player_possessions':lambda:records(leaguedashptstats.LeagueDashPtStats(player_or_team='Player',season=season,pt_measure_type='Possessions',per_mode_simple='PerGame',timeout=30).get_data_frames()[0],'PLAYER_ID'),
        'player_rebounding':lambda:records(leaguedashptstats.LeagueDashPtStats(player_or_team='Player',season=season,pt_measure_type='Rebounding',per_mode_simple='PerGame',timeout=30).get_data_frames()[0],'PLAYER_ID'),
        'player_misc':lambda:records(leaguedashplayerstats.LeagueDashPlayerStats(season=season,measure_type_detailed_defense='Misc',per_mode_detailed='PerGame',timeout=30).get_data_frames()[0],'PLAYER_ID'),
        'player_q4':lambda:records(leaguedashplayerstats.LeagueDashPlayerStats(season=season,period=4,per_mode_detailed='PerGame',timeout=30).get_data_frames()[0],'PLAYER_ID'),
        'player_clutch_advanced':lambda:records(leaguedashplayerclutch.LeagueDashPlayerClutch(season=season,measure_type_detailed_defense='Advanced',per_mode_detailed='PerGame',timeout=30).get_data_frames()[0],'PLAYER_ID'),
    }
    tasks['bench']=lambda:records(leaguedashteamstats.LeagueDashTeamStats(season=season,starter_bench_nullable='Bench',per_mode_detailed='PerGame',timeout=30).get_data_frames()[0],'TEAM_ID')
    def lineup():
        from nba_api.stats.static import teams
        values={}
        for team in teams.get_teams():
            frame=leaguedashlineups.LeagueDashLineups(season=season,group_quantity=5,team_id_nullable=team['id'],measure_type_detailed_defense='Advanced',per_mode_detailed='Totals',timeout=20).get_data_frames()[0]
            if frame.empty or len(frame)>=2000:raise ValueError('Empty or truncated team lineup response')
            for r in frame.to_dict('records'):
                token=str(r['TEAM_ID'])+':'+str(r['GROUP_ID'])
                values[token]={'group_id':str(r['GROUP_ID']),'team_id':int(r['TEAM_ID']),'minutes':float(r['MIN']),'net_rating':float(r['NET_RATING'])}
        state['lineup_team_complete']=30
        return values
    tasks['lineups']=lineup
    for pos in ('G','F','C'):
        tasks['position_'+pos]=lambda pos=pos:records(leaguedashplayerstats.LeagueDashPlayerStats(season=season,player_position_abbreviation_nullable=pos,per_mode_detailed='PerGame',timeout=30).get_data_frames()[0],'PLAYER_ID')
    for bucket,clock in [('clock_very_early','22-18 Very Early'),('clock_early24','24-22'),('clock_late','4-0 Very Late')]:
        tasks[bucket]=lambda clock=clock:records(leaguedashplayerptshot.LeagueDashPlayerPtShot(season=season,shot_clock_range_nullable=clock,per_mode_simple='PerGame',timeout=30).get_data_frames()[0],'PLAYER_ID')
    if state.get('lineup_team_complete')!=30:state['sources'].pop('lineups',None)
    if state.get('zone_counts_revision')!=2:state['sources'].pop('opponent_zones',None)
    for name,fn in tasks.items():
        if state['sources'].get(name):continue
        try:
            values=fn()
            if not values:raise ValueError('Empty source')
            state['sources'][name]=values;state['failures'].pop(name,None)
            if name=='opponent_zones':state['zone_counts_revision']=2
            print(season,name,len(values),flush=True)
        except Exception as exc:
            state['failures'][name]=str(exc)[:200];print(season,name,'FAILED',type(exc).__name__,flush=True)
        path.write_text(json.dumps(state));time.sleep(.4)
    return {k:len(v) for k,v in state['sources'].items()}

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        print(list(pool.map(fetch,['2021-22','2022-23','2023-24','2024-25','2025-26'])),flush=True)
