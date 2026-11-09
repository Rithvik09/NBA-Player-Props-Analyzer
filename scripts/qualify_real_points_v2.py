"""Separate historical research qualification; never changes prospective cutoffs."""
import json
import re
from pathlib import Path
import pandas as pd
from src.historical_inputs import load_season_logs
from src.historical_injuries import normalize_name
from src.prospective_points import digest, save_new
from scripts.build_real_points_training import validate_record
from scripts.train_models import _parse_matchup

ROOT = Path('logs/real_points_sources')


def verified_start(summary, pbp, espn):
    if str(pbp['game']['gameId']) != str(summary['gameId']):
        raise ValueError('Wrong NBA play-by-play game')
    a = pbp['game']['actions'][0]
    if (a['period'], a['clock'], a['actionType'], a['subType']) != (1, 'PT12M00.00S', 'period', 'start'):
        raise ValueError('Missing opening period marker')
    m = re.fullmatch(r'Start of 1st Period \((\d+:\d+ [AP]M) EST\)', a['description'])
    if not m:
        raise ValueError('Unknown official start format')
    scheduled = pd.Timestamp(summary['gameTimeUTC'])
    day = scheduled.tz_convert('America/New_York').date().isoformat()
    # NBA labels Eastern local clock EST even in daylight time. Require independent
    # UTC jump-ball corroboration within one minute; use the earlier lower bound.
    official = pd.Timestamp(pd.to_datetime(day+' '+m[1], format='%Y-%m-%d %I:%M %p')).tz_localize('America/New_York')
    p = espn['plays'][0]
    if p['period']['number'] != 1 or p['clock']['displayValue'] != '12:00' or p['type']['text'] != 'Jumpball' or p['homeScore'] or p['awayScore']:
        raise ValueError('Missing opening jump ball')
    jump = pd.Timestamp(p['wallclock'])
    if jump.tzinfo is None or abs((jump-official).total_seconds()) > 60:
        raise ValueError('Conflicting actual-start evidence')
    start = min(jump, official)
    if not 0 <= (start-scheduled).total_seconds() <= 3600:
        raise ValueError('Implausible actual start')
    return start


def run(quotes_path=ROOT/'probe_combined.json', out=Path('logs/real_points_qualified_v2_final'), summaries_dir=Path('logs/real_points_pilot_02')):
    out.mkdir(exist_ok=False)
    quotes = json.loads(quotes_path.read_text())['props']
    logs, evidence, accepted, rejected = {}, {}, [], []
    summaries = {str(x['gameId']):x for x in [json.loads(p.read_text()) for p in summaries_dir.glob('*_summary.json')]}
    for q in quotes:
        try:
            day = pd.Timestamp(q['game_date']); year = day.year if day.month >= 10 else day.year-1
            season = f'{year}-{str(year+1)[2:]}'
            if season not in logs:
                logs[season] = load_season_logs(season, include_postseason=True)
                logs[season]['_normalized_name'] = logs[season].PLAYER_NAME.map(normalize_name)
                logs[season]['_date'] = pd.to_datetime(logs[season].GAME_DATE)
            ep = ROOT/f"espn_summary_{q['game_id']}.json"
            espn = json.loads(ep.read_text())['payload']
            if str(espn['header']['id']) != str(q['game_id']): raise ValueError('Wrong provider event')
            provider_id = q['player_id']
            alias_nba_id = None
            if provider_id is None:
                metadata = json.loads((ROOT/'nba_carrington_identity.json').read_text())['resultSets'][0]
                bio = dict(zip(metadata['headers'], metadata['rowSet'][0]))
                if normalize_name(q['player_name']) != normalize_name(bio['PLAYERCODE'].replace('_',' ')):
                    raise ValueError('Missing provider ID without verified alias')
                alias_nba_id = int(bio['PERSON_ID'])
                roster_matches = [a['athlete'] for t in espn['boxscore']['players'] for s in t['statistics'] for a in s['athletes'] if a['athlete']['displayName']==bio['DISPLAY_FIRST_LAST']]
                if len(roster_matches)!=1: raise ValueError('Ambiguous alias roster')
                provider_id = roster_matches[0]['id']
            athletes = [(a['athlete'],t['team']['abbreviation']) for t in espn['boxscore']['players'] for s in t['statistics'] for a in s['athletes'] if str(a['athlete']['id']) == str(provider_id)]
            if len(athletes) != 1: raise ValueError('Ambiguous provider player ID')
            athlete, team = athletes[0]; team = {'WSH':'WAS','NY':'NYK','GS':'GSW','SA':'SAS','NO':'NOP','UTAH':'UTA'}.get(team,team)
            df = logs[season]; candidates = df[(df._date==day) & (df._normalized_name==normalize_name(athlete['displayName']))]
            candidates = candidates[candidates.MATCHUP.map(lambda x:_parse_matchup(x)[0]==team)]
            if len(candidates)!=1: raise ValueError('Ambiguous NBA name/team/date crosswalk')
            actual=candidates.iloc[0]
            if alias_nba_id is not None and int(actual.PLAYER_ID)!=alias_nba_id: raise ValueError('Alias NBA ID mismatch')
            gid=str(actual.GAME_ID); summary=summaries[gid]
            competitors=espn['header']['competitions'][0]['competitors']
            if {c['team']['displayName'] for c in competitors} != {summary[s]['teamCity']+' '+summary[s]['teamName'] for s in ('homeTeam','awayTeam')}:
                raise ValueError('Event teams mismatch')
            if pd.Timestamp(espn['header']['competitions'][0]['date']) != pd.Timestamp(summary['gameTimeUTC']): raise ValueError('Event scheduled start mismatch')
            pp=ROOT/f'official_timing/stats_{gid}.json'
            start=verified_start(summary,json.loads(pp.read_text()),espn)
            adjusted=dict(summary,gameTimeUTC=start.isoformat())
            validate_record(q,adjusted,set()) # entire prospective season remains excluded
            evidence[gid]={'actual_start_lower_bound':start.isoformat(),'scheduled_start':summary['gameTimeUTC'],'nba_pbp_sha256':digest(pp),'espn_sha256':digest(ep),'espn_path':str(ep),'nba_pbp_path':str(pp)}
            accepted.append({'raw_quote':q,'nba_game_id':gid,'nba_player_id':int(actual.PLAYER_ID),'nba_player_name':actual.PLAYER_NAME,'provider_display_name':athlete['displayName']})
        except (ValueError,KeyError,TypeError) as exc:
            rejected.append({'quote':q,'reason':str(exc)})
    save_new(out/'qualified.json',{'input_sha256':digest(quotes_path),'summaries_dir':str(summaries_dir),'evidence':evidence,'accepted':accepted,'rejected':rejected})
    print('qualified book quotes',len(accepted),'unique player-games',len({(r['nba_game_id'],r['nba_player_id']) for r in accepted}),'rejections',len(rejected))
    print('reasons',sorted({r['reason'] for r in rejected}))

if __name__=='__main__':run()
