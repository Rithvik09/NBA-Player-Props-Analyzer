"""Inputs available before a historical game; current snapshot tables are excluded."""
from bisect import bisect_left
from pathlib import Path
import json

import pandas as pd

PLAYER_SOURCES=('advanced','clutch','hustle','shot_prof','play_types','on_off','shot_zones','q_splits','tracking','scoring','yoy')
SNAPSHOT_SOURCES=('team_stats','team_foul','rest_splits','standings','opp_shot_zones','opp_synergy','home_away','lineup_stats','injury_status','vs_opponent','dvp_rolling','referee')
ROOT=Path(__file__).resolve().parents[1]


def previous_season(season):
    year=int(season[:4])-1
    return f'{year}-{str(year+1)[2:]}'


class HistoricalInputs:
    def __init__(self,season,raw_precomputed,player_logs,*,snapshots=True,history_window=20,audited=False):
        self.season=season;self.source_season=previous_season(season)
        self.audited=audited
        self.feature_gaps={}
        self.prediction_game_id=None
        self.history_window=history_window
        self.precomputed={key:{} for key in raw_precomputed}
        for key in ('game_team_stats','game_officials','referee_history'):
            self.precomputed[key]=raw_precomputed.get(key,{})
        self.source_rows={}
        for key in PLAYER_SOURCES:
            rows={}
            for token,value in raw_precomputed.get(key,{}).items():
                if isinstance(token,tuple) and token[1]==self.source_season:
                    rows[int(token[0])]=dict(value)
            self.source_rows[key]=len(rows) if snapshots else 0
            self.precomputed[key]={}
            if snapshots:
                for pid,row in rows.items():
                    self.precomputed[key][(pid,season)]=row
                    self.precomputed[key][pid]=row
        for key in SNAPSHOT_SOURCES:self.precomputed[key]={}
        self.logs={int(pid):frame.copy() for pid,frame in player_logs.groupby('PLAYER_ID')}
        self.team_games={int(tid):sorted(games,key=lambda g:str(g['game_date'])) for tid,games in raw_precomputed.get('game_team_stats',{}).items()}
        logs=player_logs.copy()
        self.team_boxscores={}
        if {'TEAM_ID','GAME_ID','GAME_DATE'}.issubset(logs):
            logs=logs[logs.GAME_ID.astype(str).str.startswith(('002','004','005') if audited else '002')]
            cols=[c for c in 'MIN PTS FGA FGM FG3A FG3M FTA FTM AST REB STL BLK TOV PF'.split() if c in logs]
            box=logs.groupby(['TEAM_ID','GAME_ID','GAME_DATE'],as_index=False)[cols].sum()
            box['GAME_DATE']=pd.to_datetime(box.GAME_DATE)
            self.team_boxscores={int(tid):frame.sort_values('GAME_DATE') for tid,frame in box.groupby('TEAM_ID')}
        from src.recovered_features import RecoveryContext
        self.recovery=RecoveryContext(self,player_logs) if audited else None
        if audited and not self.recovery.rich.empty:
            rich=self.recovery.rich
            post=rich[rich.GAME_ID.astype(str).str.startswith(('004','005'))]
            lookup={(r['GAME_ID'],int(r['TEAM_ID'])):r for r in rich.to_dict('records')}
            for row in post.to_dict('records'):
                other=lookup.get((row['GAME_ID'],int(row['OPP_ID'])))
                if not other:continue
                duration=float(row['MIN'])/5
                poss=.5*sum(float(g['FGA'])+.44*float(g['FTA'])-float(g.get('OREB',0))+float(g['TOV']) for g in (row,other))
                if poss<=0 or duration<=0:continue
                tid=int(row['TEAM_ID']);day=str(pd.Timestamp(row['GAME_DATE']).date())
                if any(str(g['game_date'])[:10]==day for g in self.team_games.get(tid,[])):continue
                self.team_games.setdefault(tid,[]).append({'game_date':day,'pace':48*poss/duration,'off_rating':100*float(row['PTS'])/poss,'def_rating':100*float(other['PTS'])/poss,'pts':float(row['PTS']),'opp_pts':float(other['PTS']),'source':'postseason estimated possessions from complete boxscore'})
            for games in self.team_games.values():games.sort(key=lambda g:str(g['game_date']))
        self._dates={tid:[str(g['game_date'])[:10] for g in games] for tid,games in self.team_games.items()}
        self._context_cache={}
        self.context_calls=0;self.max_source_date=None

    def catalog(self,max_players):
        path=ROOT/'logs/precompute_cache'/f'player_averages_{self.source_season}.json'
        frame=pd.DataFrame(json.loads(path.read_text()))
        frame=frame[frame['GP']>=20].sort_values(['MIN','PLAYER_ID'],ascending=[False,True]).head(max_players)
        return frame

    def player_log(self,player_id):
        frame=self.logs.get(int(player_id),pd.DataFrame()).copy()
        if not frame.empty:
            frame=frame[frame['GAME_ID'].astype(str).str.startswith(('002','004','005') if self.audited else '002')].copy()
        return frame

    def player_info(self,player_id):
        row=self.precomputed['advanced'].get(int(player_id),{})
        # No current-position/experience query. Defaults express unavailable bio.
        return pd.DataFrame([{'POSITION':'','BIRTHDATE':'','SEASON_EXP':row.get('years_experience',5),
                              'HEIGHT':'','WEIGHT':row.get('weight',220)}])

    def team_context(self,team_id,prediction_date):
        day=pd.Timestamp(prediction_date).strftime('%Y-%m-%d');key=(int(team_id),day)
        if key in self._context_cache:return self._context_cache[key]
        games=self.team_games.get(int(team_id),[])
        end=bisect_left(self._dates.get(int(team_id),[]),day)
        prior=games[max(0,end-10):end]
        self.context_calls+=1
        if prior:
            last=str(prior[-1]['game_date'])[:10]
            assert last<day
            self.max_source_date=max(self.max_source_date or last,last)
        def mean(field,default):
            values=[float(g[field]) for g in prior if g.get(field) is not None]
            return sum(values)/len(values) if values else default
        result={'pace':mean('pace',100),'offensive_rating':mean('off_rating',110),
                'defensive_rating':mean('def_rating',110),
                'days_rest':max(0,(pd.Timestamp(day)-pd.Timestamp(prior[-1]['game_date'])).days-1) if prior else 2,
                'style':{},'base':{},'league_avgs':{}}
        self._context_cache[key]=result
        return result

    def prior_team_games(self,team_id,prediction_date):
        day=pd.Timestamp(prediction_date).strftime('%Y-%m-%d')
        end=bisect_left(self._dates.get(int(team_id),[]),day)
        return self.team_games.get(int(team_id),[])[:end]

    def prior_team_boxscores(self,team_id,prediction_date):
        box=self.team_boxscores.get(int(team_id),pd.DataFrame())
        if box.empty:return box
        return box[box.GAME_DATE<pd.Timestamp(prediction_date).normalize()]


def load_season_logs(season,include_postseason=False):
    path=ROOT/'logs/points_pilot_cache'/f'players_{season}.json'
    frames=[pd.read_json(path,dtype={'GAME_ID':str})]
    if include_postseason:
        for kind in ('playoffs','playin'):
            extra=path.with_name(f'{kind}_{season}.json')
            if not extra.exists():
                completed=pd.Timestamp.now(tz='America/New_York')>=pd.Timestamp(f'{int(season[:4])+1}-07-01',tz='America/New_York')
                if completed:raise ValueError('Missing postseason replay source: '+str(extra))
                continue # Future postseason games cannot be crawled yet.
            frames.append(pd.read_json(extra,dtype={'GAME_ID':str}))
    frame=pd.concat(frames,ignore_index=True)
    if frame.duplicated(['PLAYER_ID','GAME_ID']).any():raise ValueError('Duplicate player-game source rows')
    return frame
