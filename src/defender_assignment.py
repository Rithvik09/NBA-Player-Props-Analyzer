"""Pregame defender candidates from earlier boxscores and matchup roles.

The ranker is separate from player-prop models. It must pass the existing
assignment qualification gate on a new, untouched period before use.
"""
from collections import defaultdict, deque
import numpy as np
import pandas as pd

STATS=['FGA','FG3A','REB','AST','PTS']
FEATURES=(['pair_share','pair_possessions','pair_age','def_minutes','off_minutes',
           'minutes_rank','def_games','off_games','def_age','guard_possessions']+
          [f'{prefix}_{stat}' for prefix in ['off','def','difference','guard_difference'] for stat in STATS])


class AssignmentState:
    def __init__(self):
        self.boxes=defaultdict(lambda:deque(maxlen=10))
        self.latest={}
        self.pairs={}
        self.guards={}

    def role(self,pid):
        rows=self.boxes.get(int(pid),[])
        if not rows:return None
        values=np.asarray(rows,dtype=float);minutes=values[:,0].sum()
        if minutes<=0:return None
        return values[:,1:].sum(axis=0)/minutes,float(values[:,0].mean()),len(rows)

    def candidates(self,player,opponent,day):
        day=pd.Timestamp(day);off=self.role(player)
        if off is None or off[2]<3:return [],[]
        pool=[]
        for pid,(team,date) in self.latest.items():
            if pid==int(player):continue
            if team!=int(opponent) or not 0<(day-date).days<=400:continue
            role=self.role(pid)
            if role is not None and role[2]>=3:pool.append((pid,role,date))
        pool.sort(key=lambda x:(-x[1][1],x[0]));pool=pool[:10]
        pair={}
        for pid,_,_ in pool:
            stamp,poss=self.pairs.get((int(player),pid),(day,0.))
            age=(day-stamp).days
            pair[pid]=(poss*.5**(age/90),age)
        total=sum(v[0] for v in pair.values());ids=[];features=[]
        for rank,(pid,role,date) in enumerate(pool):
            stamp,mass,sums=self.guards.get(pid,(day,0.,np.zeros(len(STATS))))
            mass=mass*.5**((day-stamp).days/90)
            # Unknown guarding profiles stay missing, never a zero role.
            guard=sums/self.guards[pid][1] if pid in self.guards and self.guards[pid][1]>0 else np.full(len(STATS),np.nan)
            poss,age=pair[pid]
            features.append([poss/total if total else 0.,poss,age if poss else np.nan,
                role[1],off[1],rank,role[2],off[2],(day-date).days,mass,
                *off[0],*role[0],*abs(off[0]-role[0]),*abs(off[0]-guard)])
            ids.append(pid)
        return ids,features

    def observe_matchups(self,part,day):
        """Call after ALL forecasts for the day, before ingesting its boxes."""
        day=pd.Timestamp(day)
        for row in part.itertuples():
            off,d=int(row.personIdOff),int(row.personIdDef);poss=float(row.partialPossessions)
            key=(off,d);stamp,old=self.pairs.get(key,(day,0.))
            self.pairs[key]=(day,old*.5**((day-stamp).days/90)+poss)
            role=self.role(off)
            if role is not None:
                stamp,mass,sums=self.guards.get(d,(day,0.,np.zeros(len(STATS))))
                weight=.5**((day-stamp).days/90)
                self.guards[d]=(day,mass*weight+poss,sums*weight+poss*role[0])

    def observe_boxes(self,part,day):
        for row in part.itertuples():
            if pd.isna(row.MIN) or float(row.MIN)<=0:continue
            pid=int(row.PLAYER_ID)
            self.boxes[pid].append([float(row.MIN)]+[float(getattr(row,s)) for s in STATS])
            self.latest[pid]=(int(row.TEAM_ID),pd.Timestamp(day))


def load_logs():
    from pathlib import Path
    logs=pd.concat([pd.read_json(p,dtype={'GAME_ID':str})
                    for p in Path('logs/points_pilot_cache').glob('players_*.json')])
    logs['GAME_DATE']=pd.to_datetime(logs.GAME_DATE)
    return logs.drop_duplicates(['GAME_ID','PLAYER_ID'])


def state_before(rows,logs,day):
    state=AssignmentState();day=pd.Timestamp(day)
    matches={d:g for d,g in rows[rows.game_date<day].groupby('game_date')}
    boxes={d:g for d,g in logs[logs.GAME_DATE<day].groupby('GAME_DATE')}
    for date in sorted(set(matches)|set(boxes)):
        if date in matches:state.observe_matchups(matches[date],date)
        if date in boxes:state.observe_boxes(boxes[date],date)
    return state


def choose(model,ids,features,threshold):
    if not ids:return None
    scores=model.predict_proba(np.asarray(features,dtype=np.float32))[:,1]
    best=int(np.argmax(scores))
    return ids[best] if scores[best]>=threshold else None
