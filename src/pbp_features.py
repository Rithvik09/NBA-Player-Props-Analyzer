"""Measured game-state/shot features from official historical play-by-play."""
from collections import defaultdict
import re
import numpy as np


def seconds(clock,period):
    m=re.fullmatch(r'PT(\d+)M([\d.]+)S',clock)
    if not m:raise ValueError('Invalid play clock')
    remaining=int(m[1])*60+float(m[2])
    length=720 if period<=4 else 300
    if not 0<=remaining<=length:raise ValueError('Invalid period clock')
    return (period-1)*720+length-remaining if period<=4 else 2880+(period-5)*300+length-remaining


def minutes(value):
    if not value:return 0.
    if ':' in str(value):
        m,s=str(value).split(':');return float(m)+float(s)/60
    return float(value)


def analyze_game(data,date):
    players={int(r['personId']):r for r in data['players']}
    teams={int(r['teamId']) for r in players.values()}
    if len(teams)!=2:raise ValueError('Expected two teams')
    home=next((int(p['teamId']) for p in data['plays'] if p.get('location')=='h' and int(p.get('teamId',0)) in teams),None)
    if home is None:raise ValueError('Cannot establish home team')
    away=next(t for t in teams if t!=home)
    rows={pid:dict(game_id=data['game_id'],game_date=date,player_id=pid,team_id=int(p['teamId']),opponent_id=away if int(p['teamId'])==home else home,minutes=minutes(p.get('minutes')),starter=bool(p.get('position')),points=0.,leading_points=0.,trailing_points=0.,tied_points=0.,overtime_points=0.,fga=0,fgm=0,shot_distance_sum=0.,known_shot_distances=0,and_ones=0,first_sub_minute=None,garbage_seconds=0.,starter_unit_seconds=0.,tracked_seconds=0.) for pid,p in players.items()}
    lineup={tid:{pid for pid,p in players.items() if int(p['teamId'])==tid and p.get('position')} for tid in teams}
    starters={tid:set(ids) for tid,ids in lineup.items()}
    valid=all(len(ids)==5 for ids in lineup.values())
    def resolve_outgoing(action):
        if action.get('outgoingPersonId') in players:return int(action['outgoingPersonId'])
        tid=int(action.get('teamId',0) or 0);name=action.get('description','').split(' FOR ',1)[-1].strip()
        def norm(v):return re.sub('[^a-z]','',str(v).lower())
        candidates=[player for player,p in players.items() if int(p['teamId'])==tid and norm(name) in (norm(p.get('familyName')),norm(p.get('nameI')),norm(p.get('firstName','')+' '+p.get('familyName','')))]
        return candidates[0] if len(candidates)==1 else None
    # Quarter changes need not have explicit substitution events. Infer each
    # period's initial five from first IN/OUT constraints and observed actors.
    # Ambiguous periods fail validation; no previous-quarter carry assumption.
    period_units={1:{tid:set(ids) for tid,ids in lineup.items()}}
    for period in sorted({int(p['period']) for p in data['plays']}):
        if period==1:continue
        first={};actors=set()
        for action in data['plays']:
            if int(action['period'])!=period:continue
            pid=int(action.get('personId',0) or 0)
            if action.get('actionType')=='Substitution':
                out=resolve_outgoing(action)
                if out is not None:first.setdefault(out,'out')
                if pid in players:first.setdefault(pid,'in')
            elif pid in players:actors.add(pid)
        candidates={pid for pid,kind in first.items() if kind=='out'} | {pid for pid in actors if pid not in first}
        period_units[period]={tid:{pid for pid in candidates if int(players[pid]['teamId'])==tid} for tid in teams}
    def interval(start,end,period,score_h,score_a):
        if not valid or end<start:return
        for tid,ids in lineup.items():
            for pid in ids:
                rows[pid]['tracked_seconds']+=end-start
                if len(ids&starters[tid])>=4:rows[pid]['starter_unit_seconds']+=end-start
                if period==4:
                    threshold=2520 if abs(score_h-score_a)>=20 else 2760 if abs(score_h-score_a)>=15 else 2880
                    rows[pid]['garbage_seconds']+=max(0,min(end,2880)-max(start,threshold))
    h=a=0;last_time=0.;last_period=1;last_made={};seen_and_one=set()
    for action in data['plays']:
        period=int(action['period']);now=seconds(action['clock'],period);elapsed=now-last_time
        if elapsed<-.1:valid=False
        if period!=last_period:
            boundary=720*(period-1) if period<=4 else 2880+300*(period-5)
            interval(last_time,boundary,last_period,h,a)
            lineup=period_units.get(period,{})
            if any(len(lineup.get(tid,set()))!=5 for tid in teams):valid=False
            last_time=boundary;elapsed=now-last_time
        interval(last_time,now,period,h,a)
        tid=int(action.get('teamId',0) or 0);pid=int(action.get('personId',0) or 0)
        new_h=int(action['scoreHome']) if str(action.get('scoreHome','')).isdigit() else h
        new_a=int(action['scoreAway']) if str(action.get('scoreAway','')).isdigit() else a
        delta_h=new_h-h;delta_a=new_a-a
        if pid in rows and tid in teams:
            row=rows[pid];margin=h-a if tid==home else a-h
            pts=delta_h if tid==home else delta_a
            if pts>0 and action.get('actionType','').lower() not in ('period','instant replay'):
                row['points']+=pts
                row['leading_points' if margin>0 else 'trailing_points' if margin<0 else 'tied_points']+=pts
                if period>=5:row['overtime_points']+=pts
            if int(action.get('isFieldGoal',0) or 0):
                row['fga']+=1
                if action.get('shotDistance') is not None:
                    row['shot_distance_sum']+=float(action['shotDistance']);row['known_shot_distances']+=1
                if action.get('shotResult')=='Made':row['fgm']+=1;last_made[pid]=(period,now)
            description=action.get('description','')
            if action.get('actionType')=='Free Throw' and '1 of 1' in action.get('subType','') and last_made.get(pid)==(period,now) and (pid,period,now) not in seen_and_one:
                row['and_ones']+=1;seen_and_one.add((pid,period,now))
            if action.get('actionType')=='Substitution':
                incoming=pid;out=resolve_outgoing(action)
                candidates=[out] if out is not None else []
                if len(candidates)==1:
                    outgoing=candidates[0]
                    if rows[outgoing]['first_sub_minute'] is None:rows[outgoing]['first_sub_minute']=now/60
                    if valid and outgoing in lineup[tid] and incoming not in lineup[tid]:lineup[tid].remove(outgoing);lineup[tid].add(incoming)
                    else:valid=False
                else:valid=False
        h,a=new_h,new_a;last_time=now;last_period=period
    # A game can have valid shots and points but incomplete substitutions.
    # Never call guessed stints observed: validate every player's minutes.
    tolerance=.51 if data.get('source','').startswith('ESPN') else .12
    expected_length=2880+max(0,last_period-4)*300
    stint_valid=valid and abs(last_time-expected_length)<.1 and all(abs(r['tracked_seconds']/60-r['minutes'])<=tolerance for r in rows.values())
    for pid,row in rows.items():
        row['points_valid']=abs(row['points']-float(players[pid].get('points',0)))<.01
        row['shots_valid']=row['fga']==int(players[pid].get('fieldGoalsAttempted',0)) and row['fgm']==int(players[pid].get('fieldGoalsMade',0))
        row['stints_valid']=stint_valid
        if not stint_valid:row['garbage_seconds']=row['starter_unit_seconds']=row['tracked_seconds']=None
    return list(rows.values())
