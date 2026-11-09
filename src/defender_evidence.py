"""Dated actual matchup observations and qualified pregame defender forecasts."""
from functools import lru_cache
from pathlib import Path
import json
import joblib
import pandas as pd

PATH=Path(__file__).resolve().parents[1]/'logs/source_blocked/defender_evidence.joblib'


def select_assignment(counts, possessions, latest_teams, opponent_id, rule):
    """Shared deterministic selection for research and feature construction."""
    if possessions < rule.get('min_possessions',20): return None
    total=sum(counts.values())
    ranked=sorted(((v,int(k)) for k,v in counts.items()
                   if latest_teams.get(int(k))==int(opponent_id)),key=lambda x:(-x[0],x[1]))
    if not ranked or total<=0:return None
    share=ranked[0][0]/total
    runner=ranked[1][0]/total if len(ranked)>1 else 0.
    if share<rule.get('min_share',.4) or share-runner<rule.get('min_margin',.15):return None
    return ranked[0][1]


def expected_defender(prior, *, player_id, opponent_id, day, latest_teams, rule=None):
    """Historical dominant matchup, never a target-game assignment lookup.

    Only measured prior pairings within 400 days and defenders whose latest
    earlier NBA appearance was for the opponent are eligible. Confidence here
    means possession concentration, not a calibrated probability of assignment.
    """
    if prior.empty:return None
    day=pd.Timestamp(day)
    rows=prior[(prior.personIdOff==int(player_id))&(prior.defTeamId==int(opponent_id))&
        (prior.game_date<day)&(prior.game_date>=day-pd.Timedelta(days=400))]
    if rows.empty:return None
    rule=rule or {}
    if rule.get('regular_season_only'):
        rows=rows[rows.game_id.str.startswith('002')]
    if rows.empty:return None
    if rule.get('last_games'):
        ids=rows.sort_values('game_date').game_id.drop_duplicates().tail(rule['last_games'])
        rows=rows[rows.game_id.isin(ids)]
    weights=rows.partialPossessions.copy()
    if rule.get('half_life_days'):
        weights=weights * (0.5 ** ((day-rows.game_date).dt.days / rule['half_life_days']))
    counts=weights.groupby(rows.personIdDef).sum().to_dict()
    return select_assignment(counts,float(rows.partialPossessions.sum()),latest_teams,opponent_id,rule)



class DefenderEvidence:
    def __init__(self,bundle):
        self.rows=bundle['rows'];self.qualifications=bundle['qualifications']
        self.switches_available=bundle.get('switches_available',False)
        self._league={}
        self.offense={int(pid):r for pid,r in self.rows.groupby('personIdOff')}
        self.defense={int(pid):r for pid,r in self.rows.groupby('personIdDef')}
        self.teams={int(tid):r for tid,r in self.rows.groupby('defTeamId')}

    def qualified(self,season,game_id=None):
        q=self.qualifications.get(season,{})
        if q and q.get('last_outcome_date','9999')>=season[:4]+'-10-01':
            raise ValueError('Defender qualification uses future outcomes')
        return str(game_id or '').startswith('002') and q.get('qualified',False)

    def features(self,ctx,player_id,opponent_id,date):
        day=pd.Timestamp(date);out={}
        # Team switching tendency is a measured defensive scheme statistic.
        rows=self.teams.get(int(opponent_id),pd.DataFrame())
        games=ctx.inputs.team_games.get(int(opponent_id),[])
        days=sorted({str(g['game_date'])[:10] for g in games if str(g['game_date'])[:10]<str(day.date())})[-20:]
        if self.switches_available and not rows.empty:
            measured=rows[(rows.game_date<day)&rows.game_date.dt.strftime('%Y-%m-%d').isin(days)]
            if measured.game_id.nunique()>=3 and measured.partialPossessions.sum()>0:
                out['defender_switching_frequency']=100*float(measured.switchesOn.sum()/measured.partialPossessions.sum())
        if not self.qualified(ctx.inputs.season,getattr(ctx.inputs,'prediction_game_id',None)):return out
        rule=self.qualifications.get(ctx.inputs.season,{}).get('rule') or {}
        cache_key=(str(day.date()),bool(rule.get('regular_season_only')))
        if not hasattr(ctx,'_defender_rosters'):ctx._defender_rosters={}
        if cache_key not in ctx._defender_rosters:
            # Qualification uses earlier appearances across season boundaries.
            # Keeping only this season's logs silently discarded that evidence
            # at opening week and diverged from the tested assignment rule.
            history=pd.concat([getattr(ctx,'previous_logs',pd.DataFrame()),ctx.logs],ignore_index=True)
            history['GAME_DATE']=pd.to_datetime(history.GAME_DATE)
            prior=history[history.GAME_DATE<day].sort_values('GAME_DATE')
            if rule.get('regular_season_only'):
                prior=prior[prior.GAME_ID.astype(str).str.startswith('002')]
            ctx._defender_rosters[cache_key]=prior.groupby('PLAYER_ID').tail(1).set_index('PLAYER_ID').TEAM_ID.to_dict()
        latest=ctx._defender_rosters[cache_key]
        own=self.offense.get(int(player_id),pd.DataFrame())
        defender=expected_defender(own,player_id=player_id,opponent_id=opponent_id,day=day,latest_teams=latest,
            rule=rule)
        if defender is None:return out
        # Previous-season measurements, never current overwritten snapshots.
        adv=ctx.inputs.precomputed.get('advanced',{}).get(defender,{})
        player_adv=ctx.inputs.precomputed.get('advanced',{}).get(int(player_id),{})
        for name,col in [('primary_defender_rating','def_rating'),('primary_defender_age','age')]:
            if adv.get(col) is not None:out[name]=adv[col]
        if adv.get('height_inches') is not None and player_adv.get('height_inches') is not None:
            out['defender_size_mismatch']=adv['height_inches']-player_adv['height_inches']
        reports=ctx.injuries(opponent_id,date)
        if reports is not None:
            identities=[ctx.resolve_name(r['player']) for r in reports]
            matches=[r for r,identity in zip(reports,identities) if identity==defender]
            if not matches and all(identity is not None for identity in identities):out['primary_defender_active']=1.
            elif matches and matches[-1]['status'] in ('Out','Available'):
                out['primary_defender_active']=float(matches[-1]['status']=='Available')
        defender_rows=self.defense.get(defender,pd.DataFrame())
        if defender_rows.empty:return out
        defender_rows=defender_rows[(defender_rows.game_date<day)&(defender_rows.game_date>=day-pd.Timedelta(days=400))].sort_values('game_date')
        ids=defender_rows.game_id.drop_duplicates().tail(20)
        twenty=defender_rows[defender_rows.game_id.isin(ids)]
        if len(ids)<3 or twenty.partialPossessions.sum()<100:return out
        rating=100*twenty.playerPoints.sum()/twenty.partialPossessions.sum()
        out['def_rating_individual']=float(rating)
        recent=twenty[twenty.game_id.isin(ids.tail(5))]
        if recent.game_id.nunique()>=3:
            out['defender_recent_form']=float(100*recent.playerPoints.sum()/recent.partialPossessions.sum()-rating)
        # Observed efficiency percentile, not an invented composite rating.
        season=ctx.inputs.season
        if season not in self._league:
            year=int(season[:4])
            league=self.rows[(self.rows.game_date<pd.Timestamp(f'{year}-10-01'))&
                (self.rows.game_date>=pd.Timestamp(f'{year-1}-10-01'))]
            league=league.groupby('personIdDef')[['playerPoints','partialPossessions']].sum()
            league=league[league.partialPossessions>=100]
            self._league[season]=100*league.playerPoints/league.partialPossessions
        values=self._league[season]
        if len(values)>=30:
            out['primary_defender_score01']=float((values>rating).mean())
        return out


@lru_cache(maxsize=1)
def _load(path,mtime):return DefenderEvidence(joblib.load(path))


def load_evidence():
    return _load(str(PATH),PATH.stat().st_mtime_ns) if PATH.exists() else None
