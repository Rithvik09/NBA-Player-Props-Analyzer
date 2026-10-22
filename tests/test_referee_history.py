"""Referee replay correctness and historical leakage boundaries."""
import pytest

from src.referee_history import ensure_history, save_game, rebuild_aggregates, historical_features, validate_game


def game(gid, date, fouls=40, win=True, pace=100):
    return {'game_id':gid,'game_date':date,'home_team_id':1,'away_team_id':2,
            'total_fouls':fouls,'home_win':win,'pace':pace,
            'officials':[{'name':f'Ref {i}','id':i} for i in (1,2,3)],
            'teams':[{'team_id':1,'pts':110,'opp_pts':100,'pace':pace,'off_rating':110,'def_rating':100},
                     {'team_id':2,'pts':100,'opp_pts':110,'pace':pace,'off_rating':100,'def_rating':110}]}


def test_replay_does_not_double_count(db_conn):
    from src.precompute_jobs import ensure_tables
    ensure_tables(db_conn)
    ensure_history(db_conn)
    for i in range(1,5):
        save_game(db_conn,'2025-26',game(f'ref-test-{i}',f'2025-11-0{i}',fouls=30+i))
    rebuild_aggregates(db_conn)
    before=db_conn.execute("SELECT games,foul_rate,home_win_pct,pace FROM referee_stats WHERE ref_name='Ref 1'").fetchone()
    save_game(db_conn,'2025-26',game('ref-test-4','2025-11-04',fouls=34))
    rebuild_aggregates(db_conn)
    after=db_conn.execute("SELECT games,foul_rate,home_win_pct,pace FROM referee_stats WHERE ref_name='Ref 1'").fetchone()
    assert before == after == (4,32.5,1.0,100.0)


def test_future_and_same_day_games_cannot_leak(db_conn):
    from src.precompute_jobs import ensure_tables
    ensure_tables(db_conn)
    ensure_history(db_conn)
    for i in range(1,4):
        save_game(db_conn,'2025-26',game(f'prior-{i}',f'2025-11-0{i}',fouls=40,pace=99))
    save_game(db_conn,'2025-26',game('target','2025-11-04',fouls=90,win=False,pace=130))
    save_game(db_conn,'2025-26',game('same-day','2025-11-04',fouls=80,win=False,pace=120))
    save_game(db_conn,'2025-26',game('future','2025-11-05',fouls=100,win=False,pace=140))
    rebuild_aggregates(db_conn)
    row=db_conn.execute("SELECT games,foul_rate,home_win_pct,pace FROM referee_game_features WHERE game_id='target' AND ref_name='Ref 1'").fetchone()
    assert row == (3,40.0,1.0,99.0)
    first=db_conn.execute("SELECT games,foul_rate FROM referee_game_features WHERE game_id='prior-1' AND ref_name='Ref 1'").fetchone()
    assert first == (0,None)


def test_training_does_not_fall_back_to_future_aggregate():
    pre={'game_officials':{'game':['ref 1']},'referee':{'ref 1':{'foul_rate':99}}}
    assert historical_features(pre,'game') == (0,.5,100)
    pre['referee_history']={('game','ref 1'):{'games':3,'foul_rate':42,'home_win_pct':.6,'pace':101}}
    assert historical_features(pre,'game') == (42,.6,101)


def test_incomplete_observation_is_rejected():
    summary={'gameId':'1','gameStatus':3,'gameEt':'2025-11-01','homeTeamId':1,'awayTeamId':2,
             'homeTeam':{'teamId':1,'score':110},'awayTeam':{'teamId':2,'score':100},
             'officials':[{'name':'Ref 1','personId':1}]}
    with pytest.raises(ValueError,match='three distinct'):
        validate_game('1',[],summary,[])
