"""Training-only synthetic-line recipes for controlled research comparisons."""
import numpy as np
from src.verified_training import classifier_frame


def offsets_from_training_quotes(frame,lines):
    center=.7*frame.recent_avg.to_numpy()+.3*frame.season_avg.to_numpy()
    delta=np.asarray(lines)-center
    if not len(delta) or not np.isfinite(delta).all():raise ValueError('Finite training quotes required')
    return np.quantile(delta,[.05,.25,.5,.75,.95]).tolist()


def expand_with_offsets(data,columns,offsets):
    frame=data['X'];center=.7*frame.recent_avg.to_numpy()+.3*frame.season_avg.to_numpy()
    if not np.isfinite(center).all():raise ValueError('Missing prior center')
    indices=[];lines=[]
    for i,c in enumerate(center):
        values=sorted({max(.5,float(np.floor(c)+.5+np.rint(o))) for o in offsets})
        indices.extend([i]*len(values));lines.extend(values)
    indices=np.asarray(indices);lines=np.asarray(lines)
    return classifier_frame(frame.iloc[indices].reset_index(drop=True),lines,columns),(data['y'][indices]>lines).astype(int),indices,lines


def feature_groups(full,compact):
    groups={k:[] for k in ['injury','shooting','matchup','role_tracking','history','other_context']}
    for c in full:
        if c in compact:continue
        if any(t in c for t in ['injury','out_count','out_unresolved','out_unknown','known_key_players','games_since_return','missed_games']):g='injury'
        elif any(t in c for t in ['vs_','matchup','head_to_head','opp_iso','opp_transition','opp_postup']):g='matchup'
        elif any(t in c for t in ['fg','shoot','shot','ts_pct','pts_','points_per','ft_','fta_','paint','restricted','and_one','iso_','pnr_','spotup','spot_up','transition','postup','cut_poss']):g='shooting'
        elif any(t in c for t in ['touch','tracking','usage','usg','minutes','mins','time_of_possession','drib','screen_assist']):g='role_tracking'
        elif any(t in c for t in ['trend','rolling','ewm','variance','volatility','consisten','hot_hand','consecutive','games_above','ceiling']):g='history'
        else:g='other_context'
        groups[g].append(c)
    return groups
