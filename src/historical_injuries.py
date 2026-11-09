"""Official timestamped injury reports; PDF extraction is an offline crawl step."""
from __future__ import annotations
from datetime import datetime
import re
from pathlib import Path
from zoneinfo import ZoneInfo

ET=ZoneInfo('America/New_York')
STATUSES={'Out','Available','Questionable','Doubtful','Probable'}


def normalize_name(name,keep_suffix=False):
    parts=name.split(',')
    if len(parts)==2:name=parts[1]+' '+parts[0]
    import unicodedata
    name=unicodedata.normalize('NFKD',name).encode('ascii','ignore').decode().lower()
    if not keep_suffix:name=re.sub(r'\b(jr|sr|ii|iii|iv)\b','',name)
    clean=re.sub('[^a-z]','',name)
    return clean if keep_suffix else re.sub(r'(jr|sr|iii|ii|iv)$','',clean)


def parse_pdf(path):
    """Read rows by PDF coordinates, not the file's internal text order.

    Missing/unsubmitted team rows are not evidence of a healthy roster.
    Header publication time, not the misleading filename, controls as-of use.
    """
    import pdfplumber
    rows=[];contexts={};publication=None;last_headers=None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text=page.extract_text(x_tolerance=2)
            compact=' '.join(text.split())
            match=re.search(r'Injury\s*Report:\s*(\d\d/\d\d/\d{2,4})\s*(\d\d:\d\d)\s*(AM|PM)',compact)
            if match:
                token=' '.join(match.groups());fmt='%m/%d/%y %I:%M %p' if len(match[1].split('/')[-1])==2 else '%m/%d/%Y %I:%M %p'
                stamp=datetime.strptime(token,fmt).replace(tzinfo=ET).isoformat()
                if publication and stamp!=publication:raise ValueError('Inconsistent PDF publication time')
                publication=stamp
            words=page.extract_words(x_tolerance=1.5,y_tolerance=2)
            headers={}
            for w in words:
                key=re.sub('[^a-z]','',w['text'].lower())
                if key in ('gamedate','gametime','matchup','team','player','playername','current','currentstatus','reason') and w['top']<140:
                    headers.setdefault(key,w)
            complete=all(k in headers for k in ('team','matchup','reason')) and ('player' in headers or 'playername' in headers) and ('current' in headers or 'currentstatus' in headers)
            if complete:
                last_headers=headers
                player=headers.get('playername',headers.get('player'));status=headers.get('currentstatus',headers.get('current'))
                body_top=status['bottom']+3
            elif last_headers:
                if not any(w['text'] in STATUSES for w in words):continue
                headers=last_headers
                player=headers.get('playername',headers.get('player'));status=headers.get('currentstatus',headers.get('current'))
                body_top=32
            else:raise ValueError('Unrecognized injury PDF header')
            xs=[0,headers.get('gametime',{'x0':headers['matchup']['x0']*.5})['x0']-2,headers['matchup']['x0']-2,headers['team']['x0']-2,player['x0']-2,status['x0']-2,headers['reason']['x0']-2,page.width]
            # Some layouts spell Game Date as two words; use the known column
            # order and only rely on Matchup/Team/Player/Status for extraction.
            status_words=sorted([w for w in words if w['text'] in STATUSES and xs[5]<=w['x0']<xs[6] and w['top']>body_top],key=lambda w:w['top'])
            data_words=[w for w in words if w['top']>body_top+2 and w['top']<page.height-25]
            for i,w in enumerate(status_words):
                top=(status_words[i-1]['top']+w['top'])/2 if i else body_top
                bottom=(w['top']+status_words[i+1]['top'])/2 if i+1<len(status_words) else page.height-25
                # Metadata is aligned with the status row; reason text may
                # wrap above/below. Context is carried across rows and pages.
                near=[x for x in data_words if abs(x['top']-w['top'])<6]
                def field(lo,hi,items):return ' '.join(x['text'] for x in sorted(items,key=lambda z:(round(z['top']/3),z['x0'])) if lo<=x['x0']<hi).strip()
                date_match=re.search(r'\d{2}/\d{2}/\d{4}',field(0,xs[2],near))
                if date_match:contexts['game_date']=datetime.strptime(date_match[0],'%m/%d/%Y').date().isoformat()
                matchup=field(xs[2],xs[3],near)
                if re.fullmatch(r'[A-Z]{2,3}@[A-Z]{2,3}',matchup):contexts['matchup']=matchup
                team=field(xs[3],xs[4],near)
                if team:contexts['team']=team
                name=field(xs[4],xs[5],near)
                reason=field(xs[6],xs[7],[x for x in data_words if top<x['top']<bottom])
                if name and all(k in contexts for k in ('game_date','matchup','team')):
                    rows.append({**contexts,'player':name,'status':w['text'],'reason':reason,'published_at':publication,'source':str(path)})
    if not publication or not rows:raise ValueError('No validated injury rows')
    for row in rows:
        row['published_at']=publication
        if row['game_date']<publication[:10]:raise ValueError('Report has a game before its publication date')
    return {'published_at':publication,'rows':rows,'source':str(path)}


def load_reports(root):
    import json
    reports=[]
    for path in Path(root).glob('injury_reports/*.json'):
        report=json.loads(path.read_text())
        if report.get('rows'):reports.append(report)
    return sorted(reports,key=lambda r:r['published_at'])
