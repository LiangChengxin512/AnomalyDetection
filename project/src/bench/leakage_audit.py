"""Dynamic feature/filename/label invariance check on a declared replay prefix."""
import copy
import numpy as np
from src.common.io_utils import read_json
from src.data.prepare_factory import load_frames
from src.bench.continuation import make_engine


def audit(prepared,fitted,scored,surrogate,cfg,site,device):
    frame=load_frames(prepared,read_json(prepared/'index.json'),'test')[0].iloc[:cfg['window']['length']+6]
    a=make_engine(prepared,fitted,scored,surrogate,cfg,site,device)
    b=make_engine(prepared,fitted,scored,surrogate,cfg,site,device)
    differences=[]
    for values in frame.itertuples(index=False,name=None):
        original=dict(zip(frame.columns,values));mutated=dict(original,label=1,label_attack=1,attack_id='canary',target_points=['invalid'],source_file='renamed.csv')
        ua,ub=a.process_frame(original),b.process_frame(mutated)
        differences.append(abs(ua['detection']['score']-ub['detection']['score']))
        if ua['alarm']['state']!=ub['alarm']['state']:raise AssertionError('Label/filename changed alarm state')
        ra=[[c['score'] for c in r['candidates']] for r in ua['rankings']]
        rb=[[c['score'] for c in r['candidates']] for r in ub['rankings']]
        if not np.allclose(ra,rb,atol=1e-6):raise AssertionError('Label/filename changed ranking')
        if ua['advice'] and ua['advice']['action_id']!=ub['advice']['action_id']:raise AssertionError('Label/filename changed selected action')
    if max(differences,default=0)>1e-6:raise AssertionError('Label/filename changed model scores')
    return dict(status='SUCCEEDED',rows=len(frame),max_score_difference=max(differences,default=0),
                scope='Declared first-segment prefix; runtime receives only registered numeric features',
                limitation='This invariance test is evidence, not proof of absence of all possible leakage')
