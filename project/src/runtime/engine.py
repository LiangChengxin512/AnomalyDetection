"""Shared causal detection -> alarm -> explanation -> advisory runtime."""
from collections import deque
import time
import numpy as np
import pandas as pd
from src.data.fit_scaler import apply_scaler
from src.detect.alarm_state import machine
from src.explain.causal import CausalRanker, model_attribution
from src.decision.safe_policy import AdvicePolicy
from src.models.train import sync


class RuntimeEngine:
    def __init__(self, detector, site, cfg, threshold, channel_thresholds, graph, catalog, reference, card,
                 variant='B00', policy_id='P03', all_methods=False, attribution_cache=None, onset_permutation=None):
        self.detector,self.site,self.cfg = detector,site,cfg
        self.threshold,self.channel_thresholds = threshold,channel_thresholds
        self.graph,self.catalog,self.reference,self.card = graph,catalog,reference,card
        self.variant,self.policy_id,self.all_methods = variant,policy_id,all_methods
        self.tags = detector.scaler['tags']
        self.cache = attribution_cache if attribution_cache is not None else {}
        self.onset_permutation = onset_permutation
        self.segment = None
        self.window = deque(maxlen=detector.window)
        self.last_explanation = None
        self.policy = AdvicePolicy(site,catalog,reference,card,cfg,variant,policy_id)
        self.sequence = 0
        self.last_seconds = None

    def reset(self, segment):
        self.segment = segment
        self.window.clear()
        self.alarm = machine(self.threshold['value'],self.cfg['alarm'],self.variant!='B01')
        self.ranker = CausalRanker(self.tags,self.graph,self.channel_thresholds,self.cfg['rootcause'])
        self.sequence,self.last_explanation,self.last_seconds = 0,None,None

    def process_frame(self, frame, precomputed=None):
        began=time.perf_counter()
        segment = (frame['source_file'],str(frame['segment_id']))
        seconds=pd.Timestamp(frame['timestamp']).timestamp()
        gap = self.last_seconds is not None and seconds-self.last_seconds != self.site['input']['sample_interval_seconds']
        if segment!=self.segment or gap:
            self.reset(segment)
        self.last_seconds = seconds
        raw = np.asarray([frame[t] for t in self.tags],np.float32)
        quality = bool(frame.get('quality_flags','')) or not np.isfinite(raw).all()
        if quality:
            self.reset(segment)
            alarm=self.alarm.update(0.,frame['sample_id'],str(frame['timestamp']),seconds,quality=True)
            return dict(detection=dict(sample_id=frame['sample_id'],timestamp=str(frame['timestamp']),score=None,threshold=self.threshold['value'],
                        quality_flags='missing_or_invalid_feature',warmup=True),alarm=alarm,rankings=[],advice=None,constraints=[],efficiency={})
        scaled=apply_scaler(raw,self.detector.scaler)
        if not self.window:
            self.window.extend([scaled.copy() for _ in range(self.detector.window-1)])
        self.window.append(scaled)
        warmup=self.sequence<self.detector.window-1
        self.sequence+=1
        window=np.asarray(self.window)
        sync(self.detector.device)
        score_start=time.perf_counter()
        score,residual,evidence = precomputed if precomputed is not None else self.detector.window_score(window)
        sync(self.detector.device)
        score_ms=(time.perf_counter()-score_start)*1000
        detection = dict(sample_id=frame['sample_id'],source_file=frame['source_file'],segment_id=str(frame['segment_id']),
                         row_in_source=int(frame['row_in_source']),timestamp=str(frame['timestamp']),score=float(score),
                         threshold=self.threshold['value'],threshold_id=self.threshold['threshold_id'],warmup=warmup,
                         quality_flags='',raw_pred=bool(score>self.threshold['value'] and not warmup))
        alarm=self.alarm.update(score,frame['sample_id'],str(frame['timestamp']),seconds,warmup=warmup)
        if not warmup:
            self.ranker.observe(evidence,seconds)
        rankings,advice,constraints=[],None,[]
        explain_ms=None
        due = alarm['opened'] or (alarm['state']=='ACTIVE' and (self.last_explanation is None or seconds-self.last_explanation>=self.cfg['attribution']['update_seconds']))
        if due:
            start=time.perf_counter()
            cache_key=(frame['sample_id'],self.all_methods)
            if cache_key not in self.cache:
                self.cache[cache_key]=model_attribution(self.detector,window,self.reference['scaled_median'],self.all_methods)
            gradient,occlusion=self.cache[cache_key]
            for method in (['R01','R02','R03','R04','R05'] if self.all_methods else ['R05']):
                candidates,margin = self.ranker.rank(evidence,gradient,occlusion,seconds,method,
                    self.variant=='B02',self.onset_permutation)
                rankings.append(dict(sample_id=frame['sample_id'],alarm_id=alarm['alarm_id'],as_of_time=str(frame['timestamp']),
                    method=method,candidates=candidates,margin=margin,confidence_kind='evidence_margin',
                    evidence_method='input_gradient' if method!='R03' else 'occlusion',
                    checkpoint_sha256=self.detector.weight_hash,abstain=margin<self.cfg['rootcause']['abstain_margin']))
                if method=='R05':
                    advice,constraints=self.policy.decide(alarm['alarm_id'],frame['sample_id'],str(frame['timestamp']),seconds,
                        dict(zip(self.tags,raw)),candidates,margin)
            self.last_explanation=seconds
            sync(self.detector.device)
            explain_ms=(time.perf_counter()-start)*1000
        if alarm['closed']:
            self.ranker.onsets[:]=np.nan
        return dict(detection=detection,alarm=alarm,rankings=rankings,advice=advice,constraints=constraints,
                    efficiency=dict(inference_ms=score_ms,explanation_ms=explain_ms,end_to_end_ms=(time.perf_counter()-began)*1000,
                                    inference_source='precomputed' if precomputed is not None else 'batch_1'))
