"""As-of-time evidence fusion; labels and attack targets are not accepted."""
from collections import deque
import numpy as np
from src.explain.attribution import input_gradient_scores, occlusion_scores


def normalize(values):
    values = np.asarray(values, float)
    total = values.sum()
    return values/total if total > 0 else np.zeros_like(values)


class CausalRanker:
    def __init__(self, tags, graph, channel_thresholds, cfg):
        self.tags, self.cfg = tags, cfg
        self.thresholds = np.asarray(channel_thresholds)
        self.graph = np.zeros((len(tags),len(tags)))
        lookup = {tag:i for i,tag in enumerate(tags)}
        for edge in graph.get('edges', []):
            source, target = edge.get('source'), edge.get('target')
            if source in lookup and target in lookup and edge.get('evidence', edge.get('source_ref', edge.get('provenance'))):
                self.graph[lookup[source],lookup[target]] = 1
                self.graph[lookup[target],lookup[source]] = 1
        self.history = {}
        self.onsets = np.full(len(tags), np.nan)

    def observe(self, evidence, seconds):
        new = (np.asarray(evidence)>self.thresholds) & np.isnan(self.onsets)
        self.onsets[new] = seconds

    def rank(self, evidence, gradient, occlusion, seconds, method='R05', no_process=False, onset_permutation=None):
        previous = self.history.setdefault(method, deque(maxlen=5))
        stability = np.zeros(len(self.tags))
        for top in previous:
            stability[top] += 1/max(1,len(previous))
        # Early onset gets greater support; unavailable evidence has zero weight.
        finite = np.isfinite(self.onsets)
        onset = np.zeros(len(self.tags))
        if finite.any():
            first = np.nanmin(self.onsets)
            onset[finite] = 1/(1+(self.onsets[finite]-first)/5)
        if onset_permutation is not None:
            onset = onset[onset_permutation]
        residual = normalize(evidence)
        transition = self.graph / np.maximum(self.graph.sum(axis=1, keepdims=True), 1)
        graph = np.zeros(len(self.tags))
        propagated = residual
        for hop in range(1, self.cfg.get('graph_hops', 2)+1):
            propagated = transition @ propagated
            graph += propagated / hop
        components = dict(residual=residual, attribution=normalize(gradient), onset=normalize(onset),
                          graph=normalize(graph), stability=normalize(stability))
        if method == 'R01':
            score = residual
        elif method == 'R02':
            score = normalize(gradient)
        elif method == 'R03':
            if occlusion is None:
                raise ValueError('Occlusion method requires genuine model occlusion evidence')
            score = normalize(occlusion)
        else:
            weights = self.cfg['weights'].copy()
            if method == 'R04' or no_process:
                weights['graph'] = 0
            weights = {k:v for k,v in weights.items() if components[k].sum() > 0}
            denom = sum(weights.values())
            score = sum((v*components[k] for k,v in weights.items()), np.zeros(len(self.tags)))/max(denom,1e-12)
        order = np.argsort(-score, kind='stable')
        margin = float(score[order[0]]-score[order[1]]) if len(order)>1 else float(score[order[0]])
        result = [dict(tag=self.tags[j], rank=i+1, score=float(score[j]),
                       **{k:float(v[j]) for k,v in components.items()},
                       onset_available=bool(finite[j]), stability_updates=len(previous),
                       graph_available=bool(self.graph[j].any())) for i,j in enumerate(order)]
        previous.append(order[:3])
        return result, margin


def model_attribution(detector, window, fill_values, include_occlusion=False):
    model = detector.model
    model.eval()
    args = (model, window[None], detector.scale, detector.cfg['model']['score_agg'], detector.cfg['model']['score_topk'])
    grad = input_gradient_scores(*args, device=str(detector.device))[0]
    batch_limit = detector.cfg['training'].get('physical_batch_size') or (16 if detector.device.type == 'mps' else 128)
    occ = occlusion_scores(*args, fill_values, device=str(detector.device), batch_limit=batch_limit)[0] if include_occlusion else None
    return grad, occ
