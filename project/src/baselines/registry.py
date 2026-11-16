"""Train-normal baselines with common fit/score persistence."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

DETECTORS = {'D01': 'RobustZ', 'D02': 'EWMA', 'D03': 'PCA', 'D04': 'IsolationForest',
             'D05': 'OCSVM', 'D06': 'TranAD', 'D07': 'TranAD_C_G_E'}
DETERMINISTIC = {'D01', 'D02'}


def fit_baseline(method, X, seed, continuous):
    median = np.median(X, axis=0)
    scale = np.maximum(1.4826*np.median(np.abs(X-median), axis=0), np.maximum(X.std(axis=0)*.01, .001))
    result = dict(method=method, median=median, scale=scale, continuous=np.asarray(continuous), discrete={})
    for j, is_cont in enumerate(continuous):
        if not is_cont:
            values, counts = np.unique(X[:, j], return_counts=True)
            result['discrete'][j] = dict(zip(values.tolist(), ((counts+1)/(len(X)+len(values)+1)).tolist()))
    if method == 'D03':
        result['estimator'] = PCA(n_components=min(20, X.shape[1], len(X)), svd_solver='full').fit(X)
    elif method == 'D04':
        result['estimator'] = IsolationForest(n_estimators=200, max_samples=min(256,len(X)), random_state=seed, n_jobs=1).fit(X)
    elif method == 'D05':
        result['estimator'] = OneClassSVM(nu=.01, gamma='scale', kernel='rbf').fit(X)
    return result


def raw_evidence(estimator, X):
    method = estimator['method']
    if method == 'D01':
        result = np.abs(X-estimator['median'])/estimator['scale']
        for j, frequency in estimator['discrete'].items():
            result[:, j] = [-np.log(frequency.get(float(v), 1e-6)) for v in X[:, j]]
        return result.astype(np.float32)
    if method == 'D02':
        mean = estimator['median'].copy()
        evidence = np.empty_like(X)
        for i, row in enumerate(X):
            evidence[i] = np.abs(row-mean)/estimator['scale']
            mean = .2*row + .8*mean
        return evidence
    model = estimator['estimator']
    if method == 'D03':
        return np.abs(X-model.inverse_transform(model.transform(X))).astype(np.float32)
    # Scalar estimators have no fabricated channel attributions.
    score = -model.score_samples(X) if method == 'D04' else -model.decision_function(X)
    return score.astype(np.float32).reshape(-1, 1)
