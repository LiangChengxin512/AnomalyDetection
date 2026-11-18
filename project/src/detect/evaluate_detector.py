"""Fit, calibrate and score explicitly supplied detector artifacts."""
from __future__ import annotations
import pickle
import time
import numpy as np
import pandas as pd
import torch
from src.common.io_utils import read_json, write_json, sha256_file
from src.common.provenance import digest
from src.data.prepare_factory import load_frames
from src.data.fit_scaler import apply_scaler
from src.data.window import build_index
from src.models.load_checkpoint import load_checkpoint
from src.models.tranad_adapter import predict_batch
from src.models.train import sync
from src.baselines.registry import fit_baseline, raw_evidence
from src.detect.score import aggregate, residual_scale
from src.detect.alarm_state import machine
from src.data.segments import KEYS


def fit_estimator(method, prepared, output, seed):
    output.mkdir(parents=True, exist_ok=True)
    index, scaler = read_json(prepared/'index.json'), read_json(prepared/'scaler.json')
    frames = load_frames(prepared, index, 'train')
    arrays, sampled = [], []
    budget = 5000 if method == 'D05' else 50000
    if method in ['D01', 'D02']:
        budget = sum(len(f) for f in frames)
    rng = np.random.default_rng(seed)
    total = sum(len(f) for f in frames)
    for frame in frames:
        n = min(len(frame), max(1, int(budget*len(frame)/total)))
        ids = np.sort(rng.choice(len(frame), n, replace=False))
        arrays.append(apply_scaler(frame[scaler['tags']].to_numpy(np.float32)[ids], scaler))
        sampled.extend(frame.sample_id.iloc[ids].tolist())
    estimator = fit_baseline(method, np.concatenate(arrays), seed, scaler['continuous_mask'])
    with (output/'estimator.pkl').open('wb') as handle:
        pickle.dump(estimator, handle)
    pd.DataFrame({'sample_id': sampled, 'sampling_seed': seed}).to_csv(output/'fit_samples.csv', index=False)
    write_json(output/'estimator_card.json', dict(method=method, fit_split='train', fit_samples=len(sampled),
        budget=budget, sampling_seed=seed, estimator_sha256=sha256_file(output/'estimator.pkl'),
        discrete_rule='negative log train frequency with unseen probability 1e-6', scale_floor='max(0.01*std,0.001)',
        dataset_fingerprint=index['dataset_fingerprint']))


class Detector:
    def __init__(self, method, directory, scaler, cfg, device):
        self.method, self.scaler, self.cfg, self.device = method, scaler, cfg, device
        self.window = cfg['window']['length']
        if method in ['D06', 'D07']:
            self.model, self.payload = load_checkpoint(directory/'best.ckpt', device_requested=str(device),
                expected={'feature_order': scaler['tags'], 'continuous_mask': scaler['continuous_mask'], 'scaler_hash': digest(scaler)})
            self.model.eval()
            self.weight_hash = sha256_file(directory/'best.ckpt')
        else:
            with (directory/'estimator.pkl').open('rb') as f:
                self.estimator = pickle.load(f)
            self.weight_hash = sha256_file(directory/'estimator.pkl')
        self.scale = None

    @torch.no_grad()
    def residuals(self, X):
        if self.method not in ['D06', 'D07']:
            return raw_evidence(self.estimator, X)
        idx = build_index([X], self.window)
        batch_size = self.cfg['training'].get('physical_batch_size') or (16 if self.device.type == 'mps' else 128)
        out = np.empty_like(X)
        for lo in range(0, len(idx), batch_size):
            ids = np.arange(lo, min(lo+batch_size,len(idx)))
            b = torch.as_tensor(idx.gather(ids), device=self.device, dtype=torch.float32)
            _, pred = predict_batch(self.model, b)
            out[ids] = (b[:, -1, :]-pred).abs().cpu().numpy()
        return out

    def scores(self, residual):
        evidence = residual / (self.scale+1e-8)
        if self.method in ['D04','D05']:
            return residual[:, 0], evidence
        return aggregate(evidence, 'max' if self.method == 'D01' else self.cfg['model']['score_agg'], self.cfg['model']['score_topk']), evidence

    @torch.no_grad()
    def window_score(self, window):
        b = torch.as_tensor(window[None], device=self.device, dtype=torch.float32)
        _, pred = predict_batch(self.model, b)
        residual = (b[:, -1]-pred).abs().cpu().numpy()
        score, evidence = self.scores(residual)
        return float(score[0]), residual[0], evidence[0]


def alarm_table(table, threshold, policy, interval, consolidate=True):
    records = []
    for _, group in table.groupby(['source_file','segment_id'], sort=False):
        state = machine(threshold, policy, consolidate)
        for row in group.itertuples():
            update = state.update(row.score, row.sample_id, str(row.timestamp), pd.Timestamp(row.timestamp).timestamp(),
                                  bool(row.warmup), bool(row.quality_flags))
            records.append(dict(sample_id=row.sample_id, alarm_state=update['state'], alarm_id=update['alarm_id'],
                                alarm_opened=update['opened'], alarm_closed=update['closed']))
    return table.merge(pd.DataFrame(records), on='sample_id', validate='one_to_one')


def calibrate_and_replay(method, fitted, prepared, output, cfg, site, device):
    output.mkdir(parents=True, exist_ok=True)
    index, scaler = read_json(prepared/'index.json'), read_json(prepared/'scaler.json')
    detector = Detector(method, fitted, scaler, cfg, device)
    training = []
    for frame in load_frames(prepared, index, 'train'):
        X = apply_scaler(frame[scaler['tags']].to_numpy(np.float32), scaler)
        training.append(detector.residuals(X)[detector.window-1:])
    raw = np.concatenate(training)
    detector.scale = residual_scale(raw, cfg['scaler']['residual_scale']) if method in ['D03','D06','D07'] else np.ones(raw.shape[1])
    write_json(output/'residual_scale.json', dict(scale=detector.scale.tolist(), fit_split='train',
               weight_sha256=detector.weight_hash, scaler_hash=digest(scaler)))
    del raw, training
    tables, evidences = {}, {}
    for split in ['val_threshold', 'test']:
        rows, values, residuals = [], [], []
        for frame in load_frames(prepared, index, split):
            X = apply_scaler(frame[scaler['tags']].to_numpy(np.float32), scaler)
            residual = detector.residuals(X)
            score, evidence = detector.scores(residual)
            if not np.isfinite(score).all():
                raise FloatingPointError(f'{method}/{split}: nonfinite anomaly score')
            table = frame[KEYS+['quality_flags']].copy()
            table['warmup'] = np.arange(len(frame)) < detector.window-1
            table['score'] = score
            rows.append(table); values.append(evidence); residuals.append(residual)
        if not rows:
            if split == 'val_threshold':
                raise ValueError('Missing normal calibration data; test fallback is prohibited')
            continue
        tables[split], evidences[split] = pd.concat(rows, ignore_index=True), np.concatenate(values)
        np.save(output/f'{split}_residual.npy', np.concatenate(residuals))
    val = tables['val_threshold']
    valid = ~val.warmup
    if not valid.any():
        raise ValueError('Empty calibration interval after warmup')
    tau = float(np.quantile(val.loc[valid,'score'], cfg['threshold']['quantile']))
    policy = cfg['alarm']
    if cfg['threshold']['method'] == 'far_per_hour':
        # Enumerated candidates, because consolidation can make event FAR nonmonotonic.
        candidates = np.unique(np.quantile(val.loc[valid,'score'], np.linspace(.9,1,101)))
        tau = float(candidates[-1])
        for candidate in candidates:
            alarms = alarm_table(val, candidate, policy, site['input']['sample_interval_seconds'])
            far = alarms.alarm_opened.sum()/(valid.sum()*site['input']['sample_interval_seconds']/3600)
            if far <= cfg['threshold']['far_per_hour']:
                tau = float(candidate); break
    channel = np.quantile(evidences['val_threshold'][valid], cfg['rootcause']['onset_quantile'], axis=0)
    threshold = dict(schema_version='2.0', value=tau, method=cfg['threshold']['method'], quantile=cfg['threshold']['quantile'],
                     calibration_split='val_threshold', weight_sha256=detector.weight_hash,
                     scaler_hash=digest(scaler), residual_scale_hash=digest(detector.scale.tolist()), alarm_policy=policy,
                     model_id=method, aggregation=cfg['model']['score_agg'], topk=cfg['model']['score_topk'],
                     calibration_fingerprint=digest(val.loc[valid,'sample_id'].tolist()))
    threshold['threshold_id'] = digest(threshold)
    write_json(output/'threshold.json', threshold)
    write_json(output/'channel_thresholds.json', dict(values=channel.tolist(), calibration_split='val_threshold', threshold_id=threshold['threshold_id']))
    for split, table in tables.items():
        table['threshold'] = tau
        table['threshold_id'] = threshold['threshold_id']
        table['raw_pred'] = (table.score > tau) & ~table.warmup
        table = alarm_table(table, tau, policy, site['input']['sample_interval_seconds'])
        table.to_parquet(output/f'{split}_scores.parquet', index=False)
        np.save(output/f'{split}_evidence.npy', evidences[split])
    val = alarm_table(val, tau, policy, site['input']['sample_interval_seconds'])
    hours = valid.sum()*site['input']['sample_interval_seconds']/3600
    write_json(output/'calibration_audit.json', dict(split='val_threshold', valid_rows=int(valid.sum()), normal_hours=hours,
        point_exceedance_rate=float((val.loc[valid,'score']>tau).mean()),
        exceeding_samples_per_hour=float((val.loc[valid,'score']>tau).sum()/hours),
        false_alert_events_per_hour=float(val.alarm_opened.sum()/hours), far_resolution=1/hours))
    if method in ['D06','D07']:
        frame=load_frames(prepared,index,'val_model')[0]
        X=apply_scaler(frame[scaler['tags']].to_numpy(np.float32),scaler)
        windows=build_index([X],detector.window,pad_mode='skip')
        timings=[]
        for i in range(min(len(windows),42)):
            window=windows.gather(np.array([i]))[0]
            sync(device);start=time.perf_counter();detector.window_score(window);sync(device)
            if i>=10:timings.append((time.perf_counter()-start)*1000)
        memory=(torch.mps.current_allocated_memory() if device.type=='mps' else
                torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        write_json(output/'efficiency.json',dict(batch=1,windows=len(timings),warmup_count=10,
            p50_ms=float(np.median(timings)) if timings else None,p95_ms=float(np.quantile(timings,.95)) if timings else None,
            throughput_windows_per_second=1000/float(np.mean(timings)) if timings else None,
            memory_kind='device_allocator' if memory is not None else 'unmeasured',memory_bytes=memory,device=str(device)))
    return threshold
