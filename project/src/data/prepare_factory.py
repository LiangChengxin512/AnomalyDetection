"""One HAI/factory adapter with label-free, portable prepared telemetry."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.common.config import PROJECT_ROOT, load_yaml
from src.common.io_utils import read_json, write_json, sha256_file
from src.common.provenance import digest, relative_file
from src.data.segments import identify, split_segments, KEYS
from src.data.split_protocol import validation_ranges
from src.data.fit_scaler import apply_scaler


def path_for(value):
    p = __import__('pathlib').Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_site(path):
    site = load_yaml(path)
    if 'tag_schema' in site:
        schema = load_yaml(site['tag_schema'])
        site['features'] = [dict(name=t['name'], source_column=t['name'],
            kind='discrete' if t['discrete'] else 'continuous', equipment_id=t.get('process'),
            role=t.get('kind'), unit=t.get('unit', 'unknown'), source_unit=t.get('unit', 'unknown'),
            transform={'scale': 1., 'offset': 0.}, hard_bounds={'min': None, 'max': None}) for t in schema['tags']]
    names = [f['name'] for f in site['features']]
    if len(set(names)) != len(names) or not names:
        raise ValueError('Features must have unique, nonempty names')
    forbidden = {'label', 'label_attack', 'attack_id', 'event_id', 'target_points'}
    if any(f['name'].lower() in forbidden or f['source_column'].lower() in forbidden for f in site['features']):
        raise ValueError('Evaluation fields cannot be model features')
    if site['input']['sample_interval_seconds'] <= 0:
        raise ValueError('sample_interval_seconds must be positive')
    return site


def site_contract(site):
    return {k: site[k] for k in ['site_id', 'input', 'features', 'model', 'process', 'decision']}


def read_telemetry(path, site):
    features = site['features']
    columns = [site['input']['timestamp_column']] + [f['source_column'] for f in features]
    try:
        df = pd.read_csv(path, usecols=columns)
    except ValueError as exc:
        raise ValueError(f'{path}: required timestamp or sensor column missing: {exc}') from exc
    out = pd.DataFrame({'timestamp': df[columns[0]]})
    for f in features:
        values = pd.to_numeric(df[f['source_column']], errors='coerce').to_numpy(float)
        transform = f.get('transform', {})
        values = values * transform.get('scale', 1.) + transform.get('offset', 0.)
        if f['kind'] == 'discrete' and f.get('allowed_values'):
            values[~np.isin(values, f['allowed_values'])] = np.nan
        out[f['name']] = values
    out['quality_flags'] = np.where(np.isfinite(out[[f['name'] for f in features]]).all(axis=1), '', 'missing_or_invalid_feature')
    out = identify(out, site.get('dataset_id', site['site_id']), __import__('pathlib').Path(path).name,
                   site['input']['sample_interval_seconds'])
    zone = site['input'].get('source_timezone')
    if zone and zone != 'unspecified_testbed_local' and out.timestamp.dt.tz is None:
        out['timestamp'] = out.timestamp.dt.tz_localize(zone, ambiguous='raise', nonexistent='raise')
    return out


def input_fingerprint(site):
    files = {str(p): sha256_file(path_for(p)) for names in site['data'].values() if isinstance(names, list) for p in names}
    telemetry_contract = {k:site[k] for k in ['site_id','input','features']}
    return digest({'files': files, 'telemetry_contract': telemetry_contract}), files


def prepare(site, cfg, dest, smoke_rows=0, smoke_replay_rows=0):
    dest.mkdir(parents=True, exist_ok=True)
    tags = [f['name'] for f in site['features']]
    W = cfg['window']['length']
    fingerprint, hashes = input_fingerprint(site)
    manifest = dict(schema_version='2.0', dataset_fingerprint=fingerprint, files=hashes, splits={}, exclusions=[], tags=tags)
    mapping = {'train': 'train_files', 'validation': 'normal_validation_files', 'test': 'replay_files'}
    for split, key in mapping.items():
        for path in site['data'].get(key, []):
            df = read_telemetry(path_for(path), site)
            if df.quality_flags.ne('').any():
                bad = int(np.flatnonzero(df.quality_flags.ne(''))[0])
                raise ValueError(f'{path}: invalid required values at row {bad + 2}; training does not impute')
            if split == 'validation':
                ranges = validation_ranges(len(df), W)
                manifest['exclusions'].append(dict(source_file=df.source_file.iloc[0], gap_rows=2*(W-1), ranges=ranges))
            else:
                ranges = {split: (0, len(df))}
            for role, (start, end) in ranges.items():
                part = df.iloc[start:end].copy()
                # Budget always disclosed. Subsets start at the declared split boundary.
                cap = smoke_replay_rows if split == 'test' and smoke_replay_rows else smoke_rows
                if cap:
                    part = part.iloc[:cap]
                part['segment_id'] = part['segment_id'].astype(str) + ':' + role
                for j, group in enumerate(split_segments(part)):
                    if len(group) < W:
                        raise ValueError(f'{path}/{role}: segment {j} shorter than window {W}')
                    name = f'{role}__{__import__("pathlib").Path(path).stem}__{j}.parquet'
                    group.to_parquet(dest / name, index=False)
                    manifest['splits'].setdefault(role, []).append(dict(parquet=name, rows=len(group),
                        source_file=group.source_file.iloc[0], segment_id=group.segment_id.iloc[0]))
    for required in ['train', 'val_model', 'val_uncertainty', 'val_threshold']:
        if not manifest['splits'].get(required):
            raise ValueError(f'Missing normal split: {required}')
    arrays = [f[tags].to_numpy(np.float32) for f in load_frames(dest, manifest, 'train')]
    X = np.concatenate(arrays)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    scaler = dict(tags=tags, min=mins.tolist(), max=maxs.tolist(), span=np.maximum(maxs-mins, 1e-8).tolist(),
                  continuous_mask=[f['kind'] == 'continuous' for f in site['features']], fit_split='train',
                  method='minmax_train_clip_-0.5_1.5', dataset_fingerprint=fingerprint)
    # Constant features use unit denominator; variation is retained in raw measurements.
    scaler['span'] = np.where(maxs-mins < 1e-8, 1., maxs-mins).tolist()
    median = np.median(X, axis=0)
    for j, f in enumerate(site['features']):
        if f['kind'] == 'discrete':
            values, counts = np.unique(X[:, j], return_counts=True)
            median[j] = values[counts.argmax()]
    reference = dict(tags=tags, raw_median=median.tolist(), scaled_median=apply_scaler(median, scaler).tolist(),
                     raw_min=mins.tolist(), raw_max=maxs.tolist(), fit_split='train')
    write_json(dest / 'index.json', manifest)
    write_json(dest / 'split_manifest.json', manifest)
    write_json(dest / 'scaler.json', scaler)
    write_json(dest / 'normal_reference.json', reference)
    return manifest


def load_frames(root, manifest, split):
    return [pd.read_parquet(relative_file(root, rec['parquet'])) for rec in manifest['splits'].get(split, [])]


def scaled_frames(root, split):
    manifest, scaler = read_json(root / 'index.json'), read_json(root / 'scaler.json')
    return [apply_scaler(f[scaler['tags']].to_numpy(np.float32), scaler) for f in load_frames(root, manifest, split)]
