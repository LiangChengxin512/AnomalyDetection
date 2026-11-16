"""One-/five-step prediction validation. Prediction is not intervention identification."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.common.io_utils import read_json, write_json
from src.data.prepare_factory import scaled_frames
from src.data.segments import transition_pairs


def fit_surrogates(prepared, output, site, cfg):
    output.mkdir(parents=True, exist_ok=True)
    train = scaled_frames(prepared,'train')
    vm = scaled_frames(prepared,'val_model')
    vu = scaled_frames(prepared,'val_uncertainty')
    X, Y = transition_pairs(train)
    # Every sampled pair remains exactly one data interval apart.
    ids = np.linspace(0,len(X)-1,min(len(X),cfg['surrogate']['fit_max_rows']),dtype=int)
    X, Y = X[ids], Y[ids]
    controls = [i for i,f in enumerate(site['features']) if f.get('role') in ['command','setpoint']]
    states = [i for i in range(X.shape[1]) if i not in controls]
    if not states:
        states, controls = list(range(X.shape[1])), []
    output_rows, models, cards = [], {}, {}
    for name, alpha in [('planner',cfg['surrogate']['alpha']),('evaluator',cfg['surrogate']['alpha']*10)]:
        model = Ridge(alpha=alpha).fit(X,Y[:,states])
        # State+control inputs are ordered exactly like the feature list.
        models[name] = dict(coef=model.coef_.tolist(), intercept=model.intercept_.tolist(), states=states, controls=controls,
                            support_min=X.min(0).tolist(),support_max=X.max(0).tolist(), alpha=alpha)
        all_pass = []
        for h in [1,5]:
            def errors(frames):
                errors_, persistence = [], []
                for frame in frames:
                    if len(frame)<=h: continue
                    current = frame[:-h].copy()
                    original = current.copy()
                    # Hold current controls; future recorded controls are never consulted.
                    for _ in range(h):
                        current[:,states] = model.predict(current)
                    errors_.append(frame[h:,states]-current[:,states])
                    persistence.append(frame[h:,states]-original[:,states])
                return np.concatenate(errors_),np.concatenate(persistence)
            uncertainty,_ = errors(vu)
            radius = np.quantile(np.abs(uncertainty), .95, axis=0)
            err, per = errors(vm)
            rmse, persistence = np.sqrt((err**2).mean(0)), np.sqrt((per**2).mean(0))
            coverage = (np.abs(err)<=radius).mean(0)
            for j, feature in enumerate(states):
                # Equality is accepted only for truly constant channels.
                qualified = bool((rmse[j]<persistence[j] or rmse[j]<1e-6) and coverage[j]>=.8)
                all_pass.append(qualified)
                output_rows.append(dict(schema_version='2.0', model_id=name, split_id='val_model',
                    uncertainty_fit_split='val_uncertainty',tag=site['features'][feature]['name'],horizon=h,
                    RMSE=rmse[j],persistence_RMSE=persistence[j],coverage=coverage[j],width=2*radius[j],qualification=qualified))
        cards[name] = dict(prediction_usable=all(all_pass), action_response_qualified=False,
            reason='Observational HAI data and unreviewed control units do not qualify intervention response',
            support_domain='train min/max of each state and control; no clipping', horizon_seconds=5*site['input']['sample_interval_seconds'],
            state_indices=states, control_indices=controls, fit_transitions=len(X),
            time_step_seconds=site['input']['sample_interval_seconds'], usable_for_policy_eval=False)
    pd.DataFrame(output_rows).to_csv(output/'surrogate_validation.csv',index=False)
    write_json(output/'models.json',models)
    write_json(output/'card.json',dict(cards['planner'],evaluator=cards['evaluator'],capabilities=['prediction_diagnostics']))
    return cards['planner']
