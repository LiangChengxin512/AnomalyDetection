"""Shared planner/evaluator rollout with explicit qualification and control transformation."""
import numpy as np
from src.decision.action_transform import transform


def rollout(model, card, state, tags, action, delta, horizon, interval=1., bypass_uncertainty=False):
    if not card.get('action_response_qualified'):
        raise ValueError('action_response_not_qualified')
    if not bypass_uncertainty and not card.get('prediction_usable'):
        raise ValueError('prediction_not_usable')
    current = np.asarray(state,float).copy()
    low,high = np.array(model['support_min']),np.array(model['support_max'])
    coef,bias = np.array(model['coef']),np.array(model['intercept'])
    trace=[]
    for h in range(1,horizon+1):
        controls = {tags[j]:current[j] for j in model['controls']}
        changed = transform(action,controls,delta if h==1 else 0.,interval)
        for j in model['controls']: current[j]=changed[tags[j]]
        if not bypass_uncertainty and ((current<low)|(current>high)).any():
            raise ValueError('outside_training_support')
        current[model['states']] = coef@current+bias
        trace.append(dict(h=h,values=current.copy(),control_input=changed,surrogate_only=True))
    return trace
