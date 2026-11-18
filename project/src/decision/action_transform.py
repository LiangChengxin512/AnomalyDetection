"""Physical actions change a single declared control input, never a measured outcome."""
import numpy as np


def transform(action, controls, delta=0., dt=1.):
    result = dict(controls)
    if action['kind'] in ['inspect','ticket','escalate','no_action','hold']:
        return result
    tag = action.get('control_tag')
    if not tag or tag not in controls or not np.isfinite(controls[tag]):
        raise ValueError('missing_control_input')
    bounds = action.get('absolute_bounds', {})
    if action.get('unit') in [None, '', 'unknown'] or bounds.get('min') is None or bounds.get('max') is None:
        raise ValueError('unreviewed_units_or_bounds')
    if abs(delta)>action['max_delta'] or abs(delta)/dt>action['rate_limit']:
        raise ValueError('delta_or_rate_limit')
    value = controls[tag]+delta
    if not bounds['min']<=value<=bounds['max']:
        raise ValueError('absolute_bounds')
    result[tag] = value
    return result
