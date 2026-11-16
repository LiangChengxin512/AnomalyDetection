"""Auditable advice selection. Default HAI capability is inspection only."""
import numpy as np
from src.common.provenance import digest
from src.decision.action_transform import transform


def check_action(action, raw, site):
    checks = []
    def add(name, passed, value=None, limit=None, unit=None):
        checks.append(dict(action_id=action['id'], constraint_name=name, passed=bool(passed), measured_value=value, limit=limit, unit=unit))
    required = action.get('required_tags', [])
    add('required_values', all(t in raw and np.isfinite(raw[t]) for t in required))
    add('review_status', action.get('review_status') == 'approved')
    add('evidence_source', bool(action.get('evidence_source')))
    for tag, bounds in action.get('requires', {}).items():
        value = raw.get(tag)
        add('requires:'+tag, value is not None and np.isfinite(value) and bounds['min']<=value<=bounds['max'], value,
            str(bounds))
    if action['kind'] not in ['inspect','ticket','escalate']:
        add('physical_adjustment_enabled', site['decision'].get('physical_adjustment_enabled', False))
        tag = action.get('control_tag')
        feature = next((f for f in site['features'] if f['name']==tag), {})
        add('registered_control', bool(tag) and tag in raw and feature.get('role') in ['command','setpoint'])
        add('known_unit', action.get('unit') not in [None,'','unknown'] and action.get('unit')==feature.get('unit'))
        add('rollback', bool(action.get('rollback')))
        bounds = action.get('absolute_bounds', {})
        add('bounds_and_rate', bounds.get('min') is not None and bounds.get('max') is not None and
            action.get('max_delta') is not None and action.get('rate_limit') is not None)
    return checks


class AdvicePolicy:
    def __init__(self, site, catalog, reference, card, cfg, variant='B00', policy_id='P03'):
        self.site, self.catalog, self.reference, self.card, self.cfg = site, catalog, reference, card, cfg
        self.variant, self.policy_id = variant, policy_id
        self.offered = {}

    def decide(self, alarm_id, sample_id, timestamp, seconds, raw, candidates, margin, quality=False):
        key = dict(alarm_id=alarm_id, as_of_time=timestamp, sample_id=sample_id, policy_id=self.policy_id,
                   variant_id=self.variant, advice_id='advice:'+digest([alarm_id,sample_id,self.variant,self.policy_id])[:24])
        result = dict(key, action_id=None, mode='abstain', reason='', legal=None, cost=None,
                      surrogate_only=False, physical_adjustment=False, description_en='', duplicate=False,
                      suppressed_duplicate=False, specific_advice_abstained=False)
        if quality:
            result['reason'] = 'data_quality_hold'; return result, []
        if self.policy_id == 'P00':
            result['reason'] = 'alert_only'; return result, []
        no_context = self.variant=='B02'
        no_rca = self.variant in ['B03','B06'] or self.policy_id=='P01'
        low_evidence = margin < self.cfg['rootcause']['abstain_margin']
        route = [] if no_context else [c['tag'] for c in candidates[:3]]
        if no_rca:
            tags = self.reference['tags']
            med = np.array(self.reference['raw_median'])
            span = np.maximum(np.array(self.reference['raw_max'])-np.array(self.reference['raw_min']), 1e-8)
            deviation = np.abs(np.array([raw[t] for t in tags])-med)/span
            route = [tags[j] for j in np.argsort(-deviation,kind='stable')[:3]]
        constraints, eligible = [], []
        for action in self.catalog['actions']:
            checks = check_action(action, raw, self.site)
            if action['kind'] not in ['inspect','ticket','escalate']:
                qualified = self.card.get('action_response_qualified', False)
                checks.append(dict(action_id=action['id'], constraint_name='action_response_qualified', passed=qualified))
                # B05 removes support/uncertainty only, never physical/catalog or response-identification checks.
                if self.variant != 'B05':
                    checks.append(dict(action_id=action['id'], constraint_name='prediction_usable', passed=self.card.get('prediction_usable',False)))
            constraints.extend(checks)
            if all(c['passed'] for c in checks):
                applies = action.get('applies_to', [])
                overlap = len(set(route)&set(applies))
                if overlap or '*' in applies:
                    eligible.append((overlap, -float(action.get('cost',0)), action['id'], action))
        if low_evidence and self.variant != 'B04' and self.policy_id=='P03':
            eligible = [r for r in eligible if '*' in r[-1].get('applies_to', []) and r[-1]['kind']=='inspect']
            result['reason'] = 'low_evidence_specific_advice_withheld'
            result['specific_advice_abstained'] = True
        if not eligible:
            result['reason'] = result['reason'] or 'no_legal_applicable_action'; return result, constraints
        action = sorted(eligible, key=lambda r:(-r[0],-r[1],r[2]))[0][-1]
        token = (alarm_id, action['id'])
        cooldown = action.get('cooldown_s',30)
        if self.variant != 'B07' and token in self.offered and seconds-self.offered[token]<cooldown:
            result.update(mode='suppressed', reason='advice_cooldown', suppressed_duplicate=True)
            return result, constraints
        result['duplicate'] = token in self.offered
        self.offered[token] = seconds
        result.update(action_id=action['id'], mode=action['kind'], legal=True, cost=float(action.get('cost',0)),
                      description_en=action['description_en'], physical_adjustment=action['kind'] not in ['inspect','ticket','escalate'])
        return result, constraints
