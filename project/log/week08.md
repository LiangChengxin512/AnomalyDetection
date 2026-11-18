# Week 8 — Action catalog and shadow policy

## Done
- `config/action_catalog.yaml` + `src/decision/action_catalog.py`: legal
  action enumeration, tag bounds, applicable conditions (draft, needs domain
  review).
- `src/decision/shadow_policy.py`: constrained short-horizon decision;
  checks catalog legality, tag-range envelope, evidence sufficiency and
  surrogate uncertainty; outputs `advice.jsonl` with selected_action|null,
  mode advisory|abstain, constraint_checks, rationale. Mock run: 17,077
  events, 7,760 abstentions, legal_action_rate = 1.0.
