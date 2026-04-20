"""
Shared constants for EnterpriseOps Gym.
Central place for corporate topology, salary bands, policies, and curriculum tiers.
"""

# ---- Corporate Topology (Replaces K8s Cluster Topology) ----
COMPANY_TOPOLOGY = {
    "engineering": ["senior_backend", "sre", "frontend_lead"],
    "product": ["pm_core", "pm_growth"],
    "sales": ["account_executive", "sales_engineer"]
}

# ---- Healthy Baseline State ----
# Used by reset() to restore budgets and policies before the negotiation starts.
HEALTHY_STATE = {
    "engineering": {
        "senior_backend": {
            "title": "Senior Backend Engineer",
            "base_band": [150000, 180000],
            "equity_band": [20000, 40000],
            "signing_bonus_cap": 15000,
            "hiring_manager": "Sarah Chen",
            "required_access": ["github_core", "aws_prod_read", "jira_eng"]
        },
        "sre": {
            "title": "Site Reliability Engineer",
            "base_band": [140000, 170000],
            "equity_band": [15000, 30000],
            "signing_bonus_cap": 10000,
            "hiring_manager": "David Kim",
            "required_access": ["aws_prod_admin", "pagerduty", "datadog"]
        }
    }
}

# ---- Active Corporate Policies (Subject to schema drift) ----
DEFAULT_POLICIES = {
    "GDPR_DATA_RETENTION_DAYS": 90,
    "ONBOARDING_SLA_HOURS": 48,
    "MANAGER_APPROVAL_LIMIT_TC": 190000, # Total Comp approval ceiling
    "EQUITY_VESTING_SCHEDULE": "4_year_1_year_cliff",
    "REMOTE_WORK_DAYS_ALLOWED": 2
}

# ---- Curriculum Tiers (Replaces K8s Fault Tiers) ----
CANDIDATE_TIERS = {
    "naive_candidate":           {"tier": 1, "min_difficulty": 0.0},
    "anchoring_candidate":       {"tier": 2, "min_difficulty": 0.25},
    "deadline_candidate":        {"tier": 3, "min_difficulty": 0.40},
    "competing_offer_candidate": {"tier": 4, "min_difficulty": 0.60},
    "adversarial_llm_candidate": {"tier": 5, "min_difficulty": 0.80},
}

# ---- Timeouts and Limits ----
MAX_STEPS = 20 # Increased slightly to accommodate 3-phase episodes
SLA_VIOLATION_PENALTY = -0.5