"""
Chaos & Schema Drift Injectors.

Injects dynamic disruptions into the corporate environment mid-episode.
This forces the agent to handle changing rules (Schema Drift) and 
unpredictable actors (Manager OOO, Ticket Floods).
"""

import logging
import random
import time

logger = logging.getLogger(__name__)

class ChaosInjector:
    """
    Mutates the corporate state to simulate production-level enterprise chaos.
    """

    def __init__(self, tool_backend):
        self.tools = tool_backend

    def inject(self, failure_type: str, params: dict) -> str:
        """Routes the chaos injection based on the scenario specification."""
        department = params.get("department", "engineering")
        
        injectors = {
            "policy_drift": self._inject_policy_drift,
            "manager_ooo": self._inject_manager_ooo,
            "ticket_flood": self._inject_ticket_flood,
            "budget_freeze": self._inject_budget_freeze,
            "calendar_api_down": self._inject_api_outage,
        }
        
        fn = injectors.get(failure_type)
        if not fn:
            return f"Unknown chaos type: {failure_type}"
        
        return fn(department)

    def _inject_policy_drift(self, department: str) -> str:
        """
        Solves Patronus AI Bonus (Schema Drift).
        Quietly alters a corporate policy. If the agent relies on its cached
        knowledge instead of re-running `policy:lookup`, it will fail the audit.
        """
        drift_options = [
            ("GDPR_DATA_RETENTION_DAYS", 30), # Down from 90
            ("ONBOARDING_SLA_HOURS", 24),     # Down from 48
            ("REMOTE_WORK_DAYS_ALLOWED", 0),  # RTO mandate
            ("MANAGER_APPROVAL_LIMIT_TC", 160000) # Lowered ceiling
        ]
        
        key, new_value = random.choice(drift_options)
        self.tools.active_policies[key] = new_value
        
        # Simulate the time it takes for an org-wide memo to propagate
        time.sleep(2) 
        
        logger.info(f"Chaos Injected: Schema Drift -> {key} is now {new_value}")
        return f"SYSTEM UPDATE: Organization policy {key} has been updated. Compliance is mandatory."

    def _inject_manager_ooo(self, department: str) -> str:
        """
        Breaks the standard `escalate:manager` tool path.
        The agent must now figure out how to route around the standard hierarchy.
        """
        self.tools.manager_ooo = True
        time.sleep(1)
        return f"OUT OF OFFICE: The hiring manager for {department} has gone on unexpected leave."

    def _inject_ticket_flood(self, department: str) -> str:
        """
        Distracts the agent with urgent IT/HR tickets that must be cleared
        before the end of the episode to maximize reward.
        """
        flood_tickets = {
            f"INC-{random.randint(1000, 9999)}": f"URGENT: Payroll system access revoked for {department} lead.",
            f"INC-{random.randint(1000, 9999)}": "SECURITY: Phishing attempt detected in candidate portal.",
            f"INC-{random.randint(1000, 9999)}": f"HR: Missing background check forms for recent {department} hires."
        }
        
        self.tools.open_tickets.update(flood_tickets)
        time.sleep(2)
        return "PAGERDUTY ALERT: Multiple critical tickets have entered the IT Helpdesk queue."

    def _inject_budget_freeze(self, department: str) -> str:
        """
        Creates an immediate financial constraint mid-negotiation.
        """
        self.tools.budget_unlocked = False
        time.sleep(1)
        return f"FINANCE ALERT: Q3 Budget freeze initiated for {department}. Discretionary bonuses suspended."

    def _inject_api_outage(self, department: str) -> str:
        """
        Simulates an external API failure (e.g., Google Calendar goes down).
        The agent must fall back to manual email coordination.
        """
        # We handle this logic in the tool_backends.py, but this triggers the state
        # For simplicity, we just inject a blocking ticket that warns the agent
        ticket_id = f"SYS-{random.randint(100, 999)}"
        self.tools.open_tickets[ticket_id] = "MAJOR OUTAGE: Calendar Scheduling API is returning 503 errors."
        return "STATUS PAGE: External API integration degraded."