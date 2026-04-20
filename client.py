"""
EnterpriseOps Gym Environment Client.
"""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import EnterpriseOpsAction, EnterpriseOpsObservation, EnterpriseOpsState

class EnterpriseOpsEnv(
    EnvClient[EnterpriseOpsAction, EnterpriseOpsObservation, EnterpriseOpsState]
):
    """
    Client for the EnterpriseOps Gym Environment.
    """

    def __init__(self, base_url: str, **kwargs):
        # Allow longer timeouts for Tier 5 LLM-driven adversarial candidate responses
        kwargs.setdefault("message_timeout_s", 300.0)
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: EnterpriseOpsAction) -> Dict:
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[EnterpriseOpsObservation]:
        obs_data = payload.get("observation", {})
        
        observation = EnterpriseOpsObservation(
            command_output=obs_data.get("command_output", ""),
            system_status_summary=obs_data.get("system_status_summary", ""),
            active_alerts=obs_data.get("active_alerts", []),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 20),
            hint=obs_data.get("hint", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EnterpriseOpsState:
        return EnterpriseOpsState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_phase=payload.get("current_phase", "discovery"),
            difficulty=payload.get("difficulty", 0.0),
            candidate_tier=payload.get("candidate_tier", "")
        )