"""
FastAPI application for the EnterpriseOps Gym Environment.
"""

import logging
import os
from openenv.core.env_server.http_server import create_app

# Import our custom models and environment
try:
    from models import EnterpriseOpsAction, EnterpriseOpsObservation
    from server.enterprise_environment import EnterpriseOpsEnvironment
except ModuleNotFoundError:
    from .models import EnterpriseOpsAction, EnterpriseOpsObservation
    from .server.enterprise_environment import EnterpriseOpsEnvironment

logger = logging.getLogger(__name__)

# Create the app with web interface integration
app = create_app(
    EnterpriseOpsEnvironment,
    EnterpriseOpsAction,
    EnterpriseOpsObservation,
    env_name="enterpriseops_gym",
    max_concurrent_envs=1, 
)

@app.get("/healthz")
async def healthz():
    """Quick health check to verify the environment can boot."""
    try:
        env = EnterpriseOpsEnvironment()
        return {"status": "ok", "message": "EnterpriseOps Environment ready"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "error": str(e)}

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EnterpriseOps Gym Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)