"""Controller-owned admission limits; a candidate manifest cannot raise them."""
from .runtime_config import get_context

DEFAULT_CEILINGS = {"timeout_seconds": 3600, "memory_bytes": 4294967296,
                    "cpus": 4, "pids_limit": 256, "workspace_bytes": 134217728,
                    "output_limit_bytes": 1048576}
DEFAULT_REQUEST = {"timeout_seconds": 300, "memory_bytes": 536870912, "cpus": 1,
                   "pids_limit": 64, "workspace_bytes": 67108864, "output_limit_bytes": 65536}


def admit_limits(limits):
    limits = dict(limits or {})
    if set(limits) - set(DEFAULT_CEILINGS):
        raise ValueError("Unknown execution resource limit")
    context = get_context()
    ceilings = dict(DEFAULT_CEILINGS)
    if context:
        execution = context.config.execution
        if execution.get("backend", "docker") not in {"docker", "oci", "ssh"}:
            raise ValueError("Unsupported execution backend")
        if execution.get("backend") == "ssh" and context.config.policy.get("data_egress") == "local_only":
            raise ValueError("Local-only policy cannot use remote SSH execution")
        if execution.get("network", "none") != "none":
            raise ValueError("Candidate network access is not supported by this release")
        if execution.get("max_parallel_jobs", 1) != 1:
            raise ValueError("This release supports one job per controller")
        supplied = execution.get("limits", {})
        if not isinstance(supplied, dict) or set(supplied) - set(ceilings):
            raise ValueError("Invalid operator execution ceilings")
        ceilings.update(supplied)
    request = dict(DEFAULT_REQUEST, **limits)
    import math
    for key, value in request.items():
        cap = ceilings[key]
        if (type(value) not in {int, float} or type(cap) not in {int, float}
                or not math.isfinite(cap) or not math.isfinite(value) or not 0 < value <= cap):
            raise ValueError(f"Execution request exceeds operator limit: {key}")
    if context:
        budget = context.config.budgets.get("experiments", {})
        requirements = {"wall_seconds": request["timeout_seconds"], "memory_bytes": request["memory_bytes"],
                        "disk_bytes": request["workspace_bytes"], "cpu_seconds": request["timeout_seconds"] * request["cpus"]}
        for key, amount in requirements.items():
            if key in budget and amount > budget[key]:
                raise ValueError(f"Execution request exceeds experiment budget: {key}")
    return request
