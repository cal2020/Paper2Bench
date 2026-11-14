"""Persist model-call budget intent without persisting prompts or credentials."""
from contextlib import contextmanager

from .bundles import canonical_json, digest
from .journal import RunJournal
from .runtime_config import load_runtime_config, runtime_scope


@contextmanager
def durable_runtime_scope(source, *, api_key=None, runs_dir="./runs", run_id=None, metadata=None):
    config = load_runtime_config(source)
    # Resolve policy before creating the ledger; no endpoint/model call occurs here.
    from .runtime_config import RunContext
    check = RunContext(config, api_key=api_key)
    for role in config.roles:
        check.resolve(role)
    limits = {f"{pool}.{key}": value for pool, budget in config.budgets.items()
              for key, value in budget.items() if key in {"requests", "input_tokens", "output_tokens"}}
    journal = RunJournal(runs_dir)
    identity = digest(canonical_json(config.to_dict()))
    if run_id is not None:
        previous = journal.get(run_id)
        if previous["spec"].get("kind") != "authoring" or previous["spec"].get("runtime_sha256") != identity:
            raise ValueError("Authoring run belongs to a different runtime configuration")
        if previous["spec"].get("generation") != metadata:
            raise ValueError("Authoring checkpoint identity changed")
        if previous["state"] not in {"failed", "blocked"}:
            raise ValueError("Authoring run is completed, active, cancelled or requires outcome reconciliation")
    else:
        run_id = journal.create({"kind": "authoring", "profile": config.profile,
                                 "runtime_sha256": identity, "generation": metadata}, limits=limits)
    attempt_id = journal.begin_attempt(run_id)
    try:
        with runtime_scope(config, api_key=api_key, budget_journal=journal, budget_run_id=run_id) as context:
            yield context
        journal.finish(run_id, attempt_id, "succeeded", {"budget": journal.get(run_id)["budget"]})
    except BaseException as error:
        # Unknown or incomplete model usage stays reserved even after the caller exits.
        status = "reconciling" if journal.get(run_id)["budget"]["reserved"] else "failed"
        journal.finish(run_id, attempt_id, status, {"error_type": type(error).__name__})
        raise
