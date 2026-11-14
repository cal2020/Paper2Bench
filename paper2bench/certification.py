"""Bounded live text-endpoint probes, separate from benchmark certification.

Only the fixed public fixture is sent. A pass establishes that one configured
role completed this fixture at the recorded time; it is not a quality, residency,
native-JSON, tool, vision, or scientific-reproduction certification.
"""
from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

from .bundles import atomic_json, canonical_json
from .evaluation import EvaluationError, parse_json_object
from .llm import chat
from .providers import ProviderError, get_adapter
from .runtime_config import (ROLES, RuntimeConfigError, get_context,
                             is_loopback_endpoint, load_runtime_config, runtime_scope)

PROBE_VERSION = "public-text-fixture/v1"
_MESSAGES = [
    {"role": "system", "content": "Return only one JSON object with the single field answer containing an integer. No prose or Markdown."},
    {"role": "user", "content": "What is 2 + 2?"},
]
_MAX_INVENTORY_BYTES = 2 * 1024 * 1024
_DIGEST = re.compile(r"(?:sha256:)?([0-9a-f]{64})")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _model_name(value):
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ProviderError("malformed_inventory", "model inventory contains an invalid model identity")
    # Ollama's omitted tag denotes :latest; do not rewrite explicit tags.
    return value if ":" in value.rsplit("/", 1)[-1] else value + ":latest"


def _inventory_get(ref, credential=None):
    """Reuse the provider transport's absolute deadline and loopback DNS guard.

    It is intentionally an internal package dependency: metadata must have the
    same connection/header/body deadline guarantees as generation, rather than
    introducing a second weaker HTTP path. No proxy or redirect is permitted.
    """
    from .providers.http import (_DeadlineState, _DeadlineHTTPHandler,
                                 _DeadlineHTTPSHandler, _NoRedirect,
                                 _TRANSPORT_SLOTS, _unique_object, _finite_float,
                                 _reject_constant)
    deadline = time.monotonic() + min(5.0, ref.timeout_seconds)
    if not _TRANSPORT_SLOTS.acquire(timeout=max(0, deadline - time.monotonic())):
        raise ProviderError("transport_capacity", "model metadata transport capacity is occupied")
    state = _DeadlineState(deadline, loopback_only=True)
    completed = queue.Queue(maxsize=1)

    def work():
        try:
            headers = {"Accept": "application/json"}
            if credential:
                headers["Authorization"] = "Bearer " + credential
            req = request.Request(ref.endpoint + "/api/tags", headers=headers, method="GET")
            opener = request.build_opener(request.ProxyHandler({}), _NoRedirect(),
                                          _DeadlineHTTPHandler(state), _DeadlineHTTPSHandler(state))
            with opener.open(req, timeout=max(0.001, deadline - time.monotonic())) as response:
                size = response.headers.get("Content-Length")
                if size is not None:
                    try:
                        declared = int(size)
                    except ValueError:
                        raise ProviderError("malformed_inventory", "invalid inventory response length") from None
                    if declared < 0 or declared > _MAX_INVENTORY_BYTES:
                        raise ProviderError("inventory_too_large", "inventory response exceeds its limit")
                chunks, received = [], 0
                while True:
                    state.check()
                    chunk = response.read1(min(65536, _MAX_INVENTORY_BYTES + 1 - received))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    received += len(chunk)
                    if received > _MAX_INVENTORY_BYTES:
                        raise ProviderError("inventory_too_large", "inventory response exceeds its limit")
                if size is not None and received != declared:
                    raise ProviderError("malformed_inventory", "inventory response ended early")
                body = json.loads(b"".join(chunks), object_pairs_hook=_unique_object,
                                  parse_float=_finite_float, parse_constant=_reject_constant)
                if not isinstance(body, dict) or not isinstance(body.get("models"), list) or len(body["models"]) > 10000:
                    raise ProviderError("malformed_inventory", "inventory does not contain a bounded model list")
            result = (True, body)
        except BaseException as exc:
            result = (False, exc)
        finally:
            state.abort()
            _TRANSPORT_SLOTS.release()
        completed.put_nowait(result)

    worker = threading.Thread(target=work, name="paper2bench-model-metadata", daemon=True)
    try:
        worker.start()
    except BaseException:
        _TRANSPORT_SLOTS.release()
        raise ProviderError("transport_capacity", "model metadata worker could not start") from None
    try:
        ok, result = completed.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty:
        state.abort()
        raise ProviderError("metadata_timeout", "absolute model metadata deadline expired") from None
    if time.monotonic() >= deadline:
        state.abort()
        raise ProviderError("metadata_timeout", "absolute model metadata deadline expired")
    if ok:
        return result
    if isinstance(result, ProviderError):
        raise result
    if isinstance(result, error.HTTPError):
        result.close()
        if result.code in {301, 302, 303, 307, 308}:
            raise ProviderError("policy_denied", "model metadata redirects are not permitted") from None
    raise ProviderError("metadata_unavailable", "model metadata could not be read safely") from None


def _ollama_identity(ref, credential=None):
    inventory = _inventory_get(ref, credential)
    expected = _model_name(ref.model)
    candidates = []
    for item in inventory["models"]:
        if not isinstance(item, dict):
            raise ProviderError("malformed_inventory", "model inventory item is not an object")
        names = [item.get(key) for key in ("name", "model") if item.get(key) is not None]
        if any(_model_name(name) == expected for name in names):
            candidates.append(item)
    if not candidates:
        raise ProviderError("unavailable_model", "configured local model is not present in the inventory")
    identities = []
    for item in candidates:
        match = _DIGEST.fullmatch(item.get("digest", "")) if isinstance(item.get("digest"), str) else None
        if not match:
            raise ProviderError("malformed_inventory", "local model lacks a valid content digest")
        details = item.get("details", {})
        if not isinstance(details, dict):
            raise ProviderError("malformed_inventory", "local model details must be an object")
        size = item.get("size")
        if size is not None and (type(size) is not int or size < 0):
            raise ProviderError("malformed_inventory", "local model size is invalid")
        identities.append({"model": expected, "digest": "sha256:" + match.group(1),
                           "size_bytes": size,
                           "quantization": details.get("quantization_level") if isinstance(details.get("quantization_level"), str) else None,
                           "parameter_size": details.get("parameter_size") if isinstance(details.get("parameter_size"), str) else None,
                           "format": details.get("format") if isinstance(details.get("format"), str) else None})
    if len({item["digest"] for item in identities}) != 1:
        raise ProviderError("ambiguous_model_identity", "local model has conflicting inventory identities")
    identity = identities[0]
    if any(isinstance(value, str) and len(value) > 1024 for value in identity.values()):
        raise ProviderError("malformed_inventory", "local model identity exceeds its bounds")
    return identity


def _error_status(error):
    code = getattr(error, "code", "probe_error")
    blocked = {"policy_denied", "authentication", "unavailable_model", "metadata_unavailable",
               "metadata_timeout", "transport_capacity", "invalid_runtime_config", "budget_exhausted",
               "context_overflow", "unsupported_capability", "ambiguous_model_identity"}
    return ("blocked" if code in blocked or getattr(error, "outcome_unknown", False) else "failed"), code


def certify_runtime(config, *, roles=None, output_path=None, probe_max_tokens=32, allow_cloud=False):
    """Perform at most one fixed text probe per selected role, without retries.

    Nonlocal/cloud probes require ``allow_cloud=True``. An active matching runtime
    context is reused, preserving its durable budget; a mismatched context is
    rejected instead of resetting accounting. Local inventory visibility is an
    operator assertion, not proof of model-server egress or scientific ability.
    """
    if type(probe_max_tokens) is not int or not 1 <= probe_max_tokens <= 128:
        raise ValueError("probe_max_tokens must be an integer in [1,128]")
    if type(allow_cloud) is not bool:
        raise ValueError("allow_cloud must be a boolean")
    output = Path(output_path) if output_path is not None else None
    if output is not None and output.exists():
        raise ValueError("Certification receipt already exists; choose a new output path")
    report = {"schema_version": 1, "probe_version": PROBE_VERSION,
              "scope": "bounded_text_endpoint_fixture_only", "status": "running",
              "created_at": _timestamp(), "fixture_sha256": _hash(_MESSAGES),
              "requested_max_output_tokens": probe_max_tokens, "probes": [],
              "model_calls": 0, "probe_calls_attempted": 0, "request_dispatch_unknown": 0,
              "limitations": ["A fixture pass is not general model-quality certification",
                              "JSON text adherence is not native structured-output enforcement",
                              "Local inventory is not an attestation of model-server egress",
                              "Model identity comes from server-reported inventory, not an independent weight measurement",
                              "Cloud aliases and operator-configured revisions may be mutable"]}

    def save():
        if output is not None:
            atomic_json(output, report)

    try:
        cfg = load_runtime_config(config)
    except (ValueError, OSError) as error:
        report.update(status="blocked", error_code=getattr(error, "code", "invalid_runtime_config"), completed_at=_timestamp())
        save()
        return report
    chosen = sorted(cfg.roles) if roles is None else list(roles) if isinstance(roles, (list, tuple)) else []
    if (not chosen or len(chosen) > len(ROLES)
            or any(not isinstance(role, str) or role not in cfg.roles for role in chosen)
            or len(set(chosen)) != len(chosen)):
        raise ValueError("roles must be a nonempty unique selection of configured roles")
    active = get_context()
    if active is not None and active.config.to_dict() != cfg.to_dict():
        raise ValueError("Certification cannot replace an active runtime's budget/configuration")
    report["configuration_sha256"] = _hash(cfg.to_dict())
    report["selected_roles"] = chosen
    save()
    scope = nullcontext(active) if active is not None else runtime_scope(cfg)
    with scope as context:
        for role in chosen:
            probe = {"role": role, "status": "preflight", "started_at": _timestamp(),
                     "request_dispatched": False, "probe_calls": 0}
            report["probes"].append(probe)
            save()
            started = time.monotonic()
            try:
                ref = context.resolve(role)
                cloud_alias = ref.provider == "ollama" and (":cloud" in ref.model.lower() or ref.model.lower().endswith("-cloud"))
                local = (ref.provider in {"ollama", "openai_compatible"}
                         and ref.hosting in {"on_device", "local_server"}
                         and ref.locality_assurance != "unknown"
                         and is_loopback_endpoint(ref.endpoint) and not cloud_alias)
                probe.update(provider=ref.provider, endpoint=ref.endpoint, requested_model=ref.model,
                             hosting=ref.hosting, locality_assurance=ref.locality_assurance,
                             configured_revision=ref.revision, revision_assurance="operator_configured" if ref.revision else "unknown")
                if not local and not allow_cloud:
                    raise ProviderError("policy_denied", "nonlocal probes require explicit cloud authorization")
                if not local and ref.credential_ref is None:
                    raise RuntimeConfigError("Nonlocal certification requires a provider-specific credential reference")
                credential = context.credential(ref)
                identity_before = _ollama_identity(ref, credential) if local and ref.provider == "ollama" else None
                if identity_before is not None:
                    probe["local_identity_before"] = identity_before
                effective_tokens = min(probe_max_tokens, ref.output_tokens or probe_max_tokens)
                probe.update(status="probing", effective_max_output_tokens=effective_tokens,
                             request_dispatched=None, probe_calls=1)
                save()  # A crash leaves an honest possibly-dispatched receipt.
                probe["effective_timeout_seconds"] = min(30.0, ref.timeout_seconds)
                response = chat(ref.model, _MESSAGES, max_tokens=effective_tokens, temperature=0, role=role,
                                timeout_seconds=probe["effective_timeout_seconds"])
                probe.update(resolved_model=response.resolved_model or response.model,
                             finish_reason=response.finish_reason, request_dispatched=True,
                             request_id_sha256=(hashlib.sha256(response.request_id.encode()).hexdigest()
                                                if isinstance(response.request_id, str) else None),
                             response_sha256=hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
                             usage=response.usage, provider_duration_seconds=response.duration_seconds)
                fixture = parse_json_object(response.content, max_bytes=4096)
                fixture_passed = set(fixture) == {"answer"} and type(fixture["answer"]) is int and fixture["answer"] == 4
                if not fixture_passed:
                    raise ProviderError("fixture_mismatch", "the fixed text fixture did not pass")
                probe["fixture_result"] = {"answer": 4}
                if identity_before is not None:
                    identity_after = _ollama_identity(ref, credential)
                    probe["local_identity_after"] = identity_after
                    if identity_after["digest"] != identity_before["digest"]:
                        raise ProviderError("model_identity_drift", "local model identity changed during the probe")
                    if _model_name(response.resolved_model or response.model) != _model_name(ref.model):
                        raise ProviderError("model_identity_mismatch", "the response identifies a different local model")
                    probe["model_identity_status"] = "digest_stable_across_probe"
                    probe["model_identity_assurance"] = "server_reported_inventory_and_response"
                else:
                    probe["model_identity_status"] = "provider_response_identity_revision_not_independently_verified"
                    probe["model_identity_assurance"] = "provider_reported_alias"
                probe["status"] = "passed"
                probe["capabilities"] = {name: ("tested_pass" if name == "text_generation" else
                                                      "unsupported" if value == "unsupported" else "unverified")
                                         for name, value in get_adapter(ref.provider).capabilities().items()}
                probe["json_text_fixture"] = "passed_prompt_adherence_only"
            except (ProviderError, RuntimeConfigError, EvaluationError, OSError, ValueError) as error:
                status, code = _error_status(error)
                probe.update(status=status, error_code=code,
                             outcome_unknown=bool(getattr(error, "outcome_unknown", False)))
                if code in {"context_overflow", "budget_exhausted", "unsupported_capability"}:
                    probe["request_dispatched"] = False
            probe.update(completed_at=_timestamp(), elapsed_seconds=time.monotonic() - started)
            save()
        report["budget"] = context.ledger.snapshot()
    states = {probe["status"] for probe in report["probes"]}
    report["status"] = "passed" if states == {"passed"} else "failed" if "failed" in states else "blocked"
    report["model_calls"] = sum(probe["request_dispatched"] is True for probe in report["probes"])
    report["probe_calls_attempted"] = sum(probe["probe_calls"] for probe in report["probes"])
    report["request_dispatch_unknown"] = sum(probe["request_dispatched"] is None for probe in report["probes"])
    report["completed_at"] = _timestamp()
    save()
    return report
