"""Explicit runtime bindings, scoped propagation, and bounded model budgets.

Nothing in this module starts a model server or reads credentials at import time.
Locality is an operator assertion unless a trusted enforcement verifier is supplied.
"""

from __future__ import annotations

import contextlib
import contextvars
import ipaddress
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Union
from urllib.parse import urlsplit


ROLES = frozenset({"author", "reference", "agent", "judge"})
PROVIDERS = frozenset({"openai", "anthropic", "ollama", "openai_compatible"})
ROLE_POOLS = {"author": "authoring", "reference": "evaluation", "judge": "evaluation", "agent": "experiments"}
DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://127.0.0.1:11434",
}


class RuntimeConfigError(ValueError):
    code = "invalid_runtime_config"


class PolicyDenied(RuntimeConfigError):
    code = "policy_denied"


class BudgetExceeded(RuntimeConfigError):
    code = "budget_exhausted"


def _mapping(value: Any, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise RuntimeConfigError(f"{label} must be a mapping")
    return dict(value)


def _positive_int(value: Any, label: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        raise RuntimeConfigError(f"{label} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RuntimeConfigError(f"{label} must be a boolean")
    return value


def _public_copy(value: Any) -> Any:
    """Copy configuration metadata while redacting accidental execution secrets."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower()
            secret = normalized in {"api_key", "apikey", "password", "secret", "secrets", "credentials", "authorization", "access_token", "token"} or normalized.endswith(("_api_key", "_password", "_secret", "_access_token"))
            result[key] = "[redacted]" if secret else _public_copy(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_public_copy(item) for item in value]
    return value


def is_loopback_endpoint(endpoint: str) -> bool:
    host = urlsplit(endpoint).hostname or ""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_endpoint(endpoint: str, *, allow_insecure_http: bool = False) -> str:
    if not isinstance(endpoint, str) or not endpoint or len(endpoint) > 2048 or any(ord(c) < 32 for c in endpoint):
        raise RuntimeConfigError("model endpoint must be a nonempty URL of at most 2048 characters")
    parsed = urlsplit(endpoint)
    try:
        parsed.port
    except ValueError as exc:
        raise RuntimeConfigError("invalid endpoint port") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeConfigError("model endpoint must use http or https with a host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeConfigError("endpoint must not contain credentials, query parameters, or a fragment")
    if parsed.scheme == "http" and not is_loopback_endpoint(endpoint) and not allow_insecure_http:
        raise PolicyDenied("remote model endpoints require HTTPS")
    return endpoint.rstrip("/")


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str
    endpoint: str
    hosting: str = "managed_cloud_api"
    credential_ref: Optional[str] = None
    locality_assurance: str = "unknown"
    context_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    timeout_seconds: float = 120.0
    required_capabilities: tuple = ("text_generation",)
    revision: Optional[str] = None
    quantization: Optional[str] = None
    parameters: dict = field(default_factory=dict, repr=False)
    max_response_bytes: int = 2 * 1024 * 1024
    allow_insecure_http: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRef":
        data = _mapping(value, "model binding")
        allowed = {"provider", "model", "model_from_env", "endpoint", "hosting", "credential_ref", "locality_assurance", "context_tokens", "output_tokens", "timeout_seconds", "required_capabilities", "revision", "quantization", "parameters", "max_response_bytes", "allow_insecure_http"}
        if set(data) - allowed:
            raise RuntimeConfigError("unknown model binding fields: " + ", ".join(sorted(set(data) - allowed)))
        provider = data.get("provider")
        if provider not in PROVIDERS:
            raise RuntimeConfigError("provider must be openai, anthropic, ollama, or openai_compatible")
        if "model" in data and "model_from_env" in data:
            raise RuntimeConfigError("choose model or model_from_env, not both")
        model = data.get("model")
        if "model_from_env" in data:
            env_name = data["model_from_env"]
            if not isinstance(env_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
                raise RuntimeConfigError("model_from_env must name an environment variable")
            model = os.getenv(env_name)
            if not model:
                raise RuntimeConfigError(f"model environment variable {env_name} is unset")
        if not isinstance(model, str) or not model.strip() or len(model) > 512 or any(ord(c) < 32 for c in model):
            raise RuntimeConfigError("model must be a nonempty model ID")
        credential = data.get("credential_ref")
        if isinstance(credential, Mapping):
            if set(credential) != {"env"}:
                raise RuntimeConfigError("credential_ref supports only an env reference")
            credential = credential["env"]
        if credential is not None and (not isinstance(credential, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", credential)):
            raise RuntimeConfigError("credential_ref must name an environment variable, never a key value")
        insecure = _boolean(data.get("allow_insecure_http", False), "allow_insecure_http")
        endpoint = validate_endpoint(data.get("endpoint", DEFAULT_ENDPOINTS.get(provider)), allow_insecure_http=insecure)
        for name in ("context_tokens", "output_tokens"):
            if data.get(name) is not None:
                _positive_int(data[name], name)
        timeout = data.get("timeout_seconds", 120.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise RuntimeConfigError("timeout_seconds must be finite and between zero and 3600")
        capabilities = data.get("required_capabilities", ["text_generation"])
        if not isinstance(capabilities, (list, tuple)) or any(not isinstance(c, str) for c in capabilities):
            raise RuntimeConfigError("required_capabilities must be a list of strings")
        hosting = data.get("hosting", "local_server" if provider == "ollama" else "managed_cloud_api")
        if hosting not in {"on_device", "local_server", "private_cluster", "managed_cloud_api", "self_managed_cloud"}:
            raise RuntimeConfigError("unknown hosting location")
        assurance = data.get("locality_assurance", "unknown")
        if assurance not in {"unknown", "operator_trusted", "egress_enforced"}:
            raise RuntimeConfigError("locality_assurance must be unknown, operator_trusted, or egress_enforced")
        parameters = _mapping(data.get("parameters", {}), "parameters")
        if set(parameters) - {"temperature", "max_tokens_parameter", "seed", "api_version"}:
            raise RuntimeConfigError("unsupported provider parameter override")
        if parameters.get("max_tokens_parameter", "max_tokens") not in {"max_tokens", "max_completion_tokens"}:
            raise RuntimeConfigError("max_tokens_parameter must be max_tokens or max_completion_tokens")
        if "temperature" in parameters and parameters["temperature"] is not None:
            t = parameters["temperature"]
            if isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or not 0 <= t <= 2:
                raise RuntimeConfigError("temperature override must be null or between zero and two")
        if "seed" in parameters and type(parameters["seed"]) is not int:
            raise RuntimeConfigError("seed must be an integer")
        if "api_version" in parameters and (provider != "anthropic" or not isinstance(parameters["api_version"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", parameters["api_version"])):
            raise RuntimeConfigError("api_version must be an Anthropic API version date")
        return cls(provider=provider, model=model, endpoint=endpoint, hosting=hosting, credential_ref=credential,
                   locality_assurance=assurance, context_tokens=data.get("context_tokens"), output_tokens=data.get("output_tokens"),
                   timeout_seconds=float(timeout), required_capabilities=tuple(capabilities), revision=data.get("revision"),
                   quantization=data.get("quantization"), parameters=parameters,
                   max_response_bytes=_positive_int(data.get("max_response_bytes", 2 * 1024 * 1024), "max_response_bytes"),
                   allow_insecure_http=insecure)

    def to_dict(self) -> dict:
        # Credential references are identifiers only. Never serialize resolved values.
        return {"provider": self.provider, "model": self.model, "endpoint": self.endpoint, "hosting": self.hosting,
                "credential_ref": self.credential_ref, "locality_assurance": self.locality_assurance,
                "context_tokens": self.context_tokens, "output_tokens": self.output_tokens,
                "timeout_seconds": self.timeout_seconds, "required_capabilities": list(self.required_capabilities),
                "revision": self.revision, "quantization": self.quantization, "parameters": dict(self.parameters),
                "max_response_bytes": self.max_response_bytes, "allow_insecure_http": self.allow_insecure_http}


@dataclass(frozen=True)
class RuntimeConfig:
    models: dict[str, ModelRef]
    roles: dict[str, str]
    policy: dict = field(default_factory=lambda: {"data_egress": "allow", "verified_offline_required": False})
    budgets: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    retry: dict = field(default_factory=dict)
    profile: str = "explicit"
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        data = _mapping(value, "runtime config")
        allowed = {"schema_version", "profile", "models", "roles", "policy", "budgets", "execution", "retry", "routing"}
        if set(data) - allowed:
            raise RuntimeConfigError("unknown runtime fields: " + ", ".join(sorted(set(data) - allowed)))
        if type(data.get("schema_version", 1)) is not int or data.get("schema_version", 1) != 1:
            raise RuntimeConfigError("unsupported runtime schema version")
        raw_models = _mapping(data.get("models"), "models")
        if not raw_models or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", name) for name in raw_models):
            raise RuntimeConfigError("models require nonempty stable binding names")
        models = {name: ModelRef.from_dict(item) for name, item in raw_models.items()}
        roles = _mapping(data.get("roles"), "roles")
        if not roles or set(roles) - ROLES or any(not isinstance(ref, str) or ref not in models for ref in roles.values()):
            raise RuntimeConfigError("roles must bind known roles to declared model names")
        policy = {"data_egress": "allow", "verified_offline_required": False, "telemetry_content": False}
        policy.update(_mapping(data.get("policy", {}), "policy"))
        if set(policy) - {"data_egress", "verified_offline_required", "telemetry_content"}:
            raise RuntimeConfigError("unknown policy field")
        if policy["data_egress"] not in {"allow", "local_only"}:
            raise RuntimeConfigError("data_egress must be allow or local_only")
        _boolean(policy["verified_offline_required"], "verified_offline_required")
        _boolean(policy["telemetry_content"], "telemetry_content")
        if policy["verified_offline_required"] and policy["data_egress"] != "local_only":
            raise RuntimeConfigError("verified_offline_required requires local_only")
        if policy["data_egress"] == "local_only" and policy["telemetry_content"]:
            raise PolicyDenied("local_only disallows content-bearing telemetry")
        budgets = _mapping(data.get("budgets", {}), "budgets")
        if set(budgets) - set(ROLE_POOLS.values()):
            raise RuntimeConfigError("unknown budget pool")
        known_limits = {"requests", "input_tokens", "output_tokens", "wall_seconds", "cpu_seconds", "memory_bytes", "disk_bytes", "agent_turns", "search_trials"}
        for pool, raw_limits in budgets.items():
            limits = _mapping(raw_limits, f"budget {pool}")
            if set(limits) - known_limits:
                raise RuntimeConfigError(f"unknown limits in budget {pool}")
            for key, limit in limits.items():
                _positive_int(limit, f"budgets.{pool}.{key}", zero=True)
            budgets[pool] = limits
        routing = _mapping(data.get("routing", {}), "routing")
        if set(routing) - {"fallback"} or routing.get("fallback", "disabled") != "disabled":
            raise RuntimeConfigError("automatic model fallback is not supported; use a separate explicit condition")
        retry = _mapping(data.get("retry", {}), "retry")
        if set(retry) - {"max_attempts_per_step", "max_total_attempts"}:
            raise RuntimeConfigError("unknown retry field")
        for key, limit in retry.items():
            _positive_int(limit, key)
        return cls(models=models, roles=roles, policy=policy, budgets=budgets,
                   execution=_mapping(data.get("execution", {}), "execution"), retry=retry,
                   profile=str(data.get("profile", "explicit")))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "profile": self.profile, "policy": dict(self.policy),
                "models": {name: ref.to_dict() for name, ref in self.models.items()}, "roles": dict(self.roles),
                "budgets": _public_copy(self.budgets), "execution": _public_copy(self.execution),
                "retry": _public_copy(self.retry), "routing": {"fallback": "disabled"}}


ConfigSource = Union[RuntimeConfig, Mapping[str, Any], str, Path]


def load_runtime_config(source: ConfigSource) -> RuntimeConfig:
    if isinstance(source, RuntimeConfig):
        return source
    if isinstance(source, Mapping):
        return RuntimeConfig.from_dict(source)
    path = Path(source)
    if path.stat().st_size > 1024 * 1024:
        raise RuntimeConfigError("runtime configuration exceeds 1 MiB")
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RuntimeConfigError("invalid runtime YAML") from exc
    return RuntimeConfig.from_dict(data)


@dataclass
class Reservation:
    pool: str
    input_tokens: int
    output_tokens: int
    settled: bool = False
    key: str = field(default_factory=lambda: "model:" + uuid.uuid4().hex)


class BudgetLedger:
    """Model-call reservations with optional restart-durable RunJournal backing.

    Supplying a journal requires an existing budget run whose core limits use
    ``<pool>.<dimension>`` names. Unknown/partial outcomes stay reserved in full.
    A successful provider response with known usage settles once; actual overrun
    is persisted before it raises BudgetExceeded.
    """
    def __init__(self, limits: Mapping[str, Mapping[str, int]], *, journal=None, run_id=None):
        self.limits = {pool: dict(values) for pool, values in limits.items()}
        self._used: dict[str, dict] = {}
        self._lock = threading.Lock()
        if (journal is None) != (run_id is None):
            raise RuntimeConfigError("durable model budgets require both budget_journal and budget_run_id")
        self.journal, self.run_id = journal, run_id
        if journal is not None:
            if not isinstance(run_id, str) or not run_id:
                raise RuntimeConfigError("budget_run_id must name an existing journal run")
            budget = journal.get(run_id)["budget"]
            for pool, values in self.limits.items():
                for dimension in ("requests", "input_tokens", "output_tokens"):
                    if dimension in values and budget["limits"].get(f"{pool}.{dimension}") != values[dimension]:
                        raise RuntimeConfigError("durable model budget limits do not match runtime configuration")

    @staticmethod
    def _flatten(pool, amounts):
        return {f"{pool}.{dimension}": amount for dimension, amount in amounts.items()}

    def _durable_overage(self):
        # An observed overage with incomplete usage cannot be settled honestly.
        # Keep its reservation and prevent later dispatch across restarts until
        # the owner reconciles the unknown outcome in the journal.
        events = self.journal.events(self.run_id)
        settled = {e["payload"]["key"] for e in events if e["kind"] == "budget_settled"}
        return any(e["kind"] == "model_usage_overage_unknown" and e["payload"]["key"] not in settled for e in events)

    def reserve(self, role: str, input_tokens: int, output_tokens: int) -> Reservation:
        if role not in ROLE_POOLS:
            raise RuntimeConfigError("unknown budget role")
        _positive_int(input_tokens, "input token reservation", zero=True)
        _positive_int(output_tokens, "output token reservation", zero=True)
        pool = ROLE_POOLS[role]
        with self._lock:
            # Explicit profiles must bound every model role they dispatch.
            if pool not in self.limits:
                raise BudgetExceeded(f"role {role} has no configured {pool} budget")
            limits = self.limits[pool]
            if not {"requests", "input_tokens", "output_tokens"} <= set(limits):
                raise BudgetExceeded(f"{pool} needs requests, input_tokens, and output_tokens limits")
            amounts = {"requests": 1, "input_tokens": input_tokens, "output_tokens": output_tokens}
            reservation = Reservation(pool, input_tokens, output_tokens)
            if self.journal is not None:
                from .journal import JournalError
                state = self.journal.get(self.run_id)["state"]
                if state not in {"pending", "running"}:
                    raise BudgetExceeded("model budget run is no longer active")
                if self._durable_overage():
                    raise BudgetExceeded("model budget requires reconciliation of partial overage")
                try:
                    self.journal.reserve(self.run_id, reservation.key, self._flatten(pool, amounts))
                except JournalError as exc:
                    raise BudgetExceeded("durable model budget exhausted or unavailable") from exc
                return reservation
            used = self._used.setdefault(pool, {"requests": 0, "input_tokens": 0, "output_tokens": 0, "usage_unknown_requests": 0})
            if any(used[key] + amount > limits[key] for key, amount in amounts.items()):
                raise BudgetExceeded(f"{pool} model budget would be exceeded")
            for key, amount in amounts.items():
                used[key] += amount
            return reservation

    def settle(self, reservation: Reservation, input_tokens: Optional[int], output_tokens: Optional[int]) -> None:
        with self._lock:
            if reservation.settled:
                raise RuntimeConfigError("budget reservation already settled")
            for name, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
                if value is not None:
                    _positive_int(value, name, zero=True)
            if self.journal is not None:
                if input_tokens is None or output_tokens is None:
                    overage = ((input_tokens is not None and input_tokens > reservation.input_tokens)
                               or (output_tokens is not None and output_tokens > reservation.output_tokens))
                    self.journal.event(self.run_id, "model_usage_overage_unknown" if overage else "model_usage_unknown",
                                       {"key": reservation.key, "pool": reservation.pool,
                                        "input_tokens": input_tokens, "output_tokens": output_tokens})
                    # No partial settlement/refund: an interrupted or unreported
                    # request may still complete and be billed externally.
                    reservation.settled = True
                    if overage:
                        raise BudgetExceeded("partial measured model usage exceeds its reservation; reconciliation required")
                    return
                actual = self._flatten(reservation.pool, {"requests": 1, "input_tokens": input_tokens, "output_tokens": output_tokens})
                self.journal.settle(self.run_id, reservation.key, actual)
                reservation.settled = True
                budget = self.journal.get(self.run_id)["budget"]
                if any(budget["used"].get(key, 0) + budget["reserved"].get(key, 0) > limit
                       for key, limit in budget["limits"].items()):
                    raise BudgetExceeded("measured model usage exceeded its durable budget")
                return
            used = self._used[reservation.pool]
            if input_tokens is None or output_tokens is None:
                used["usage_unknown_requests"] += 1
                # Retain both conservative bounds when either dimension is
                # unknown; a known partial overage may increase that bound.
                if input_tokens is not None:
                    used["input_tokens"] += max(0, input_tokens - reservation.input_tokens)
                if output_tokens is not None:
                    used["output_tokens"] += max(0, output_tokens - reservation.output_tokens)
            else:
                used["input_tokens"] += input_tokens - reservation.input_tokens
                used["output_tokens"] += output_tokens - reservation.output_tokens
            reservation.settled = True
            limits = self.limits[reservation.pool]
            if any(used[key] > limits[key] for key in ("requests", "input_tokens", "output_tokens")):
                raise BudgetExceeded(f"measured {reservation.pool} usage exceeded its configured budget")

    def snapshot(self) -> dict:
        with self._lock:
            if self.journal is not None:
                budget = self.journal.get(self.run_id)["budget"]
                usage, used, reserved = {}, {}, {}
                for pool in self.limits:
                    spent = {key: budget["used"].get(f"{pool}.{key}", 0) for key in ("requests", "input_tokens", "output_tokens")}
                    held = {key: budget["reserved"].get(f"{pool}.{key}", 0) for key in ("requests", "input_tokens", "output_tokens")}
                    if any(spent.values()) or any(held.values()):
                        used[pool], reserved[pool] = spent, held
                        usage[pool] = {key: spent[key] + held[key] for key in spent}
                        usage[pool]["usage_unknown_requests"] = held["requests"]
                return {"scope": "durable_run", "run_id": self.run_id, "usage": usage, "used": used,
                        "reserved": reserved, "limits": {pool: dict(values) for pool, values in self.limits.items()},
                        "reconciliation_required": any(v["requests"] for v in reserved.values()) or self._durable_overage()}
            return {"scope": "in_memory_run", "usage": {pool: dict(values) for pool, values in self._used.items()},
                    "limits": {pool: dict(values) for pool, values in self.limits.items()}}


@dataclass
class RunContext:
    config: RuntimeConfig
    api_key: Optional[str] = field(default=None, repr=False)
    offline_verifier: Optional[Callable[[ModelRef], bool]] = field(default=None, repr=False)
    budget_journal: Any = field(default=None, repr=False)
    budget_run_id: Optional[str] = field(default=None, repr=False)
    ledger: BudgetLedger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ledger = BudgetLedger(self.config.budgets, journal=self.budget_journal, run_id=self.budget_run_id)
        if self.api_key:
            self._validate_generic_credential()

    def _validate_generic_credential(self) -> None:
        targets = {(self.config.models[name].provider, self.config.models[name].endpoint)
                   for name in self.config.roles.values()}
        if len(targets) != 1:
            raise RuntimeConfigError("a generic api_key is ambiguous across providers or endpoints; configure credential_ref per binding")

    def resolve(self, role: str) -> ModelRef:
        if role not in ROLES or role not in self.config.roles:
            raise RuntimeConfigError(f"no model binding configured for role {role}")
        ref = self.config.models[self.config.roles[role]]
        if self.config.policy.get("data_egress") == "local_only":
            known_cloud_model = ref.provider == "ollama" and (":cloud" in ref.model.lower() or ref.model.lower().endswith("-cloud"))
            if ref.provider in {"openai", "anthropic"} or ref.hosting not in {"on_device", "local_server"} or not is_loopback_endpoint(ref.endpoint) or known_cloud_model:
                raise PolicyDenied("local_only forbids this provider, model, or nonlocal endpoint")
            if ref.locality_assurance == "unknown":
                raise PolicyDenied("local_only requires an explicit operator-trusted residency assertion or enforced egress boundary")
            if self.config.policy.get("verified_offline_required") and not (self.offline_verifier and self.offline_verifier(ref) is True):
                raise PolicyDenied("verified offline requires a trusted model-server egress enforcement verifier; configuration alone is insufficient")
        return ref

    def credential(self, ref: ModelRef, explicit: Optional[str] = None) -> Optional[str]:
        if ref.credential_ref:
            value = os.getenv(ref.credential_ref)
            if not value:
                raise RuntimeConfigError(f"credential environment variable {ref.credential_ref} is unset")
            return value
        if explicit or self.api_key:
            self._validate_generic_credential()
            return explicit or self.api_key
        if ref.provider in {"openai", "anthropic"}:
            name = "OPENAI_API_KEY" if ref.provider == "openai" else "ANTHROPIC_API_KEY"
            value = os.getenv(name)
            if not value:
                raise RuntimeConfigError(f"{name} is unset")
            return value
        # Never leak a host cloud key to a compatible or local endpoint.
        return None


_CONTEXT: contextvars.ContextVar[Optional[RunContext]] = contextvars.ContextVar("paper2bench_runtime", default=None)
_ROLE: contextvars.ContextVar[str] = contextvars.ContextVar("paper2bench_role", default="author")


def get_context() -> Optional[RunContext]:
    return _CONTEXT.get()


def get_role() -> str:
    return _ROLE.get()


@contextlib.contextmanager
def runtime_scope(source: ConfigSource, *, api_key: Optional[str] = None,
                  offline_verifier: Optional[Callable[[ModelRef], bool]] = None,
                  budget_journal=None, budget_run_id: Optional[str] = None) -> Iterator[RunContext]:
    context = RunContext(load_runtime_config(source), api_key=api_key, offline_verifier=offline_verifier,
                         budget_journal=budget_journal, budget_run_id=budget_run_id)
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


@contextlib.contextmanager
def role_scope(role: str) -> Iterator[None]:
    if role not in ROLES:
        raise RuntimeConfigError("unknown model role")
    token = _ROLE.set(role)
    try:
        yield
    finally:
        _ROLE.reset(token)
