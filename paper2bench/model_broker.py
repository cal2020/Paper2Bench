"""Scoped target-model calls over files, without worker keys or networking.

A candidate writes model_requests.json. The trusted controller consumes only a
typed generate operation against operator-selected target bindings, then writes
model_responses.json for the next isolated workspace snapshot. Durable receipts
prevent a repeated request ID from silently generating another billed response.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import time
from contextlib import contextmanager
from pathlib import Path

from .bundles import atomic_json, canonical_json, read_json
from .llm import chat
from .providers import ProviderError
from .runtime_config import (RunContext, RuntimeConfigError, get_context,
                             load_runtime_config, runtime_scope)

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_RECEIPT = re.compile(r"[a-f0-9]{64}\.json")
_CORE_LIMITS = {"requests", "input_tokens", "output_tokens"}
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024


class ModelBrokerError(ValueError):
    def __init__(self, code):
        super().__init__("Target-model broker rejected operation: " + code)
        self.code = code


def _hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


class ModelBroker:
    def __init__(self, config=None, *, target_bindings, receipt_dir,
                 max_requests=8, max_batch_requests=8, max_input_tokens=8192,
                 max_output_tokens=512, max_batch_bytes=262144,
                 max_response_bytes=262144, max_tokens_per_request=None,
                 max_total_output_tokens=None):
        supplied_context = config if isinstance(config, RunContext) else None
        active = supplied_context or get_context()
        source = supplied_context.config if supplied_context else config or (active.config if active else None)
        if source is None:
            raise ModelBrokerError("runtime_configuration_required")
        # Detach mutable dictionaries from caller-held dataclasses.
        self.config = load_runtime_config(load_runtime_config(source).to_dict())
        self._context = supplied_context
        self.config_sha256 = _hash(self.config.to_dict())
        if (not isinstance(target_bindings, dict) or not 1 <= len(target_bindings) <= 32
                or any(not isinstance(public, str) or not _ID.fullmatch(public)
                       or not isinstance(binding, str) or binding not in self.config.models
                       for public, binding in target_bindings.items())):
            raise ModelBrokerError("invalid_target_bindings")
        if not _CORE_LIMITS <= set(self.config.budgets.get("experiments", {})):
            raise ModelBrokerError("experiment_model_budget_required")
        self.target_bindings = dict(target_bindings)
        if max_tokens_per_request is not None:
            if max_output_tokens != 512 and max_output_tokens != max_tokens_per_request:
                raise ModelBrokerError("conflicting_output_token_limits")
            max_output_tokens = max_tokens_per_request
        for name, value, maximum in (("max_requests", max_requests, 1024),
                ("max_batch_requests", max_batch_requests, 32),
                ("max_input_tokens", max_input_tokens, 131072),
                ("max_output_tokens", max_output_tokens, 16384),
                ("max_batch_bytes", max_batch_bytes, 2097152),
                ("max_response_bytes", max_response_bytes, 2097152)):
            if not _integer(value, 1, maximum):
                raise ModelBrokerError("invalid_" + name)
            setattr(self, name, value)
        self.max_total_output_tokens = (self.max_requests * self.max_output_tokens
                                        if max_total_output_tokens is None else max_total_output_tokens)
        if not _integer(self.max_total_output_tokens, 1, 2**31 - 1):
            raise ModelBrokerError("invalid_max_total_output_tokens")
        if self.max_response_bytes < self.max_batch_requests * 1024 + 1024:
            raise ModelBrokerError("response_budget_too_small_for_batch")
        root = Path(receipt_dir)
        if root.is_symlink():
            raise ModelBrokerError("receipt_directory_symlink")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = root.resolve()
        info = self.root.stat()
        if stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid():
            raise ModelBrokerError("receipt_directory_not_private")
        self.receipts = self.root / "receipts"
        self.receipts.mkdir(mode=0o700, exist_ok=True)
        if self.receipts.is_symlink():
            raise ModelBrokerError("receipt_directory_symlink")

    @property
    def names(self):
        return tuple(sorted(self.target_bindings))

    def instructions(self):
        return ("Target models are available only through controller-scoped file requests. "
                "Write model_requests.json as {\"schema_version\":1,\"requests\":[{\"id\":\"unique_trial_id\","
                "\"target\":\"allowed_target\",\"operation\":\"generate\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"public experiment prompt\"}],\"max_tokens\":32,\"temperature\":0}]}. "
                "Allowed target IDs: " + ", ".join(self.names) + ". "
                f"At most {self.max_batch_requests} requests per batch, {self.max_requests} distinct request IDs, "
                f"{self.max_output_tokens} output tokens per request and {self.max_total_output_tokens} total reserved/known output tokens. "
                "After a successful code block, read model_responses.json in the next code block. "
                "Reuse an ID only for the identical request; replies are cached historical artifacts. "
                "A new independent trial needs a new ID. Unknown outcomes are never automatically replayed. "
                "Workers receive no model credentials and remain network-disabled.")

    @contextmanager
    def _locked(self):
        try:
            import fcntl
        except ImportError as error:
            raise ModelBrokerError("posix_locking_required") from error
        fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ModelBrokerError("invalid_receipt_lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ModelBrokerError("broker_busy") from error
            yield
        finally:
            os.close(fd)

    def _active_context(self):
        context = get_context() or self._context
        if context is None or context.budget_journal is None or not context.budget_run_id:
            raise ModelBrokerError("durable_runtime_context_required")
        if _hash(context.config.to_dict()) != self.config_sha256:
            raise ModelBrokerError("runtime_configuration_changed")
        if self._context is not None and (context.budget_run_id != self._context.budget_run_id
                or context.budget_journal.path != self._context.budget_journal.path):
            raise ModelBrokerError("budget_run_changed")
        return context

    def _identity(self, context):
        return {"schema_version": 1, "protocol": "target_model_file_broker/v1",
                "runtime_sha256": self.config_sha256, "budget_run_id": context.budget_run_id,
                "target_bindings": self.target_bindings,
                "limits": {name: getattr(self, name) for name in ("max_requests", "max_batch_requests",
                           "max_input_tokens", "max_output_tokens", "max_batch_bytes", "max_response_bytes",
                           "max_total_output_tokens")}}

    def _validate_batch(self, data):
        if (not isinstance(data, dict) or set(data) != {"schema_version", "requests"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1
                or not isinstance(data["requests"], list) or len(data["requests"]) > self.max_batch_requests):
            raise ModelBrokerError("invalid_request_batch")
        seen = set()
        requests = []
        for original in data["requests"]:
            if (not isinstance(original, dict) or set(original) - {"id", "target", "operation", "messages", "max_tokens", "temperature"}
                    or not {"id", "target", "operation", "messages", "max_tokens"} <= set(original)):
                raise ModelBrokerError("invalid_request_fields")
            identifier, target = original["id"], original["target"]
            if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in seen:
                raise ModelBrokerError("invalid_or_duplicate_request_id")
            seen.add(identifier)
            if not isinstance(target, str) or target not in self.target_bindings:
                raise ModelBrokerError("target_not_allowed")
            if original["operation"] != "generate":
                raise ModelBrokerError("operation_not_allowed")
            messages = original["messages"]
            if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
                raise ModelBrokerError("invalid_messages")
            for message in messages:
                if (not isinstance(message, dict) or set(message) != {"role", "content"}
                        or not isinstance(message["role"], str) or message["role"] not in {"system", "user", "assistant"}
                        or not isinstance(message["content"], str)):
                    raise ModelBrokerError("invalid_message_fields")
            estimate = sum(len(m["content"].encode("utf-8")) + 64 for m in messages)
            if estimate > self.max_input_tokens:
                raise ModelBrokerError("request_input_limit")
            maximum = original["max_tokens"]
            if not _integer(maximum, 1, self.max_output_tokens):
                raise ModelBrokerError("request_output_limit")
            temperature = original.get("temperature", 0)
            try:
                valid_temperature = type(temperature) in {float, int} and math.isfinite(temperature) and 0 <= temperature <= 2
            except OverflowError:
                valid_temperature = False
            if not valid_temperature:
                raise ModelBrokerError("invalid_temperature")
            requests.append(dict(original, temperature=temperature))
        return requests

    @staticmethod
    def _public_error(request, code, *, replay=False, unknown=False):
        return {"id": request["id"], "target": request["target"],
                "status": "reconciliation_required" if unknown else "blocked",
                "error_code": code, "cache_hit": replay, "request_dispatched": None if unknown else False}

    def _request(self, request, context, identity_sha256, deadline):
        path = self.receipts / (_hash({"id": request["id"]}) + ".json")
        request_hash = _hash(request)
        if path.exists():
            receipt = read_json(path, limit=_MAX_RECEIPT_BYTES)
            if (receipt.get("request_sha256") != request_hash or receipt.get("broker_sha256") != identity_sha256):
                return self._public_error(request, "request_id_conflict", replay=True)
            if receipt.get("state") == "completed":
                result = receipt.get("response")
                if not isinstance(result, dict) or _hash(result) != receipt.get("response_sha256"):
                    raise ModelBrokerError("response_receipt_integrity")
                return dict(result, cache_hit=True, cache_semantics="historical_reply_reuse")
            return self._public_error(request, "request_outcome_unknown", replay=True, unknown=True)
        if time.monotonic() >= deadline:
            return self._public_error(request, "broker_deadline")
        existing = list(self.receipts.iterdir())
        if len(existing) >= self.max_requests:
            return self._public_error(request, "broker_request_limit")
        if any(p.is_symlink() or not p.is_file() or not _RECEIPT.fullmatch(p.name) for p in existing):
            raise ModelBrokerError("invalid_receipt_directory")
        consumed = 0
        for previous in existing:
            record = read_json(previous, limit=_MAX_RECEIPT_BYTES)
            if record.get("broker_sha256") != identity_sha256 or record.get("state") not in {"dispatching", "unknown", "completed"}:
                raise ModelBrokerError("invalid_output_reservation_receipt")
            reserved = record.get("reserved_output_tokens")
            if not _integer(reserved, 1, self.max_output_tokens):
                raise ModelBrokerError("invalid_output_reservation_receipt")
            charge = reserved
            if record.get("state") == "completed":
                response_data = record.get("response")
                if not isinstance(response_data, dict) or _hash(response_data) != record.get("response_sha256"):
                    raise ModelBrokerError("response_receipt_integrity")
                measured = response_data.get("usage", {}).get("output_tokens")
                if type(measured) is int and measured >= 0:
                    charge = measured
            consumed += charge
        if consumed + request["max_tokens"] > self.max_total_output_tokens:
            return self._public_error(request, "broker_output_token_limit")
        binding_name = self.target_bindings[request["target"]]
        ref = self.config.models[binding_name]
        # Native cloud adapters must use a target-specific reference. A generic
        # authoring key, even for the same provider, is deliberately not carried.
        if ref.provider in {"openai", "anthropic"} and ref.credential_ref is None:
            return self._public_error(request, "target_credential_reference_required")
        role_config = self.config.to_dict()
        role_config["roles"] = {"agent": binding_name}
        role_config["models"][binding_name]["max_response_bytes"] = min(
            ref.max_response_bytes, self.max_response_bytes * 6 + 8192)
        with runtime_scope(role_config, budget_journal=context.budget_journal,
                           budget_run_id=context.budget_run_id) as target_context:
            try:
                selected = target_context.resolve("agent")
                secret = target_context.credential(selected)
            except RuntimeConfigError as error:
                return self._public_error(request, getattr(error, "code", "target_preflight_blocked"))
            intent = {"schema_version": 1, "id": request["id"], "target": request["target"],
                      "broker_sha256": identity_sha256, "request_sha256": request_hash,
                      "state": "dispatching", "reserved_output_tokens": request["max_tokens"], "created_at": time.time()}
            atomic_json(path, intent)  # Durable intent precedes every model call.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = self._public_error(request, "broker_deadline")
                intent.update(state="completed", response=result, response_sha256=_hash(result), finished_at=time.time())
                atomic_json(path, intent)
                return result
            try:
                response = chat(selected.model, request["messages"], role="agent",
                                max_tokens=request["max_tokens"], temperature=request["temperature"],
                                timeout_seconds=remaining)
            except Exception as error:
                code = getattr(error, "code", "target_call_error")
                # Even an error with a nominally known HTTP status may have billed
                # work. Keep its intent unresolved and never replay automatically.
                intent.update(state="unknown", error_code=code, finished_at=time.time())
                atomic_json(path, intent)
                return self._public_error(request, code, unknown=True)
            content = response.content
            redacted = bool(secret and secret in content)
            if redacted:
                content = content.replace(secret, "[redacted]")
            if len(content.encode("utf-8")) > self.max_response_bytes:
                result = self._public_error(request, "target_response_limit")
                result["request_dispatched"] = True
            else:
                result = {"id": request["id"], "target": request["target"], "status": "completed",
                          "content": content, "usage": response.usage, "finish_reason": response.finish_reason,
                          "cache_hit": False, "request_dispatched": True, "credential_redaction": redacted,
                          "scope": "target_model_text_only", "target_identity_sha256": _hash(ref.to_dict()),
                          "generation_parameters": {"requested_temperature": request["temperature"],
                              "effective_temperature": ref.parameters.get("temperature", request["temperature"]),
                              "configured_seed": ref.parameters.get("seed"), "max_tokens": request["max_tokens"]}}
            intent.update(state="completed", response=result, response_sha256=_hash(result), finished_at=time.time(),
                          model_response_sha256=hashlib.sha256(response.content.encode()).hexdigest())
            atomic_json(path, intent)  # Private known response is durable first.
            return result

    def process(self, workspace, *, timeout_seconds=None):
        """Consume a bounded candidate request batch; return only public statuses."""
        timeout_seconds = 300 if timeout_seconds is None else timeout_seconds
        try:
            valid_timeout = (type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
                             and 0 < timeout_seconds <= 86400)
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ModelBrokerError("invalid_batch_deadline")
        deadline = time.monotonic() + timeout_seconds
        root = Path(workspace)
        if root.is_symlink() or not root.is_dir():
            raise ModelBrokerError("invalid_workspace")
        root = root.resolve()
        if self.root.is_relative_to(root) or root.is_relative_to(self.root):
            raise ModelBrokerError("receipt_workspace_overlap")
        source = root / "model_requests.json"
        if not source.exists() and not source.is_symlink():
            return {"status": "no_requests", "requests": 0, "responses": 0, "model_calls": 0}
        context = self._active_context()
        identity = self._identity(context)
        identity_sha = _hash(identity)
        with self._locked():
            state_path = self.root / "broker.json"
            if state_path.exists():
                stored = read_json(state_path)
                if stored != identity:
                    raise ModelBrokerError("broker_identity_changed")
            else:
                atomic_json(state_path, identity)
            try:
                if source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1:
                    raise ModelBrokerError("invalid_request_file")
                requests = self._validate_batch(read_json(source, limit=self.max_batch_bytes))
            except ValueError as error:
                code = getattr(error, "code", "invalid_request_json")
                public = {"schema_version": 1, "status": "blocked", "responses": [], "error_code": code}
                atomic_json(root / "model_responses.json", public)
                return {"status": "blocked", "requests": 0, "responses": 0, "model_calls": 0, "error_code": code}
            results, retained = [], 256
            for index, request in enumerate(requests):
                result = self._request(request, context, identity_sha, deadline)
                remaining_error_space = (len(requests) - index - 1) * 1024
                if retained + len(canonical_json(result)) + remaining_error_space > self.max_response_bytes:
                    # Never truncate subject text into an apparently valid answer.
                    # The known reply remains privately cached for explicit review.
                    replacement = self._public_error(request, "batch_response_limit", replay=result["cache_hit"])
                    replacement["request_dispatched"] = result["request_dispatched"]
                    result = replacement
                retained += len(canonical_json(result))
                results.append(result)
            public = {"schema_version": 1, "responses": results,
                      "scope": "operator_scoped_target_models_no_worker_network_or_keys"}
            # A public file write failure cannot cause duplicate model calls:
            # every known response already has its private receipt.
            atomic_json(root / "model_responses.json", public)
            return {"status": "complete" if all(r["status"] == "completed" for r in results) else "partial",
                    "requests": len(requests), "responses": len(results),
                    "model_calls": sum(r["request_dispatched"] is True and not r["cache_hit"] for r in results),
                    "cached_replies": sum(r["cache_hit"] for r in results),
                    "unknown_outcomes": sum(r["status"] == "reconciliation_required" for r in results),
                    "public_response_sha256": _hash(public)}
