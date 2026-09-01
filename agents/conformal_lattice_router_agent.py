"""Train-only conformal selector over the unchanged PWM top-three subset lattice."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from agents.conformal_router_features import (
    baseline_ranked_items,
    choose_mask,
    feature_names,
    feature_vector,
)
from agents.process_workflow_memory_agent import ProcessWorkflowMemoryAgent as _Parent


class ConformalLatticeRouterAgent(_Parent):
    """Select a learned memory subset, abstaining to original PWM on uncertainty."""

    router_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "conformal_lattice_router"
        / "router.json"
    )

    def __init__(
        self,
        client,
        system_prompt,
        tools,
        tool_handlers,
        runtime_context=None,
        **kwargs,
    ):
        super().__init__(
            client,
            system_prompt,
            tools,
            tool_handlers,
            runtime_context,
            **kwargs,
        )
        self._domain = str(getattr(runtime_context, "domain", ""))
        configured = os.environ.get("STATE_BENCH_CONFORMAL_ROUTER_PATH", "").strip()
        path = Path(configured) if configured else self.router_path
        self._artifact: dict[str, Any] = {}
        self._artifact_error: str | None = None
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if artifact.get("schema_version") != "conformal_lattice_router_v1":
                raise ValueError("unsupported_router_schema")
            models = artifact.get("models")
            if not isinstance(models, dict) or set(models) != {str(mask) for mask in range(8)}:
                raise ValueError("incomplete_router_models")
            expected_metrics = {"completion", "state", "task", "ux", "utility"}
            width = len(feature_names())
            for metrics in models.values():
                if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
                    raise ValueError("invalid_router_metrics")
                for weights in metrics.values():
                    if (
                        not isinstance(weights, list)
                        or len(weights) != width
                        or any(not isinstance(value, (int, float)) for value in weights)
                    ):
                        raise ValueError("invalid_router_weights")
            self._artifact = artifact
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._artifact_error = type(exc).__name__
        self._telemetry_path = os.environ.get(
            "STATE_BENCH_CONFORMAL_ROUTER_TELEMETRY_PATH", ""
        ).strip()
        # Lattice interventions are fixed for an entire trajectory.  Lock the
        # first decision so deployment matches the train-time intervention.
        self._active_mask: int | None = None

    def _write_telemetry(
        self,
        card_ids: list[str],
        mask: int,
        reason: str,
        predictions: dict[int, dict[str, float]],
    ) -> None:
        if not self._telemetry_path:
            return
        record = {
            "run_uuid": uuid.uuid4().hex,
            "domain": self._domain,
            "card_ids": card_ids,
            "selected_mask": mask,
            "selection_reason": reason,
            "predictions": {
                str(candidate): {
                    metric: round(value, 8) for metric, value in metrics.items()
                }
                for candidate, metrics in predictions.items()
            },
        }
        try:
            path = Path(self._telemetry_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            descriptor = os.open(
                path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
            )
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        if self._domain != "shopping_assistant" or self._artifact_error:
            return super().retrieve_learnings(query, top_k)
        ranked = baseline_ranked_items(self, query, top_k)
        if not ranked:
            return []
        if self._active_mask is None:
            try:
                features = feature_vector(query, ranked)
                mask, predictions, reason = choose_mask(
                    self._artifact, features, require_deployment=True
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                mask, predictions, reason = 7, {}, "runtime_validation_fallback"
            self._active_mask = mask
            card_ids = [str(item.get("id", "")) for _score, item in ranked]
            self._write_telemetry(card_ids, mask, reason, predictions)
        else:
            mask = self._active_mask
        if mask == 7:
            return super().retrieve_learnings(query, top_k)
        if mask == 0:
            return []
        text_key = {
            "hybrid": "text",
            "awm_only": "awm_text",
            "process_only": "process_text",
        }[self.mode]
        return [
            str(item.get(text_key, item.get("text", "")))[:2200]
            for index, (_score, item) in enumerate(ranked)
            if mask & (1 << index)
        ]
