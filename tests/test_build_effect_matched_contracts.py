from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.build_effect_matched_contracts import (
    AtomSchemaError,
    TrainTrace,
    analyze_pair_availability,
    build_artifact,
    build_contrast_set,
    compile_atoms,
    induce_atoms_one,
    induce_one,
    load_traces,
    merge_contracts,
    normalize_payload,
    normalize_atom_payload,
    normalize_selector,
    repair_checkpoint,
    resolved_terminal_label,
    split_is_validation,
    validation_gate_intercepts,
    validate_contracts,
)


def trace_with_same_effect() -> TrainTrace:
    conversation = [
        {
            "role": "user",
            "content": "Please cancel my reservation and explain the fee.",
        },
        {
            "role": "assistant",
            "content": "It is cancelled. The fee is $50.",
            "tool_calls": [
                {
                    "name": "cancel_booking",
                    "arguments": {"booking_id": "BK-TRAIN", "confirm": True},
                    "result": {"status": "cancelled", "fee": 50},
                }
            ],
        },
        {"role": "user", "content": "That does not explain why the fee applies."},
        {
            "role": "assistant",
            "content": "The fee applies because the free window has passed.",
        },
        {"role": "user", "content": "Thanks, that answers it. [TASK_DONE]"},
    ]
    return TrainTrace(
        domain="travel",
        task_id="train-a",
        path=Path("train-a.json"),
        conversation=conversation,
        source_sha256="a" * 64,
    )


def valid_payload():
    return {
        "terminal_assessment": {
            "label": "explicit_acceptance",
            "reason": "the terminal feedback explicitly confirms resolution",
        },
        "candidate_labels": [
            {
                "checkpoint_id": "cp_1",
                "label": "closure_repair",
                "reason": "the user explicitly requests the omitted rationale",
            }
        ],
        "repair_abstentions": [],
        "contracts": [
            {
                "source_checkpoint_id": "cp_1",
                "family": "fee_rationale_before_closure",
                "title": "Grounded fee rationale",
                "intent": "resolve a fee-bearing operation",
                "keywords": ["fee", "reason", "explain"],
                "confidence": 0.9,
                "applicability": {
                    "mode": "all",
                    "unknown_policy": "require_resolution",
                    "unknown_description": "authoritative fee rationale is unresolved",
                    "predicates": [
                        {
                            "source": "tool_result",
                            "tool": "cancel_*",
                            "path": "fee",
                            "operator": "exists",
                        }
                    ],
                },
                "obligations": [
                    {
                        "id": "explain_fee",
                        "deadline": "before_final",
                        "type": "explanation_rationale",
                        "requirement": "Explain why the applicable fee applies using current evidence.",
                        "priority": 10,
                        "evidence_requirements": [
                            {
                                "description": "authoritative fee outcome",
                                "required": True,
                                "any_of": [
                                    {
                                        "source": "tool_result",
                                        "tool": "cancel_*",
                                        "path": "fee",
                                        "operator": "exists",
                                    }
                                ],
                            }
                        ],
                        "response_requirements": [
                            {
                                "kind": "mention_evidence",
                                "description": "mention the authoritative fee outcome",
                                "selectors": [
                                    {
                                        "source": "tool_result",
                                        "tool": "cancel_*",
                                        "path": "fee",
                                        "operator": "exists",
                                    }
                                ],
                                "value_mode": "numeric",
                                "min_mentions": 1,
                            },
                            {
                                "kind": "causal_explanation",
                                "description": "state the causal rationale",
                            },
                        ],
                    }
                ],
            }
        ],
    }


def valid_atom_payload():
    return {
        "terminal_assessment": {
            "label": "explicit_acceptance",
            "reason": "the terminal feedback explicitly confirms resolution",
        },
        "candidate_labels": [
            {
                "checkpoint_id": "cp_1",
                "label": "closure_repair",
                "reason": "the user explicitly requests the omitted rationale",
            }
        ],
        "repair_abstentions": [],
        "closure_atoms": [
            {
                "source_checkpoint_id": "cp_1",
                "title": "Grounded fee rationale",
                "intent": "resolve a fee-bearing operation",
                "keywords": ["fee", "reason", "explain"],
                "confidence": 0.9,
                "deadline": "before_final",
                "type": "explanation_rationale",
                "requirement": "Explain why an applicable fee applies using authoritative evidence.",
                "trigger_candidates": [
                    {
                        "source": "tool_result",
                        "tool": "cancel_*",
                        "path": "fee",
                        "operator": "exists",
                    }
                ],
                "bindings": [
                    {
                        "id": "fee_outcome",
                        "description": "authoritative fee outcome",
                        "required": True,
                        "selectors": [
                            {
                                "source": "tool_result",
                                "tool": "cancel_*",
                                "path": "fee",
                                "operator": "exists",
                            }
                        ],
                    }
                ],
                "discharge": [
                    {
                        "kind": "causal_explanation",
                        "binding_ids": ["fee_outcome"],
                    }
                ],
            }
        ],
    }


def test_contrast_requires_exactly_unchanged_realized_effect() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    assert contrast is not None
    assert [item.id for item in contrast.candidates] == ["cp_1"]
    assert contrast.terminal.id == "cp_3"
    assert contrast.candidates[0].effect_signature == contrast.terminal.effect_signature
    repaired = repair_checkpoint(contrast, contrast.candidates[0])
    assert repaired.id == "cp_3"
    assert repaired.effect_signature == contrast.candidates[0].effect_signature


def test_text_selector_rejects_trajectory_position_paths() -> None:
    trace = trace_with_same_effect()
    selector = {
        "source": "user_text",
        "tool": "*",
        "path": "M3",
        "operator": "contains",
        "value": "confirm",
    }

    assert normalize_selector(selector, trace=trace) is None
    selector["path"] = "content"
    assert normalize_selector(selector, trace=trace)["path"] == "content"


def test_numeric_selector_requires_typed_observed_structural_constant() -> None:
    trace = trace_with_same_effect()
    trace.conversation[1]["tool_calls"][0]["result"]["policy_threshold"] = 3
    selector = {
        "source": "tool_result",
        "tool": "get_policy",
        "path": "item_count",
        "operator": "gte",
        "value": 3,
    }

    assert normalize_selector(selector, trace=trace) is None
    selector["value_kind"] = "structural_constant"
    selector["value_evidence"] = {
        "tool": "cancel_*",
        "path": "policy_threshold",
    }
    normalized = normalize_selector(selector, trace=trace)
    assert normalized["value"] == 3
    assert normalized["value_kind"] == "structural_constant"

    selector["value"] = 7
    assert normalize_selector(selector, trace=trace) is None


def test_structural_constant_is_stamped_only_after_cross_task_support() -> None:
    first = trace_with_same_effect()
    first.conversation[1]["tool_calls"][0]["result"]["policy_threshold"] = 3
    second = replace(
        trace_with_same_effect(),
        task_id="train-b",
        source_sha256="b" * 64,
        path=Path("train-b.json"),
    )
    second.conversation[1]["tool_calls"][0]["arguments"]["booking_id"] = "BK-OTHER"
    second.conversation[1]["tool_calls"][0]["result"]["policy_threshold"] = 3
    payload = valid_payload()
    payload["contracts"][0]["applicability"]["predicates"] = [
        {
            "source": "tool_result",
            "tool": "cancel_*",
            "path": "fee",
            "operator": "gte",
            "value": 3,
            "value_kind": "structural_constant",
            "value_evidence": {"tool": "cancel_*", "path": "policy_threshold"},
        }
    ]

    candidates = [
        *normalize_payload(payload, build_contrast_set(first)),
        *normalize_payload(payload, build_contrast_set(second)),
    ]
    merged = merge_contracts(candidates, min_support=2)
    selector = merged[0]["applicability"]["predicates"][0]
    assert selector["value_support"] == 2
    assert selector["value_provenance_sha256"]

    assert merge_contracts(candidates[:1], min_support=1) == []


def test_analyze_only_reports_pair_availability_without_model_calls() -> None:
    pairable = trace_with_same_effect()
    no_terminal_base = trace_with_same_effect()
    no_terminal = replace(
        no_terminal_base,
        task_id="no-terminal",
        source_sha256="c" * 64,
        conversation=no_terminal_base.conversation[:-1],
    )

    report = analyze_pair_availability([pairable, no_terminal])

    assert report["trajectories"] == 2
    assert report["terminal_trajectories"] == 1
    assert report["pairable_trajectories"] == 1
    assert report["selected_candidate_checkpoints"] == 1
    assert report["local_effect_stable_repair_pairs"] == 1
    assert report["api_calls"] == 0


def test_new_mutation_between_responses_prevents_pairing() -> None:
    sample = trace_with_same_effect()
    sample.conversation[3]["tool_calls"] = [
        {
            "name": "update_booking",
            "arguments": {"booking_id": "BK-TRAIN", "confirm": True},
            "result": {"status": "updated"},
        }
    ]
    assert build_contrast_set(sample) is None


def test_normalization_keeps_only_explicit_closure_repair_contracts() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    contracts = normalize_payload(valid_payload(), contrast)
    assert len(contracts) == 1
    assert contracts[0]["family"] == "fee_rationale_before_closure"
    payload = valid_payload()
    payload["candidate_labels"][0]["label"] = "normal_progress"
    assert normalize_payload(payload, contrast) == []


def test_atomic_extraction_separates_binding_from_qualitative_discharge() -> None:
    atoms = normalize_atom_payload(
        valid_atom_payload(), build_contrast_set(trace_with_same_effect())
    )

    assert len(atoms) == 1
    assert atoms[0]["bindings"][0]["selectors"][0]["source"] == "tool_result"
    assert atoms[0]["discharge"] == [
        {
            "kind": "causal_explanation",
            "binding_ids": ["fee_outcome"],
        }
    ]


def test_atomic_binding_cannot_self_ground_in_assistant_text() -> None:
    payload = valid_atom_payload()
    payload["closure_atoms"][0]["bindings"][0]["selectors"][0] = {
        "source": "assistant_text",
        "tool": "*",
        "path": "content",
        "operator": "contains",
        "value": "fee",
    }

    with pytest.raises(AtomSchemaError, match="assistant_text cannot ground"):
        normalize_atom_payload(payload, build_contrast_set(trace_with_same_effect()))


def test_repair_only_phrase_cannot_become_an_applicability_trigger() -> None:
    payload = valid_atom_payload()
    payload["closure_atoms"][0]["trigger_candidates"] = [
        {
            "source": "user_text",
            "tool": "*",
            "path": "content",
            "operator": "contains",
            "value": "does not explain",
        }
    ]

    with pytest.raises(AtomSchemaError, match="not true before the rejected draft"):
        normalize_atom_payload(payload, build_contrast_set(trace_with_same_effect()))


def test_atom_and_semantic_abstention_must_be_disjoint() -> None:
    payload = valid_atom_payload()
    payload["repair_abstentions"] = [
        {"checkpoint_id": "cp_1", "reason": "cannot generalize"}
    ]

    with pytest.raises(AtomSchemaError, match="atom and abstention overlap"):
        normalize_atom_payload(payload, build_contrast_set(trace_with_same_effect()))


def test_cross_task_atomic_compilation_does_not_require_free_form_family_match() -> None:
    first = trace_with_same_effect()
    second = replace(
        trace_with_same_effect(),
        task_id="train-b",
        source_sha256="b" * 64,
        path=Path("train-b.json"),
    )
    second.conversation[1]["tool_calls"][0]["arguments"]["booking_id"] = "BK-OTHER"
    atoms = [
        *normalize_atom_payload(valid_atom_payload(), build_contrast_set(first)),
        *normalize_atom_payload(valid_atom_payload(), build_contrast_set(second)),
    ]

    contracts = compile_atoms(atoms, min_support=2)

    assert len(contracts) == 1
    assert contracts[0]["support"] == 2
    clauses = contracts[0]["obligations"][0]["response_requirements"]
    assert [item["kind"] for item in clauses] == ["causal_explanation"]
    assert contracts[0]["provenance"]["atomic_induction"] is True

    artifact = build_artifact(
        [first, second],
        atoms,
        model="model",
        validation_percent=0,
        min_support=2,
    )
    assert artifact["version"] == 5
    assert artifact["method"].startswith("effect_stable_atomic_delta")
    assert artifact["stats"]["raw_atoms"] == 2
    assert artifact["stats"]["atomic_clusters"] == 1
    assert artifact["atomic_induction_audit"]["contains_raw_conversations"] is False
    assert artifact["atomic_induction_audit"]["compiled_clusters"] == 1


def test_atomic_compiler_keeps_distinct_binding_slots_separate() -> None:
    first = trace_with_same_effect()
    second = replace(
        trace_with_same_effect(),
        task_id="train-b",
        source_sha256="b" * 64,
        path=Path("train-b.json"),
    )
    atoms = [
        *normalize_atom_payload(valid_atom_payload(), build_contrast_set(first)),
        *normalize_atom_payload(valid_atom_payload(), build_contrast_set(second)),
    ]
    for atom in atoms:
        atom["type"] = "comparison"
        atom["title"] = "Compare operation outcome and fee"
        atom["intent"] = "compare two authoritative operation properties"
        atom["requirement"] = "Compare the authoritative fee with the operation state."
        atom["bindings"].append(
            {
                "id": "operation_state",
                "description": "authoritative operation state",
                "required": True,
                "selectors": [
                    {
                        "source": "tool_result",
                        "tool": "cancel_*",
                        "path": "status",
                        "operator": "exists",
                    }
                ],
            }
        )
        atom["discharge"] = [
            {
                "kind": "comparison",
                "binding_ids": ["fee_outcome", "operation_state"],
            }
        ]

    contract = compile_atoms(atoms, min_support=2)[0]

    evidence = contract["obligations"][0]["evidence_requirements"]
    assert len(evidence) == 2
    assert {item["any_of"][0]["path"] for item in evidence} == {"fee", "status"}


def test_terminal_marker_alone_is_not_treated_as_success_but_can_bound_discharge() -> None:
    sample = trace_with_same_effect()
    sample.conversation[-1]["content"] = "[TASK_DONE]"
    contrast = build_contrast_set(sample)
    payload = valid_payload()
    payload["terminal_assessment"] = {
        "label": "protocol_only",
        "reason": "the marker has no semantic acceptance text",
    }

    assert len(normalize_payload(payload, contrast)) == 1


def test_marker_only_terminal_model_label_is_deterministically_protocol_only() -> None:
    sample = trace_with_same_effect()
    sample.conversation[-1]["content"] = "[TASK_DONE]"
    contrast = build_contrast_set(sample)

    assert resolved_terminal_label(valid_payload(), contrast) == "protocol_only"
    assert len(normalize_payload(valid_payload(), contrast)) == 1


def test_protocol_only_with_semantic_feedback_does_not_suppress_local_pair() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_payload()
    payload["terminal_assessment"]["label"] = "protocol_only"

    assert resolved_terminal_label(payload, contrast) == "ambiguous"
    assert len(normalize_payload(payload, contrast)) == 1


def test_adverse_terminal_is_audit_label_not_local_pair_gate() -> None:
    sample = trace_with_same_effect()
    sample.conversation[-1][
        "content"
    ] = "That matches, even though approval happened out of order. [TASK_DONE]"
    contrast = build_contrast_set(sample)

    payload = valid_payload()
    assert resolved_terminal_label(payload, contrast) == "qualified_or_adverse"
    assert len(normalize_payload(payload, contrast)) == 1


def test_qualified_terminal_does_not_suppress_discharged_local_repair() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_payload()
    payload["terminal_assessment"] = {
        "label": "qualified_or_adverse",
        "reason": "the user notes that approval happened out of order",
    }

    assert len(normalize_payload(payload, contrast)) == 1


def test_contract_must_be_observably_discharged_at_local_repair_boundary() -> None:
    sample = trace_with_same_effect()
    sample.conversation[3]["content"] = "Okay."
    contrast = build_contrast_set(sample)

    with pytest.raises(ValueError, match="machine-checkable"):
        normalize_payload(valid_payload(), contrast)


def test_every_repair_checkpoint_requires_contract_or_explicit_abstention() -> None:
    sample = trace_with_same_effect()
    sample.conversation[-1]["content"] = "Also confirm the final cancellation status."
    sample.conversation.extend(
        [
            {"role": "assistant", "content": "The cancellation is complete."},
            {"role": "user", "content": "Thanks. [TASK_DONE]"},
        ]
    )
    contrast = build_contrast_set(sample)
    payload = valid_payload()
    payload["candidate_labels"].append(
        {
            "checkpoint_id": "cp_3",
            "label": "closure_repair",
            "reason": "the user requests omitted final-state confirmation",
        }
    )

    with pytest.raises(ValueError, match="lack contract or explicit abstention"):
        normalize_payload(payload, contrast)

    payload["repair_abstentions"] = [
        {
            "checkpoint_id": "cp_3",
            "reason": "no safe evidence selector represents this repair",
        }
    ]
    assert len(normalize_payload(payload, contrast)) == 1


def test_unrepresentable_contract_is_cached_as_auditable_abstention(tmp_path) -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_payload()
    payload["contracts"][0]["obligations"][0][
        "requirement"
    ] = "Tell BK-TRAIN that the exact fee is $50."

    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload))
                    )
                ]
            )

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = induce_one(client, "model", contrast, tmp_path, retries=1)

    assert result.contracts == ()
    assert result.abstention_reason == "unrepresentable_machine_checkable_contract"
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    cache = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert cache["status"] == "abstained"


def test_invalid_atomic_schema_is_not_mislabeled_as_semantic_abstention(
    tmp_path,
) -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_atom_payload()
    payload["closure_atoms"][0]["bindings"][0]["selectors"][0] = {
        "source": "assistant_text",
        "tool": "*",
        "path": "content",
        "operator": "contains",
        "value": "fee",
    }

    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
                ]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    result = induce_atoms_one(client, "model", contrast, tmp_path, retries=1)

    assert result.atoms == ()
    assert result.semantic_abstentions == ()
    assert "assistant_text cannot ground" in result.schema_failure
    cache = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert cache["status"] == "invalid_atom_schema"

    cached = induce_atoms_one(
        SimpleNamespace(chat=None), "model", contrast, tmp_path, retries=1
    )
    assert cached == result


def test_terminal_assessment_is_required() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_payload()
    payload.pop("terminal_assessment")

    with pytest.raises(ValueError, match="terminal assessment"):
        normalize_payload(payload, contrast)


def test_normalization_rejects_train_specific_answers() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_payload()
    payload["contracts"][0]["obligations"][0][
        "requirement"
    ] = "Tell BK-TRAIN that the exact fee is $50."
    with pytest.raises(ValueError, match="machine-checkable"):
        normalize_payload(payload, contrast)


def test_normalization_rejects_the_harness_acceptance_marker() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_payload()
    payload["contracts"][0]["obligations"][0][
        "requirement"
    ] = "Wait for [TASK_DONE] before closing the interaction."

    with pytest.raises(ValueError, match="machine-checkable"):
        normalize_payload(payload, contrast)


def test_normalization_rejects_copied_train_entity_names() -> None:
    sample = trace_with_same_effect()
    sample.conversation[1]["tool_calls"][0]["result"][
        "product_name"
    ] = "TechPhone Ultra"
    contrast = build_contrast_set(sample)
    payload = valid_payload()
    payload["contracts"][0]["obligations"][0][
        "requirement"
    ] = "Explain why TechPhone Ultra has the applicable fee."
    with pytest.raises(ValueError, match="machine-checkable"):
        normalize_payload(payload, contrast)


def test_normalization_rejects_assistant_text_as_its_own_grounding() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    payload = valid_payload()
    payload["contracts"][0]["obligations"][0]["response_requirements"][0][
        "selectors"
    ] = [
        {
            "source": "assistant_text",
            "path": "content",
            "operator": "contains",
            "value": "fee",
        }
    ]

    with pytest.raises(ValueError, match="machine-checkable"):
        normalize_payload(payload, contrast)


def test_loader_rejects_oracle_like_fields(tmp_path: Path) -> None:
    domain = tmp_path / "travel"
    domain.mkdir()
    (domain / "one.json").write_text(
        json.dumps(
            {
                "conversation": [{"role": "user", "content": "hello"}],
                "metadata": {"task_summary": "hidden answer"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="oracle-like"):
        load_traces(tmp_path)


def test_merge_requires_recurrence_and_artifact_records_no_oracle_use() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    first = normalize_payload(valid_payload(), contrast)[0]
    second = json.loads(json.dumps(first))
    second["source_task"] = "train-b"
    second["source_sha256"] = "b" * 64
    second["source_pair"]["id"] = "pair_b"
    assert merge_contracts([first], min_support=2) == []
    merged = merge_contracts([first, second], min_support=2)
    assert len(merged) == 1
    assert merged[0]["support"] == 2
    artifact = build_artifact(
        [contrast.trace],
        [first, second],
        model="mock-model",
        validation_percent=0,
        min_support=2,
    )
    assert artifact["source"]["conversation_only"] is True
    assert artifact["source"]["uses_task_summary"] is False
    assert artifact["source"]["uses_task_score"] is False
    assert artifact["source"]["uses_test_data"] is False


def test_same_family_with_incompatible_triggers_is_not_false_support() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    first = normalize_payload(valid_payload(), contrast)[0]
    second = json.loads(json.dumps(first))
    second["source_task"] = "train-b"
    second["source_pair"]["id"] = "pair-b"
    second["applicability"]["predicates"][0]["tool"] = "get_unrelated_profile"
    assert merge_contracts([first, second], min_support=2) == []


def test_opposite_values_on_the_same_trigger_are_not_merged() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    first = normalize_payload(valid_payload(), contrast)[0]
    second = json.loads(json.dumps(first))
    first["applicability"]["predicates"][0].update(
        {"operator": "equals", "value": True}
    )
    second["applicability"]["predicates"][0].update(
        {"operator": "equals", "value": False}
    )
    second["source_task"] = "train-b"
    second["source_pair"]["id"] = "pair-b"

    assert merge_contracts([first, second], min_support=2) == []


def test_each_merged_obligation_must_independently_meet_support() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    first = normalize_payload(valid_payload(), contrast)[0]
    second = json.loads(json.dumps(first))
    second["source_task"] = "train-b"
    second["source_pair"]["id"] = "pair-b"
    second["obligations"][0].update(
        {
            "type": "final_state_reporting",
            "requirement": "Report the final operation state from authoritative evidence.",
        }
    )
    third = json.loads(json.dumps(first))
    third["source_task"] = "train-c"
    third["source_pair"]["id"] = "pair-c"

    merged = merge_contracts([first, second, third], min_support=2)

    assert len(merged) == 1
    assert len(merged[0]["obligations"]) == 1
    assert merged[0]["obligations"][0]["support"] == 2
    assert merged[0]["obligations"][0]["type"] == "explanation_rationale"


def test_heldout_candidates_validate_but_do_not_enter_contract_provenance() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    base = normalize_payload(valid_payload(), contrast)[0]
    train_ids = []
    validation_id = None
    for index in range(1000):
        task_id = f"split-{index}"
        if split_is_validation(task_id, 20) and validation_id is None:
            validation_id = task_id
        elif not split_is_validation(task_id, 20) and len(train_ids) < 2:
            train_ids.append(task_id)
        if validation_id and len(train_ids) == 2:
            break
    candidates = []
    for position, task_id in enumerate([*train_ids, validation_id]):
        item = json.loads(json.dumps(base))
        item["source_task"] = task_id
        item["source_sha256"] = str(position) * 64
        item["source_pair"]["id"] = f"pair_{position}"
        candidates.append(item)
    traces = [
        TrainTrace(
            domain="travel",
            task_id=item["source_task"],
            path=Path(f"{item['source_task']}.json"),
            conversation=contrast.trace.conversation,
            source_sha256=item["source_sha256"],
        )
        for item in candidates
    ]
    artifact = build_artifact(
        traces, candidates, model="mock-model", validation_percent=20, min_support=2,
    )
    assert artifact["validation"]["heldout_candidates"] == 1
    assert "pair_2" not in artifact["contracts"][0]["provenance"]["source_pairs"]
    serialized_contracts = json.dumps(artifact["contracts"])
    assert "validation_conversation" not in serialized_contracts
    assert "validation_draft_text" not in serialized_contracts
    assert "validation_tool_calls" not in serialized_contracts
    assert "validation_nonrepair_boundaries" not in serialized_contracts

    strict = build_artifact(
        traces,
        candidates,
        model="mock-model",
        validation_percent=20,
        min_support=2,
        min_contract_validation_retrievals=2,
        min_contract_validation_negative_retrievals=2,
        min_contract_validation_precision=0.5,
        min_contract_validation_specificity=0.8,
    )
    assert strict["contracts"] == []
    assert len(strict["monitor_contracts"]) == 1
    assert strict["monitor_contracts"][0]["runtime_eligible"] is False
    assert strict["stats"]["monitor_only_contracts"] == 1


def test_heldout_retrieval_uses_opening_request_not_induced_labels() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    base = normalize_payload(valid_payload(), contrast)[0]
    second = json.loads(json.dumps(base))
    second["source_task"] = "train-b"
    second["source_pair"]["id"] = "pair-b"
    contracts = merge_contracts([base, second], min_support=2)
    heldout = json.loads(json.dumps(base))
    heldout["opening_request"] = "paint a watercolor landscape"
    heldout["intent"] = contracts[0]["intent"]
    heldout["keywords"] = contracts[0]["keywords"]
    metrics = validate_contracts(contracts, [heldout])
    assert metrics["retrievals"] == 0
    assert metrics["coverage"] == 0.0


def test_heldout_validation_requires_obligation_semantics_not_only_type() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    base = normalize_payload(valid_payload(), contrast)[0]
    second = json.loads(json.dumps(base))
    second["source_task"] = "train-b"
    second["source_pair"]["id"] = "pair-b"
    contracts = merge_contracts([base, second], min_support=2)
    heldout = json.loads(json.dumps(base))
    heldout["obligations"][0][
        "requirement"
    ] = "Explain how available delivery windows differ for a future shipment."

    metrics = validate_contracts(contracts, [heldout])

    assert metrics["retrievals"] == 1
    assert metrics["semantically_relevant_retrievals"] == 0
    assert metrics["relevant_retrievals"] == 0


def test_heldout_validation_replays_pre_action_boundary() -> None:
    contrast = build_contrast_set(trace_with_same_effect())
    candidate = normalize_payload(valid_payload(), contrast)[0]
    candidate["applicability"] = {
        "mode": "all",
        "unknown_policy": "inactive",
        "unknown_description": "",
        "predicates": [
            {
                "source": "tool_argument",
                "tool": "cancel_booking",
                "path": "confirm",
                "operator": "truthy",
                "quantifier": "any",
            }
        ],
    }
    candidate["obligations"][0]["deadline"] = "before_action"

    assert validation_gate_intercepts(candidate, candidate) is True
