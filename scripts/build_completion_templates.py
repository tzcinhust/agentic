"""Induce an independent completion-template artifact from fixed train trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from agents.completion_templates import tokens


PROMPT_VERSION = "completion_induction_v6_counterfactual_20260829"
PHASES = {"pre_claim", "pre_action", "final"}
KINDS = {"achievement", "invariant"}
TYPES = {
    "comparison",
    "explanation_rationale",
    "cost_amount_reporting",
    "proactive_disclosure",
    "user_confirmation_choice",
    "boundary_must_not",
    "final_state_reporting",
    "evidence_grounding",
    "execution",
}
ID_LITERAL = re.compile(r"\b[A-Z]{1,8}[-_][A-Z0-9]{2,}\b")
MONEY_LITERAL = re.compile(r"(?:[$£€]\s*\d|\b\d+(?:\.\d+)?\s*(?:USD|dollars?|euros?|pounds?)\b)", re.I)
DATE_LITERAL = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
    re.I,
)
MUTATION_TOOL = re.compile(
    r"^(add|apply|book|cancel|create|exchange|process|redeem|refund|remove|replace|return|set|submit|update)_",
    re.I,
)
SIGNAL_TERMS = {
    "profile_benefit": {"first-time", "new customer", "tier", "loyalty", "balance", "benefit"},
    "boundary_transition": {"boundary", "cutoff", "deadline", "threshold", "window", "next tier", "crossing", "waiting"},
    "multi_entity_relation": {"multiple", "each", "combined", "aggregate", "bundle", "group", "all requested"},
    "protected_state": {"unchanged", "protected", "excluded", "preserve", "leave active", "only requested"},
}


def compact(value: Any, limit: int = 1000) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def stable_id(*parts: str, prefix: str) -> str:
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def parse_json_object(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model returned no JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("templates"), list):
        raise ValueError("model response is missing templates")
    return payload


@dataclass(frozen=True)
class TrainTrace:
    domain: str
    task_id: str
    path: Path
    conversation: list[dict[str, Any]]
    source_sha256: str

    @property
    def observed_tools(self) -> list[str]:
        return sorted(
            {
                str(call.get("name", ""))
                for message in self.conversation
                if message.get("role") == "assistant"
                for call in message.get("tool_calls") or []
                if call.get("name")
            }
        )

    @property
    def opening_request(self) -> str:
        return next(
            (
                compact(message.get("content", ""), 1000)
                for message in self.conversation
                if message.get("role") == "user"
                and "[TASK_DONE]" not in str(message.get("content", ""))
            ),
            "",
        )

    @property
    def latent_signals(self) -> list[str]:
        keys: set[str] = set()
        values: list[str] = []
        mutation_keys: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    keys.add(str(key).casefold())
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, str):
                values.append(value.casefold())

        user_text = " ".join(
            str(message.get("content", ""))
            for message in self.conversation
            if message.get("role") == "user" and "[TASK_DONE]" not in str(message.get("content", ""))
        ).casefold()
        for message in self.conversation:
            for call in message.get("tool_calls") or []:
                if MUTATION_TOOL.match(str(call.get("name", ""))):
                    arguments = {
                        key: value
                        for key, value in (call.get("arguments") or {}).items()
                        if key not in {"confirm", "dry_run", "preview"}
                    }
                    mutation_keys.add(
                        json.dumps(
                            [call.get("name", ""), arguments],
                            ensure_ascii=True,
                            sort_keys=True,
                            default=str,
                        )
                    )
                visit(call.get("result"))

        signals = []
        if any(
            re.search(r"(?:^|_)(?:is_first_time|tier|loyalty|points|balance|credit|membership)(?:$|_)", key)
            for key in keys
        ):
            signals.append("profile_benefit")
        boundary_text = " ".join(values)
        if any(re.search(r"(?:deadline|threshold|cutoff|window|expir)", key) for key in keys) or re.search(
            r"\b(?:within|more than|less than|before|after|at least|fewer than)\b[^.]{0,60}\b\d+\s*(?:hours?|days?|nights?)\b|"
            r"\b\d+\+?\s*(?:hours?|days?|nights?)\b[^.]{0,30}\b(?:before|after|window)\b",
            boundary_text,
        ):
            signals.append("boundary_transition")
        if len(mutation_keys) >= 2 or re.search(r"\b(?:both|multiple|all|two|three|several)\b", user_text):
            signals.append("multi_entity_relation")
        if re.search(r"\b(?:keep|leave|remain)\b[^.]{0,50}\b(?:active|unchanged|alone)\b|\bonly\b", user_text):
            signals.append("protected_state")
        return signals[:3]

    def render(self) -> str:
        lines = []
        for index, message in enumerate(self.conversation):
            role = message.get("role")
            content = str(message.get("content", ""))
            if role in {"user", "assistant"} and content:
                lines.append(f"M{index} {role}: {compact(content, 900)}")
            for call_index, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                lines.append(
                    f"T{index}.{call_index} {call.get('name', '')} "
                    f"args={compact(call.get('arguments', {}), 320)} "
                    f"result={compact(call.get('result'), 800)}"
                )
        rendered = "\n".join(lines)
        if len(rendered) <= 26000:
            return rendered
        return rendered[:10000] + "\n...[middle evidence compacted]...\n" + rendered[-15960:]


def load_traces(root: Path, limit: int | None = None) -> list[TrainTrace]:
    traces = []
    for path in sorted(root.glob("*/*.json")):
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        conversation = payload.get("conversation")
        if not isinstance(conversation, list):
            continue
        traces.append(
            TrainTrace(
                domain=path.parent.name,
                task_id=path.stem,
                path=path,
                conversation=conversation,
                source_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        if limit and len(traces) >= limit:
            break
    if not traces:
        raise ValueError(f"no train trajectories found under {root}")
    return traces


def induction_prompt(trace: TrainTrace) -> str:
    schema = {
        "templates": [
            {
                "family": "stable_snake_case_completion_family",
                "title": "short reusable completion condition family",
                "trigger": {
                    "intent": "generalized user intent",
                    "observable_when": ["conditions inferable from current user messages or tool results"],
                },
                "keywords": ["retrieval terms and paraphrases"],
                "confidence": 0.0,
                "obligations": [
                    {
                        "id": "short_snake_case_name",
                        "phase": "pre_claim|pre_action|final",
                        "kind": "achievement|invariant",
                        "type": "one allowed type",
                        "requirement": "generalized condition, never a procedure or copied answer",
                        "activation": "observable condition that activates it",
                        "required_evidence": ["evidence needed before discharge"],
                        "discharge": "observable evidence/utterance that satisfies or invalidates it",
                        "priority": 0,
                    }
                ],
            }
        ]
    }
    return f"""Induce reusable Task-Closure Memory from one fixed training trajectory.

Task-Closure Memory answers WHAT still has to be true before the task is done. It does not answer HOW to call
tools. Do not output a workflow, tool sequence, next action, or task-specific answer. The trajectory can contain
mistakes; infer only completion conditions supported by the interaction. Prefer conditions that explain why a
state-correct agent can still fail the user's task: missing comparison, rationale, exact grounded amount,
proactive benefit/warning, confirmation, protected boundary, or final-state report.

Phases:
- pre_claim: evidence/invariant required before stating a concrete policy, amount, eligibility, or outcome.
- pre_action: consent/boundary required before an irreversible action. Describe the prerequisite, not a tool.
- final: communication or reporting required if the agent is otherwise ready to finish.

Kinds:
- achievement: something must become true.
- invariant: something must never be falsely claimed or changed; any historical violation remains violated.

Allowed types: {sorted(TYPES)}

Generalization rules:
- Never copy entity IDs, customer names, product names, exact dates, or exact monetary answers.
- Express amounts relationally, e.g. "report the exact next-tier fee from current authoritative evidence".
- Do not create a generic obligation merely because a tool result contains a money field.
- Proactive conditions may be learned even when the user's wording has no lexical overlap.
- Run two induction passes internally: (A) conditions demonstrated by the response, and (B) latent closure
  gaps signaled by observable but unused state. A latent signal can be a boolean/profile flag, tier or balance,
  threshold/deadline, protected entity, relational group, incompatibility, or side effect. If such a signal can
  materially change correctness, induce a generalized condition to resolve and disclose it even when this
  particular assistant omitted it. Keep unknown values as required evidence; never invent the answer.
- Structural examples are domain-general: a threshold may require the current tier plus the consequence of
  crossing the next tier; an account/profile flag may require resolving an applicable benefit before reporting
  a final total; several related items may activate an aggregate or bundle condition; a protected sibling entity
  may require an unchanged-state report.
- If the observable trace contains one of those latent signals, include at least one corresponding latent-gap
  template instead of spending every template on generic execution/final-state reporting. In particular, a
  textual threshold/deadline reason should induce a conditional boundary template that explains the current side
  and, when delaying could materially change the outcome, requires the exact next consequence from current
  authoritative evidence. A dormant account/profile benefit signal should induce eligibility resolution plus
  proactive disclosure. A multi-item relation should induce any applicable aggregate/bundle condition.
- Keep only obligations likely to matter for judging task completion; 1-4 obligations per template.
- Return 1-3 non-overlapping templates. JSON only, matching this schema:
{json.dumps(schema, ensure_ascii=False)}

Domain: {trace.domain}
Observed tools (retrieval metadata only): {json.dumps(trace.observed_tools)}
Detected latent structural signals that require explicit coverage when present: {json.dumps(trace.latent_signals)}
Fixed train trajectory:
{trace.render()}
"""


def missing_signal_coverage(
    templates: list[dict[str, Any]], signals: list[str]
) -> list[str]:
    semantic_views = [
        {
            "title": item.get("title"),
            "trigger": item.get("trigger"),
            "keywords": item.get("keywords"),
            "obligations": item.get("obligations"),
        }
        for item in templates
    ]
    text = json.dumps(semantic_views, ensure_ascii=False).casefold()
    missing = []
    for signal in signals:
        terms = SIGNAL_TERMS[signal]
        if not any(term in text for term in terms):
            missing.append(signal)
            continue
        if signal == "boundary_transition" and not any(
            term in text for term in {"next tier", "crossing", "waiting", "if delayed", "future consequence"}
        ):
            missing.append(signal)
    return missing


def annotate_signal_coverage(
    templates: list[dict[str, Any]], trace: TrainTrace
) -> list[dict[str, Any]]:
    for item in templates:
        item["latent_signal_categories"] = [
            signal
            for signal in trace.latent_signals
            if not missing_signal_coverage([item], [signal])
        ]
    return templates


def latent_repair_prompt(trace: TrainTrace, missing: list[str]) -> str:
    return (
        induction_prompt(trace)
        + "\n\nA prior induction pass failed to cover these detected latent structural signals: "
        + json.dumps(missing)
        + "\nReturn 1-3 templates focused only on those missing signals. Do not return generic execution or "
        "final-state templates. For boundary_transition, encode both the current-side rationale and the exact "
        "evidence-grounded next consequence when waiting/delay can cross the boundary. For profile_benefit, "
        "encode eligibility resolution before a concrete total plus proactive disclosure of any applicable "
        "account-linked benefit. For multi_entity_relation, encode the applicable aggregate/bundle or all-entities "
        "completion condition. For protected_state, encode preservation plus final unchanged-state reporting."
    )


def unsafe_specific_literal(text: str) -> bool:
    return bool(ID_LITERAL.search(text) or MONEY_LITERAL.search(text) or DATE_LITERAL.search(text))


def normalize_obligation(raw: dict[str, Any], family: str, position: int) -> dict[str, Any] | None:
    phase = str(raw.get("phase", ""))
    kind = str(raw.get("kind", ""))
    item_type = str(raw.get("type", ""))
    requirement = compact(raw.get("requirement", ""), 520)
    if phase not in PHASES or kind not in KINDS or item_type not in TYPES or not requirement:
        return None
    fields = [
        requirement,
        compact(raw.get("activation", ""), 320),
        compact(raw.get("discharge", ""), 320),
        *[compact(item, 180) for item in raw.get("required_evidence", []) if item],
    ]
    if any(unsafe_specific_literal(value) for value in fields):
        return None
    raw_id = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id", "")).lower()).strip("_")
    obligation_id = raw_id or stable_id(family, requirement, str(position), prefix="o")
    return {
        "id": obligation_id[:80],
        "phase": phase,
        "kind": kind,
        "type": item_type,
        "requirement": requirement,
        "activation": compact(raw.get("activation", ""), 320),
        "required_evidence": [compact(item, 180) for item in raw.get("required_evidence", []) if item][:6],
        "discharge": compact(raw.get("discharge", ""), 320),
        "priority": min(100, max(0, int(raw.get("priority", 50)))),
    }


def normalize_template(raw: dict[str, Any], trace: TrainTrace, position: int) -> dict[str, Any] | None:
    family = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("family", "")).lower()).strip("_")
    title = compact(raw.get("title", ""), 180)
    trigger = raw.get("trigger") if isinstance(raw.get("trigger"), dict) else {}
    intent = compact(trigger.get("intent", ""), 180)
    observable_when = [compact(item, 240) for item in trigger.get("observable_when", []) if item][:8]
    observable_when = [item for item in observable_when if not unsafe_specific_literal(item)]
    if (
        not family
        or any(character.isdigit() for character in family)
        or not title
        or not intent
        or unsafe_specific_literal(title + " " + intent)
    ):
        return None
    obligations = [
        item
        for index, value in enumerate(raw.get("obligations", []))
        if isinstance(value, dict)
        and (item := normalize_obligation(value, family, index)) is not None
    ]
    if not obligations:
        return None
    keywords = [
        item
        for value in raw.get("keywords", [])
        if value and not unsafe_specific_literal(item := compact(value, 80))
    ][:20]
    return {
        "domain": trace.domain,
        "family": family,
        "title": title,
        "trigger": {"intent": intent, "observable_when": observable_when},
        "keywords": keywords,
        "confidence": min(1.0, max(0.0, float(raw.get("confidence", 0.6)))),
        "obligations": obligations,
        "observed_tools": trace.observed_tools,
        "source_tasks": [trace.task_id],
        "source_sha256": [trace.source_sha256],
        "opening_requests": [trace.opening_request],
        "latent_signal_categories": [],
        "induction_position": position,
    }


def request_templates(client: Any, model: str, trace: TrainTrace, prompt: str) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3500,
    )
    text = response.choices[0].message.content or ""
    payload = parse_json_object(text)
    normalized = [
        item
        for index, raw in enumerate(payload["templates"][:3])
        if isinstance(raw, dict)
        and (item := normalize_template(raw, trace, index)) is not None
    ]
    return annotate_signal_coverage(normalized, trace)


def call_model(client: Any, model: str, trace: TrainTrace, retries: int) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            normalized = request_templates(client, model, trace, induction_prompt(trace))
            if not normalized:
                raise ValueError("all returned templates failed validation")
            missing = missing_signal_coverage(normalized, trace.latent_signals)
            if missing:
                repaired = request_templates(client, model, trace, latent_repair_prompt(trace, missing))
                normalized.extend(repaired)
                still_missing = missing_signal_coverage(normalized, missing)
                if still_missing:
                    raise ValueError(f"latent signal coverage missing after repair: {still_missing}")
            return normalized
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"induction failed for {trace.domain}/{trace.task_id}: {last_error}")


def induce_one(client: Any, model: str, trace: TrainTrace, cache_dir: Path, retries: int) -> list[dict[str, Any]]:
    cache_key = stable_id(PROMPT_VERSION, model, trace.source_sha256, prefix="cache")
    cache_path = cache_dir / trace.domain / f"{trace.task_id}.{cache_key}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("prompt_version") == PROMPT_VERSION and isinstance(payload.get("templates"), list):
            normalized = [
                item
                for index, raw in enumerate(payload["templates"])
                if isinstance(raw, dict)
                and (item := normalize_template(raw, trace, index)) is not None
            ]
            if normalized:
                return annotate_signal_coverage(normalized, trace)
    templates = call_model(client, model, trace, retries)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "source_sha256": trace.source_sha256,
                "templates": templates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return templates


def obligation_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if (left["phase"], left["kind"], left["type"]) != (right["phase"], right["kind"], right["type"]):
        return 0.0
    left_text = left["requirement"].casefold()
    right_text = right["requirement"].casefold()
    left_tokens, right_tokens = set(tokens(left_text)), set(tokens(right_text))
    jaccard = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    return max(jaccard, SequenceMatcher(None, left_text, right_text).ratio())


def merge_templates(raw_templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for template in raw_templates:
        groups.setdefault((template["domain"], template["family"]), []).append(template)

    output = []
    for (domain, family), group in sorted(groups.items()):
        representative = max(group, key=lambda item: item["confidence"])
        obligations: list[dict[str, Any]] = []
        for template in sorted(group, key=lambda item: item["confidence"], reverse=True):
            for obligation in template["obligations"]:
                match = next(
                    (item for item in obligations if obligation_similarity(item, obligation) >= 0.45),
                    None,
                )
                if match is None:
                    obligations.append(dict(obligation))
                elif obligation["priority"] < match["priority"]:
                    match.update(obligation)
        obligations = sorted(obligations, key=lambda item: (item["priority"], item["phase"], item["id"]))[:10]
        for index, obligation in enumerate(obligations):
            obligation["id"] = stable_id(family, obligation["phase"], obligation["type"], obligation["requirement"], str(index), prefix="obl")

        source_tasks = sorted({task for item in group for task in item["source_tasks"]})
        observed_tools = sorted({tool for item in group for tool in item["observed_tools"]})
        keywords = list(dict.fromkeys(keyword for item in group for keyword in item["keywords"]))[:40]
        opening_requests = list(dict.fromkeys(request for item in group for request in item["opening_requests"]))[:12]
        latent_signals = sorted(
            {signal for item in group for signal in item.get("latent_signal_categories", [])}
        )
        search_text = compact(
            " ".join(
                [
                    family.replace("_", " "),
                    representative["title"],
                    representative["trigger"]["intent"],
                    *representative["trigger"]["observable_when"],
                    *keywords,
                    *[signal.replace("_", " ") for signal in latent_signals],
                    *observed_tools,
                    *opening_requests,
                ]
            ),
            6000,
        )
        template_id = stable_id(domain, family, representative["title"], prefix="ct")
        output.append(
            {
                "id": template_id,
                "domain": domain,
                "family": family,
                "title": representative["title"],
                "trigger": representative["trigger"],
                "keywords": keywords,
                "observed_tools": observed_tools,
                "support": len(source_tasks),
                "confidence": round(sum(item["confidence"] for item in group) / len(group), 4),
                "source_tasks": source_tasks,
                "latent_signal_categories": latent_signals,
                "source_sha256": sorted({value for item in group for value in item["source_sha256"]}),
                "obligations": obligations,
                "search_text": search_text,
                "tokens": tokens(search_text),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("STATE_BENCH_AGENT_BASE_URL") or os.environ.get("NOVA_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("STATE_BENCH_AGENT_API_KEY") or os.environ.get("NOVA_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("STATE_BENCH_AGENT_MODEL", "gpt-5.4"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise ValueError("base URL and API key are required via arguments or environment")

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key, timeout=180, max_retries=2)
    traces = load_traces(args.input_root, args.limit)
    raw_templates: list[dict[str, Any]] = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(induce_one, client, args.model, trace, args.cache_dir, args.retries): trace
            for trace in traces
        }
        completed = 0
        for future in as_completed(futures):
            trace = futures[future]
            try:
                raw_templates.extend(future.result())
            except Exception as error:
                failures.append(f"{trace.domain}/{trace.task_id}: {error}")
            completed += 1
            if completed % 10 == 0 or completed == len(traces):
                print(f"induced {completed}/{len(traces)} trajectories; failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError("template induction incomplete:\n" + "\n".join(failures))

    templates = merge_templates(raw_templates)
    artifact = {
        "version": 2,
        "kind": "independent_task_completion_templates",
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "source": {
            "trajectory_count": len(traces),
            "domains": sorted({trace.domain for trace in traces}),
            "conversation_only": True,
            "uses_task_summary": False,
            "uses_task_requirements": False,
        },
        "stats": {
            "raw_templates": len(raw_templates),
            "merged_templates": len(templates),
            "obligations": sum(len(item["obligations"]) for item in templates),
        },
        "templates": templates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact["stats"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
