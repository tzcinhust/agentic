"""Build a PatchCore-style memory of compliant local agent transitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.transition_patch_memory import build_transition_artifact, normalize_text


PHASES = {"pre_write", "post_write", "pre_final"}
CACHE_VERSION = 2
WRITE_TOOL_NAMES = {
    "create_booking",
    "update_booking",
    "cancel_booking",
    "book_hotel",
    "cancel_hotel_reservation",
    "book_car_rental",
    "cancel_car_rental",
    "process_return",
    "process_exchange",
    "process_refund",
    "process_warranty_claim",
    "process_shipping_claim",
    "cancel_order",
    "add_to_cart",
    "remove_from_cart",
    "update_cart_item",
    "apply_promo_code",
    "remove_promo_code",
    "redeem_loyalty_points",
    "set_shipping_option",
    "add_to_wishlist",
    "remove_from_wishlist",
}


def _compact(value: Any, limit: int = 800) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _load_case(path: Path, domain: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "task_id": path.stem,
        "domain": domain,
        "conversation": payload.get("conversation", []),
    }


def _format_case(case: dict[str, Any]) -> str:
    lines = [f"TRAIN TASK {case['task_id']}"]
    for item in case["conversation"]:
        role = str(item.get("role", ""))
        content = str(item.get("content", "")).strip()
        if role == "user" and "[TASK_DONE]" not in content:
            lines.append(f"USER: {content[:700]}")
        if role != "assistant":
            continue
        for call in item.get("tool_calls") or []:
            name = str(call.get("name", ""))
            arguments = sorted((call.get("arguments") or {}).keys())
            lines.append(
                f"TOOL: {name}({', '.join(arguments)}) -> {_compact(call.get('result'), 550)}"
            )
        if content:
            lines.append(f"ASSISTANT: {content[:700]}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM did not return a JSON object")
    return json.loads(match.group(0))


def _llm_patches(client: Any, model: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    domain = cases[0]["domain"]
    examples = "\n\n=====\n\n".join(_format_case(case) for case in cases)
    prompt = f"""You are converting public training data into nominal local-transition patches for a
stateful tool-using agent. The patches will be used for nearest-neighbor anomaly detection.

Domain: {domain}

The observed trajectories can be noisy or unsuccessful. Extract only reusable transitions directly
supported by user authorization, live tool-policy output, live state fields, or an observed grounded
response after a tool call. Do not infer a rule merely because the agent happened to do something.
Abstract away all
customer IDs, order IDs, product names, exact prices, dates, and other task-specific values. Never
invent a policy or a tool. A patch must describe one local decision boundary, not the whole workflow.

Phases:
- pre_write: facts, consent, or choices required immediately before a state-changing tool call.
- post_write: refreshes or observed side effects required immediately after a state-changing call.
- pre_final: grounded disclosures, refusals, calculations, or unfinished user requests before answering.

Return JSON with key "patches". Each patch must contain exactly:
- source_task: one supplied train task id
- phase: pre_write, post_write, or pre_final
- trigger: observable user intent and local state conditions
- observed_tools: tool names whose outputs establish the local context
- state_cues: abstract result fields or state changes relevant to the decision
- expected_action: the compliant next tool action or response behavior
- obligations: locally evidenced checks, confirmations, refreshes, or disclosures
- forbidden: locally invalid actions or claims
- keywords: short retrieval terms

Produce 0-5 non-overlapping patches per task. Omit a task when no grounded local transition can be
abstracted. Prefer transitions anchored by explicit tool policy, explicit user authorization, or
material fields in a write result. Do not use task definitions or assume hidden evaluator criteria.

{examples}

Return JSON only."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You extract concise, grounded runtime transition specifications. JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=6000,
    )
    return _extract_json(response.choices[0].message.content or "").get("patches", [])


def _fallback_patches(case: dict[str, Any]) -> list[dict[str, Any]]:
    patches = []
    users = []
    observed_tools = []
    observed_state_cues = []
    for item in case["conversation"]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role == "user" and "[TASK_DONE]" not in content:
            users.append(content)
            continue
        if role != "assistant":
            continue
        calls = item.get("tool_calls") or []
        names = [str(call.get("name", "")) for call in calls]
        write_calls = [call for call in calls if str(call.get("name", "")) in WRITE_TOOL_NAMES]
        context = " ".join(users[-3:])
        if write_calls:
            write_names = [str(call.get("name", "")) for call in write_calls]
            patches.append(
                {
                    "source_task": case["task_id"],
                    "phase": "pre_write",
                    "trigger": context,
                    "observed_tools": observed_tools,
                    "state_cues": observed_state_cues[-30:],
                    "expected_action": f"call {' '.join(write_names)} using grounded live fields",
                    "obligations": [
                        "follow the latest explicit user instruction and use identifiers from live tool results"
                    ],
                    "forbidden": ["do not copy task-specific values from memory"],
                    "keywords": write_names,
                }
            )
            result_cues = sorted(
                {
                    str(key)
                    for call in write_calls
                    if isinstance(call.get("result"), dict)
                    for key in call["result"]
                }
            )
            if content and result_cues:
                patches.append(
                    {
                        "source_task": case["task_id"],
                        "phase": "post_write",
                        "trigger": context,
                        "observed_tools": [*observed_tools, *write_names],
                        "state_cues": result_cues,
                        "expected_action": content,
                        "obligations": [
                            "ground the completion report in material fields returned by the write"
                        ],
                        "forbidden": ["do not claim a write effect that the tool did not return"],
                        "keywords": write_names,
                    }
                )
        elif content:
            result_cues = sorted(
                {
                    str(key)
                    for call in calls
                    if isinstance(call.get("result"), dict)
                    for key in call["result"]
                }
            )
            patches.append(
                {
                    "source_task": case["task_id"],
                    "phase": "pre_final",
                    "trigger": context,
                    "observed_tools": [*observed_tools, *names],
                    "state_cues": [*observed_state_cues[-20:], *result_cues],
                    "expected_action": content,
                    "obligations": ["answer using the observed live tool evidence"],
                    "forbidden": ["do not invent unobserved state or policy facts"],
                    "keywords": names,
                }
            )
        for call in calls:
            observed_tools.append(str(call.get("name", "")))
            if isinstance(call.get("result"), dict):
                observed_state_cues.extend(map(str, call["result"].keys()))
    return patches


def _validate_patch(raw: dict[str, Any], case_ids: set[str], domain: str) -> dict[str, Any] | None:
    source_task = str(raw.get("source_task", ""))
    phase = str(raw.get("phase", ""))
    if source_task not in case_ids or phase not in PHASES:
        return None
    observed_tools = [normalize_text(value) for value in raw.get("observed_tools", []) if str(value)]
    state_cues = [normalize_text(value) for value in raw.get("state_cues", []) if str(value)]
    obligations = [normalize_text(value) for value in raw.get("obligations", []) if str(value)]
    forbidden = [normalize_text(value) for value in raw.get("forbidden", []) if str(value)]
    keywords = [normalize_text(value) for value in raw.get("keywords", []) if str(value)]
    trigger = normalize_text(raw.get("trigger", ""))
    expected_action = normalize_text(raw.get("expected_action", ""))
    if not trigger or not expected_action or not obligations:
        return None
    context_text = normalize_text(" ".join(
        [
            f"phase {phase}",
            f"trigger {trigger}",
            f"observed tools {' '.join(observed_tools)}",
            f"state cues {' '.join(state_cues)}",
            f"keywords {' '.join(keywords)}",
        ]
    ))
    transition_text = normalize_text(" ".join(
        [
            f"expected {expected_action}",
            f"obligations {' '.join(obligations)}",
        ]
    ))
    digest = hashlib.sha1(normalize_text(transition_text).encode("utf-8")).hexdigest()[:10]
    return {
        "id": f"{domain}:{source_task}:{phase}:{digest}",
        "domain": domain,
        "phase": phase,
        "source_task": source_task,
        "trigger": trigger,
        "observed_tools": observed_tools,
        "state_cues": state_cues,
        "expected_action": expected_action,
        "obligations": obligations,
        "forbidden": forbidden,
        "keywords": keywords,
        "context_text": context_text,
        "transition_text": transition_text,
    }


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build(args: argparse.Namespace) -> dict[str, Any]:
    client = None
    if not args.no_llm:
        from openai import OpenAI

        api_key = os.environ.get(args.llm_api_key_env)
        if not api_key:
            raise ValueError(f"Set {args.llm_api_key_env} or pass --no-llm")
        client = OpenAI(
            base_url=args.llm_base_url.rstrip("/") + "/",
            api_key=api_key,
            timeout=args.llm_timeout,
            max_retries=1,
        )

    all_patches = []
    jobs = []
    for domain in [value.strip() for value in args.domains.split(",") if value.strip()]:
        cases = [
            _load_case(path, domain)
            for path in sorted((args.data_root / domain).glob("*.json"))
        ]
        jobs.extend((domain, batch) for batch in _chunks(cases, args.cases_per_request))

    def process(job: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        domain, cases = job
        task_ids = [case["task_id"] for case in cases]
        digest = hashlib.sha1("|".join(task_ids).encode("utf-8")).hexdigest()[:12]
        cache_key = f"{task_ids[0][:48]}__{digest}"
        cache_path = args.cache_dir / domain / f"{cache_key}.json"
        generator = "fallback" if client is None else "llm"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("version") == CACHE_VERSION
                and cached.get("model") == args.llm_model
                and cached.get("generator") == generator
            ):
                raw_patches = cached.get("patches", [])
            else:
                raw_patches = []
        else:
            raw_patches = []
        if not raw_patches and client is None:
            raw_patches = [patch for case in cases for patch in _fallback_patches(case)]
        elif not raw_patches:
            raw_patches = _llm_patches(client, args.llm_model, cases)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "model": args.llm_model,
                    "generator": generator,
                    "patches": raw_patches,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        case_ids = {case["task_id"] for case in cases}
        return [
            patch
            for raw in raw_patches
            if isinstance(raw, dict)
            if (patch := _validate_patch(raw, case_ids, domain)) is not None
        ]

    with ThreadPoolExecutor(max_workers=max(1, args.llm_workers)) as executor:
        futures = {executor.submit(process, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            all_patches.extend(future.result())
            print(json.dumps({"completed": completed, "total": len(jobs)}))

    artifact = build_transition_artifact(
        all_patches,
        coreset_ratio=args.coreset_ratio,
        min_per_group=args.min_per_group,
        max_per_group=args.max_per_group,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=True, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/train_task_trajectories"))
    parser.add_argument("--output", type=Path, default=Path("outputs/memory/transition_patches.json"))
    parser.add_argument("--domains", default="travel,customer_support,shopping_assistant")
    parser.add_argument("--cases-per-request", type=int, default=4)
    parser.add_argument("--coreset-ratio", type=float, default=0.35)
    parser.add_argument("--min-per-group", type=int, default=6)
    parser.add_argument("--max-per-group", type=int, default=48)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-base-url", default=os.environ.get("WORKFLOW_LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("WORKFLOW_LLM_MODEL", "gpt-5.4"))
    parser.add_argument("--llm-api-key-env", default="WORKFLOW_LLM_API_KEY")
    parser.add_argument("--llm-workers", type=int, default=6)
    parser.add_argument("--llm-timeout", type=float, default=240.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/memory/transition_patch_cache"))
    args = parser.parse_args()
    artifact = build(args)
    print(json.dumps({"output": str(args.output), "patches": len(artifact["patches"]), "stats": artifact["stats"]}))


if __name__ == "__main__":
    main()
