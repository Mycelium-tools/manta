"""
Shared helpers for MANTA gold standard answers (GSAs).

GSAs are stored as full conversation traces so they can be replayed as
target-model-role assistant messages without re-running the dynamic follow-up
generator. This keeps oracle replay deterministic once a trace is selected.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.dataset import Sample
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalPlanStep,
    EvalSample,
    EvalSpec,
    EvalStats,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateConfig, ModelOutput

from dynamic_multiturn_solver import PRESSURE_DESCRIPTIONS
from manta_scorer import MANTA_DIMENSIONS


SCHEMA_VERSION = 1
DEFAULT_GSA_DIR = Path("dataset/gold_standard_answers")
DEFAULT_CANDIDATES_DIR = DEFAULT_GSA_DIR / "candidates"
DEFAULT_SELECTED_PATH = DEFAULT_GSA_DIR / "selected.eval"
DEFAULT_VALIDATION_DIR = DEFAULT_GSA_DIR / "validation"

TAG_ALIASES = {
    "Value Stability": "Animal Welfare Value Stability",
    "Contextual Welfare Salience": "Animal Welfare Moral Sensitivity",
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    input: str
    target: str
    metadata: dict[str, Any]


def parse_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return []


def normalize_tags(tags: list[str]) -> list[str]:
    return [TAG_ALIASES.get(tag, tag) for tag in tags]


def dimension_catalog(include_10_turn: bool = True) -> dict[str, str]:
    dimensions = {
        key: value["description"]
        for key, value in MANTA_DIMENSIONS.items()
    }
    if not include_10_turn:
        dimensions.pop("10-Turn Value Stability", None)
    return dimensions


def pressure_catalog() -> dict[str, str]:
    return dict(PRESSURE_DESCRIPTIONS)


def load_sample_records(samples_file: str | Path = "samples.json") -> list[SampleRecord]:
    with Path(samples_file).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records: list[SampleRecord] = []
    for row in payload["all"]:
        tags = normalize_tags(parse_list(row.get("tags", [])))
        pressure = row.get("pressure") or ["economic", "social", "pragmatic"]
        animals = row.get("animals") or []
        base_metadata = {
            "tags": tags,
            "language": row.get("language", "en"),
            "pressure": pressure,
            "reference_answer": row.get("reference_answer", ""),
        }

        if animals:
            for animal in animals:
                sample_id = f"{row['id']}_{animal}"
                question = row["question"].replace("{{animal}}", animal)
                records.append(SampleRecord(
                    sample_id=sample_id,
                    input=question,
                    target=json.dumps({"tags": tags}),
                    metadata={**base_metadata, "animal": animal, "base_id": str(row["id"])},
                ))
        else:
            records.append(SampleRecord(
                sample_id=str(row["id"]),
                input=row["question"],
                target=json.dumps({"tags": tags}),
                metadata=base_metadata,
            ))

    return records


def filter_records(
    records: list[SampleRecord],
    ids: list[str] | None = None,
    sample_range: tuple[int, int] | None = None,
) -> list[SampleRecord]:
    selected = records
    if ids:
        wanted = set(ids)
        selected = [record for record in selected if record.sample_id in wanted]
    if sample_range:
        start, end = sample_range
        selected = selected[start:end]
    return selected


def to_inspect_sample(record: SampleRecord) -> Sample:
    return Sample(
        input=record.input,
        target=record.target,
        id=record.sample_id,
        metadata=record.metadata,
    )


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_candidate_traces(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".eval":
        log = read_eval_log(path)
        candidates = []
        for sample in log.samples or []:
            candidate = (sample.metadata or {}).get("gsa_candidate")
            if candidate:
                candidates.append(candidate)
        return candidates
    return read_jsonl(path)


def selected_payload(selected: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Curated MANTA gold standard answer traces keyed by expanded sample id. "
            "Replay uses these stored user and assistant turns without generation."
        ),
        "selected": selected or {},
    }


def trace_to_chat_messages(trace: dict[str, Any]) -> list[ChatMessageUser | ChatMessageAssistant]:
    messages = []
    for message in trace.get("messages", []):
        if message.get("role") == "user":
            messages.append(ChatMessageUser(content=message["content"]))
        elif message.get("role") == "assistant":
            messages.append(ChatMessageAssistant(
                content=message["content"],
                model=message.get("source_model") or trace.get("source_model"),
                metadata={
                    "candidate_id": trace.get("candidate_id"),
                    "turn": message.get("turn"),
                    "pressure_type": message.get("pressure_type"),
                },
            ))
    return messages


def _trace_question(trace: dict[str, Any]) -> str:
    for message in trace.get("messages", []):
        if message.get("turn") == 1 and message.get("role") == "user":
            return message["content"]
    return trace.get("question", "")


def _selected_payload_from_eval(path: Path) -> dict[str, Any]:
    log = read_eval_log(path)
    selected = {}
    for sample in log.samples or []:
        trace = (sample.metadata or {}).get("gsa_candidate")
        if not trace:
            continue
        selected[str(trace.get("sample_id") or sample.id)] = trace
    return selected_payload(selected)


def load_selected(path: str | Path = DEFAULT_SELECTED_PATH) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".eval":
        if not path.exists():
            legacy_json = path.with_suffix(".json")
            if legacy_json.exists():
                return load_selected(legacy_json)
            return selected_payload()
        return _selected_payload_from_eval(path)

    payload = read_json(path, default=None)
    if payload is None:
        return selected_payload()
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported GSA schema version in {path}: {payload.get('schema_version')}"
        )
    payload.setdefault("selected", {})
    return payload


def write_selected(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.suffix != ".eval":
        write_json(path, payload)
        return

    traces = payload.get("selected", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    samples = []
    for sample_id, trace in sorted(traces.items()):
        messages = trace_to_chat_messages(trace)
        output_text = messages[-1].text if messages else ""
        samples.append(EvalSample(
            id=str(sample_id),
            epoch=1,
            input=_trace_question(trace),
            target=json.dumps({"tags": trace.get("tags", [])}),
            messages=messages,
            output=ModelOutput.from_content(
                model=trace.get("source_model") or "manta-gsa-oracle",
                content=output_text,
            ),
            metadata={
                "sample_id": str(sample_id),
                "candidate_id": trace.get("candidate_id"),
                "source_model": trace.get("source_model"),
                "tags": trace.get("tags", []),
                "pressure": trace.get("pressure", []),
                "pressure_types_used": trace.get("pressure_types_used", []),
                "gsa_candidate": trace,
                "selected_gsa": True,
            },
        ))

    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created=now,
            task="selected_gold_standard_answers",
            task_file="gold_standard.py",
            task_display_name="Selected Gold Standard Answers",
            solver="gold_standard_oracle_solver",
            solver_args={},
            tags=["manta", "gold-standard-answer", "selected"],
            dataset=EvalDataset(
                name="manta_selected_gsas",
                location=str(path),
                samples=len(samples),
                sample_ids=[sample.id for sample in samples],
            ),
            model="manta-gsa-oracle",
            model_generate_config=GenerateConfig(),
            config=EvalConfig(epochs=1),
            metadata={
                "schema_version": SCHEMA_VERSION,
                "output_kind": "selected_gsa_traces",
                "format_note": (
                    "Each EvalSample is one selected GSA trace. "
                    "The replayable trace is in sample.metadata.gsa_candidate."
                ),
            },
        ),
        plan=EvalPlan(
            steps=[
                EvalPlanStep(
                    solver="gold_standard_oracle_solver",
                    params={},
                )
            ],
            config=GenerateConfig(),
        ),
        stats=EvalStats(started_at=now, completed_at=now),
        samples=samples,
        location=str(path),
    )
    write_eval_log(log, path, format="eval")


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def assistant_turns(trace: dict[str, Any]) -> dict[int, str]:
    turns: dict[int, str] = {}
    for message in trace.get("messages", []):
        if message.get("role") == "assistant":
            turns[int(message["turn"])] = message_text(message)
    return turns


def user_turns(trace: dict[str, Any]) -> dict[int, str]:
    turns: dict[int, str] = {}
    for message in trace.get("messages", []):
        if message.get("role") == "user":
            turns[int(message["turn"])] = message_text(message)
    return turns


def validate_trace(trace: dict[str, Any], turn_count: int | None = None) -> list[str]:
    errors: list[str] = []
    if trace.get("schema_version") != SCHEMA_VERSION:
        errors.append("trace has unsupported or missing schema_version")
    if not trace.get("sample_id"):
        errors.append("trace is missing sample_id")

    expected_turns = int(turn_count or trace.get("turn_count") or 5)
    messages = trace.get("messages")
    if not isinstance(messages, list):
        errors.append("trace messages must be a list")
        return errors

    by_key = {(m.get("turn"), m.get("role")): m for m in messages}
    for turn in range(1, expected_turns + 1):
        if (turn, "user") not in by_key:
            errors.append(f"missing user message for turn {turn}")
        if (turn, "assistant") not in by_key:
            errors.append(f"missing assistant message for turn {turn}")

    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            errors.append(f"invalid message role: {role}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"empty content for turn {message.get('turn')} {role}")

    pressure_used = trace.get("pressure_types_used", [])
    if expected_turns >= 3 and len(pressure_used) != expected_turns - 2:
        errors.append(
            f"pressure_types_used should have {expected_turns - 2} entries, got {len(pressure_used)}"
        )
    return errors


def validate_selected_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    failures = {}
    for sample_id, trace in payload.get("selected", {}).items():
        errors = validate_trace(trace)
        if sample_id != str(trace.get("sample_id")):
            errors.append("selected key does not match trace sample_id")
        if errors:
            failures[sample_id] = errors
    return failures


def candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, float]:
    validation = candidate.get("validation") or {}
    awvs = validation.get("3turn_mean")
    if awvs is None:
        awvs = validation.get("awvs_score")
    awms = validation.get("awms_score")
    return (
        float(awvs if awvs is not None else -1.0),
        float(awms if awms is not None else -1.0),
    )


def choose_best_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        sample_id = str(candidate["sample_id"])
        incumbent = winners.get(sample_id)
        if incumbent is None or candidate_rank_key(candidate) > candidate_rank_key(incumbent):
            winners[sample_id] = candidate
    return winners
