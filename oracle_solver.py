"""
Inspect solver for replaying selected MANTA gold standard answer traces.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput
from inspect_ai.solver import solver

from gold_standard import DEFAULT_SELECTED_PATH, load_selected, validate_trace


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.text if hasattr(item, "text") else str(item)
            for item in content
        )
    return str(content)


@solver
def gold_standard_oracle_solver(
    turn_count: int = 5,
    selected_path: str = str(DEFAULT_SELECTED_PATH),
    allow_missing: bool = False,
):
    """
    Replay a selected GSA trace as assistant messages.

    The solver does not call generate(). It appends assistant messages from the
    selected trace at each target-model answer slot, and appends stored user
    follow-up messages for turns 2..N.
    """

    async def solve(state, generate):
        payload = load_selected(Path(selected_path))
        sample_id = str(state.sample_id)
        trace = payload.get("selected", {}).get(sample_id)
        if trace is None:
            if allow_missing:
                transcript().info({"oracle_missing": sample_id})
                return state
            raise ValueError(
                f"No selected gold standard trace for sample {sample_id} in {selected_path}"
            )

        errors = validate_trace(trace, turn_count=turn_count)
        if errors:
            raise ValueError(
                f"Invalid gold standard trace for sample {sample_id}: " + "; ".join(errors)
            )

        trace_messages = trace["messages"]
        first_user = next(
            message for message in trace_messages
            if message["turn"] == 1 and message["role"] == "user"
        )
        if state.messages and _content_text(state.messages[0].content) != first_user["content"]:
            raise ValueError(
                f"Selected GSA trace for sample {sample_id} does not match the task input"
            )

        pressure_types_used = []
        for turn in range(1, turn_count + 1):
            if turn > 1:
                user_message = next(
                    message for message in trace_messages
                    if message["turn"] == turn and message["role"] == "user"
                )
                state.messages.append(ChatMessageUser(content=user_message["content"]))

            assistant_message = next(
                message for message in trace_messages
                if message["turn"] == turn and message["role"] == "assistant"
            )
            state.messages.append(ChatMessageAssistant(
                content=assistant_message["content"],
                model=assistant_message.get("source_model") or "manta-gsa-oracle",
                metadata={
                    "gsa_oracle": True,
                    "source_candidate_id": trace.get("candidate_id"),
                    "turn": turn,
                },
            ))

            if turn >= 3:
                pressure_type = assistant_message.get("pressure_type")
                if pressure_type:
                    pressure_types_used.append(pressure_type)

        if not pressure_types_used:
            pressure_types_used = trace.get("pressure_types_used", [])

        state.metadata["pressure_types"] = pressure_types_used
        state.metadata["dynamic_followups"] = False
        state.metadata["gold_standard_oracle"] = True
        state.metadata["gold_standard_candidate_id"] = trace.get("candidate_id")
        state.metadata["anchor_turn"] = 2
        state.output = ModelOutput(
            model="manta-gsa-oracle",
            completion=state.messages[-1].text,
        )
        return state

    return solve
