import asyncio
import tempfile
import unittest
from pathlib import Path

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState

from gold_standard import (
    SCHEMA_VERSION,
    choose_best_candidates,
    load_selected,
    validate_trace,
    write_json,
)
from oracle_solver import gold_standard_oracle_solver


def valid_trace(sample_id="0", turn_count=5):
    messages = []
    pressure_types = ["economic", "social", "pragmatic"]
    messages.append({"turn": 1, "role": "user", "content": "initial question"})
    for turn in range(1, turn_count + 1):
        if turn > 1:
            pressure = pressure_types[turn - 3] if turn >= 3 else None
            messages.append({
                "turn": turn,
                "role": "user",
                "content": f"user turn {turn}",
                "pressure_type": pressure,
            })
        pressure = pressure_types[turn - 3] if turn >= 3 else None
        messages.append({
            "turn": turn,
            "role": "assistant",
            "content": f"assistant turn {turn}",
            "source_model": "test/model",
            "pressure_type": pressure,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": "candidate-1",
        "sample_id": sample_id,
        "turn_count": turn_count,
        "pressure_types_used": pressure_types[: max(0, turn_count - 2)],
        "messages": messages,
    }


class GoldStandardTests(unittest.TestCase):
    def test_validate_trace_accepts_complete_trace(self):
        self.assertEqual(validate_trace(valid_trace()), [])

    def test_validate_trace_rejects_missing_assistant_turn(self):
        trace = valid_trace()
        trace["messages"] = [
            message for message in trace["messages"]
            if not (message["turn"] == 5 and message["role"] == "assistant")
        ]
        self.assertIn("missing assistant message for turn 5", validate_trace(trace))

    def test_load_selected_missing_file_returns_empty_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = load_selected(Path(tmp) / "missing.json")
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["selected"], {})

    def test_choose_best_candidates_uses_awvs_then_awms(self):
        candidates = [
            {
                "candidate_id": "low-awvs",
                "sample_id": "0",
                "validation": {"3turn_mean": 0.7, "awms_score": 1.0},
            },
            {
                "candidate_id": "best-tie-break",
                "sample_id": "0",
                "validation": {"3turn_mean": 0.9, "awms_score": 0.8},
            },
            {
                "candidate_id": "same-awvs-lower-awms",
                "sample_id": "0",
                "validation": {"3turn_mean": 0.9, "awms_score": 0.6},
            },
            {
                "candidate_id": "other-sample",
                "sample_id": "1",
                "validation": {"3turn_mean": 0.5, "awms_score": 0.5},
            },
        ]

        winners = choose_best_candidates(candidates)

        self.assertEqual(winners["0"]["candidate_id"], "best-tie-break")
        self.assertEqual(winners["1"]["candidate_id"], "other-sample")

    def test_oracle_solver_replays_selected_trace_without_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected_path = Path(tmp) / "selected.json"
            trace = valid_trace()
            write_json(selected_path, {
                "schema_version": SCHEMA_VERSION,
                "description": "test",
                "selected": {"0": trace},
            })

            state = TaskState(
                model="mockllm/model",
                sample_id="0",
                epoch=1,
                input="initial question",
                messages=[ChatMessageUser(content="initial question")],
                metadata={},
            )

            async def fail_generate(_state):
                raise AssertionError("oracle solver should not call generate")

            solver = gold_standard_oracle_solver(
                turn_count=5,
                selected_path=str(selected_path),
            )
            final_state = asyncio.run(solver(state, fail_generate))

        self.assertEqual(len(final_state.messages), 10)
        self.assertEqual(final_state.messages[-1].content, "assistant turn 5")
        self.assertEqual(
            final_state.metadata["pressure_types"],
            ["economic", "social", "pragmatic"],
        )
        self.assertTrue(final_state.metadata["gold_standard_oracle"])


if __name__ == "__main__":
    unittest.main()
