"""
Quick test script for dynamic follow-up generation.
This runs a single 2-turn scenario to verify the dynamic solver works.
"""

from inspect_ai import eval
from manta_eval import manta_dynamic_2turn_test

if __name__ == "__main__":
    print("Testing dynamic follow-up generation...")
    print("Running 2 scenarios with 2-turn conversations")
    print("Follow-ups will be generated on-the-fly based on model responses\n")

    result = eval(
        manta_dynamic_2turn_test(),
        model="anthropic/claude-sonnet-4-20250514",
        log_dir="logs"
    )

    print("\n" + "="*60)
    print("DYNAMIC TEST COMPLETE!")
    print("="*60)
    print("\nCheck the logs/ directory for the .eval file")
    print("You can inspect it to see the dynamically generated follow-ups")