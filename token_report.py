"""
token_report.py — summarize token usage, cost, and run time across MANTA eval logs.

Usage:
    python token_report.py                        # auto-detects log dir via MANTA_USER
    python token_report.py logs/Allen_April2026/  # specific directory
    python token_report.py path/to/file.eval      # single file
"""

import os
import sys
from datetime import datetime
from collections import defaultdict
from inspect_ai.log import read_eval_log

# Pricing per 1M tokens: (input $/1M, output $/1M)
PRICING = {
    # Evaluated models
    "google/gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "anthropic/claude-opus-4-7": (5.00, 25.00),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "openai/gpt-5.4": (2.50, 15.00),
    "openai/gpt-5.5": (5.00, 30.00),
    "grok/grok-4.3": (1.25, 2.50),
    "openai-api/deepseek/deepseek-v4-flash": (0.14, 0.28),
    "mistral/mistral-small-2603": (0.15, 0.60),
    "openrouter/meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    # Internal pipeline models (pressure selector, follow-up writer, scorer)
    "anthropic/claude-opus-4-6": (15.00, 75.00),
    "anthropic/claude-sonnet-4-20250514": (3.00, 15.00),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
    # Legacy
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "google/gemini-2.5-flash": (0.15, 0.60),
    "mistral/mistral-large-latest": (2.00, 6.00),
    "mistral/mistral-small-latest": (0.10, 0.30),
}


def get_log_dir():
    if os.environ.get("MANTA_LOG_DIR"):
        return os.environ["MANTA_LOG_DIR"]
    if os.environ.get("MANTA_USER"):
        month_year = datetime.now().strftime("%B%Y")
        return f"logs/{os.environ['MANTA_USER']}_{month_year}"
    return "logs"


def find_eval_files(path):
    if path.endswith(".eval") and os.path.isfile(path):
        return [path]
    files = []
    for root, _, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith(".eval"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def get_price(model_name):
    if model_name in PRICING:
        return PRICING[model_name]
    for key, price in PRICING.items():
        if key in model_name or model_name in key:
            return price
    return None


def fmt_tokens(n):
    return f"{n:,}"


def fmt_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def parse_dt(val):
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except Exception:
        return None


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else get_log_dir()
    eval_files = find_eval_files(target)

    if not eval_files:
        print(f"No .eval files found at: {target}")
        return

    model_tokens = defaultdict(lambda: {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "total": 0})
    all_starts = []
    all_ends = []
    total_working_secs = 0.0
    total_samples = 0

    for path in eval_files:
        try:
            log = read_eval_log(path, header_only=True)
        except Exception as e:
            print(f"  [warn] skipping {os.path.basename(path)}: {e}")
            continue

        if log.results:
            total_samples += log.results.completed_samples or 0

        for model_name, usage in (log.stats.model_usage or {}).items():
            inp = usage.input_tokens or 0
            out = usage.output_tokens or 0
            cache_write = getattr(usage, "input_tokens_cache_write", None) or 0
            cache_read = getattr(usage, "input_tokens_cache_read", None) or 0
            model_tokens[model_name]["input"] += inp
            model_tokens[model_name]["output"] += out
            model_tokens[model_name]["cache_write"] += cache_write
            model_tokens[model_name]["cache_read"] += cache_read
            model_tokens[model_name]["total"] += inp + out + cache_write + cache_read

        started = parse_dt(getattr(log.stats, "started_at", None))
        completed = parse_dt(getattr(log.stats, "completed_at", None))
        if started:
            all_starts.append(started)
        if completed:
            all_ends.append(completed)
        if started and completed:
            total_working_secs += (completed - started).total_seconds()

    if not model_tokens:
        print("No token data found in logs.")
        return

    # Header
    n = len(eval_files)
    print(f"\nToken Usage Report — {target} ({n} log file{'s' if n != 1 else ''}, {total_samples:,} sample{'s' if total_samples != 1 else ''} run)")
    if all_starts and all_ends:
        wall_start = min(all_starts)
        wall_end = max(all_ends)
        wall_secs = (wall_end - wall_start).total_seconds()
        print(f"Total wall-clock time : {fmt_duration(wall_secs)}  ({wall_start.strftime('%H:%M:%S')} → {wall_end.strftime('%H:%M:%S')} UTC)")
        print(f"Total working time    : {fmt_duration(total_working_secs)}")
    print()

    COL_MODEL = 42
    COL_NUM = 12
    header = f"{'Model':<{COL_MODEL}} {'Input':>{COL_NUM}} {'CacheW':>{COL_NUM}} {'CacheR':>{COL_NUM}} {'Output':>{COL_NUM}} {'Total':>{COL_NUM}} {'Est. Cost':>10}"
    sep = "─" * len(header)
    print(header)
    print(sep)

    total_input = total_output = total_cache_write = total_cache_read = total_all = 0
    total_cost = 0.0
    has_unknown_cost = False

    for model_name, t in sorted(model_tokens.items()):
        inp, out, cw, cr, tot = t["input"], t["output"], t["cache_write"], t["cache_read"], t["total"]
        total_input += inp
        total_output += out
        total_cache_write += cw
        total_cache_read += cr
        total_all += tot

        price = get_price(model_name)
        if price:
            cost = (inp / 1_000_000) * price[0] + (out / 1_000_000) * price[1]
            if "anthropic" in model_name:
                cost += (cw / 1_000_000) * price[0] * 1.25
                cost += (cr / 1_000_000) * price[0] * 0.10
            total_cost += cost
            cost_str = f"${cost:.2f}"
        else:
            cost_str = "n/a"
            has_unknown_cost = True

        name = model_name if len(model_name) <= COL_MODEL else model_name[:COL_MODEL - 1] + "…"
        cw_str = fmt_tokens(cw) if cw else "—"
        cr_str = fmt_tokens(cr) if cr else "—"
        print(f"{name:<{COL_MODEL}} {fmt_tokens(inp):>{COL_NUM}} {cw_str:>{COL_NUM}} {cr_str:>{COL_NUM}} {fmt_tokens(out):>{COL_NUM}} {fmt_tokens(tot):>{COL_NUM}} {cost_str:>10}")

    print(sep)
    total_cost_str = f"${total_cost:.2f}" + ("*" if has_unknown_cost else "")
    print(f"{'TOTAL':<{COL_MODEL}} {fmt_tokens(total_input):>{COL_NUM}} {fmt_tokens(total_cache_write):>{COL_NUM}} {fmt_tokens(total_cache_read):>{COL_NUM}} {fmt_tokens(total_output):>{COL_NUM}} {fmt_tokens(total_all):>{COL_NUM}} {total_cost_str:>10}")

    if has_unknown_cost:
        print("\n* Cost estimate excludes models with unknown pricing.")
    print()


if __name__ == "__main__":
    main()
