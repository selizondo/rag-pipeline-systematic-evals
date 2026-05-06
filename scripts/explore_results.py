"""
Explore committed experiment results without any API calls or data download.

Usage:
    python scripts/explore_results.py              # full leaderboard
    python scripts/explore_results.py --top 5      # top N only
    python scripts/explore_results.py --metric mrr # sort by different metric
    python scripts/explore_results.py --dim chunk  # breakdown by dimension

Reads from experiments/*.json — all 24 result files are committed to the repo.
No OpenAI key, no PDF, no setup beyond `pip install` required.
"""

import argparse
import json
import sys
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"

METRIC_KEYS = {
    "mrr": ("mrr", None),
    "recall1": ("recall_at_k", "1"),
    "recall3": ("recall_at_k", "3"),
    "recall5": ("recall_at_k", "5"),
    "ndcg5": ("ndcg_at_k", "5"),
    "map": ("map_score", None),
}

DIMENSIONS = ("chunk", "embed", "retrieval")


def load_results(experiments_dir: Path) -> list[dict]:
    results = []
    for path in sorted(experiments_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            results.append(data)
        except Exception as e:
            print(f"[warn] could not load {path.name}: {e}", file=sys.stderr)
    return results


def get_metric(result: dict, metric: str) -> float:
    key, subkey = METRIC_KEYS[metric]
    val = result.get("metrics", {}).get(key, 0.0)
    if subkey is not None:
        val = val.get(subkey, 0.0) if isinstance(val, dict) else 0.0
    return float(val)


def label_config(result: dict) -> str:
    cfg = result.get("config", {})
    chunk_strat = cfg.get("chunk", {}).get("strategy", "?")
    embed_model = cfg.get("embed", {}).get("model", "?").replace("text-embedding-3-", "")
    retrieval = cfg.get("retrieval", {}).get("method", "?")
    return f"{chunk_strat:<10}  {embed_model:<6}  {retrieval}"


def print_leaderboard(results: list[dict], metric: str, top: int) -> None:
    ranked = sorted(results, key=lambda r: get_metric(r, metric), reverse=True)[:top]

    metric_label = metric.upper().replace("_", "@").replace("AT", "@")
    header = f"{'Rank':<5}  {'Config (chunk / embed / retrieval)':<38}  {metric_label:>7}  {'MRR':>7}  {'R@5':>7}  {'NDCG@5':>7}  {'Queries':>7}"
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for i, r in enumerate(ranked, 1):
        mrr = get_metric(r, "mrr")
        r5 = get_metric(r, "recall5")
        ndcg5 = get_metric(r, "ndcg5")
        primary = get_metric(r, metric)
        queries = r.get("metrics", {}).get("total_queries", "?")
        cfg_label = label_config(r)
        print(f"{i:<5}  {cfg_label:<38}  {primary:>7.3f}  {mrr:>7.3f}  {r5:>7.2f}  {ndcg5:>7.3f}  {queries:>7}")

    print(f"{'─' * len(header)}")
    print(f"  {len(results)} total experiments  |  sorted by {metric_label}  |  all values from committed experiments/\n")


def print_dimension_breakdown(results: list[dict], metric: str, dim: str) -> None:
    groups: dict[str, list[float]] = {}
    for r in results:
        cfg = r.get("config", {})
        if dim == "chunk":
            key = cfg.get("chunk", {}).get("strategy", "?")
        elif dim == "embed":
            key = cfg.get("embed", {}).get("model", "?").replace("text-embedding-3-", "")
        else:
            key = cfg.get("retrieval", {}).get("method", "?")
        groups.setdefault(key, []).append(get_metric(r, metric))

    metric_label = metric.upper()
    print(f"\n  Breakdown by {dim} dimension — avg {metric_label} across all other dims\n")
    print(f"  {'Value':<25}  {'Avg ' + metric_label:>10}  {'Min':>7}  {'Max':>7}  {'N':>4}")
    print(f"  {'─' * 60}")
    for key, vals in sorted(groups.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        avg = sum(vals) / len(vals)
        print(f"  {key:<25}  {avg:>10.3f}  {min(vals):>7.3f}  {max(vals):>7.3f}  {len(vals):>4}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=None, help="Show top N configs (default: all)")
    parser.add_argument("--metric", choices=list(METRIC_KEYS), default="mrr", help="Primary sort metric (default: mrr)")
    parser.add_argument("--dim", choices=DIMENSIONS, default=None, help="Show per-dimension breakdown instead of full leaderboard")
    args = parser.parse_args()

    if not EXPERIMENTS_DIR.exists() or not list(EXPERIMENTS_DIR.glob("*.json")):
        print(f"No experiment results found in {EXPERIMENTS_DIR}/", file=sys.stderr)
        print("Run `make eval` to generate results, or ensure you're in the repo root.", file=sys.stderr)
        sys.exit(1)

    results = load_results(EXPERIMENTS_DIR)
    print(f"\n  Loaded {len(results)} experiments from {EXPERIMENTS_DIR}/")

    if args.dim:
        print_dimension_breakdown(results, args.metric, args.dim)
    else:
        top = args.top or len(results)
        print_leaderboard(results, args.metric, top)


if __name__ == "__main__":
    main()
