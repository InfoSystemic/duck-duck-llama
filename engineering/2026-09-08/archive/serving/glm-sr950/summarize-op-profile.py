#!/usr/bin/env python3
import collections
import re
import sys


GRAPH = re.compile(
    r"CPU_OP_PROFILE index=(\d+) cpu=(-?\d+).* total=([0-9.]+) ms "
    r"first='([^']*)' last='([^']*)'"
)
NODE = re.compile(
    r"CPU_OP_PROFILE cpu=(-?\d+).* node=(\d+) op=([^ ]+) "
    r"time=([0-9.]+) ms name='([^']*)'"
)
LAYER_SUFFIX = re.compile(r"-(\d+)(?=$|[ (])")


def normalized_name(name: str) -> str:
    return LAYER_SUFFIX.sub("-*", name)


def print_groups(
    title: str,
    groups: dict[str, list[float]],
    measured_ms: float,
    limit: int | None = None,
) -> None:
    print(f"\n{title}")
    print(f"{'group':<44} {'calls':>7} {'total ms':>11} {'share':>8} {'mean us':>10} {'max ms':>9}")
    rows = sorted(groups.items(), key=lambda item: sum(item[1]), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    for name, samples in rows:
        total = sum(samples)
        share = 100.0 * total / measured_ms if measured_ms else 0.0
        print(
            f"{name[:44]:<44} {len(samples):>7} {total:>11.3f} "
            f"{share:>7.2f}% {1000.0 * total / len(samples):>10.2f} "
            f"{max(samples):>9.3f}"
        )


def main() -> int:
    graph_samples: list[tuple[int, int, float, str, str]] = []
    by_op: dict[str, list[float]] = collections.defaultdict(list)
    by_name: dict[str, list[float]] = collections.defaultdict(list)
    graph_by_cpu: dict[int, list[float]] = collections.defaultdict(list)

    for line in sys.stdin:
        graph_match = GRAPH.search(line)
        if graph_match:
            index, cpu, total, first, last = graph_match.groups()
            sample = (int(index), int(cpu), float(total), first, last)
            graph_samples.append(sample)
            graph_by_cpu[sample[1]].append(sample[2])
            continue

        node_match = NODE.search(line)
        if node_match:
            _cpu, _node, op, elapsed, name = node_match.groups()
            value = float(elapsed)
            by_op[op].append(value)
            by_name[f"{op}:{normalized_name(name)}"].append(value)

    if not graph_samples:
        print("no CPU_OP_PROFILE graph records on stdin", file=sys.stderr)
        return 1

    graph_ms = sum(sample[2] for sample in graph_samples)
    measured_ms = sum(sum(samples) for samples in by_op.values())
    indexes = [sample[0] for sample in graph_samples]
    print(
        f"graphs={len(graph_samples)} indexes={min(indexes)}..{max(indexes)} "
        f"graph_total_ms={graph_ms:.3f} recorded_node_ms={measured_ms:.3f} "
        f"unrecorded_ms={graph_ms - measured_ms:.3f}"
    )

    print("\nper CPU graph totals")
    print(f"{'cpu':>5} {'graphs':>7} {'total ms':>11} {'mean ms':>10} {'max ms':>9}")
    for cpu, samples in sorted(graph_by_cpu.items()):
        print(
            f"{cpu:>5} {len(samples):>7} {sum(samples):>11.3f} "
            f"{sum(samples) / len(samples):>10.3f} {max(samples):>9.3f}"
        )

    print_groups("by operation", by_op, measured_ms)
    print_groups("top normalized tensor names", by_name, measured_ms, limit=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
