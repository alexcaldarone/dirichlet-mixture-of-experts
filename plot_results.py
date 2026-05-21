import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DirMoE experiment metrics from artifact folders.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/plots"))
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit run directories. Defaults to all artifact subdirectories with metrics.jsonl.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_runs(args: argparse.Namespace) -> list[dict]:
    run_dirs = args.runs
    if run_dirs is None:
        run_dirs = sorted(path for path in args.artifacts_dir.iterdir() if path.is_dir())

    runs = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        metadata = read_json(run_dir / "run_metadata.json")
        metrics = read_metrics(metrics_path)
        if not metrics:
            continue
        config = metadata.get("config", {})
        label = metadata.get("experiment_name") or run_dir.name
        runs.append({
            "dir": run_dir,
            "label": label,
            "metadata": metadata,
            "config": config,
            "metrics": metrics,
        })
    return runs


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs = [row["step"] for row in rows if key in row]
    ys = [row[key] for row in rows if key in row]
    return xs, ys


def save_loss_plot(runs: list[dict], output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for run in runs:
        xs, ys = series(run["metrics"], "lm_loss")
        if xs:
            plt.plot(xs, ys, label=f"{run['label']} train")
        xs, ys = series(run["metrics"], "val_lm_loss")
        if xs:
            plt.plot(xs, ys, linestyle="--", label=f"{run['label']} val")
    plt.xlabel("step")
    plt.ylabel("LM loss")
    plt.title("Language Modeling Loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=180)
    plt.close()


def save_experts_sweep_plot(runs: list[dict], output_dir: Path) -> None:
    sweep_runs = sorted(runs, key=lambda run: (run["config"].get("num_experts", 0), run["label"]))
    plt.figure(figsize=(10, 6))
    for run in sweep_runs:
        num_experts = run["config"].get("num_experts", "?")
        xs, ys = series(run["metrics"], "simpson_index")
        if xs:
            plt.plot(xs, ys, label=f"E={num_experts} ({run['label']})")
    plt.xlabel("step")
    plt.ylabel("Simpson index")
    plt.title("Effect of Number of Experts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "experts_sweep_simpson.png", dpi=180)
    plt.close()


def save_target_active_plot(runs: list[dict], output_dir: Path) -> None:
    sweep_runs = sorted(runs, key=lambda run: (run["config"].get("k", 0), run["label"]))
    plt.figure(figsize=(10, 6))
    for run in sweep_runs:
        target_k = run["config"].get("k", "?")
        xs, ys = series(run["metrics"], "active_mean")
        if xs:
            plt.plot(xs, ys, label=f"k={target_k} ({run['label']})")
    plt.xlabel("step")
    plt.ylabel("mean active experts")
    plt.title("Effect of Target Active Experts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "target_active_sweep.png", dpi=180)
    plt.close()


def save_router_sparsity_diagnostics(runs: list[dict], output_dir: Path) -> None:
    keys = [
        ("z_sum_mean", "soft z sum"),
        ("z_gt_0_5_count_mean", "count z > 0.5"),
        ("r_gt_0_05_count_mean", "count r > 0.05"),
        ("r_top1_mass", "top-1 routed mass"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (key, title) in zip(axes.flat, keys):
        for run in runs:
            xs, ys = series(run["metrics"], key)
            if xs:
                axis.plot(xs, ys, label=run["label"])
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.set_ylabel(key)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle("Router Sparsity Diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "router_sparsity_diagnostics.png", dpi=180)
    plt.close(fig)


def save_loss_components_plot(runs: list[dict], output_dir: Path) -> None:
    keys = [
        ("lm_loss", "LM loss"),
        ("recon_loss", "reconstruction loss"),
        ("sparsity_loss", "sparsity loss"),
        ("sparsity_to_recon_ratio", "sparsity / reconstruction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (key, title) in zip(axes.flat, keys):
        for run in runs:
            xs, ys = series(run["metrics"], key)
            if xs:
                axis.plot(xs, ys, label=run["label"])
        axis.set_title(title)
        axis.set_xlabel("step")
        axis.set_ylabel(key)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.suptitle("Loss Components")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "loss_components.png", dpi=180)
    plt.close(fig)


def latest_source_routing(run: dict) -> dict:
    for row in reversed(run["metrics"]):
        source_routing = row.get("source_routing")
        if source_routing:
            return source_routing
    return {}


def average_layers(matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return []
    num_layers = len(matrix)
    num_experts = len(matrix[0])
    return [
        sum(layer[expert_idx] for layer in matrix) / num_layers
        for expert_idx in range(num_experts)
    ]


def save_domain_specialization_heatmap(runs: list[dict], output_dir: Path) -> None:
    if not runs:
        return
    run = runs[0]
    for candidate in runs:
        if latest_source_routing(candidate):
            run = candidate
            break

    source_routing = latest_source_routing(run)
    if not source_routing:
        return

    sources = sorted(source_routing)
    heatmap = []
    for source in sources:
        heatmap.append(average_layers(source_routing[source].get("r_mean_by_layer", [])))
    if not heatmap or not heatmap[0]:
        return

    num_experts = len(heatmap[0])
    plt.figure(figsize=(max(8, num_experts * 0.6), max(4, len(sources) * 0.7)))
    image = plt.imshow(heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(image, label="mean routed mass")
    plt.xticks(range(num_experts), [f"E{i}" for i in range(num_experts)])
    plt.yticks(range(len(sources)), sources)
    plt.xlabel("expert")
    plt.ylabel("source")
    plt.title(f"Domain Specialization: {run['label']}")
    plt.tight_layout()
    plt.savefig(output_dir / "domain_specialization_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args)
    if not runs:
        raise SystemExit("No runs with metrics.jsonl found.")

    save_loss_plot(runs, args.output_dir)
    save_experts_sweep_plot(runs, args.output_dir)
    save_target_active_plot(runs, args.output_dir)
    save_router_sparsity_diagnostics(runs, args.output_dir)
    save_loss_components_plot(runs, args.output_dir)
    save_domain_specialization_heatmap(runs, args.output_dir)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
