import argparse
import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW

from dirichlet_mixture_of_experts.model import DirMoE, DirMoEConfig, DirMoELoss

SequenceRecord = dict[str, list[int] | str]


class ByteTokenizer:
    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        token_ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local DirMoE shakeout trainer")
    parser.add_argument(
        "--data-path",
        type=Path,
        nargs="+",
        required=True,
        help="Input corpus files (.jsonl, .json, or .txt). JSONL rows should contain {'text': ..., 'source': ...}.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--d-ffn", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--lambda-q", type=float, default=20.0)
    parser.add_argument("--tau-z-start", type=float, default=2.0)
    parser.add_argument("--tau-z-end", type=float, default=0.3)
    parser.add_argument("--lambda-p-start", type=float, default=0.5)
    parser.add_argument("--lambda-p-end", type=float, default=0.3)
    parser.add_argument("--alpha-hi-prior", type=float, default=1.985)
    parser.add_argument("--alpha-lo-prior", type=float, default=0.005)
    parser.add_argument("--beta-theta", type=float, default=0.01)
    parser.add_argument("--lambda-sparsity", type=float, default=0.01)
    parser.add_argument("--disable-recon-loss", action="store_true")
    parser.add_argument("--disable-kl-loss", action="store_true")
    parser.add_argument(
        "--max-tokens-per-source",
        type=int,
        default=12_500_000,
        help="Cap per source to keep local corpora small.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.98,
        help="Fraction of packed sequences used for training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return Path("artifacts") / datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    rows.append({"text": item, "source": path.stem})
                else:
                    rows.append({"text": item["text"], "source": item.get("source", path.stem)})
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = []
            for item in payload:
                if isinstance(item, str):
                    rows.append({"text": item, "source": path.stem})
                else:
                    rows.append({"text": item["text"], "source": item.get("source", path.stem)})
            return rows
        if isinstance(payload, dict) and "text" in payload:
            return [{"text": payload["text"], "source": payload.get("source", path.stem)}]
        raise ValueError(f"Unsupported JSON payload in {path}")
    if path.suffix == ".txt":
        return [{"text": path.read_text(encoding="utf-8"), "source": path.stem}]
    raise ValueError(f"Unsupported input file type: {path}")


def load_corpus(
    paths: list[Path],
    tokenizer: ByteTokenizer,
    max_tokens_per_source: int,
    seq_len: int,
) -> tuple[list[SequenceRecord], dict[str, dict[str, int]]]:
    tokens_by_source: dict[str, list[int]] = {}
    stats: dict[str, dict[str, int]] = {}
    for path in paths:
        for row in _read_rows(path):
            source = row["source"]
            source_tokens = tokens_by_source.setdefault(source, [])
            if len(source_tokens) >= max_tokens_per_source:
                continue
            encoded = tokenizer.encode(row["text"], add_bos=True, add_eos=True)
            remaining = max_tokens_per_source - len(source_tokens)
            if remaining <= 0:
                continue
            if len(encoded) > remaining:
                encoded = encoded[:remaining]
                if encoded[-1] != tokenizer.eos_token_id:
                    encoded[-1] = tokenizer.eos_token_id
            source_tokens.extend(encoded)
            stat = stats.setdefault(source, {"tokens": 0, "documents": 0})
            stat["tokens"] += len(encoded)
            stat["documents"] += 1

    packed_sequences: list[SequenceRecord] = []
    packed_length = seq_len + 1
    for source, token_ids in tokens_by_source.items():
        packed_count = 0
        for start in range(0, len(token_ids) - packed_length + 1, packed_length):
            stop = start + packed_length
            if stop > len(token_ids):
                break
            sequence = token_ids[start:stop]
            if len(sequence) == packed_length:
                packed_sequences.append({"tokens": sequence, "source": source})
                packed_count += 1
        stats[source]["packed_sequences"] = packed_count
    return packed_sequences, stats


def split_sequences(
    sequences: list[SequenceRecord],
    train_split: float,
) -> tuple[list[SequenceRecord], list[SequenceRecord]]:
    random.shuffle(sequences)
    split_idx = max(1, int(len(sequences) * train_split))
    split_idx = min(split_idx, len(sequences) - 1) if len(sequences) > 1 else len(sequences)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:] if split_idx < len(sequences) else sequences[:1]
    return train_sequences, val_sequences


def make_batch(
    sequences: list[SequenceRecord],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    batch = random.sample(sequences, k=min(batch_size, len(sequences)))
    inputs = [item["tokens"][:seq_len] for item in batch]
    targets = [item["tokens"][1:seq_len + 1] for item in batch]
    sources = [str(item["source"]) for item in batch]
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
        sources,
    )


def cosine_decay(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(max(step / (total_steps - 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def learning_rate(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    return cosine_decay(step - warmup_steps, max(1, total_steps - warmup_steps), base_lr, base_lr * 0.1)


def collect_router_metrics(aux_list: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    active_counts = []
    active_maxes = []
    expert_loads = []
    expert_load_entropies = []
    simpson_indices = []
    route_entropies = []
    route_top1_masses = []
    route_top2_masses = []
    route_gt_005_counts = []
    route_gt_010_counts = []
    z_sum_stds = []
    z_gt_01_counts = []
    z_gt_05_counts = []
    z_means = []
    for aux in aux_list:
        z = aux["z"].float()
        r = aux["r"].float()
        z_sum = z.sum(dim=-1)
        expert_load = z.mean(dim=0)
        load_distribution = expert_load / expert_load.sum().clamp_min(1e-8)
        sorted_r = r.sort(dim=-1, descending=True).values

        active_counts.append(z_sum.mean())
        active_maxes.append(z_sum.max())
        expert_loads.append(expert_load)
        expert_load_entropies.append(-(load_distribution * load_distribution.clamp_min(1e-8).log()).sum())
        simpson_indices.append((r.pow(2).sum(dim=-1)).mean())
        route_entropies.append(-(r * r.clamp_min(1e-8).log()).sum(dim=-1).mean())
        route_top1_masses.append(sorted_r[..., 0].mean())
        route_top2_masses.append(sorted_r[..., :2].sum(dim=-1).mean())
        route_gt_005_counts.append((r > 0.05).float().sum(dim=-1).mean())
        route_gt_010_counts.append((r > 0.10).float().sum(dim=-1).mean())
        z_sum_stds.append(z_sum.std(unbiased=False))
        z_gt_01_counts.append((z > 0.1).float().sum(dim=-1).mean())
        z_gt_05_counts.append((z > 0.5).float().sum(dim=-1).mean())
        z_means.append(z.mean())
    mean_active = torch.stack(active_counts).mean().item()
    max_active = torch.stack(active_maxes).max().item()
    mean_loads = torch.stack(expert_loads).mean(dim=0)
    load_mean = mean_loads.mean()
    load_std = mean_loads.std(unbiased=False)
    return {
        "active_mean": mean_active,
        "active_max": max_active,
        "z_sum_mean": mean_active,
        "z_sum_std": torch.stack(z_sum_stds).mean().item(),
        "z_mean": torch.stack(z_means).mean().item(),
        "z_gt_0_1_count_mean": torch.stack(z_gt_01_counts).mean().item(),
        "z_gt_0_5_count_mean": torch.stack(z_gt_05_counts).mean().item(),
        "simpson_index": torch.stack(simpson_indices).mean().item(),
        "r_entropy": torch.stack(route_entropies).mean().item(),
        "r_top1_mass": torch.stack(route_top1_masses).mean().item(),
        "r_top2_mass": torch.stack(route_top2_masses).mean().item(),
        "r_gt_0_05_count_mean": torch.stack(route_gt_005_counts).mean().item(),
        "r_gt_0_10_count_mean": torch.stack(route_gt_010_counts).mean().item(),
        "max_expert_load": mean_loads.max().item(),
        "min_expert_load": mean_loads.min().item(),
        "expert_load_cv": (load_std / load_mean.clamp_min(1e-8)).item(),
        "expert_load_entropy": torch.stack(expert_load_entropies).mean().item(),
    }


def collect_source_routing_metrics(
    aux_list: list[dict[str, torch.Tensor]],
    sources: list[str],
    seq_len: int,
) -> dict[str, dict[str, list[list[float]]]]:
    stats = {}
    for source in sorted(set(sources)):
        batch_indices = [idx for idx, item in enumerate(sources) if item == source]
        if not batch_indices:
            continue
        layer_r_means = []
        layer_z_means = []
        for aux in aux_list:
            num_experts = aux["r"].shape[-1]
            r = aux["r"].float().view(len(sources), seq_len, num_experts)
            z = aux["z"].float().view(len(sources), seq_len, num_experts)
            source_index = torch.tensor(batch_indices, device=r.device)
            layer_r_means.append(r.index_select(0, source_index).mean(dim=(0, 1)).cpu().tolist())
            layer_z_means.append(z.index_select(0, source_index).mean(dim=(0, 1)).cpu().tolist())
        stats[source] = {
            "r_mean_by_layer": layer_r_means,
            "z_mean_by_layer": layer_z_means,
        }
    return stats


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


@torch.no_grad()
def evaluate(
    model: DirMoE,
    criterion: DirMoELoss,
    sequences: list[SequenceRecord],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    inputs, targets, _ = make_batch(sequences, batch_size=batch_size, seq_len=seq_len, device=device)
    logits, aux_list = model(inputs)
    components = criterion(logits, targets, aux_list, return_components=True)
    metrics = collect_router_metrics(aux_list)
    metrics.update({
        "total_loss": components["total_loss"].item(),
        "lm_loss": components["lm_loss"].item(),
        "recon_loss": components["recon_loss"].item(),
        "kl_loss": components["kl_loss"].item(),
        "sparsity_loss": components["sparsity_loss"].item(),
        "dirmoe_loss": components["dirmoe_loss"].item(),
    })
    return metrics


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: DirMoE,
    optimizer: AdamW,
    config: DirMoEConfig,
    tokenizer: ByteTokenizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
        output_dir / f"checkpoint_step_{step}.pt",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = resolve_output_dir(args.output_dir)

    tokenizer = ByteTokenizer()
    packed_sequences, source_stats = load_corpus(
        paths=args.data_path,
        tokenizer=tokenizer,
        max_tokens_per_source=args.max_tokens_per_source,
        seq_len=args.seq_len,
    )
    if not packed_sequences:
        raise RuntimeError("No packed sequences were built. Increase --max-tokens-per-source or provide more text.")

    train_sequences, val_sequences = split_sequences(packed_sequences, train_split=args.train_split)

    config = DirMoEConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        d_ffn=args.d_ffn,
        max_seq_len=args.seq_len,
        num_experts=args.num_experts,
        k=args.k,
        hidden_dim_router=args.router_hidden_dim,
        lambda_p=args.lambda_p_start,
        lambda_q=args.lambda_q,
        tau_z=args.tau_z_start,
        alpha_hi_prior=args.alpha_hi_prior,
        alpha_lo_prior=args.alpha_lo_prior,
        tau_z_min=args.tau_z_end,
        lambda_p_min=args.lambda_p_end,
        beta_theta=args.beta_theta,
        lambda_sparsity=args.lambda_sparsity,
    )

    model = DirMoE(config).to(device)
    criterion = DirMoELoss(
        k=config.k,
        beta_theta=config.beta_theta,
        lambda_sparsity=config.lambda_sparsity,
        sigma2=config.sigma2,
        use_recon_loss=not args.disable_recon_loss,
        use_kl_loss=not args.disable_kl_loss,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": asdict(config),
        "experiment_name": args.experiment_name,
        "output_dir": str(output_dir),
        "loss_ablation": {
            "disable_recon_loss": args.disable_recon_loss,
            "disable_kl_loss": args.disable_kl_loss,
        },
        "source_stats": source_stats,
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"

    print("Loaded corpus:")
    for source, stats in sorted(source_stats.items()):
        print(
            f"  {source}: docs={stats['documents']} tokens={stats['tokens']} packed_sequences={stats['packed_sequences']}"
        )

    for step in range(args.steps):
        model.train()

        tau_z = cosine_decay(step, args.steps, args.tau_z_start, args.tau_z_end)
        lambda_p = cosine_decay(step, args.steps, args.lambda_p_start, args.lambda_p_end)
        model.set_router_schedule(tau_z=tau_z, lambda_p=lambda_p)

        lr = learning_rate(step, args.steps, args.warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets, sources = make_batch(train_sequences, batch_size=args.batch_size, seq_len=args.seq_len, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux_list = model(inputs)
        components = criterion(logits, targets, aux_list, return_components=True)
        loss = components["total_loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            train_metrics = collect_router_metrics(aux_list)
            source_metrics = collect_source_routing_metrics(aux_list, sources=sources, seq_len=args.seq_len)
            val_metrics = evaluate(
                model=model,
                criterion=criterion,
                sequences=val_sequences,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
            )
            metrics_row = {
                "step": step,
                "experiment_name": args.experiment_name,
                "total_loss": loss.item(),
                "lm_loss": components["lm_loss"].item(),
                "recon_loss": components["recon_loss"].item(),
                "kl_loss": components["kl_loss"].item(),
                "sparsity_loss": components["sparsity_loss"].item(),
                "sparsity_to_recon_ratio": (
                    components["sparsity_loss"] / components["recon_loss"].clamp_min(1e-8)
                ).item(),
                "dirmoe_loss": components["dirmoe_loss"].item(),
                "val_total_loss": val_metrics["total_loss"],
                "val_lm_loss": val_metrics["lm_loss"],
                "lr": lr,
                "tau_z": tau_z,
                "lambda_p": lambda_p,
                "lambda_sparsity": args.lambda_sparsity,
                "k": args.k,
                "num_experts": args.num_experts,
                "grad_norm": float(grad_norm),
                **train_metrics,
                "source_routing": source_metrics,
            }
            append_jsonl(metrics_path, metrics_row)
            print(
                "step={step} loss={loss:.4f} val_loss={val_loss:.4f} lm_loss={lm_loss:.4f} "
                "lr={lr:.6f} tau_z={tau_z:.3f} lambda_p={lambda_p:.3f} grad_norm={grad_norm:.3f} "
                "active_mean={active_mean:.3f} simpson={simpson:.3f} "
                "expert_load_range=[{min_load:.3f}, {max_load:.3f}]".format(
                    step=step,
                    loss=loss.item(),
                    val_loss=val_metrics["total_loss"],
                    lm_loss=components["lm_loss"].item(),
                    lr=lr,
                    tau_z=tau_z,
                    lambda_p=lambda_p,
                    grad_norm=float(grad_norm),
                    active_mean=train_metrics["active_mean"],
                    simpson=train_metrics["simpson_index"],
                    min_load=train_metrics["min_expert_load"],
                    max_load=train_metrics["max_expert_load"],
                )
            )

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_checkpoint(output_dir, step + 1, model, optimizer, config, tokenizer)


if __name__ == "__main__":
    main()
