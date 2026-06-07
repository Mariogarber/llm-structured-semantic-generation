from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import train_kubernetes_two_head_ordinal_positional_sft as positional_sft
from llm_structured_semantic_generation.dataset_io import read_jsonl


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "two_head_ordinal_positional_sft_kubernetes_v1"
    / "two-head-ordinal-positional-v1-512-64-gap05-mlp-lr3-threshold-lr50-20260527"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether Positional Head V1 uses line position as a temporal shortcut."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default="checkpoint-step-80")
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "train.jsonl")
    parser.add_argument("--validation-file", type=Path, default=REPO_ROOT / "data" / "processed" / "kubernetes_v1" / "sft" / "validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--level-class-count", type=int, default=positional_sft.DEFAULT_LEVEL_CLASS_COUNT)
    parser.add_argument("--position-bin-size", type=int, default=4)
    parser.add_argument("--run-model-ablations", action="store_true")
    parser.add_argument("--ablation-modes", default="normal,zero,shuffle")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpu-memory", default="4.8GiB")
    parser.add_argument("--cpu-memory", default="32GiB")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0 for _ in values]
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for offset in range(start, end + 1):
            result[order[offset]] = rank
        start = end + 1
    return result


def load_predictions(predictions_path: Path, checkpoint: str, max_samples: int | None) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_jsonl(predictions_path)
        if str(row.get("checkpoint")) == checkpoint
    ]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"no_predictions_for_checkpoint:{checkpoint}:{predictions_path}")
    return rows


def update_group(group: dict[str, Any], *, pred: int, gold: int | None, z: float | None) -> None:
    group["n"] += 1
    group["pred_sum"] += pred
    group["pred_counts"][pred] += 1
    if pred == 0:
        group["pred0"] += 1
    if gold is not None:
        group["gold_sum"] += gold
        group["gold_n"] += 1
        group["gold_counts"][gold] += 1
        group["abs_error_sum"] += abs(pred - gold)
        if pred == gold:
            group["exact"] += 1
        if gold == 0:
            group["gold0"] += 1
        if gold > 0 and pred == 0:
            group["gold_gt0_pred0"] += 1
    if z is not None:
        group["z_values"].append(float(z))


def empty_group() -> dict[str, Any]:
    return {
        "n": 0,
        "pred0": 0,
        "gold0": 0,
        "gold_gt0_pred0": 0,
        "exact": 0,
        "pred_sum": 0,
        "gold_sum": 0,
        "gold_n": 0,
        "abs_error_sum": 0,
        "pred_counts": Counter(),
        "gold_counts": Counter(),
        "z_values": [],
    }


def group_row(name: str, group: dict[str, Any], *, level_class_count: int) -> dict[str, Any]:
    n = int(group["n"])
    gold_n = int(group["gold_n"])
    z_values = [float(value) for value in group["z_values"]]
    row: dict[str, Any] = {
        "group": name,
        "n": n,
        "pred0_rate": group["pred0"] / n if n else None,
        "gold0_rate": group["gold0"] / gold_n if gold_n else None,
        "gold_gt0_pred0_rate": group["gold_gt0_pred0"] / gold_n if gold_n else None,
        "mean_pred_level": group["pred_sum"] / n if n else None,
        "mean_gold_level": group["gold_sum"] / gold_n if gold_n else None,
        "mae": group["abs_error_sum"] / gold_n if gold_n else None,
        "exact_match": group["exact"] / gold_n if gold_n else None,
        "z_mean": mean(z_values),
        "z_min": min(z_values) if z_values else None,
        "z_p25": quantile(z_values, 0.25),
        "z_p50": quantile(z_values, 0.50),
        "z_p75": quantile(z_values, 0.75),
        "z_max": max(z_values) if z_values else None,
    }
    for level in range(level_class_count):
        row[f"pred_level_{level}"] = int(group["pred_counts"].get(level, 0))
        row[f"gold_level_{level}"] = int(group["gold_counts"].get(level, 0))
    return row


def summarize_predictions(predictions: list[dict[str, Any]], *, level_class_count: int) -> dict[str, Any]:
    quartiles = {index: empty_group() for index in range(4)}
    lines: dict[int, dict[str, Any]] = defaultdict(empty_group)
    flat_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        gold_levels = [int(level) for level in prediction.get("gold_levels", [])]
        blocks = prediction.get("predicted_blocks") or []
        block_count = len(blocks)
        for block_offset, block in enumerate(blocks):
            pred = int(block["level"])
            line_index = int(block["line_index"])
            gold = gold_levels[line_index] if 0 <= line_index < len(gold_levels) else None
            z = block.get("ordinal_score")
            quartile = min(3, int(block_offset * 4 / max(block_count, 1)))
            update_group(quartiles[quartile], pred=pred, gold=gold, z=z)
            update_group(lines[line_index], pred=pred, gold=gold, z=z)
            if gold is not None and z is not None:
                flat_rows.append(
                    {
                        "line_position": float(line_index),
                        "pred_level": float(pred),
                        "gold_level": float(gold),
                        "z": float(z),
                    }
                )
    quartile_rows = [
        group_row(f"Q{index + 1}", quartiles[index], level_class_count=level_class_count)
        for index in range(4)
    ]
    line_rows = [
        {"line_index": line_index, **group_row(str(line_index), lines[line_index], level_class_count=level_class_count)}
        for line_index in sorted(lines)
    ]
    line_positions = [row["line_position"] for row in flat_rows]
    pred_levels = [row["pred_level"] for row in flat_rows]
    gold_levels = [row["gold_level"] for row in flat_rows]
    z_values = [row["z"] for row in flat_rows]
    correlations = {
        "n": len(flat_rows),
        "pearson_line_position_z": pearson(line_positions, z_values),
        "spearman_line_position_z": pearson(ranks(line_positions), ranks(z_values)) if flat_rows else None,
        "pearson_line_position_pred_level": pearson(line_positions, pred_levels),
        "pearson_line_position_gold_level": pearson(line_positions, gold_levels),
        "pearson_z_gold_level": pearson(z_values, gold_levels),
        "pearson_pred_gold_level": pearson(pred_levels, gold_levels),
    }
    return {
        "quartile_rows": quartile_rows,
        "line_rows": line_rows,
        "correlations": correlations,
    }


def checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    candidate = run_dir / "checkpoints" / checkpoint
    if candidate.exists():
        return candidate
    candidate = run_dir / checkpoint
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"checkpoint_not_found:{checkpoint}")


def torch_load_weights(path: Path, *, map_location: str = "cpu") -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ordinal_head_weight_balance(run_dir: Path, checkpoint: str, config: dict[str, Any]) -> dict[str, Any]:
    import torch

    state_path = checkpoint_path(run_dir, checkpoint) / "ordinal_level_head.pt"
    state = torch_load_weights(state_path, map_location="cpu")
    weight = state.get("final_projector.weight")
    payload: dict[str, Any] = {"checkpoint": checkpoint, "state_path": str(state_path)}
    if weight is not None:
        vector = weight.detach().float().reshape(-1)
        position_dim = int(config["ordinal_level_head"]["position_encoding"]["dim"])
        hidden_vector = vector[:-position_dim]
        position_vector = vector[-position_dim:]
        hidden_norm = float(torch.linalg.vector_norm(hidden_vector))
        position_norm = float(torch.linalg.vector_norm(position_vector))
        payload.update(
            {
                "final_projector_total_dim": int(vector.numel()),
                "final_projector_hidden_dim": int(hidden_vector.numel()),
                "final_projector_position_dim": int(position_vector.numel()),
                "final_projector_hidden_norm": hidden_norm,
                "final_projector_position_norm": position_norm,
                "final_projector_position_to_hidden_norm_ratio": (
                    position_norm / hidden_norm if hidden_norm else None
                ),
                "final_projector_hidden_abs_mean": float(hidden_vector.abs().mean()),
                "final_projector_position_abs_mean": float(position_vector.abs().mean()),
            }
        )
    if "raw_tau0" in state:
        raw_tau0 = state["raw_tau0"].detach().float()
        raw_deltas = state.get("raw_deltas")
        if raw_deltas is not None and raw_deltas.numel():
            thresholds = torch.cat(
                [raw_tau0.reshape(1), raw_tau0 + torch.cumsum(torch.nn.functional.softplus(raw_deltas.detach().float()), dim=0)]
            )
        else:
            thresholds = raw_tau0.reshape(1)
        payload["thresholds"] = [float(value) for value in thresholds.tolist()]
    return payload


def majority(counter: Counter[int], fallback: int = 0) -> int:
    if not counter:
        return fallback
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def collect_position_maps(rows: list[dict[str, Any]], *, bin_size: int) -> dict[str, Any]:
    by_abs: dict[int, Counter[int]] = defaultdict(Counter)
    by_bin: dict[int, Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()
    for row in rows:
        for line_position, block in enumerate(positional_sft.deserialize_training_blocks(str(row["target"]))):
            level = int(block.level)
            by_abs[line_position][level] += 1
            by_bin[line_position // bin_size][level] += 1
            global_counts[level] += 1
    global_majority = majority(global_counts)
    return {
        "absolute": {position: majority(counts, global_majority) for position, counts in by_abs.items()},
        "bin": {position_bin: majority(counts, global_majority) for position_bin, counts in by_bin.items()},
        "global_majority": global_majority,
    }


def evaluate_position_only(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    bin_size: int,
    level_class_count: int,
) -> dict[str, Any]:
    maps = collect_position_maps(train_rows, bin_size=bin_size)
    strategies = {
        "absolute_line_position": lambda line_position: maps["absolute"].get(line_position, maps["global_majority"]),
        "fixed_width_position_bin": lambda line_position: maps["bin"].get(line_position // bin_size, maps["global_majority"]),
    }
    payload: dict[str, Any] = {
        "bin_size": bin_size,
        "global_majority_level": maps["global_majority"],
    }
    for strategy_name, predictor in strategies.items():
        predictions = []
        for row in validation_rows:
            gold_levels = [
                int(block.level)
                for block in positional_sft.deserialize_training_blocks(str(row["target"]))
            ]
            pred_blocks = [
                {"line_index": index, "level": int(predictor(index))}
                for index, _level in enumerate(gold_levels)
            ]
            predictions.append({"gold_levels": gold_levels, "predicted_blocks": pred_blocks, "evaluation": {}})
        diagnostics = positional_sft.level_diagnostic_metrics(predictions, level_class_count=level_class_count)
        errors = []
        exact = 0
        total = 0
        pred0 = 0
        gold0 = 0
        quartiles = {index: empty_group() for index in range(4)}
        for prediction in predictions:
            gold_levels = prediction["gold_levels"]
            pred_blocks = prediction["predicted_blocks"]
            for index, block in enumerate(pred_blocks):
                pred = int(block["level"])
                gold = int(gold_levels[index])
                total += 1
                errors.append(abs(pred - gold))
                exact += int(pred == gold)
                pred0 += int(pred == 0)
                gold0 += int(gold == 0)
                quartile = min(3, int(index * 4 / max(len(gold_levels), 1)))
                update_group(quartiles[quartile], pred=pred, gold=gold, z=None)
        payload[strategy_name] = {
            "total_lines": total,
            "mae": mean([float(value) for value in errors]),
            "exact_match": exact / total if total else None,
            "pred0_rate": pred0 / total if total else None,
            "gold0_rate": gold0 / total if total else None,
            "quartiles": [
                group_row(f"Q{index + 1}", quartiles[index], level_class_count=level_class_count)
                for index in range(4)
            ],
            "level_diagnostics": diagnostics,
        }
    return payload


class PositionFeatureAblation:
    def __init__(self, head: Any, *, mode: str, seed: int) -> None:
        self.head = head
        self.mode = mode
        self.seed = seed
        self.original = None

    def __enter__(self) -> None:
        if self.mode == "normal":
            return
        import torch

        self.original = self.head.position_features
        generator = torch.Generator().manual_seed(int(self.seed))

        def patched(inner_self: Any, line_positions: Any, *, dtype: Any, device: Any) -> Any:
            if self.mode == "zero":
                return torch.zeros((*line_positions.shape, inner_self.position_dim), dtype=dtype, device=device)
            if self.mode == "shuffle":
                flat = line_positions.detach().cpu().reshape(-1)
                if flat.numel() > 1:
                    flat = flat[torch.randperm(flat.numel(), generator=generator)]
                shuffled = flat.reshape(tuple(line_positions.shape)).to(device=line_positions.device)
                return self.original(shuffled, dtype=dtype, device=device)
            raise ValueError(f"unsupported_ablation_mode:{self.mode}")

        self.head.position_features = MethodType(patched, self.head)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.original is not None:
            self.head.position_features = self.original


def model_args_from_config(config: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    head = config["ordinal_level_head"]
    position = head["position_encoding"]
    lora = config["lora"]
    return argparse.Namespace(
        base_model_path=Path(config["base_model_path"]),
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
        lora_r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        lora_target_modules=",".join(lora["target_modules"]),
        level_class_count=int(head["class_count"]),
        ordinal_head_hidden_dims=",".join(str(value) for value in head["hidden_dims"]),
        ordinal_head_dropouts=",".join(str(value) for value in head["dropouts"]),
        level_density_weights=list(head.get("density_weights", {}).get("weights", [1.0] * int(head["class_count"]))),
        lambda_level=float(head["lambda_level"]),
        initial_threshold_center=float(head["threshold_initialization"]["center"]),
        initial_threshold_gap=float(head["threshold_initialization"]["gap"]),
        ordinal_position_encoding=str(position["type"]),
        ordinal_position_dim=int(position["dim"]),
        ordinal_position_frequencies=",".join(str(value) for value in position["frequencies"]),
        ordinal_position_injection=str(position["injection"]),
    )


def load_checkpoint_for_ablation(config: dict[str, Any], args: argparse.Namespace, checkpoint_dir: Path) -> tuple[Any, Any]:
    import torch
    from peft import set_peft_model_state_dict

    tokenizer, model = positional_sft.load_model_and_tokenizer(model_args_from_config(config, args))
    adapter_path = checkpoint_dir / "adapter"
    if (adapter_path / "adapter_model.safetensors").exists():
        from safetensors.torch import load_file

        adapter_state = load_file(adapter_path / "adapter_model.safetensors")
    else:
        adapter_state = torch_load_weights(adapter_path / "adapter_model.bin", map_location="cpu")
    set_peft_model_state_dict(model.backbone, adapter_state, adapter_name="default")
    model.ordinal_level_head.load_state_dict(torch_load_weights(checkpoint_dir / "ordinal_level_head.pt", map_location="cpu"))
    model.eval()
    return tokenizer, model


def recompute_prediction_levels(
    *,
    prediction: dict[str, Any],
    row_by_unit_id: dict[str, dict[str, Any]],
    tokenizer: Any,
    model: Any,
) -> dict[str, Any]:
    unit_id = str(prediction["unit_id"])
    source_row = row_by_unit_id[unit_id]
    content_blocks = prediction.get("predicted_content_blocks") or []
    lightweight_blocks = [
        SimpleNamespace(
            document_index=int(block["document_index"]),
            line_index=int(block["line_index"]),
            line_text=str(block["line_text"]),
            level=0,
        )
        for block in content_blocks
    ]
    content_text, spans = positional_sft.build_content_text_and_spans(lightweight_blocks, include_level_labels=False)
    predicted_blocks = positional_sft.predict_levels_for_content(
        tokenizer=tokenizer,
        model=model,
        prompt=positional_sft.build_content_only_prompt(str(source_row["prompt"])),
        content_blocks=content_blocks,
        content_text=content_text,
        spans=spans,
    )
    evaluation = positional_sft.evaluate_blocks_prediction(
        str(source_row["target_yaml_normalized"]),
        predicted_blocks,
        prompt_text=str(source_row["prompt"]),
    )
    updated = dict(prediction)
    updated["predicted_blocks"] = predicted_blocks
    updated["evaluation"] = evaluation.to_dict()
    return updated


def run_model_ablations(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_path(args.run_dir, args.checkpoint)
    tokenizer, model = load_checkpoint_for_ablation(config, args, checkpoint_dir)
    row_by_unit_id = {positional_sft.build_unit_id(row): row for row in validation_rows}
    metrics_by_mode: dict[str, Any] = {}
    for mode in split_csv(args.ablation_modes):
        mode_predictions = []
        with PositionFeatureAblation(model.ordinal_level_head, mode=mode, seed=args.shuffle_seed):
            for prediction in predictions:
                mode_predictions.append(
                    recompute_prediction_levels(
                        prediction=prediction,
                        row_by_unit_id=row_by_unit_id,
                        tokenizer=tokenizer,
                        model=model,
                    )
                )
        for row in mode_predictions:
            row["checkpoint"] = f"{args.checkpoint}-{mode}"
            row["position_ablation"] = mode
        metrics = positional_sft.derive_validation_metrics(
            run_id=f"{args.run_dir.name}-audit",
            predictions=mode_predictions,
            checkpoint=f"{args.checkpoint}-{mode}",
        )
        metrics["position_ablation"] = mode
        write_jsonl(output_dir / f"ablation_{mode}_predictions.jsonl", mode_predictions)
        write_json(output_dir / f"ablation_{mode}_metrics.json", metrics)
        metrics_by_mode[mode] = metrics
    return metrics_by_mode


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "positional_v1_audit" / args.checkpoint
    predictions_path = args.predictions_path or run_dir / "intermediate_validation_predictions.jsonl"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    predictions = load_predictions(predictions_path, args.checkpoint, args.max_samples)
    train_rows = positional_sft.load_sft_rows(args.train_file, max_samples=None, split_name="train")
    validation_rows = positional_sft.load_sft_rows(args.validation_file, max_samples=None, split_name="validation")

    prediction_summary = summarize_predictions(predictions, level_class_count=args.level_class_count)
    quartile_fields = list(prediction_summary["quartile_rows"][0].keys())
    line_fields = list(prediction_summary["line_rows"][0].keys()) if prediction_summary["line_rows"] else []
    write_csv(output_dir / "quartile_level_z_summary.csv", prediction_summary["quartile_rows"], quartile_fields)
    if line_fields:
        write_csv(output_dir / "line_index_level_z_summary.csv", prediction_summary["line_rows"], line_fields)
    write_json(output_dir / "correlations.json", prediction_summary["correlations"])

    weight_balance = ordinal_head_weight_balance(run_dir, args.checkpoint, config)
    write_json(output_dir / "ordinal_head_weight_balance.json", weight_balance)

    position_only = evaluate_position_only(
        train_rows,
        validation_rows,
        bin_size=args.position_bin_size,
        level_class_count=args.level_class_count,
    )
    write_json(output_dir / "position_only_baseline.json", position_only)

    ablation_metrics = None
    if args.run_model_ablations:
        ablation_metrics = run_model_ablations(
            args=args,
            config=config,
            predictions=predictions,
            validation_rows=validation_rows,
            output_dir=output_dir,
        )

    write_json(
        output_dir / "audit_summary.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": args.checkpoint,
            "predictions_path": str(predictions_path),
            "prediction_count": len(predictions),
            "outputs": {
                "quartile_summary": str(output_dir / "quartile_level_z_summary.csv"),
                "line_index_summary": str(output_dir / "line_index_level_z_summary.csv"),
                "correlations": str(output_dir / "correlations.json"),
                "weight_balance": str(output_dir / "ordinal_head_weight_balance.json"),
                "position_only_baseline": str(output_dir / "position_only_baseline.json"),
            },
            "model_ablations_executed": bool(args.run_model_ablations),
            "model_ablation_metrics": ablation_metrics,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "model_ablations_executed": bool(args.run_model_ablations)}, indent=2))


if __name__ == "__main__":
    main()
