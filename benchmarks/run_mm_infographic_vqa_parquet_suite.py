import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import mm_infographic_vqa_baseline as baseline


SUMMARY_COLUMNS = [
    "variant",
    "status",
    "requested_compression",
    "writer_compression",
    "observed_compressions",
    "payload_format",
    "image_storage",
    "read_pattern",
    "read_block_size",
    "input_bytes",
    "input_mib",
    "ratio_vs_uncompressed",
    "ratio_vs_upstream",
    "dataset_load_time_s",
    "samples_per_s",
    "tokens_per_s",
    "data_time_fraction",
    "compute_time_fraction",
    "optimizer_time_fraction",
    "move_time_fraction",
    "data_time_mean_s",
    "compute_time_mean_s",
    "optimizer_time_mean_s",
    "loader_batches_per_s",
    "loader_batch_time_mean_s",
    "rows_profiled",
    "dataset_total_rows",
    "benchmark_json",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run loader benchmarks across prepared mm_infographic_vqa Parquet variants."
    )
    parser.add_argument("--manifest", default="data/variants/mm_infographic_vqa/manifest.json")
    parser.add_argument("--output-dir", default="results/mm_infographic_vqa_parquet_suite")
    parser.add_argument("--loader", choices=["pyarrow", "datasets", "rowpack"], default="pyarrow")
    parser.add_argument("--parquet-batch-size", type=int, default=32)
    parser.add_argument("--rowpack-native-dir", default=None)
    parser.add_argument("--rowpack-native-decode-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rowpack-direct-vqa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--read-pattern", choices=["sequential", "random_block"], default="sequential")
    parser.add_argument("--read-block-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--loader-benchmark-batches", type=int, default=0)
    parser.add_argument("--loader-benchmark-warmup-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--materialized", action="store_true", help="Disable streaming and benchmark via HF Arrow cache.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--variants", nargs="+", default=None)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in SUMMARY_COLUMNS})


def write_markdown(rows: list[dict[str, Any]], path: Path):
    columns = [
        "variant",
        "status",
        "requested_compression",
        "observed_compressions",
        "payload_format",
        "image_storage",
        "read_pattern",
        "input_mib",
        "ratio_vs_uncompressed",
        "dataset_load_time_s",
        "samples_per_s",
        "data_time_fraction",
        "loader_batches_per_s",
        "compute_time_fraction",
        "error",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def benchmark_variant(args, variant: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    name = variant["name"]
    if variant.get("status") != "ready":
        return with_read_pattern(
            row_from_manifest(variant, status=variant.get("status", "skipped"), error=variant.get("error")),
            args,
        )

    paths = [normalize_path(path) for path in variant.get("paths", [])]
    if not paths:
        return with_read_pattern(row_from_manifest(variant, status="error", error="No paths in manifest"), args)

    benchmark_json = output_dir / "runs" / f"{name}.json"
    benchmark_json.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and benchmark_json.exists():
        result = json.loads(benchmark_json.read_text(encoding="utf-8"))
    else:
        try:
            result = run_benchmark_in_process(args, name, paths, benchmark_json)
        except Exception as exc:
            return with_read_pattern(
                row_from_manifest(
                    variant,
                    status="benchmark_error",
                    benchmark_json=str(benchmark_json),
                    error=repr(exc),
                ),
                args,
            )

    summary = result["training"]["summary"]
    loader_summary = (result.get("loader_benchmark") or {}).get("summary") or {}
    storage = result["storage"]
    return with_read_pattern({
        **row_from_manifest(variant, status="ready", benchmark_json=str(benchmark_json)),
        "dataset_load_time_s": result.get("dataset_load_time_s"),
        "samples_per_s": summary.get("samples_per_s"),
        "tokens_per_s": summary.get("tokens_per_s"),
        "data_time_fraction": summary.get("data_time_fraction"),
        "compute_time_fraction": summary.get("compute_time_fraction"),
        "optimizer_time_fraction": summary.get("optimizer_time_fraction"),
        "move_time_fraction": summary.get("move_time_fraction"),
        "data_time_mean_s": summary.get("data_time_s", {}).get("mean"),
        "compute_time_mean_s": summary.get("compute_time_s", {}).get("mean"),
        "optimizer_time_mean_s": summary.get("optimizer_time_s", {}).get("mean"),
        "loader_batches_per_s": loader_summary.get("batches_per_s"),
        "loader_batch_time_mean_s": loader_summary.get("batch_time_s", {}).get("mean"),
        "rows_profiled": storage.get("rows_profiled"),
        "dataset_total_rows": storage.get("dataset_total_rows"),
    }, args)


def with_read_pattern(row: dict[str, Any], args) -> dict[str, Any]:
    row["read_pattern"] = args.read_pattern
    row["read_block_size"] = args.read_block_size
    return row


def run_benchmark_in_process(args, name: str, paths: list[str], benchmark_json: Path) -> dict[str, Any]:
    bench_args = SimpleNamespace(
        dataset="nimapourjafar/mm_infographic_vqa",
        split="train",
        data_files=paths,
        loader=args.loader,
        parquet_batch_size=args.parquet_batch_size,
        rowpack_native_dir=args.rowpack_native_dir,
        rowpack_native_decode_images=args.rowpack_native_decode_images,
        rowpack_direct_vqa=args.rowpack_direct_vqa,
        read_pattern=args.read_pattern,
        read_block_size=args.read_block_size,
        max_rows=args.max_rows,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        loader_benchmark_batches=args.loader_benchmark_batches,
        loader_benchmark_warmup_batches=args.loader_benchmark_warmup_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=args.shuffle,
        shuffle_buffer=args.shuffle_buffer,
        streaming=not args.materialized,
        seed=args.seed,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        max_images=args.max_images,
        lr=args.lr,
        skip_hub_size=True,
        size_path=[f"{name}={paths[0] if len(paths) == 1 else str(Path(paths[0]).parent)}"],
        output=str(benchmark_json),
    )

    dataset, dataset_label, load_time, total_rows = baseline.load_split(bench_args)
    base_cfg = baseline.VLMConfig()
    tokenizer = baseline.get_tokenizer(base_cfg.lm_tokenizer, base_cfg.vlm_extra_tokens, base_cfg.lm_chat_template)
    vlm_cfg = baseline.build_tiny_vlm_config(tokenizer, bench_args)
    loader = baseline.make_dataloader(dataset, tokenizer, vlm_cfg, bench_args)
    storage = baseline.dataset_storage_report(dataset, bench_args, dataset_label, total_rows)
    loader_metrics = baseline.benchmark_loader_only(loader, bench_args)
    if loader_metrics is not None:
        loader = baseline.make_dataloader(dataset, tokenizer, vlm_cfg, bench_args)
    train_metrics = baseline.train_smoke(loader, tokenizer, vlm_cfg, bench_args)

    result = {
        "config": {
            "dataset": bench_args.dataset,
            "split": bench_args.split,
            "data_files": bench_args.data_files,
            "loader": bench_args.loader,
            "parquet_batch_size": bench_args.parquet_batch_size,
            "rowpack_native_dir": bench_args.rowpack_native_dir,
            "rowpack_native_decode_images": bench_args.rowpack_native_decode_images,
            "rowpack_direct_vqa": bench_args.rowpack_direct_vqa,
            "read_pattern": bench_args.read_pattern,
            "read_block_size": bench_args.read_block_size,
            "max_rows": bench_args.max_rows,
            "steps": bench_args.steps,
            "warmup_steps": bench_args.warmup_steps,
            "loader_benchmark_batches": bench_args.loader_benchmark_batches,
            "loader_benchmark_warmup_batches": bench_args.loader_benchmark_warmup_batches,
            "batch_size": bench_args.batch_size,
            "num_workers": bench_args.num_workers,
            "prefetch_factor": bench_args.prefetch_factor,
            "shuffle": bench_args.shuffle,
            "shuffle_buffer": bench_args.shuffle_buffer,
            "streaming": bench_args.streaming,
            "sequence_length": bench_args.sequence_length,
            "image_size": bench_args.image_size,
            "max_images": bench_args.max_images,
            "seed": bench_args.seed,
        },
        "dataset_load_time_s": load_time,
        "storage": storage,
        "loader_benchmark": loader_metrics,
        "training": train_metrics,
    }
    benchmark_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def row_from_manifest(
    variant: dict[str, Any],
    status: str,
    benchmark_json: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    input_bytes = variant.get("bytes")
    return {
        "variant": variant.get("name"),
        "status": status,
        "requested_compression": variant.get("requested_compression"),
        "writer_compression": variant.get("writer_compression"),
        "observed_compressions": ", ".join(variant.get("observed_compressions", [])),
        "payload_format": variant.get("payload_format"),
        "image_storage": variant.get("image_storage"),
        "read_pattern": None,
        "read_block_size": None,
        "input_bytes": input_bytes,
        "input_mib": input_bytes / (1024 * 1024) if input_bytes else None,
        "ratio_vs_uncompressed": variant.get("ratio_vs_uncompressed"),
        "ratio_vs_upstream": variant.get("ratio_vs_upstream"),
        "benchmark_json": benchmark_json,
        "error": error,
    }


def normalize_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve().as_posix()


def make_charts(rows: list[dict[str, Any]], charts_dir: Path):
    ready = [row for row in rows if row.get("status") == "ready" and row.get("samples_per_s") is not None]
    if not ready:
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    def save_bar(key: str, title: str, ylabel: str, filename: str, transform=None, label_format="{:.2f}"):
        labels = [row["variant"] for row in ready]
        values = [row.get(key) or 0 for row in ready]
        if transform is not None:
            values = [transform(value) for value in values]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 4.8))
        bars = ax.bar(labels, values, color=[bar_color(label) for label in labels])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        apply_zoomed_ylim(ax, values)
        label_bars(ax, bars, values, label_format)
        fig.tight_layout()
        path = charts_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        chart_paths.append(str(path))

    save_bar("input_mib", "Dataset Variant File Size", "MiB", "file_size_mib.png", label_format="{:.1f}")
    save_bar(
        "dataset_load_time_s",
        "Dataset Construction Time",
        "milliseconds",
        "dataset_load_time_s.png",
        transform=lambda value: value * 1000,
        label_format="{:.2f}",
    )
    save_bar("samples_per_s", "Training Throughput", "samples/s", "samples_per_s.png", label_format="{:.2f}")
    if any(row.get("loader_batches_per_s") is not None for row in ready):
        save_bar("loader_batches_per_s", "Loader-Only Throughput", "batches/s", "loader_batches_per_s.png", label_format="{:.2f}")
    save_bar("data_time_fraction", "Fraction Waiting On Data", "fraction", "data_time_fraction.png", label_format="{:.3f}")

    labels = [row["variant"] for row in ready]
    stacks = [
        ("data", "data_time_fraction", "#376da8"),
        ("compute", "compute_time_fraction", "#58a55c"),
        ("optimizer", "optimizer_time_fraction", "#d89b34"),
    ]
    move_values = [row.get("move_time_fraction") or 0 for row in ready]
    if max(move_values, default=0) >= 0.01:
        stacks.append(("move", "move_time_fraction", "#8f6bb3"))
    bottoms = [0.0] * len(ready)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 5.2))
    for label, key, color in stacks:
        values = [row.get(key) or 0 for row in ready]
        ax.bar(labels, values, bottom=bottoms, label=label, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title("Measured Step Time Fractions")
    ax.set_ylabel("fraction")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = charts_dir / "step_time_fractions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    chart_paths.append(str(path))

    return chart_paths


def bar_color(label: str) -> str:
    normalized = label.lower()
    if normalized == "rowpack_lzav_hi" or normalized.endswith("_lzav_hi"):
        return "#f28e2b"
    return "#376da8"


def apply_zoomed_ylim(ax, values: list[float]):
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return

    min_value = min(finite_values)
    max_value = max(finite_values)
    if min_value == max_value:
        margin = max(abs(min_value) * 0.1, 0.01)
    else:
        margin = (max_value - min_value) * 0.3

    lower = min_value - margin
    upper = max_value + margin
    if min_value >= 0:
        lower = max(0, lower)
        if lower == 0 and min_value > 0:
            lower = max(0, min_value * 0.9)
    ax.set_ylim(lower, upper)


def label_bars(ax, bars, values: list[float], label_format: str):
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin
    for bar, value in zip(bars, values):
        if value is None:
            continue
        label = label_format.format(value)
        y = value - y_span * 0.05
        va = "top"
        if y < ymin + y_span * 0.03:
            y = value + y_span * 0.03
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            rotation=0,
            fontsize=8,
            color="white" if va == "top" else "black",
        )


def main():
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.variants) if args.variants else None

    rows = []
    for variant in manifest["variants"]:
        if selected is not None and variant["name"] not in selected:
            continue
        print(f"Benchmarking {variant['name']} ({variant.get('status')})...", flush=True)
        rows.append(benchmark_variant(args, variant, output_dir))

    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    charts = make_charts(rows, output_dir / "charts")

    print(json.dumps({
        "summary_csv": str(csv_path),
        "summary_markdown": str(md_path),
        "charts": charts,
        "ready_variants": [row["variant"] for row in rows if row.get("status") == "ready"],
        "non_ready": {row["variant"]: row.get("status") for row in rows if row.get("status") != "ready"},
    }, indent=2))


if __name__ == "__main__":
    main()
