import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COLUMNS = [
    "label",
    "loader",
    "payload_format",
    "input_mib",
    "dataset_load_time_ms",
    "samples_per_s",
    "data_time_fraction",
    "compute_time_fraction",
    "data_time_mean_ms",
    "compute_time_mean_ms",
    "skipped_empty_batches",
    "benchmark_json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare selected benchmark JSON outputs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Benchmark JSON to include, e.g. parquet_uncompressed=results/foo/runs/uncompressed.json",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_row(label: str, path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["training"]["summary"]
    config = data["config"]
    storage = data["storage"]
    extra_sizes = storage.get("extra_size_paths") or {}
    input_bytes = next(iter(extra_sizes.values())).get("bytes") if extra_sizes else None

    return {
        "label": label,
        "loader": config.get("loader"),
        "payload_format": payload_format_from_json(data),
        "input_mib": input_bytes / (1024 * 1024) if input_bytes else None,
        "dataset_load_time_ms": data.get("dataset_load_time_s", 0.0) * 1000,
        "samples_per_s": summary.get("samples_per_s"),
        "data_time_fraction": summary.get("data_time_fraction"),
        "compute_time_fraction": summary.get("compute_time_fraction"),
        "data_time_mean_ms": summary.get("data_time_s", {}).get("mean", 0.0) * 1000,
        "compute_time_mean_ms": summary.get("compute_time_s", {}).get("mean", 0.0) * 1000,
        "skipped_empty_batches": data["training"].get("skipped_empty_batches"),
        "benchmark_json": str(path),
    }


def payload_format_from_json(data: dict[str, Any]) -> str:
    data_files = data.get("config", {}).get("data_files") or []
    if not data_files:
        return ""
    first = Path(data_files[0])
    if first.suffix != ".rowpack" or not first.exists():
        return ""
    try:
        from rowpack.format import HEADER_SIZE, unpack_header

        with first.open("rb") as handle:
            header = unpack_header(handle.read(HEADER_SIZE))
            handle.seek(header.metadata_offset)
            metadata = json.loads(handle.read(header.metadata_size).decode("utf-8"))
        return metadata.get("payload_format", "")
    except Exception:
        return ""


def write_csv(rows: list[dict[str, Any]], path: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})


def write_markdown(rows: list[dict[str, Any]], path: Path):
    lines = [
        "| " + " | ".join(COLUMNS[:-1]) + " |",
        "| " + " | ".join(["---"] * (len(COLUMNS) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in COLUMNS[:-1]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def make_charts(rows: list[dict[str, Any]], output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    def save_bar(key: str, title: str, ylabel: str, filename: str, label_format: str):
        labels = [row["label"] for row in rows]
        values = [row.get(key) or 0 for row in rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
        bars = ax.bar(labels, values, color=[bar_color(label) for label in labels])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        apply_zoomed_ylim(ax, values)
        label_bars(ax, bars, values, label_format)
        fig.tight_layout()
        path = chart_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        chart_paths.append(str(path))

    save_bar("samples_per_s", "Training Throughput", "samples/s", "samples_per_s.png", "{:.2f}")
    save_bar("data_time_mean_ms", "Mean Data Wait", "milliseconds", "data_time_mean_ms.png", "{:.2f}")
    save_bar("dataset_load_time_ms", "Dataset Construction Time", "milliseconds", "dataset_load_time_ms.png", "{:.2f}")
    save_bar("input_mib", "Input File Size", "MiB", "input_mib.png", "{:.1f}")
    return chart_paths


def bar_color(label: str) -> str:
    normalized = label.lower()
    if normalized == "rowpack_lzav_hi" or normalized.endswith("_lzav_hi"):
        return "#f28e2b"
    return "#376da8"


def apply_zoomed_ylim(ax, values: list[float]):
    if not values:
        return
    min_value = min(values)
    max_value = max(values)
    margin = max((max_value - min_value) * 0.3, max_value * 0.03, 0.01)
    lower = max(0, min_value - margin)
    upper = max_value + margin
    if lower == 0 and min_value > 0:
        lower = min_value * 0.9
    ax.set_ylim(lower, upper)


def label_bars(ax, bars, values: list[float], label_format: str):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar, value in zip(bars, values):
        label = label_format.format(value)
        y = value - span * 0.05
        va = "top"
        if y < ymin + span * 0.03:
            y = value + span * 0.03
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in args.run:
        label, sep, raw_path = item.partition("=")
        if not sep:
            raise ValueError(f"--run must be LABEL=PATH, got {item!r}")
        rows.append(load_row(label, Path(raw_path)))

    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    charts = make_charts(rows, output_dir)
    print(json.dumps({"summary_csv": str(csv_path), "summary_markdown": str(md_path), "charts": charts}, indent=2))


if __name__ == "__main__":
    main()
