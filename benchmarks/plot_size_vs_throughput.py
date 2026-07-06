"""Render a combined File Size + Training Throughput mega chart from baseline JSONs.

The chart stacks two panels sharing the x-axis (one bar per variant):
  * Top panel:    training throughput (samples/s, linear).
  * Bottom panel: file size (MiB, log scale, inverted so bars grow downward).

Numeric labels are placed just outside each bar so the absolute value can be read
at a glance despite the log scale on the size panel.

Inputs are produced by ``benchmarks/mm_infographic_vqa_baseline.py`` (the same
JSON files consumed by ``benchmarks/compare_benchmark_runs.py``).

Example::

    python benchmarks/plot_size_vs_throughput.py \\
        --output-dir results/mm_infographic_vqa_comparison \\
        --run parquet_jpeg_uncompressed=path/to/uncompressed.json \\
        --run rowpack_jpeg_cista_lzav_hi=path/to/rowpack_cista.json \\
        ...

Use ``--group LABEL=COLOR`` to override the default per-group color palette.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Default palette: blue family = Parquet, orange family = RowPack.
# Each image encoding gets a distinct shade so the three "compression columns"
# within a group are visually grouped while still showing a clear contrast
# between storage families.
DEFAULT_GROUP_COLORS = {
    "parquet_rgb": "#a6cee3",
    "parquet_png": "#1f78b4",
    "parquet_jpeg": "#08306b",
    "rowpack_rgb": "#fdbf6f",
    "rowpack_qoi": "#ff7f00",
    "rowpack_jpeg": "#8c2d04",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Add one benchmark JSON to the chart. May be repeated; order is preserved.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="GROUP=COLOR",
        help="Override the color for a group prefix (e.g. parquet_jpeg=#08306b).",
    )
    parser.add_argument(
        "--title",
        default="Parquet vs RowPack: File Size and Training Throughput",
    )
    parser.add_argument("--filename", default="size_vs_throughput.png")
    parser.add_argument("--figure-width", type=float, default=14.0)
    parser.add_argument("--figure-height", type=float, default=8.5)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_metrics(label: str, path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["training"]["summary"]
    storage = data["storage"]
    extra_sizes = storage.get("extra_size_paths") or {}
    size_entry = extra_sizes.get(label)
    if not isinstance(size_entry, dict):
        raise ValueError(
            f"{path} does not contain a storage.extra_size_paths entry for {label!r}; "
            f"available entries are {sorted(extra_sizes)}"
        )
    input_bytes = size_entry.get("bytes")
    size_path = size_entry.get("path")
    if size_path and Path(size_path).exists():
        actual_bytes = Path(size_path).stat().st_size
        if input_bytes != actual_bytes:
            raise ValueError(
                f"{path} has a stale size for {label!r}: "
                f"{input_bytes!r} recorded, {actual_bytes!r} on disk at {size_path}"
            )
    return {
        "label": label,
        "samples_per_s": summary.get("samples_per_s"),
        "input_mib": input_bytes / (1024 * 1024) if input_bytes else None,
        "data_time_mean_ms": summary.get("data_time_s", {}).get("mean", 0.0) * 1000,
        "dataset_load_time_ms": data.get("dataset_load_time_s", 0.0) * 1000,
        "benchmark_json": str(path),
    }


def color_for_label(label: str, palette: dict[str, str]) -> str:
    # Match the longest registered prefix so e.g. "parquet_jpeg_gzip" picks
    # "parquet_jpeg" rather than a hypothetical "parquet" entry.
    matches = [prefix for prefix in palette if label.startswith(prefix)]
    if not matches:
        return "#888888"
    return palette[max(matches, key=len)]


def parse_overrides(overrides: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in overrides:
        prefix, sep, color = item.partition("=")
        if not sep:
            raise ValueError(f"--group must be PREFIX=COLOR, got {item!r}")
        result[prefix] = color
    return result


def make_chart(rows: list[dict[str, Any]], output_path: Path, *, title: str,
               palette: dict[str, str], figure_width: float, figure_height: float, dpi: int):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels = [row["label"] for row in rows]
    throughputs = [row["samples_per_s"] or 0.0 for row in rows]
    sizes_mib = [row["input_mib"] or 0.0 for row in rows]
    colors = [color_for_label(label, palette) for label in labels]
    positions = list(range(len(labels)))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        sharex=True,
        figsize=(figure_width, figure_height),
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.04},
    )

    # ---- Top: throughput (linear, bars grow upward) ----
    top_bars = ax_top.bar(positions, throughputs, color=colors, edgecolor="white", linewidth=0.5)
    ax_top.set_ylabel("Training Throughput\n(samples / s)")
    ax_top.set_ylim(0, max(throughputs) * 1.18 if throughputs else 1)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.axhline(0, color="black", linewidth=0.8)
    ax_top.set_title(title, fontsize=12)

    span_top = ax_top.get_ylim()[1] - ax_top.get_ylim()[0]
    for bar, value in zip(top_bars, throughputs):
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            value + span_top * 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ---- Bottom: file size (log, bars grow downward) ----
    # We plot positive sizes on a log y-axis and then invert the axis so larger
    # values appear lower on the figure; this is visually equivalent to the
    # "negative log" the request asks for without matplotlib's symlog quirks.
    safe_sizes = [max(value, 1e-3) for value in sizes_mib]  # log can't take zero
    bottom_bars = ax_bottom.bar(
        positions, safe_sizes, color=colors, edgecolor="white", linewidth=0.5
    )
    ax_bottom.set_yscale("log")
    ax_bottom.invert_yaxis()
    ax_bottom.set_ylabel("File Size\n(MiB, log scale, grows down)")
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)
    ax_bottom.spines["bottom"].set_visible(False)
    ax_bottom.axhline(safe_sizes and min(safe_sizes) or 1, color="white", linewidth=0)
    ax_bottom.set_xticks(positions)
    ax_bottom.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax_bottom.tick_params(axis="x", which="both", top=True, labeltop=False)

    if safe_sizes:
        max_size = max(safe_sizes)
        min_size = min(safe_sizes)
        # Pad limits one decade beyond the data on each side so labels fit
        # comfortably even after the y-axis inversion.
        ax_bottom.set_ylim(top=min_size * 0.6, bottom=max_size * 2.5)

    for bar, value in zip(bottom_bars, sizes_mib):
        if value <= 0:
            continue
        # Bar grows downward on an inverted log axis; place the label just past
        # the bar end (i.e. below it in screen space).
        ax_bottom.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.18,
            f"{value:,.0f} MiB" if value >= 10 else f"{value:.2f} MiB",
            ha="center",
            va="top",
            fontsize=8,
        )

    # ---- Legend (group -> color) ----
    seen_groups: dict[str, str] = {}
    for label, color in zip(labels, colors):
        matches = [prefix for prefix in palette if label.startswith(prefix)]
        if not matches:
            continue
        group = max(matches, key=len)
        if group not in seen_groups:
            seen_groups[group] = color
    legend_handles = [Patch(facecolor=color, label=group) for group, color in seen_groups.items()]
    if legend_handles:
        ax_top.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(len(legend_handles), 6),
            frameon=False,
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    palette = dict(DEFAULT_GROUP_COLORS)
    palette.update(parse_overrides(args.group))

    rows: list[dict[str, Any]] = []
    for item in args.run:
        label, sep, raw_path = item.partition("=")
        if not sep:
            raise ValueError(f"--run must be LABEL=PATH, got {item!r}")
        rows.append(load_metrics(label, Path(raw_path)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / args.filename
    make_chart(
        rows,
        chart_path,
        title=args.title,
        palette=palette,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
    )

    csv_path = output_dir / (Path(args.filename).stem + ".csv")
    import csv
    fieldnames = ["label", "samples_per_s", "input_mib", "data_time_mean_ms",
                  "dataset_load_time_ms", "benchmark_json"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(json.dumps({"chart": str(chart_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
