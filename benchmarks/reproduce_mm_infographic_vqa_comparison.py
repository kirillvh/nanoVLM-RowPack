"""Reproducible "Parquet vs RowPack" benchmark for nimapourjafar/mm_infographic_vqa.

This is the canonical RowPack benchmark methodology. Running this script with
its defaults will:

  1. Generate 9 Parquet variants (3 image encodings x 3 container codecs):
       parquet_jpeg_{uncompressed,gzip,brotli}
       parquet_png_{uncompressed,gzip,brotli}
       parquet_rgb_{uncompressed,gzip,brotli}
  2. Generate 6 RowPack variants (3 image storages x 2 payload formats),
     all using the LZAV_HI block codec for an apples-to-apples comparison:
       rowpack_jpeg_json_lzav_hi      rowpack_jpeg_cista_lzav_hi
       rowpack_qoi_json_lzav_hi       rowpack_qoi_cista_lzav_hi
       rowpack_rgb_json_lzav_hi       rowpack_rgb_cista_lzav_hi
  3. Run an identical training-smoke benchmark (nanoVLM tiny config) on each
     variant via ``benchmarks/mm_infographic_vqa_baseline.py``.
  4. Plot the combined File Size + Training Throughput mega chart via
     ``benchmarks/plot_size_vs_throughput.py``.

Each step can be skipped independently with the ``--skip-*`` flags so you can
re-iterate on charting without re-running the (expensive) benchmark step.

Disk requirements: the 15 variants together can exceed 20 GiB for the full
2,118-row dataset. Use ``--max-rows`` to subset for a quick smoke test.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# (group_label, image_encoding, container_compression)
#
# NOTE: The mm_infographic_vqa source already stores images as JPEG bytes, so
# the "parquet_jpeg" group uses image_encoding="source" (zero-copy) to compare
# the canonical JPEG payload exactly as-is. The "png" and "rgb" groups re-encode
# every image via PIL.
PARQUET_PLAN = [
    ("parquet_jpeg", "source", "uncompressed"),
    ("parquet_jpeg", "source", "gzip"),
    ("parquet_jpeg", "source", "brotli"),
    ("parquet_png", "png", "uncompressed"),
    ("parquet_png", "png", "gzip"),
    ("parquet_png", "png", "brotli"),
    ("parquet_rgb", "rgb", "uncompressed"),
    ("parquet_rgb", "rgb", "gzip"),
    ("parquet_rgb", "rgb", "brotli"),
]

# (group_label, image_storage, payload_format)
ROWPACK_PLAN = [
    ("rowpack_jpeg", "encoded", "json"),
    ("rowpack_jpeg", "encoded", "cista"),
    ("rowpack_qoi", "qoi_lossless", "json"),
    ("rowpack_qoi", "qoi_lossless", "cista"),
    ("rowpack_rgb", "raw_rgb", "json"),
    ("rowpack_rgb", "raw_rgb", "cista"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="results/mm_infographic_vqa_comparison",
                        help="Top-level results directory.")
    parser.add_argument("--data-root", default="data/variants/mm_infographic_vqa_comparison",
                        help="Top-level data directory for generated variants.")
    parser.add_argument("--dataset", default="nimapourjafar/mm_infographic_vqa")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--source-parquet",
        nargs="+",
        default=None,
        help="Optional local Parquet source files for preparing all Parquet and RowPack variants.",
    )
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap variants to this many rows (default: full dataset).")
    parser.add_argument("--bench-max-rows", type=int, default=256,
                        help="Rows scanned by the benchmark inner loop.")
    parser.add_argument("--bench-steps", type=int, default=32,
                        help="Training-smoke steps measured per variant.")
    parser.add_argument("--bench-warmup-steps", type=int, default=4)
    parser.add_argument("--bench-batch-size", type=int, default=1)
    parser.add_argument("--bench-num-workers", type=int, default=0)
    parser.add_argument("--bench-sequence-length", type=int, default=128)
    parser.add_argument("--bench-image-size", type=int, default=32)
    parser.add_argument("--read-pattern", choices=["sequential", "random_block"], default="random_block")
    parser.add_argument("--read-block-size", type=int, default=16)
    parser.add_argument("--rows-per-block", type=int, default=64,
                        help="RowPack rows per compression block.")
    parser.add_argument("--row-group-size", type=int, default=64,
                        help="Parquet rows per row group.")
    parser.add_argument("--rowpack-native-dir", default="rowpack_build_py")
    parser.add_argument(
        "--rowpack-native-decode-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For CISTA RowPack rows, decode supported image payloads inside "
            "rowpack_native before returning them to Python. The default keeps "
            "stored bytes so the benchmark matches the generic Python image path."
        ),
    )
    parser.add_argument("--block-codec", default="lzav_hi", help="RowPack block codec.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true",
                        help="Force regeneration of existing variants.")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip variant generation, reuse existing files.")
    parser.add_argument("--skip-bench", action="store_true",
                        help="Skip benchmarking, reuse existing JSON outputs.")
    parser.add_argument("--skip-chart", action="store_true",
                        help="Skip the final mega chart.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="If set, only run the named variants (e.g. parquet_jpeg_gzip).")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def expected_size_path(data_files: list[Path]) -> Path:
    return data_files[0] if len(data_files) == 1 else data_files[0].parent


def run_json_matches(
    output_path: Path,
    *,
    variant_label: str,
    loader: str,
    data_files: list[Path],
    args,
) -> tuple[bool, str]:
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"could not read JSON: {exc!r}"

    config = data.get("config") or {}
    if config.get("loader") != loader:
        return False, f"loader mismatch: {config.get('loader')!r} != {loader!r}"

    actual_files = config.get("data_files") or []
    if len(actual_files) != len(data_files):
        return False, f"data file count mismatch: {len(actual_files)} != {len(data_files)}"
    for actual, expected in zip(actual_files, data_files):
        if not same_path(actual, expected):
            return False, f"data file mismatch: {actual!r} != {str(expected)!r}"

    expected_config = {
        "read_pattern": args.read_pattern,
        "read_block_size": args.read_block_size,
        "max_rows": args.bench_max_rows,
        "steps": args.bench_steps,
        "warmup_steps": args.bench_warmup_steps,
        "batch_size": args.bench_batch_size,
        "num_workers": args.bench_num_workers,
        "sequence_length": args.bench_sequence_length,
        "image_size": args.bench_image_size,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            return False, f"config {key} mismatch: {config.get(key)!r} != {expected!r}"

    storage = data.get("storage") or {}
    extra_sizes = storage.get("extra_size_paths") or {}
    size_entry = extra_sizes.get(variant_label)
    if not isinstance(size_entry, dict):
        return False, f"missing size entry for {variant_label!r}"

    expected_path = expected_size_path(data_files)
    actual_size_path = size_entry.get("path")
    if actual_size_path is None or not same_path(actual_size_path, expected_path):
        return False, f"size path mismatch: {actual_size_path!r} != {str(expected_path)!r}"
    if expected_path.exists() and size_entry.get("bytes") != expected_path.stat().st_size:
        return False, (
            f"size mismatch for {expected_path}: "
            f"{size_entry.get('bytes')!r} != {expected_path.stat().st_size!r}"
        )

    return True, "ok"


def require_rowpack_native_symbols(args, planned_variants: set[str]) -> None:
    required: set[str] = set()
    for _group_label, image_storage, payload_format in ROWPACK_PLAN:
        variant = rowpack_variant_name(image_storage, payload_format, args.block_codec)
        if variant not in planned_variants:
            continue
        if args.block_codec != "none":
            required.update({"lzav_compress", "lzav_decompress"})
        if payload_format == "cista":
            required.update({"encode_cista_payload", "decode_cista_vqa_payload"})
        if image_storage == "qoi_lossless":
            required.update({"qoi_encode_rgb", "qoi_decode_rgb"})

    if not required:
        return

    from rowpack.native import load_native

    native = load_native(args.rowpack_native_dir)
    missing = sorted(name for name in required if not hasattr(native, name))
    if missing:
        native_path = getattr(native, "__file__", "<unknown>")
        raise SystemExit(
            "rowpack_native is missing symbols required by this benchmark: "
            f"{missing}\n"
            f"Loaded native module: {native_path}\n"
            "This usually means the CMake build directory has a stale cache or "
            "was configured without the bundled third-party include directories. "
            "Rerun benchmarks/run_mega_benchmark.sh or delete the build directory "
            "and configure RowPack again."
        )


def parquet_variant_name(group_label: str, container: str) -> str:
    # group_label already encodes the image format (e.g. "parquet_jpeg"), which
    # may differ from the underlying image_encoding flag (e.g. "source" for the
    # jpeg group, since the source already is JPEG and we want a zero-copy bench).
    return f"{group_label}_{container}"


def rowpack_variant_name(image_storage: str, payload_format: str, block_codec: str) -> str:
    image_label = {"encoded": "jpeg", "raw_rgb": "rgb", "qoi_lossless": "qoi"}[image_storage]
    return f"rowpack_{image_label}_{payload_format}_{block_codec}"


def prepare_parquet_group(args, group_label: str, image_encoding: str, containers: list[str]) -> None:
    out_dir = Path(args.data_root) / group_label
    cmd = [
        args.python, "benchmarks/prepare_mm_infographic_vqa_variants.py",
        "--dataset", args.dataset,
        "--split", args.split,
        "--output-dir", str(out_dir),
        "--image-encoding", image_encoding,
        "--row-group-size", str(args.row_group_size),
        "--variants", *containers,
    ]
    if args.source_parquet:
        cmd += ["--data-files", *args.source_parquet]
    if args.max_rows is not None:
        cmd += ["--max-rows", str(args.max_rows)]
    if args.overwrite:
        cmd += ["--overwrite"]
    run(cmd)

    manifest_path = out_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read Parquet prepare manifest for {group_label}: {manifest_path}: {exc!r}")

    variants = {variant.get("name"): variant for variant in manifest.get("variants", [])}
    missing_or_failed = {}
    for container in containers:
        variant = variants.get(container)
        if variant is None:
            missing_or_failed[container] = "missing from manifest"
        elif variant.get("status") != "ready":
            missing_or_failed[container] = variant.get("error") or variant.get("status")
        elif not parquet_path(args, group_label, container).exists():
            missing_or_failed[container] = f"output file missing: {parquet_path(args, group_label, container)}"
    if missing_or_failed:
        raise RuntimeError(f"Parquet prepare failed for {group_label}: {missing_or_failed}")


def prepare_rowpack_variant(args, group_label: str, image_storage: str, payload_format: str) -> Path:
    variant = rowpack_variant_name(image_storage, payload_format, args.block_codec)
    out_dir = Path(args.data_root) / group_label
    cmd = [
        args.python, "benchmarks/prepare_mm_infographic_vqa_rowpack.py",
        "--dataset", args.dataset,
        "--split", args.split,
        "--output-dir", str(out_dir),
        "--variant-name", variant,
        "--payload-format", payload_format,
        "--image-storage", image_storage,
        "--block-codec", args.block_codec,
        "--rows-per-block", str(args.rows_per_block),
        "--rowpack-native-dir", args.rowpack_native_dir,
    ]
    if args.source_parquet:
        cmd += ["--data-files", *args.source_parquet]
    if args.max_rows is not None:
        cmd += ["--max-rows", str(args.max_rows)]
    if args.overwrite:
        cmd += ["--overwrite"]
    run(cmd)
    return out_dir / f"{variant}.rowpack"


def parquet_path(args, group_label: str, container: str) -> Path:
    return Path(args.data_root) / group_label / f"{container}.parquet"


def variant_loader_and_files(args, variant: str) -> tuple[str, list[Path]]:
    for group_label, _image_encoding, container in PARQUET_PLAN:
        if variant == parquet_variant_name(group_label, container):
            return "pyarrow", [parquet_path(args, group_label, container)]
    for group_label, image_storage, payload_format in ROWPACK_PLAN:
        expected = rowpack_variant_name(image_storage, payload_format, args.block_codec)
        if variant == expected:
            return "rowpack", [Path(args.data_root) / group_label / f"{variant}.rowpack"]
    raise KeyError(variant)


def benchmark_variant(args, variant_label: str, loader: str, data_files: list[Path]) -> Path:
    runs_dir = Path(args.root) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_path = runs_dir / f"{variant_label}.json"
    if output_path.exists() and not args.overwrite:
        matches, reason = run_json_matches(
            output_path,
            variant_label=variant_label,
            loader=loader,
            data_files=data_files,
            args=args,
        )
        if matches:
            print(f"  reuse {output_path}")
            return output_path
        print(f"  stale {output_path}: {reason}; rerunning")

    cmd = [
        args.python, "benchmarks/mm_infographic_vqa_baseline.py",
        "--dataset", args.dataset,
        "--split", args.split,
        "--loader", loader,
        "--read-pattern", args.read_pattern,
        "--read-block-size", str(args.read_block_size),
        "--max-rows", str(args.bench_max_rows),
        "--steps", str(args.bench_steps),
        "--warmup-steps", str(args.bench_warmup_steps),
        "--batch-size", str(args.bench_batch_size),
        "--num-workers", str(args.bench_num_workers),
        "--sequence-length", str(args.bench_sequence_length),
        "--image-size", str(args.bench_image_size),
        "--output", str(output_path),
        "--size-path", f"{variant_label}={data_files[0] if len(data_files) == 1 else data_files[0].parent}",
        "--data-files", *[str(p) for p in data_files],
    ]
    if loader == "rowpack":
        cmd += ["--rowpack-native-dir", args.rowpack_native_dir]
        if args.rowpack_native_decode_images:
            cmd += ["--rowpack-native-decode-images"]
    run(cmd)
    return output_path


def step_prepare(args, planned_variants: set[str]) -> None:
    # Group Parquet variants by image encoding so we only re-encode the source
    # table once per group.
    parquet_groups: dict[tuple[str, str], list[str]] = {}
    for group_label, image_encoding, container in PARQUET_PLAN:
        if parquet_variant_name(group_label, container) not in planned_variants:
            continue
        parquet_groups.setdefault((group_label, image_encoding), []).append(container)

    for (group_label, image_encoding), containers in parquet_groups.items():
        print(f"[prepare] {group_label}: {containers}")
        prepare_parquet_group(args, group_label, image_encoding, containers)

    for group_label, image_storage, payload_format in ROWPACK_PLAN:
        variant = rowpack_variant_name(image_storage, payload_format, args.block_codec)
        if variant not in planned_variants:
            continue
        print(f"[prepare] {variant}")
        prepare_rowpack_variant(args, group_label, image_storage, payload_format)


def step_benchmark(args, planned_variants: set[str]) -> dict[str, Path]:
    run_paths: dict[str, Path] = {}
    for group_label, image_encoding, container in PARQUET_PLAN:
        variant = parquet_variant_name(group_label, container)
        if variant not in planned_variants:
            continue
        path = parquet_path(args, group_label, container)
        if not path.exists():
            raise FileNotFoundError(f"Prepared Parquet variant is missing for {variant}: {path}")
        print(f"[bench] {variant} (parquet)")
        run_paths[variant] = benchmark_variant(args, variant, "pyarrow", [path])

    for group_label, image_storage, payload_format in ROWPACK_PLAN:
        variant = rowpack_variant_name(image_storage, payload_format, args.block_codec)
        if variant not in planned_variants:
            continue
        path = Path(args.data_root) / group_label / f"{variant}.rowpack"
        if not path.exists():
            raise FileNotFoundError(f"Prepared RowPack variant is missing for {variant}: {path}")
        print(f"[bench] {variant} (rowpack)")
        run_paths[variant] = benchmark_variant(args, variant, "rowpack", [path])
    return run_paths


def step_chart(args, run_paths: dict[str, Path]) -> None:
    ordered_variants = []
    for group_label, _image_encoding, container in PARQUET_PLAN:
        ordered_variants.append(parquet_variant_name(group_label, container))
    for _, image_storage, payload_format in ROWPACK_PLAN:
        ordered_variants.append(rowpack_variant_name(image_storage, payload_format, args.block_codec))

    cmd = [
        args.python, "benchmarks/plot_size_vs_throughput.py",
        "--output-dir", args.root,
        "--filename", "size_vs_throughput.png",
    ]
    for variant in ordered_variants:
        path = run_paths.get(variant)
        if path is None and args.skip_bench:
            # Try to pick up an already-existing run JSON only when the user
            # explicitly skipped benchmarking, and validate that it still points
            # at the intended variant file.
            candidate = Path(args.root) / "runs" / f"{variant}.json"
            if candidate.exists():
                loader, data_files = variant_loader_and_files(args, variant)
                matches, reason = run_json_matches(
                    candidate,
                    variant_label=variant,
                    loader=loader,
                    data_files=data_files,
                    args=args,
                )
                if not matches:
                    raise RuntimeError(f"Existing run JSON is stale for {variant}: {candidate}: {reason}")
                path = candidate
        if path is None:
            raise FileNotFoundError(
                f"Missing benchmark output for planned variant {variant}. "
                "Rerun without --skip-bench or pass --overwrite to regenerate stale runs."
            )
        cmd += ["--run", f"{variant}={path}"]
    run(cmd)


def main():
    args = parse_args()
    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    Path(args.root).mkdir(parents=True, exist_ok=True)

    all_variants = set()
    for group_label, _image_encoding, container in PARQUET_PLAN:
        all_variants.add(parquet_variant_name(group_label, container))
    for _, image_storage, payload_format in ROWPACK_PLAN:
        all_variants.add(rowpack_variant_name(image_storage, payload_format, args.block_codec))

    planned_variants = set(args.only) if args.only else all_variants
    unknown = planned_variants - all_variants
    if unknown:
        raise SystemExit(f"Unknown variants in --only: {sorted(unknown)}")

    require_rowpack_native_symbols(args, planned_variants)

    started = time.perf_counter()
    if not args.skip_prepare:
        step_prepare(args, planned_variants)
    run_paths: dict[str, Path] = {}
    if not args.skip_bench:
        run_paths = step_benchmark(args, planned_variants)
    if not args.skip_chart:
        step_chart(args, run_paths)

    manifest_path = Path(args.root) / "reproduce_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "config": vars(args),
            "duration_s": time.perf_counter() - started,
            "variants": sorted(planned_variants),
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "root": args.root}, indent=2))


if __name__ == "__main__":
    main()
