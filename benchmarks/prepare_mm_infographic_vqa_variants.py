import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

IMAGE_ENCODINGS = ("source", "jpeg", "png", "rgb")


VARIANT_SPECS = [
    {"name": "upstream", "kind": "upstream", "requested_compression": "UPSTREAM"},
    {"name": "uncompressed", "kind": "generated", "requested_compression": "UNCOMPRESSED", "writer_compression": "NONE"},
    {"name": "snappy", "kind": "generated", "requested_compression": "SNAPPY", "writer_compression": "SNAPPY"},
    {"name": "zstd", "kind": "generated", "requested_compression": "ZSTD", "writer_compression": "ZSTD"},
    {"name": "gzip", "kind": "generated", "requested_compression": "GZIP", "writer_compression": "GZIP"},
    {"name": "lz4_raw", "kind": "generated", "requested_compression": "LZ4_RAW", "writer_compression": "LZ4_RAW"},
    {"name": "brotli", "kind": "generated", "requested_compression": "BROTLI", "writer_compression": "BROTLI"},
    {"name": "lzo", "kind": "generated", "requested_compression": "LZO", "writer_compression": "LZO"},
]
PYARROW_PARQUET_WRITER_CODECS = {"NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Parquet compression variants for nimapourjafar/mm_infographic_vqa."
    )
    parser.add_argument("--dataset", default="nimapourjafar/mm_infographic_vqa")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Optional local Parquet files to use as the source table instead of loading the Hub dataset.",
    )
    parser.add_argument("--output-dir", default="data/variants/mm_infographic_vqa")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--row-group-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--image-encoding",
        choices=IMAGE_ENCODINGS,
        default="source",
        help=(
            "Re-encode every image in the 'images' column before writing Parquet. "
            "'source' keeps the original bytes (JPEG for mm_infographic_vqa). "
            "'jpeg'/'png' re-encode via PIL; 'rgb' stores raw uint8 RGB pixels."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality when --image-encoding=jpeg.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[spec["name"] for spec in VARIANT_SPECS],
        help="Subset of variants to prepare.",
    )
    return parser.parse_args()


def parquet_metadata(paths: list[Path]) -> dict[str, Any]:
    files = []
    total_bytes = 0
    total_rows = 0
    row_groups = 0
    compressions = set()

    for path in paths:
        size = path.stat().st_size
        metadata = pq.ParquetFile(path).metadata
        total_bytes += size
        total_rows += metadata.num_rows
        row_groups += metadata.num_row_groups
        for rg_idx in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_idx)
            for col_idx in range(row_group.num_columns):
                compressions.add(row_group.column(col_idx).compression)
        files.append({"path": str(path), "bytes": size, "rows": metadata.num_rows})

    return {
        "paths": [str(path) for path in paths],
        "files": files,
        "bytes": total_bytes,
        "rows": total_rows,
        "row_groups": row_groups,
        "observed_compressions": sorted(compressions),
    }


def writer_support(compression: str) -> tuple[bool, str | None]:
    if compression.upper() not in PYARROW_PARQUET_WRITER_CODECS:
        return (
            False,
            f"PyArrow {pa.__version__} ParquetWriter supports "
            f"{sorted(PYARROW_PARQUET_WRITER_CODECS)}, not {compression!r}.",
        )
    try:
        table = pa.table({"x": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "probe.parquet"
            pq.write_table(table, path, compression=compression)
        return True, None
    except Exception as exc:
        return False, repr(exc)


def find_upstream_parquet_files(dataset_name: str, split: str) -> tuple[str, list[str]]:
    info = HfApi().repo_info(dataset_name, repo_type="dataset", files_metadata=True)
    parquet_files = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.endswith(".parquet")
    )
    preferred = [
        filename for filename in parquet_files
        if Path(filename).name.startswith(f"{split}-") or Path(filename).stem.startswith(split)
    ]
    return info.sha, preferred or parquet_files


def copy_upstream_files(dataset_name: str, split: str, output_dir: Path, overwrite: bool) -> dict[str, Any]:
    revision, filenames = find_upstream_parquet_files(dataset_name, split)
    if not filenames:
        raise FileNotFoundError(f"No upstream Parquet files found for {dataset_name}")

    variant_dir = output_dir / "upstream"
    variant_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    started = time.perf_counter()
    for filename in filenames:
        local_source = Path(hf_hub_download(dataset_name, filename=filename, repo_type="dataset"))
        destination = variant_dir / Path(filename).name
        if overwrite or not destination.exists():
            shutil.copy2(local_source, destination)
        paths.append(destination)

    return {
        "name": "upstream",
        "kind": "upstream",
        "status": "ready",
        "requested_compression": "UPSTREAM",
        "writer_compression": None,
        "hub_revision": revision,
        "source_filenames": filenames,
        "duration_s": time.perf_counter() - started,
        **parquet_metadata(paths),
    }


def load_source_table(dataset_name: str, split: str, max_rows: int | None, data_files: list[str] | None = None):
    if data_files:
        tables = [pq.read_table(path) for path in data_files]
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        total_rows = table.num_rows
        if max_rows is not None:
            table = table.slice(0, min(max_rows, total_rows))
        return table, total_rows, table.num_rows

    dataset = load_dataset(dataset_name, split=split)
    total_rows = len(dataset)
    if max_rows is not None:
        dataset = dataset.select(range(min(max_rows, total_rows)))
    return dataset.data.table, total_rows, len(dataset)


def reencode_images_in_table(table: pa.Table, encoding: str, jpeg_quality: int) -> pa.Table:
    """Replace each row's `images` column entries with the requested encoding.

    Source rows already store images as `{"bytes": <encoded_or_raw>, "path": None}`.
    For "source", the table is returned unchanged. For "jpeg"/"png" we re-encode
    via PIL. For "rgb" we store raw uint8 RGB pixel bytes (height*width*3).
    """
    if encoding == "source":
        return table
    if "images" not in table.column_names:
        raise ValueError("expected an 'images' column to re-encode")

    images_column = table.column("images").to_pylist()
    new_images_column: list[list[Any]] = []
    for image_list in images_column:
        new_list = []
        for image in image_list or []:
            encoded = reencode_single_image(image, encoding, jpeg_quality)
            if encoding == "rgb" and isinstance(encoded, dict):
                # The RGB benchmark is a synthetic "raw pixels in Parquet" test,
                # so store it as list<binary> instead of the nested HF
                # list<struct<bytes,path>> image feature. PyArrow 13 on Linux
                # has been observed to fail writing the nested raw-RGB variant
                # with compression=NONE, while the simple binary layout is both
                # more direct and closer to what this benchmark is measuring.
                new_list.append(encoded.get("bytes"))
            else:
                new_list.append(encoded)
        new_images_column.append(new_list)

    arrays = []
    names = []
    images_type = table.schema.field("images").type
    for name in table.column_names:
        names.append(name)
        if name == "images":
            if encoding == "rgb":
                arrays.append(pa.array(new_images_column, type=pa.list_(pa.binary())))
            else:
                arrays.append(pa.array(new_images_column, type=images_type))
        else:
            arrays.append(table.column(name).combine_chunks())

    # Rebuild the table from concrete Arrays rather than mixing the original
    # ChunkedArrays with a freshly encoded nested image Array. Some PyArrow
    # builds have produced invalid nested list/struct offsets for the raw RGB
    # uncompressed writer path when the mixed table is written directly.
    return pa.Table.from_arrays(arrays, names=names)


def reencode_single_image(image: dict[str, Any], encoding: str, jpeg_quality: int) -> dict[str, Any]:
    if image is None:
        return image
    src_bytes = image.get("bytes")
    src_path = image.get("path")
    if src_bytes is not None:
        pil_image = Image.open(io.BytesIO(src_bytes))
    elif src_path is not None:
        pil_image = Image.open(src_path)
    else:
        return image
    pil_image = pil_image.convert("RGB")
    if encoding == "jpeg":
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=jpeg_quality)
        return {"bytes": buf.getvalue(), "path": None}
    if encoding == "png":
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", optimize=False)
        return {"bytes": buf.getvalue(), "path": None}
    if encoding == "rgb":
        # Store raw uint8 RGB pixel bytes preceded by a small magic header so
        # the downstream loader can reconstruct image dimensions without a
        # separate schema change. Layout (little-endian):
        #   bytes  0..7  : magic b"RGB8\x00\x03\x00"  (RGB, channels=3, version=0)
        #   bytes  7..11 : uint32 height
        #   bytes 11..15 : uint32 width
        #   bytes 15..   : raw RGB pixels (height * width * 3)
        w, h = pil_image.size
        header = b"RGB8\x00\x03\x00" + h.to_bytes(4, "little") + w.to_bytes(4, "little")
        return {"bytes": header + pil_image.tobytes(), "path": None}
    raise ValueError(f"unsupported image encoding {encoding!r}")


def parquet_matches_image_encoding(path: Path, encoding: str) -> bool:
    """Return whether an existing variant appears to contain the requested image encoding.

    This guards against stale valid Parquet files left by older benchmark
    scripts. Without this check, a previous JPEG-sized file named
    `parquet_rgb/gzip.parquet` could be reused forever when `--overwrite` is
    omitted, producing a chart with correct labels but wrong data.
    """
    if encoding == "source":
        return True

    sample = first_image_bytes(path)
    if sample is None:
        return False
    if encoding == "rgb":
        return sample.startswith(b"RGB8\x00\x03\x00")
    if encoding == "png":
        return sample.startswith(b"\x89PNG\r\n\x1a\n")
    if encoding == "jpeg":
        return sample.startswith(b"\xff\xd8")
    return True


def first_image_bytes(path: Path) -> bytes | None:
    parquet_file = pq.ParquetFile(path)
    if "images" not in parquet_file.schema_arrow.names:
        return None
    for batch in parquet_file.iter_batches(batch_size=1, columns=["images"]):
        rows = batch.to_pylist()
        if not rows:
            continue
        images = rows[0].get("images") if isinstance(rows[0], dict) else None
        if not images:
            continue
        image = images[0]
        if isinstance(image, dict):
            payload = image.get("bytes")
            return bytes(payload) if payload is not None else None
        if isinstance(image, (bytes, bytearray, memoryview)):
            return bytes(image)
        return None
    return None


def write_generated_variant(
    table,
    spec: dict[str, Any],
    output_dir: Path,
    row_group_size: int,
    overwrite: bool,
    image_encoding: str,
) -> dict[str, Any]:
    compression = spec["writer_compression"]
    supported, error = writer_support(compression)
    path = output_dir / f"{spec['name']}.parquet"

    result = {
        "name": spec["name"],
        "kind": spec["kind"],
        "requested_compression": spec["requested_compression"],
        "writer_compression": compression,
    }
    if not supported:
        return {
            **result,
            "status": "unsupported",
            "error": error,
        }

    started = time.perf_counter()
    existing_metadata = None
    if path.exists() and not overwrite:
        try:
            existing_metadata = parquet_metadata([path])
            if not parquet_matches_image_encoding(path, image_encoding):
                existing_metadata = None
        except Exception:
            # A failed previous write can leave a partial Parquet file behind.
            # Treat invalid existing outputs as stale and regenerate them.
            existing_metadata = None

    if existing_metadata is None:
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            pq.write_table(
                table,
                tmp_path,
                compression=compression,
                row_group_size=row_group_size,
            )
            parquet_metadata([tmp_path])
            tmp_path.replace(path)
            written_metadata = parquet_metadata([path])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        metadata = written_metadata
    else:
        metadata = existing_metadata

    return {
        **result,
        "status": "ready",
        "duration_s": time.perf_counter() - started,
        **metadata,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.variants)
    unknown = selected - {spec["name"] for spec in VARIANT_SPECS}
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    manifest: dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "output_dir": str(output_dir),
        "data_files": args.data_files,
        "max_rows": args.max_rows,
        "row_group_size": args.row_group_size,
        "image_encoding": args.image_encoding,
        "pyarrow_version": pa.__version__,
        "variants": [],
    }

    source_table = None
    source_total_rows = None
    source_rows_written = None

    for spec in VARIANT_SPECS:
        if spec["name"] not in selected:
            continue

        print(f"Preparing {spec['name']}...")
        try:
            if spec["kind"] == "upstream":
                variant = copy_upstream_files(args.dataset, args.split, output_dir, args.overwrite)
            else:
                if source_table is None:
                    source_table, source_total_rows, source_rows_written = load_source_table(
                        args.dataset, args.split, args.max_rows, args.data_files
                    )
                    if args.image_encoding != "source":
                        print(f"Re-encoding images as {args.image_encoding}...")
                        source_table = reencode_images_in_table(
                            source_table, args.image_encoding, args.jpeg_quality
                        )
                    manifest["source_total_rows"] = source_total_rows
                    manifest["source_rows_written"] = source_rows_written
                    manifest["image_encoding"] = args.image_encoding
                    if args.image_encoding == "jpeg":
                        manifest["jpeg_quality"] = args.jpeg_quality
                variant = write_generated_variant(
                    source_table,
                    spec,
                    output_dir,
                    args.row_group_size,
                    args.overwrite,
                    args.image_encoding,
                )
        except Exception as exc:
            variant = {
                "name": spec["name"],
                "kind": spec["kind"],
                "requested_compression": spec.get("requested_compression"),
                "writer_compression": spec.get("writer_compression"),
                "status": "error",
                "error": repr(exc),
            }
        manifest["variants"].append(variant)

    ready_sizes = {
        variant["name"]: variant["bytes"]
        for variant in manifest["variants"]
        if variant.get("status") == "ready" and variant.get("bytes")
    }
    uncompressed_bytes = ready_sizes.get("uncompressed")
    upstream_bytes = ready_sizes.get("upstream")
    for variant in manifest["variants"]:
        if variant.get("status") != "ready" or not variant.get("bytes"):
            continue
        if uncompressed_bytes:
            variant["ratio_vs_uncompressed"] = variant["bytes"] / uncompressed_bytes
        if upstream_bytes:
            variant["ratio_vs_upstream"] = variant["bytes"] / upstream_bytes

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "ready": [v["name"] for v in manifest["variants"] if v.get("status") == "ready"],
        "unsupported": [v["name"] for v in manifest["variants"] if v.get("status") == "unsupported"],
        "errors": {v["name"]: v.get("error") for v in manifest["variants"] if v.get("status") == "error"},
    }, indent=2))


if __name__ == "__main__":
    main()
