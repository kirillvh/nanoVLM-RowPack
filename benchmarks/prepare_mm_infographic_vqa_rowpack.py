import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from PIL import Image

from rowpack import RowPackReader, RowPackWriter
from rowpack.native import load_native
try:
    from rowpack.make_list import write_rowpack_list
except ModuleNotFoundError:
    def write_rowpack_list(paths: list[Path], output: str | Path, *, absolute: bool = False) -> None:
        """Local fallback for older RowPack checkouts without rowpack.make_list."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        base = output.resolve().parent
        lines = []
        for path in paths:
            resolved = Path(path).resolve()
            if absolute:
                lines.append(str(resolved))
                continue
            try:
                lines.append(str(resolved.relative_to(base)))
            except ValueError:
                lines.append(str(resolved))
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a RowPack variant for nimapourjafar/mm_infographic_vqa."
    )
    parser.add_argument("--dataset", default="nimapourjafar/mm_infographic_vqa")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Optional local Parquet files to convert instead of loading the Hub dataset.",
    )
    parser.add_argument("--output-dir", default="data/variants/mm_infographic_vqa_rowpack")
    parser.add_argument("--variant-name", default="rowpack_uncompressed")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rows-per-block", type=int, default=32)
    parser.add_argument("--block-codec", choices=["none", "lzav_default", "lzav_hi"], default="none")
    parser.add_argument("--parquet-batch-size", type=int, default=32)
    parser.add_argument("--payload-format", choices=["json", "cista"], default="json")
    parser.add_argument(
        "--image-storage",
        choices=["encoded", "raw_rgb", "qoi_lossless"],
        default="encoded",
        help=(
            "Image bytes stored in each row. 'encoded' keeps the source bytes "
            "(JPEG for mm_infographic_vqa). 'raw_rgb' decodes to raw uint8 pixels. "
            "'qoi_lossless' encodes raw pixels through QOI. Applies to both 'json' "
            "and 'cista' payload formats."
        ),
    )
    parser.add_argument("--rowpack-native-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_source(args):
    if args.data_files:
        dataset = LocalParquetRows(args.data_files, max_rows=args.max_rows, batch_size=args.parquet_batch_size)
        return dataset, "local_parquet", dataset.total_rows
    else:
        dataset = load_dataset(args.dataset, split=args.split)
        label = args.dataset

    total_rows = len(dataset)
    if args.max_rows is not None:
        dataset = dataset.select(range(min(args.max_rows, total_rows)))
    return dataset, label, total_rows


class LocalParquetRows:
    def __init__(self, paths: list[str], *, max_rows: int | None, batch_size: int):
        self.paths = paths
        self.max_rows = max_rows
        self.batch_size = batch_size
        self.total_rows = self._total_rows()

    def __len__(self):
        if self.max_rows is None:
            return self.total_rows
        return min(self.max_rows, self.total_rows)

    def __iter__(self):
        import pyarrow.parquet as pq

        yielded = 0
        for path in self.paths:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                for row in batch.to_pylist():
                    if self.max_rows is not None and yielded >= self.max_rows:
                        return
                    yielded += 1
                    yield row

    def _total_rows(self) -> int:
        import pyarrow.parquet as pq

        return sum(pq.ParquetFile(path).metadata.num_rows for path in self.paths)


def write_rowpack(args, dataset, source_label: str, source_total_rows: int) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rowpack_path = output_dir / f"{args.variant_name}.rowpack"

    started = time.perf_counter()
    metadata = {
        "dataset": args.dataset,
        "source": source_label,
        "split": args.split,
        "source_total_rows": source_total_rows,
        "data_files": args.data_files,
    }

    json_qoi_encoder = None
    if args.image_storage == "qoi_lossless":
        native = load_native(args.rowpack_native_dir)
        if not hasattr(native, "qoi_encode_rgb"):
            raise RuntimeError(
                "rowpack_native was built without QOI support, but --image-storage qoi_lossless "
                "was requested. Reconfigure RowPack with the bundled QOI include directory or rerun "
                "benchmarks/run_mega_benchmark.sh to refresh the CMake cache."
            )
        if args.payload_format == "json":
            # CISTA path runs QOI encoding inside the native writer. JSON path needs
            # to embed already-encoded QOI bytes, so resolve the encoder up-front.
            json_qoi_encoder = native.qoi_encode_rgb

    if rowpack_path.exists() and not args.overwrite:
        try:
            return rowpack_variant_metadata(
                args,
                rowpack_path,
                started=started,
                rows_written=None,
            )
        except Exception:
            # A failed previous write can leave a partial RowPack behind. Regenerate
            # invalid existing outputs through a temp file and replace on success.
            pass

    tmp_path = rowpack_path.with_name(f".{rowpack_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    rows_written = 0
    try:
        with RowPackWriter(
            tmp_path,
            rows_per_block=args.rows_per_block,
            metadata=metadata,
            payload_format=args.payload_format,
            block_codec=args.block_codec,
            native_module_dir=args.rowpack_native_dir,
            overwrite=True,
        ) as writer:
            for idx, row in enumerate(dataset):
                if args.image_storage in {"raw_rgb", "qoi_lossless"}:
                    if args.payload_format == "cista":
                        row = with_decoded_rgb_images(row, storage=args.image_storage)
                    else:  # json
                        row = with_json_storage_images(
                            row,
                            storage=args.image_storage,
                            qoi_encoder=json_qoi_encoder,
                        )
                writer.append_row(row, name=f"row_{idx:08d}")
                rows_written += 1

        rowpack_variant_metadata(args, tmp_path, started=started, rows_written=rows_written)
        tmp_path.replace(rowpack_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return rowpack_variant_metadata(args, rowpack_path, started=started, rows_written=rows_written)


def rowpack_variant_metadata(
    args,
    rowpack_path: Path,
    *,
    started: float,
    rows_written: int | None,
) -> dict[str, Any]:
    with RowPackReader(rowpack_path) as reader:
        block_count = len(reader.blocks)
        row_count = len(reader)
        observed_compressions = [
            manifest_codec_name(codec) for codec in reader.metadata.get("observed_compressions", [args.block_codec])
        ]

    return {
        "name": args.variant_name,
        "kind": "generated",
        "status": "ready",
        "requested_compression": manifest_codec_name(args.block_codec),
        "writer_compression": manifest_codec_name(args.block_codec),
        "observed_compressions": observed_compressions,
        "payload_format": args.payload_format,
        "image_storage": args.image_storage,
        "block_codec": args.block_codec,
        "paths": [str(rowpack_path)],
        "bytes": rowpack_path.stat().st_size,
        "rows": row_count,
        "row_groups": block_count,
        "rows_per_block": args.rows_per_block,
        "rowpack_native_dir": args.rowpack_native_dir,
        "duration_s": time.perf_counter() - started,
        "source_rows_written": rows_written if rows_written is not None else row_count,
    }


def manifest_codec_name(codec: str) -> str:
    return codec.upper() if codec != "none" else "NONE"


def main():
    args = parse_args()
    dataset, source_label, source_total_rows = load_source(args)
    variant = write_rowpack(args, dataset, source_label, source_total_rows)

    manifest_path = Path(args.output_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variants = [item for item in manifest.get("variants", []) if item.get("name") != variant["name"]]
        variants.append(variant)
        manifest["variants"] = variants
        manifest["source_rows_written"] = max(
            manifest.get("source_rows_written") or 0,
            variant["source_rows_written"],
        )
    else:
        manifest = {
            "dataset": args.dataset,
            "split": args.split,
            "output_dir": args.output_dir,
            "max_rows": args.max_rows,
            "rows_per_block": args.rows_per_block,
            "source_total_rows": source_total_rows,
            "source_rows_written": variant["source_rows_written"],
            "variants": [variant],
        }
    manifest["payload_formats"] = sorted({item.get("payload_format", "json") for item in manifest["variants"]})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_dir = Path(args.output_dir)
    variant_paths = [Path(path) for path in variant["paths"]]
    list_path = output_dir / "rowpacks.txt"
    variant_list_path = output_dir / f"rowpacks_{variant['name']}.txt"
    write_rowpack_list(variant_paths, list_path)
    write_rowpack_list(variant_paths, variant_list_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "rowpack_list": str(list_path),
        "variant_rowpack_list": str(variant_list_path),
        "variant": variant["name"],
        "path": variant["paths"][0],
        "bytes": variant["bytes"],
        "rows": variant["rows"],
        "row_groups": variant["row_groups"],
    }, indent=2))


def with_decoded_rgb_images(row: dict[str, Any], *, storage: str) -> dict[str, Any]:
    row = dict(row)
    row["images"] = [to_decoded_rgb_image(image, storage=storage) for image in row.get("images") or []]
    return row


def with_json_storage_images(row: dict[str, Any], *, storage: str, qoi_encoder=None) -> dict[str, Any]:
    """Pre-encode image bytes for the JSON payload format.

    The JSON path serialises whatever bytes it receives. `cista` performs raw/QOI
    transformations in the native writer; for `json` we have to embed the final
    bytes ourselves so the on-disk layout matches the requested storage mode.
    """
    row = dict(row)
    new_images = []
    for image in row.get("images") or []:
        decoded = to_decoded_rgb_image(image, storage="raw_rgb")
        if storage == "raw_rgb":
            new_images.append(decoded)
        elif storage == "qoi_lossless":
            if qoi_encoder is None:
                raise RuntimeError("qoi_encoder required for json+qoi_lossless")
            encoded = qoi_encoder(
                decoded["bytes"], decoded["height"], decoded["width"], decoded["channels"]
            )
            new_images.append({
                "bytes": bytes(encoded),
                "height": decoded["height"],
                "width": decoded["width"],
                "channels": decoded["channels"],
                "storage": "qoi_lossless",
            })
        else:
            raise ValueError(f"unsupported json image storage {storage!r}")
    row["images"] = new_images
    return row


def to_decoded_rgb_image(image: Any, *, storage: str) -> dict[str, Any]:
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            pil_image = Image.open(io.BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            pil_image = Image.open(image["path"])
        else:
            raise TypeError(f"Unsupported image payload: {image!r}")
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise TypeError(f"Unsupported image payload: {type(image)!r}")

    pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    return {
        "bytes": pil_image.tobytes(),
        "height": height,
        "width": width,
        "channels": 3,
        "storage": storage,
    }


if __name__ == "__main__":
    main()
