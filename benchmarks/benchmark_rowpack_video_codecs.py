from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rowpack import MetadataBuilder, RowPackDatasetBuilder
from rowpack.video import FfmpegVideoEncoder, VideoFrame


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RowPack video chunk codec choices on synthetic RGB frames.")
    parser.add_argument("--output-dir", default="results/rowpack_video_codec_benchmark")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--codecs", nargs="+", default=["avif", "h264", "h265"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_frames(count: int, width: int, height: int, fps: float) -> list[VideoFrame]:
    frames: list[VideoFrame] = []
    for frame_index in range(count):
        data = bytearray(width * height * 3)
        cursor = 0
        for y in range(height):
            for x in range(width):
                data[cursor] = (x + frame_index * 3) % 256
                data[cursor + 1] = (y * 2 + frame_index * 5) % 256
                data[cursor + 2] = ((x // 4) ^ (y // 4) ^ frame_index) % 256
                cursor += 3
        frames.append(
            VideoFrame(
                timestamp_ns=int(frame_index * 1_000_000_000 / fps),
                data=bytes(data),
                height=height,
                width=width,
                channels=3,
            )
        )
    return frames


def jpeg_frame_payloads(frames: list[VideoFrame], quality: int) -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("JPEG frame benchmark requires Pillow") from exc

    payloads = []
    for index, frame in enumerate(frames):
        image = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        payloads.append(
            {
                "bytes": buffer.getvalue(),
                "name": f"frame_{index:06d}.jpg",
                "mime_type": "image/jpeg",
                "role": "video_frame",
                "codec": "jpeg",
                "frame_index": index,
                "timestamp_ns": frame.timestamp_ns,
            }
        )
    return payloads


def write_jpeg_rowpack(frames: list[VideoFrame], output: Path, quality: int, overwrite: bool) -> dict[str, Any]:
    start = time.perf_counter()
    payloads = jpeg_frame_payloads(frames, quality)
    encode_time_s = time.perf_counter() - start
    with RowPackDatasetBuilder(
        output,
        metadata=base_metadata("jpeg_frames"),
        payload_format="json",
        block_codec="none",
        overwrite=overwrite,
    ) as builder:
        builder.append_file_row(
            payloads,
            extra={"codec_layout": "independent_jpeg_frames", "frame_count": len(frames)},
            name="jpeg_frames",
        )
    return {
        "variant": "jpeg_frames",
        "codec": "jpeg",
        "status": "ready",
        "encoded_bytes": sum(len(payload["bytes"]) for payload in payloads),
        "rowpack_bytes": output.stat().st_size,
        "encode_time_s": encode_time_s,
        "rowpack_path": str(output),
        "error": "",
    }


def write_video_chunk_rowpack(
    frames: list[VideoFrame],
    output: Path,
    *,
    codec: str,
    fps: float,
    crf: int,
    ffmpeg: str,
    overwrite: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    encoded = FfmpegVideoEncoder(executable=ffmpeg, codec=codec, crf=crf).encode(frames, fps=fps)
    encode_time_s = time.perf_counter() - start
    with RowPackDatasetBuilder(
        output,
        metadata=base_metadata(f"{codec}_chunk"),
        payload_format="json",
        block_codec="none",
        overwrite=overwrite,
    ) as builder:
        builder.append_video_chunk_row(
            stream="synthetic_camera",
            chunk=encoded,
            chunk_index=0,
            codec=codec,
            mime_type=str(encoded.get("mime_type")),
            start_timestamp_ns=int(encoded.get("start_timestamp_ns") or 0),
            end_timestamp_ns=int(encoded.get("end_timestamp_ns") or 0),
            frame_count=int(encoded.get("frame_count") or len(frames)),
            fps=fps,
        )
    return {
        "variant": f"{codec}_chunk",
        "codec": codec,
        "status": "ready",
        "encoded_bytes": len(encoded["bytes"]),
        "rowpack_bytes": output.stat().st_size,
        "encode_time_s": encode_time_s,
        "rowpack_path": str(output),
        "error": "",
    }


def base_metadata(name: str) -> MetadataBuilder:
    return (
        MetadataBuilder()
        .dataset_name(name)
        .description("Synthetic RowPack video codec benchmark")
        .row_field("files", "file[]", "Encoded frame files or video chunks")
        .row_field("_rowpack_continuation", "json", "Video chunk continuation metadata when present")
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = ["variant", "codec", "status", "encoded_bytes", "rowpack_bytes", "encode_time_s", "rowpack_path", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| variant | codec | status | encoded MiB | rowpack MiB | encode seconds | error |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {codec} | {status} | {encoded:.3f} | {rowpack:.3f} | {time:.3f} | {error} |".format(
                variant=row.get("variant", ""),
                codec=row.get("codec", ""),
                status=row.get("status", ""),
                encoded=(row.get("encoded_bytes") or 0) / (1024 * 1024),
                rowpack=(row.get("rowpack_bytes") or 0) / (1024 * 1024),
                time=row.get("encode_time_s") or 0,
                error=row.get("error", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_charts(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    ready = [row for row in rows if row.get("status") == "ready"]
    if not ready:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def bar(key: str, title: str, ylabel: str, filename: str, scale: float = 1.0):
        labels = [row["variant"] for row in ready]
        values = [(row.get(key) or 0) / scale for row in ready]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.5))
        bars = ax.bar(labels, values, color="#376da8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        for item, value in zip(bars, values):
            ax.text(item.get_x() + item.get_width() / 2, item.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        path = charts_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    bar("rowpack_bytes", "RowPack File Size", "MiB", "rowpack_mib.png", scale=1024 * 1024)
    bar("encode_time_s", "Encode Time", "seconds", "encode_time_s.png")
    return paths


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    frames = make_frames(args.frames, args.width, args.height, args.fps)
    rows: list[dict[str, Any]] = []
    rows.append(write_jpeg_rowpack(frames, variants_dir / "jpeg_frames.rowpack", args.jpeg_quality, args.overwrite))

    for codec in args.codecs:
        try:
            rows.append(
                write_video_chunk_rowpack(
                    frames,
                    variants_dir / f"{codec}_chunk.rowpack",
                    codec=codec,
                    fps=args.fps,
                    crf=args.crf,
                    ffmpeg=args.ffmpeg,
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:
            rows.append({"variant": f"{codec}_chunk", "codec": codec, "status": "error", "error": repr(exc)})

    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    charts = write_charts(rows, output_dir)
    print(json.dumps({"summary_csv": str(csv_path), "summary_markdown": str(md_path), "charts": charts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
