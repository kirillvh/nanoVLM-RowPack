import argparse
import itertools
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from data.collators import VQACollator
from data.datasets import VQADataset
from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel
from rowpack import NativeCistaVQARows, RowPackRows
from rowpack.format import HEADER_SIZE, unpack_header


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class MMInfographicVQAAdapter(Dataset):
    """Adapt mm_infographic_vqa rows to the repo's VQA dataset schema."""

    def __init__(self, dataset, max_images: int = 1):
        self.dataset = dataset
        self.max_images = max_images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return adapt_mm_infographic_row(self.dataset[idx], self.max_images)


class MMInfographicVQAIterableAdapter(IterableDataset):
    """Streaming adapter for direct Parquet/HF iterable access."""

    def __init__(self, dataset, max_images: int = 1):
        self.dataset = dataset
        self.max_images = max_images

    def __iter__(self):
        source = self.dataset
        worker_info = get_worker_info()
        if worker_info is not None:
            if hasattr(source, "shard"):
                source = source.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            else:
                source = itertools.islice(source, worker_info.id, None, worker_info.num_workers)

        for row in source:
            yield adapt_mm_infographic_row(row, self.max_images)


class VQAIterableDataset(IterableDataset):
    """Apply the existing VQA processing path to an adapted iterable dataset."""

    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length, max_images: int):
        self.dataset = MMInfographicVQAIterableAdapter(dataset, max_images=max_images)
        self.processor = VQADataset([], tokenizer, image_processor, mp_image_token_length)

    def __iter__(self):
        for item in self.dataset:
            yield self.processor._process_data(item)


class RowPackDirectVQAIterableDataset(IterableDataset):
    """Process native CISTA RowPack VQA tuples without reconstructing HF rows."""

    def __init__(self, rowpack_rows, tokenizer, image_processor, mp_image_token_length, max_images: int):
        self.rowpack_rows = rowpack_rows
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.max_images = max_images
        self.processor = VQADataset([], tokenizer, image_processor, mp_image_token_length)

    def __iter__(self):
        source = self.rowpack_rows
        worker_info = get_worker_info()
        if worker_info is not None:
            source = itertools.islice(source, worker_info.id, None, worker_info.num_workers)

        for _row_id, text_pairs, image_payloads in source:
            yield self._process_native_vqa(text_pairs, image_payloads)

    def _process_native_vqa(self, text_pairs, image_payloads):
        processed_images = []
        splitted_image_counts = []
        if image_payloads:
            processed_images, splitted_image_counts = self._process_image_payloads(image_payloads)

        messages = self._messages_from_pairs(text_pairs, splitted_image_counts)
        if not messages:
            return None

        input_ids, mask, attention_mask = self.processor._prepare_inputs_and_loss_mask(messages)
        labels = self.processor._get_labels(input_ids, mask)
        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _process_image_payloads(self, image_payloads):
        processed_images = []
        splitted_image_counts = []
        for payload in image_payloads[: self.max_images]:
            if is_qoi_payload(payload):
                payload = qoi_payload_to_raw_rgb_payload(payload)
            if is_raw_rgb_payload(payload):
                processed_image, splitted_image_count = self._process_raw_rgb_payload(payload)
            else:
                pil_image = image_payload_to_pil(payload)
                processed_image, splitted_image_count = self.image_processor(pil_image)

            if (
                not hasattr(self.tokenizer, "global_image_token")
                and splitted_image_count[0] * splitted_image_count[1] == len(processed_image) - 1
            ):
                processed_image = processed_image[1:]
            processed_images.append(processed_image)
            splitted_image_counts.append(splitted_image_count)

        return processed_images, splitted_image_counts

    def _process_raw_rgb_payload(self, payload):
        image = raw_rgb_payload_to_tensor(payload)
        transforms = getattr(self.image_processor, "transforms", None)
        if transforms is None or len(transforms) < 3:
            raise TypeError("Raw RGB RowPack images require the standard image processor Compose")

        image = transforms[0](image)
        return transforms[2](image)

    def _messages_from_pairs(self, text_pairs, splitted_image_counts):
        messages = []
        for user_text, assistant_text in text_pairs:
            user_text = str(user_text).strip()
            assistant_text = str(assistant_text).strip()
            if not user_text or not assistant_text:
                continue
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

        if not messages:
            return messages

        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logging.warning("Found and removed an image token in RowPack text before adding image string.")
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if splitted_image_counts:
            image_string = get_image_string(self.tokenizer, splitted_image_counts, self.mp_image_token_length)
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages


class PyArrowParquetRows:
    """Row iterator over local Parquet files using PyArrow directly."""

    def __init__(
        self,
        paths: list[str],
        max_rows: int | None = None,
        batch_size: int = 32,
        read_pattern: str = "sequential",
        read_block_size: int = 32,
        seed: int = 0,
    ):
        self.paths = paths
        self.max_rows = max_rows
        self.batch_size = batch_size
        self.read_pattern = read_pattern
        self.read_block_size = max(1, read_block_size)
        self.seed = seed
        self.total_rows = parquet_total_rows(paths)

    def __iter__(self):
        if self.read_pattern == "random_block":
            yield from self._iter_random_blocks()
        else:
            yield from self._iter_sequential()

    def _iter_sequential(self):
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

    def _iter_random_blocks(self):
        if self.total_rows is None:
            yield from self._iter_sequential()
            return

        target_rows = self.max_rows if self.max_rows is not None else self.total_rows
        yielded = 0
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        while yielded < target_rows:
            window_size = min(self.read_block_size, target_rows - yielded)
            max_start = max(0, self.total_rows - window_size)
            start = int(torch.randint(max_start + 1, (1,), generator=generator).item()) if max_start else 0
            for row in self._iter_range(start, window_size):
                yield row
                yielded += 1
                if yielded >= target_rows:
                    break

    def _iter_range(self, start: int, length: int):
        import pyarrow.parquet as pq

        remaining = length
        absolute_offset = 0
        for path in self.paths:
            parquet_file = pq.ParquetFile(path)
            file_rows = parquet_file.metadata.num_rows
            file_start = max(0, start - absolute_offset)
            if file_start >= file_rows:
                absolute_offset += file_rows
                continue

            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                batch_rows = batch.num_rows
                if file_start >= batch_rows:
                    file_start -= batch_rows
                    continue

                take = min(remaining, batch_rows - file_start)
                for row in batch.slice(file_start, take).to_pylist():
                    yield row
                remaining -= take
                file_start = 0
                if remaining <= 0:
                    return

            absolute_offset += file_rows

    def __len__(self):
        if self.max_rows is not None and self.total_rows is not None:
            return min(self.max_rows, self.total_rows)
        if self.total_rows is None:
            raise TypeError("Unknown Parquet row count")
        return self.total_rows


def adapt_mm_infographic_row(row, max_images: int):
    return {
        "images": decode_images(row.get("images") or [], max_images),
        "texts": conversation_pairs(row.get("data") or []),
    }


def _decode_image_bytes(buf: bytes):
    # Match the RGB8 magic header written by prepare_mm_infographic_vqa_variants.py
    # for the "rgb" (raw-pixel) encoding. Layout (little-endian):
    #   bytes  0..7  : magic b"RGB8\x00\x03\x00"  (RGB, channels=3, version=0)
    #   bytes  7..11 : uint32 height
    #   bytes 11..15 : uint32 width
    #   bytes 15..   : raw RGB pixels (height * width * 3)
    if len(buf) >= 15 and buf[:7] == b"RGB8\x00\x03\x00":
        height = int.from_bytes(buf[7:11], "little")
        width = int.from_bytes(buf[11:15], "little")
        return Image.frombytes("RGB", (width, height), buf[15:])
    return Image.open(io.BytesIO(buf)).convert("RGB")


_QOI_DECODER = None


def _decode_qoi_to_pil(buf: bytes):
    global _QOI_DECODER
    if _QOI_DECODER is None:
        from rowpack.native import load_native
        _QOI_DECODER = load_native().qoi_decode_rgb
    decoded = _QOI_DECODER(bytes(buf))
    height = int(decoded["height"])
    width = int(decoded["width"])
    return Image.frombytes("RGB", (width, height), bytes(decoded["bytes"]))


def decode_images(images, max_images: int):
    decoded = []
    for image in images[:max_images]:
        if isinstance(image, Image.Image):
            decoded.append(image.convert("RGB"))
            continue

        if isinstance(image, (bytes, bytearray, memoryview)):
            decoded.append(_decode_image_bytes(bytes(image)))
            continue

        if isinstance(image, dict):
            storage = image.get("storage")
            if storage == "raw_rgb" and image.get("bytes") is not None:
                height = int(image["height"])
                width = int(image["width"])
                decoded.append(Image.frombytes("RGB", (width, height), bytes(image["bytes"])))
                continue
            if storage == "qoi_lossless" and image.get("bytes") is not None:
                decoded.append(_decode_qoi_to_pil(image["bytes"]))
                continue
            if image.get("bytes") is not None:
                decoded.append(_decode_image_bytes(image["bytes"]))
                continue
            if image.get("path") is not None:
                decoded.append(Image.open(image["path"]).convert("RGB"))
                continue

        raise TypeError(f"Unsupported image payload: {type(image)!r}")

    return decoded


def is_raw_rgb_payload(payload) -> bool:
    return isinstance(payload, dict) and payload.get("storage") == "raw_rgb"


def is_qoi_payload(payload) -> bool:
    return isinstance(payload, dict) and payload.get("storage") == "qoi_lossless"


def qoi_payload_to_raw_rgb_payload(payload):
    global _QOI_DECODER
    if _QOI_DECODER is None:
        from rowpack.native import load_native
        _QOI_DECODER = load_native().qoi_decode_rgb
    return _QOI_DECODER(bytes(payload["bytes"]))


def raw_rgb_payload_to_tensor(payload):
    height = int(payload["height"])
    width = int(payload["width"])
    channels = int(payload.get("channels") or 3)
    if channels != 3:
        raise ValueError(f"Expected raw RGB payload with 3 channels, got {channels}")

    array = np.frombuffer(payload["bytes"], dtype=np.uint8, count=height * width * channels)
    array = array.reshape((height, width, channels))
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).to(torch.float32)
    return tensor.div_(255.0)


def image_payload_to_pil(payload):
    if isinstance(payload, dict):
        image_bytes = payload["bytes"]
    else:
        image_bytes = payload
    return Image.open(io.BytesIO(bytes(image_bytes))).convert("RGB")


def conversation_pairs(turns):
    pairs = []
    pending_user = []

    for turn in turns:
        if turn.get("modality") != "text":
            continue

        role = turn.get("role")
        text = str(turn.get("data", "")).strip()
        if not text:
            continue

        if role == "user":
            pending_user.append(text)
        elif role == "assistant":
            user_text = "\n".join(pending_user).strip()
            if user_text:
                pairs.append({"user": user_text, "assistant": text})
            pending_user = []

    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-train nanoVLM on mm_infographic_vqa and measure data-loader overhead."
    )
    parser.add_argument("--dataset", default="nimapourjafar/mm_infographic_vqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-files", nargs="+", default=None)
    parser.add_argument("--loader", choices=["datasets", "pyarrow", "rowpack"], default="datasets")
    parser.add_argument("--parquet-batch-size", type=int, default=32)
    parser.add_argument("--rowpack-native-dir", default=None)
    parser.add_argument("--rowpack-native-decode-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rowpack-direct-vqa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For CISTA RowPack files, use the native direct VQA tuple reader instead of reconstructing HF-like rows.",
    )
    parser.add_argument("--read-pattern", choices=["sequential", "random_block"], default="sequential")
    parser.add_argument("--read-block-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--loader-benchmark-batches", type=int, default=0)
    parser.add_argument("--loader-benchmark-warmup-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use datasets streaming mode so iteration reads from Parquet/HF source instead of a materialized Arrow cache.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip-hub-size", action="store_true")
    parser.add_argument(
        "--size-path",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Add a local dataset version to size reporting, e.g. uncompressed=data/foo.parquet.",
    )
    parser.add_argument(
        "--output",
        default="results/mm_infographic_vqa_baseline.json",
        help="Where to write the benchmark JSON artifact.",
    )
    return parser.parse_args()


def build_tiny_vlm_config(tokenizer, args) -> VLMConfig:
    vit_patch_size = 16
    mp_pixel_shuffle_factor = 2
    image_tokens_per_patch = (args.image_size // vit_patch_size // mp_pixel_shuffle_factor) ** 2
    if image_tokens_per_patch < 1:
        raise ValueError("--image-size must be at least 32 for the tiny baseline config")

    base_cfg = VLMConfig()
    return VLMConfig(
        vit_model_type="testing",
        vit_hidden_dim=32,
        vit_inter_dim=64,
        vit_patch_size=vit_patch_size,
        vit_img_size=args.image_size,
        vit_n_heads=4,
        vit_dropout=0.0,
        vit_n_blocks=1,
        vit_ln_eps=1e-6,
        vit_cls_flag=False,
        lm_model_type="testing",
        lm_hidden_dim=64,
        lm_inter_dim=128,
        lm_rms_eps=1e-5,
        lm_re_base=10000,
        lm_max_position_embeddings=args.sequence_length,
        lm_base_vocab_size=len(tokenizer),
        extra_token_amount=0,
        lm_vocab_size=len(tokenizer),
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_dropout=0.0,
        lm_n_blocks=1,
        lm_attn_scaling=1.0,
        lm_max_length=args.sequence_length,
        lm_use_tokens=False,
        lm_tie_weights=True,
        lm_tokenizer=base_cfg.lm_tokenizer,
        lm_chat_template=base_cfg.lm_chat_template,
        mp_pixel_shuffle_factor=mp_pixel_shuffle_factor,
        mp_image_token_length=image_tokens_per_patch,
        max_img_size=args.image_size,
        resize_to_max_side_len=True,
        vlm_extra_tokens=base_cfg.vlm_extra_tokens,
        vlm_load_backbone_weights=False,
        hf_repo_name=None,
    )


def load_split(args):
    load_start = time.perf_counter()
    if args.data_files and args.loader == "pyarrow":
        dataset = PyArrowParquetRows(
            args.data_files,
            max_rows=args.max_rows,
            batch_size=args.parquet_batch_size,
            read_pattern=args.read_pattern,
            read_block_size=args.read_block_size,
            seed=args.seed,
        )
        return dataset, "pyarrow_parquet", time.perf_counter() - load_start, dataset.total_rows

    if args.data_files and args.loader == "rowpack":
        payload_format = rowpack_payload_format(args.data_files[0])
        if args.rowpack_direct_vqa and payload_format == "cista":
            dataset = NativeCistaVQARows(
                args.data_files,
                max_rows=args.max_rows,
                read_pattern=args.read_pattern,
                read_block_size=args.read_block_size,
                seed=args.seed,
                native_module_dir=args.rowpack_native_dir,
                native_decode_images=args.rowpack_native_decode_images,
            )
            return dataset, "rowpack_cista_direct", time.perf_counter() - load_start, dataset.total_rows
        else:
            dataset = RowPackRows(
                args.data_files,
                max_rows=args.max_rows,
                read_pattern=args.read_pattern,
                read_block_size=args.read_block_size,
                seed=args.seed,
                native_module_dir=args.rowpack_native_dir,
            )
            return dataset, "rowpack", time.perf_counter() - load_start, dataset.total_rows

    if args.data_files:
        data_files = {args.split: args.data_files}
        dataset = load_dataset("parquet", data_files=data_files, split=args.split, streaming=args.streaming)
        dataset_label = "local_parquet"
    else:
        dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
        dataset_label = args.dataset

    total_rows = None
    if args.streaming:
        total_rows = parquet_total_rows(args.data_files) if args.data_files else None
        if args.shuffle and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        if args.max_rows is not None:
            dataset = dataset.take(args.max_rows)
    else:
        total_rows = len(dataset) if hasattr(dataset, "__len__") else None
        if args.shuffle:
            dataset = dataset.shuffle(seed=args.seed)
        if args.max_rows is not None and total_rows is not None:
            dataset = dataset.select(range(min(args.max_rows, total_rows)))

    return dataset, dataset_label, time.perf_counter() - load_start, total_rows


def parquet_total_rows(paths: list[str] | None) -> int | None:
    if not paths:
        return None
    try:
        import pyarrow.parquet as pq

        return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    except Exception:
        return None


def rowpack_payload_format(path: str) -> str:
    candidate = Path(path)
    with candidate.open("rb") as handle:
        header = unpack_header(handle.read(HEADER_SIZE))
        handle.seek(header.metadata_offset)
        metadata = json.loads(handle.read(header.metadata_size).decode("utf-8"))
    return metadata.get("payload_format", "json")


def safe_len(obj) -> int | None:
    try:
        return len(obj)
    except Exception:
        return None


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    raise FileNotFoundError(path)


def dataset_storage_report(dataset, args, dataset_label: str, total_rows: int | None) -> dict[str, Any]:
    cache_files = []
    for cache_file in getattr(dataset, "cache_files", []) or []:
        filename = cache_file.get("filename")
        if not filename:
            continue
        path = Path(filename)
        cache_files.append({
            "path": str(path),
            "bytes": path.stat().st_size if path.exists() else None,
        })

    rows_profiled = safe_len(dataset)
    if rows_profiled is None and (args.streaming or args.loader in {"pyarrow", "rowpack"}):
        rows_profiled = args.max_rows

    report: dict[str, Any] = {
        "dataset": dataset_label,
        "dataset_total_rows": total_rows,
        "rows_profiled": rows_profiled,
        "cache_files": cache_files,
        "cache_bytes": sum(item["bytes"] or 0 for item in cache_files),
        "extra_size_paths": {},
    }

    for entry in args.size_path:
        label, _, raw_path = entry.partition("=")
        if not label or not raw_path:
            raise ValueError(f"--size-path must be LABEL=PATH, got {entry!r}")
        report["extra_size_paths"][label] = {
            "path": raw_path,
            "bytes": path_size(Path(raw_path)),
        }

    if not args.skip_hub_size and args.data_files is None:
        try:
            from huggingface_hub import HfApi

            info = HfApi().repo_info(args.dataset, repo_type="dataset", files_metadata=True)
            files = [
                {"path": sibling.rfilename, "bytes": sibling.size}
                for sibling in info.siblings
                if sibling.size is not None
            ]
            report["hub_revision"] = info.sha
            report["hub_files"] = files
            report["hub_bytes"] = sum(item["bytes"] for item in files)
            report["hub_parquet_bytes"] = sum(
                item["bytes"] for item in files if item["path"].endswith(".parquet")
            )
        except Exception as exc:  # Network size metadata is useful, not required.
            report["hub_size_error"] = repr(exc)

    parquet_bytes = report.get("hub_parquet_bytes") or 0
    if parquet_bytes and report["cache_bytes"]:
        report["arrow_cache_to_hub_parquet_ratio"] = report["cache_bytes"] / parquet_bytes

    return report


def make_dataloader(dataset, tokenizer, vlm_cfg, args):
    image_processor = get_image_processor(
        vlm_cfg.max_img_size,
        vlm_cfg.vit_img_size,
        vlm_cfg.resize_to_max_side_len,
    )
    if isinstance(dataset, NativeCistaVQARows):
        vqa_dataset = RowPackDirectVQAIterableDataset(
            dataset,
            tokenizer,
            image_processor,
            vlm_cfg.mp_image_token_length,
            max_images=args.max_images,
        )
    elif args.streaming or args.loader in {"pyarrow", "rowpack"}:
        vqa_dataset = VQAIterableDataset(
            dataset,
            tokenizer,
            image_processor,
            vlm_cfg.mp_image_token_length,
            max_images=args.max_images,
        )
    else:
        adapted = MMInfographicVQAAdapter(dataset, max_images=args.max_images)
        vqa_dataset = VQADataset(
            adapted,
            tokenizer,
            image_processor,
            vlm_cfg.mp_image_token_length,
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False if (args.streaming or args.loader in {"pyarrow", "rowpack"}) else args.shuffle,
        "num_workers": args.num_workers,
        "collate_fn": VQACollator(tokenizer, vlm_cfg.lm_max_length),
        "drop_last": False,
        "generator": generator,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0 and args.prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(vqa_dataset, **dataloader_kwargs)


def is_valid_batch(batch) -> bool:
    return bool(batch) and hasattr(batch.get("input_ids"), "numel") and batch["input_ids"].numel() > 0


def next_valid_batch(iterator):
    skipped = 0
    while True:
        batch = next(iterator)
        if is_valid_batch(batch):
            return batch, skipped
        skipped += 1


def summarize(values):
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def train_smoke(loader, tokenizer, vlm_cfg, args):
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    model = VisionLanguageModel(vlm_cfg, load_backbone=False).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    iterator = iter(loader)
    rows = []
    skipped_batches = 0

    for step_idx in range(args.warmup_steps + args.steps):
        data_start = time.perf_counter()
        batch, skipped = next_valid_batch(iterator)
        data_time = time.perf_counter() - data_start
        skipped_batches += skipped

        move_start = time.perf_counter()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["images"]
        move_time = time.perf_counter() - move_start

        compute_start = time.perf_counter()
        _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)
        loss.backward()
        compute_time = time.perf_counter() - compute_start

        optim_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optim_time = time.perf_counter() - optim_start

        if step_idx < args.warmup_steps:
            continue

        tokens = int(attention_mask.sum().item())
        image_count = sum(len(image_pack) for image_pack in images)
        total_time = data_time + move_time + compute_time + optim_time
        rows.append({
            "step": step_idx - args.warmup_steps,
            "loss": float(loss.item()),
            "samples": int(input_ids.shape[0]),
            "tokens": tokens,
            "images": image_count,
            "data_time_s": data_time,
            "move_time_s": move_time,
            "compute_time_s": compute_time,
            "optimizer_time_s": optim_time,
            "total_measured_time_s": total_time,
            "samples_per_s": float(input_ids.shape[0] / total_time) if total_time else 0.0,
            "tokens_per_s": float(tokens / total_time) if total_time else 0.0,
        })

    data_times = [row["data_time_s"] for row in rows]
    compute_times = [row["compute_time_s"] for row in rows]
    optimizer_times = [row["optimizer_time_s"] for row in rows]
    move_times = [row["move_time_s"] for row in rows]
    total_times = [row["total_measured_time_s"] for row in rows]
    total_data = sum(data_times)
    total_compute = sum(compute_times)
    total_optim = sum(optimizer_times)
    total_move = sum(move_times)
    total_measured = sum(total_times)

    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "steps": rows,
        "skipped_empty_batches": skipped_batches,
        "summary": {
            "loss_mean": mean([row["loss"] for row in rows]) if rows else 0.0,
            "samples": sum(row["samples"] for row in rows),
            "tokens": sum(row["tokens"] for row in rows),
            "images": sum(row["images"] for row in rows),
            "data_time_s": summarize(data_times),
            "move_time_s": summarize(move_times),
            "compute_time_s": summarize(compute_times),
            "optimizer_time_s": summarize(optimizer_times),
            "total_measured_time_s": summarize(total_times),
            "data_time_fraction": total_data / total_measured if total_measured else 0.0,
            "compute_time_fraction": total_compute / total_measured if total_measured else 0.0,
            "optimizer_time_fraction": total_optim / total_measured if total_measured else 0.0,
            "move_time_fraction": total_move / total_measured if total_measured else 0.0,
            "samples_per_s": (
                sum(row["samples"] for row in rows) / total_measured if total_measured else 0.0
            ),
            "tokens_per_s": (
                sum(row["tokens"] for row in rows) / total_measured if total_measured else 0.0
            ),
        },
    }


def benchmark_loader_only(loader, args):
    target_batches = max(0, int(args.loader_benchmark_batches))
    if target_batches <= 0:
        return None

    iterator = iter(loader)
    warmup_valid_batches = 0
    warmup_skipped_batches = 0
    for _ in range(max(0, int(args.loader_benchmark_warmup_batches))):
        try:
            _batch, skipped = next_valid_batch(iterator)
        except StopIteration:
            break
        warmup_valid_batches += 1
        warmup_skipped_batches += skipped

    rows = []
    skipped_batches = 0
    for batch_idx in range(target_batches):
        start = time.perf_counter()
        try:
            batch, skipped = next_valid_batch(iterator)
        except StopIteration:
            break
        elapsed = time.perf_counter() - start
        skipped_batches += skipped
        samples = int(batch["input_ids"].shape[0])
        tokens = int(batch["attention_mask"].sum().item())
        image_count = sum(len(image_pack) for image_pack in batch["images"])
        rows.append({
            "batch": batch_idx,
            "samples": samples,
            "tokens": tokens,
            "images": image_count,
            "batch_time_s": elapsed,
            "samples_per_s": samples / elapsed if elapsed else 0.0,
            "tokens_per_s": tokens / elapsed if elapsed else 0.0,
        })

    batch_times = [row["batch_time_s"] for row in rows]
    total_time = sum(batch_times)
    total_samples = sum(row["samples"] for row in rows)
    total_tokens = sum(row["tokens"] for row in rows)
    total_images = sum(row["images"] for row in rows)
    return {
        "requested_batches": target_batches,
        "warmup_batches": warmup_valid_batches,
        "warmup_skipped_empty_batches": warmup_skipped_batches,
        "batches": rows,
        "skipped_empty_batches": skipped_batches,
        "summary": {
            "batches": len(rows),
            "samples": total_samples,
            "tokens": total_tokens,
            "images": total_images,
            "batch_time_s": summarize(batch_times),
            "total_time_s": total_time,
            "batches_per_s": len(rows) / total_time if total_time else 0.0,
            "samples_per_s": total_samples / total_time if total_time else 0.0,
            "tokens_per_s": total_tokens / total_time if total_time else 0.0,
            "images_per_s": total_images / total_time if total_time else 0.0,
        },
    }


def main():
    args = parse_args()
    dataset, dataset_label, load_time, total_rows = load_split(args)

    base_cfg = VLMConfig()
    tokenizer = get_tokenizer(base_cfg.lm_tokenizer, base_cfg.vlm_extra_tokens, base_cfg.lm_chat_template)
    vlm_cfg = build_tiny_vlm_config(tokenizer, args)
    loader = make_dataloader(dataset, tokenizer, vlm_cfg, args)

    storage = dataset_storage_report(dataset, args, dataset_label, total_rows)
    loader_metrics = benchmark_loader_only(loader, args)
    if loader_metrics is not None:
        loader = make_dataloader(dataset, tokenizer, vlm_cfg, args)
    train_metrics = train_smoke(loader, tokenizer, vlm_cfg, args)

    result = {
        "config": {
            "dataset": args.dataset,
            "split": args.split,
            "data_files": args.data_files,
            "loader": args.loader,
            "parquet_batch_size": args.parquet_batch_size,
            "rowpack_native_dir": args.rowpack_native_dir,
            "rowpack_native_decode_images": args.rowpack_native_decode_images,
            "rowpack_direct_vqa": args.rowpack_direct_vqa,
            "read_pattern": args.read_pattern,
            "read_block_size": args.read_block_size,
            "max_rows": args.max_rows,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "loader_benchmark_batches": args.loader_benchmark_batches,
            "loader_benchmark_warmup_batches": args.loader_benchmark_warmup_batches,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "shuffle": args.shuffle,
            "shuffle_buffer": args.shuffle_buffer,
            "streaming": args.streaming,
            "sequence_length": args.sequence_length,
            "image_size": args.image_size,
            "max_images": args.max_images,
            "seed": args.seed,
        },
        "dataset_load_time_s": load_time,
        "storage": storage,
        "loader_benchmark": loader_metrics,
        "training": train_metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = train_metrics["summary"]
    print(json.dumps({
        "output": str(output),
        "dataset_load_time_s": load_time,
        "dataset_total_rows": storage["dataset_total_rows"],
        "rows_profiled": storage["rows_profiled"],
        "cache_bytes": storage["cache_bytes"],
        "hub_parquet_bytes": storage.get("hub_parquet_bytes"),
        "arrow_cache_to_hub_parquet_ratio": storage.get("arrow_cache_to_hub_parquet_ratio"),
        "samples_per_s": summary["samples_per_s"],
        "tokens_per_s": summary["tokens_per_s"],
        "data_time_fraction": summary["data_time_fraction"],
        "compute_time_fraction": summary["compute_time_fraction"],
        "optimizer_time_fraction": summary["optimizer_time_fraction"],
        "data_time_mean_s": summary["data_time_s"]["mean"],
        "compute_time_mean_s": summary["compute_time_s"]["mean"],
        "loader_batches_per_s": loader_metrics["summary"]["batches_per_s"] if loader_metrics else None,
        "loader_batch_time_mean_s": loader_metrics["summary"]["batch_time_s"]["mean"] if loader_metrics else None,
        "loss_mean": summary["loss_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
