import io
import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from data.datasets import VQADataset
from data.processors import get_image_string


class RowPackNativeVQADataset(IterableDataset):
    """Adapt RowPack native VQA tuples to nanoVLM training samples."""

    def __init__(self, rowpack_rows, tokenizer, image_processor, mp_image_token_length, max_images: int = 1):
        self.rowpack_rows = rowpack_rows
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.max_images = max_images
        self.processor = VQADataset([], tokenizer, image_processor, mp_image_token_length)

    def __len__(self):
        return len(self.rowpack_rows)

    def __iter__(self):
        for _row_id, text_pairs, image_payloads in self.rowpack_rows:
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


def is_raw_rgb_payload(payload) -> bool:
    return isinstance(payload, dict) and payload.get("storage") == "raw_rgb"


def is_qoi_payload(payload) -> bool:
    return isinstance(payload, dict) and payload.get("storage") == "qoi_lossless"


def qoi_payload_to_raw_rgb_payload(payload):
    """Decode an in-memory QOI payload to a raw-RGB payload using the native module."""
    try:
        from rowpack.native import load_native
        native = load_native()
    except Exception as exc:  # pragma: no cover - surfaced as RuntimeError below
        native = None
        native_error = exc
    else:
        native_error = None
    if native is None or not hasattr(native, "qoi_decode_rgb"):
        raise RuntimeError(
            "QOI image payloads require the rowpack_native module built with QOI support "
            "(rowpack_native.qoi_decode_rgb is unavailable)."
        ) from native_error
    return native.qoi_decode_rgb(bytes(payload["bytes"]))


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
