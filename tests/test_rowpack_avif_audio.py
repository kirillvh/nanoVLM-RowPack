"""Integration tests for RowPack's AVIF image-sequence codec and audio file support.

These exercise:
  - native ``avif_encode_rgb_sequence`` / ``avif_decode_rgb_sequence`` round-trips
  - the high-level ``LibAvifVideoEncoder`` / ``LibAvifVideoDecoder`` wrappers
  - persisting an AVIF chunk into a ``.rowpack`` file and reading it back
  - persisting WAV / Opus / FLAC / raw PCM audio attachments
  - nanoVLM-side decoding of single-frame AVIF stored as an "encoded" image
    in CISTA payloads (verifying ``data/rowpack_datasets.py`` handles it)

Run with::

    PYTHONPATH= ROWPACK_NATIVE_DIR=$PWD/rowpack_build_avif \
        python -m pytest tests/test_rowpack_avif_audio.py -v
"""

from __future__ import annotations

import io
import math
import os
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module unless an AVIF-capable native build is available.
NATIVE_DIR = os.environ.get("ROWPACK_NATIVE_DIR")
if not NATIVE_DIR:
    pytest.skip(
        "ROWPACK_NATIVE_DIR must point at a RowPack build (e.g. rowpack_build_avif)",
        allow_module_level=True,
    )

from rowpack import RowPackReader  # noqa: E402
from rowpack.authoring import MetadataBuilder, RowPackDatasetBuilder  # noqa: E402
from rowpack.native import load_native  # noqa: E402
from rowpack.video import (  # noqa: E402
    LibAvifVideoDecoder,
    LibAvifVideoEncoder,
    VideoFrame,
    libavif_available,
    libavif_decode_available,
)


def _native():
    return load_native(NATIVE_DIR)


def _require_libavif():
    if not hasattr(_native(), "avif_encode_rgb_sequence"):
        pytest.skip("native module was built without libavif (-DROWPACK_ENABLE_LIBAVIF=ON)")


def _gradient_frame(width: int, height: int, phase: int) -> bytes:
    """Generate a deterministic 24-bit RGB frame with a smooth gradient.

    libavif uses YUV subsampling internally, so true pixel-exact round-trips
    are not expected; smooth content keeps quantization artifacts low so we
    can assert tight PSNR bounds.
    """
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    r = (xx * 255 // max(1, width - 1)).astype(np.uint8)
    g = (yy * 255 // max(1, height - 1)).astype(np.uint8)
    b = np.full_like(r, (phase * 17) & 0xFF)
    return np.stack([r, g, b], axis=-1).tobytes()


def _psnr(reference: bytes, candidate: bytes) -> float:
    a = np.frombuffer(reference, dtype=np.uint8).astype(np.float32)
    b = np.frombuffer(candidate, dtype=np.uint8).astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return math.inf
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


# ---------------------------------------------------------------------------
# Native AVIF binding tests
# ---------------------------------------------------------------------------


def test_native_avif_runtime_info_reports_libavif_and_codecs():
    _require_libavif()
    info = _native().avif_runtime_info()
    assert isinstance(info, dict)
    assert "version" in info
    assert "codec_versions" in info
    # We built against system aom + dav1d; both should appear.
    codecs = info["codec_versions"].lower()
    assert "aom" in codecs or "dav1d" in codecs


def test_native_avif_single_frame_round_trip_is_high_psnr():
    _require_libavif()
    native = _native()
    width, height = 96, 64
    frame = _gradient_frame(width, height, phase=0)

    payload = native.avif_encode_rgb_sequence(
        [frame], height, width, 1.0, 90, 6, 1, "yuv420"
    )
    payload_bytes = bytes(payload)
    assert len(payload_bytes) > 64
    # First 4 bytes are the box size; bytes 4..8 should be "ftyp".
    assert payload_bytes[4:8] == b"ftyp"

    decoded = native.avif_decode_rgb_sequence(payload_bytes, 1)
    assert decoded["width"] == width
    assert decoded["height"] == height
    assert decoded["channels"] == 3
    assert decoded["frame_count"] == 1
    out = bytes(decoded["frames"][0])
    assert len(out) == width * height * 3
    psnr = _psnr(frame, out)
    assert psnr >= 35.0, f"AVIF single-frame PSNR {psnr:.2f} dB is too low"


def test_native_avif_multi_frame_sequence_round_trip_preserves_count_and_fps():
    _require_libavif()
    native = _native()
    width, height = 64, 48
    fps = 12.0
    frames = [_gradient_frame(width, height, phase=i) for i in range(8)]

    payload = native.avif_encode_rgb_sequence(
        frames, height, width, fps, 80, 8, 1, "yuv420"
    )
    payload_bytes = bytes(payload)

    decoded = native.avif_decode_rgb_sequence(payload_bytes, 1)
    assert decoded["frame_count"] == len(frames)
    assert decoded["width"] == width and decoded["height"] == height
    # libavif rounds FPS through a timescale; allow a small relative tolerance.
    assert abs(decoded["fps"] - fps) < 0.5
    # Per-frame PSNR should still be reasonable across the whole sequence.
    for ref, out_frame in zip(frames, decoded["frames"]):
        assert len(out_frame) == width * height * 3
        assert _psnr(ref, bytes(out_frame)) >= 30.0


@pytest.mark.parametrize("yuv_format", ["yuv420", "yuv422", "yuv444"])
def test_native_avif_supports_all_yuv_formats(yuv_format):
    _require_libavif()
    native = _native()
    width, height = 48, 32
    frame = _gradient_frame(width, height, phase=3)

    payload = native.avif_encode_rgb_sequence(
        [frame], height, width, 1.0, 85, 8, 1, yuv_format
    )
    decoded = native.avif_decode_rgb_sequence(bytes(payload), 1)
    assert decoded["frame_count"] == 1
    out = bytes(decoded["frames"][0])
    # 4:4:4 should be at least as accurate as 4:2:0.
    psnr = _psnr(frame, out)
    assert psnr >= 30.0, f"{yuv_format} PSNR {psnr:.2f} dB is too low"


# ---------------------------------------------------------------------------
# High-level video.py wrapper tests
# ---------------------------------------------------------------------------


def test_libavif_helpers_report_availability():
    _require_libavif()
    assert libavif_available(NATIVE_DIR) is True
    assert libavif_decode_available(NATIVE_DIR) is True


def test_libavif_video_encoder_and_decoder_roundtrip():
    _require_libavif()
    width, height = 64, 48
    frames = [
        VideoFrame(
            timestamp_ns=int(i * 1e8),
            data=_gradient_frame(width, height, phase=i),
            height=height,
            width=width,
            channels=3,
        )
        for i in range(4)
    ]

    encoder = LibAvifVideoEncoder(
        quality=85, speed=8, max_threads=1, yuv_format="yuv420",
        native_module_dir=NATIVE_DIR,
    )
    chunk = encoder.encode(frames, fps=10.0)
    assert chunk["codec"] == "avif"
    assert chunk["mime_type"] == "image/avif"
    assert chunk["frame_count"] == 4
    assert chunk["height"] == height and chunk["width"] == width

    decoder = LibAvifVideoDecoder(max_threads=1, native_module_dir=NATIVE_DIR)
    decoded = decoder.decode(chunk)
    assert decoded["frame_count"] == 4
    for ref, out_frame in zip(frames, decoded["frames"]):
        assert _psnr(ref.data, out_frame) >= 30.0


# ---------------------------------------------------------------------------
# AVIF + RowPack writer round-trip
# ---------------------------------------------------------------------------


def test_rowpack_writes_and_reads_freshly_encoded_avif_chunk(tmp_path):
    _require_libavif()
    width, height = 64, 48
    frames = [
        VideoFrame(
            timestamp_ns=int(i * 1e8),
            data=_gradient_frame(width, height, phase=i),
            height=height,
            width=width,
            channels=3,
        )
        for i in range(3)
    ]
    encoder = LibAvifVideoEncoder(
        quality=80, speed=8, max_threads=1, yuv_format="yuv420",
        native_module_dir=NATIVE_DIR,
    )
    chunk = encoder.encode(frames, fps=15.0)

    path = tmp_path / "fresh_avif.rowpack"
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder().dataset_name("fresh_avif"),
        payload_format="cista",
        block_codec="lzav_hi",
        native_module_dir=NATIVE_DIR,
        overwrite=True,
    ) as builder:
        builder.append_video_chunk_row(
            stream="front_camera",
            chunk=chunk,
            chunk_index=0,
            codec="avif",
            mime_type="image/avif",
            start_timestamp_ns=int(frames[0].timestamp_ns),
            end_timestamp_ns=int(frames[-1].timestamp_ns),
            frame_count=len(frames),
            fps=15.0,
        )

    with RowPackReader(path, native_module_dir=NATIVE_DIR) as reader:
        assert len(reader) == 1
        row = reader.read_row(0)
        assert row["_rowpack_continuation"]["kind"] == "video_chunk"
        file_payload = row["files"][0]
        assert file_payload["codec"] == "avif"
        assert file_payload["mime_type"] == "image/avif"
        assert file_payload["frame_count"] == 3
        # Decode the file payload bytes back to RGB and verify.
        decoded = LibAvifVideoDecoder(native_module_dir=NATIVE_DIR).decode(
            {"bytes": file_payload["bytes"]}
        )
        assert decoded["frame_count"] == 3
        for ref, out_frame in zip(frames, decoded["frames"]):
            assert _psnr(ref.data, out_frame) >= 30.0


# ---------------------------------------------------------------------------
# Audio attachment tests
# ---------------------------------------------------------------------------


def _make_wav_bytes(sample_rate: int = 16000, duration_s: float = 0.25, freq: float = 440.0) -> bytes:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float32) / sample_rate
    samples = (0.4 * np.sin(2 * math.pi * freq * t) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_raw_pcm_bytes(sample_rate: int = 16000, duration_s: float = 0.1) -> bytes:
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float32) / sample_rate
    samples = (0.3 * np.sin(2 * math.pi * 1000.0 * t) * 32767.0).astype(np.int16)
    return samples.tobytes()


@pytest.mark.parametrize("payload_format", ["json", "cista"])
def test_rowpack_stores_wav_audio_file_attachment(tmp_path, payload_format):
    """WAV audio survives a full RowPack write/read cycle as a file attachment."""
    audio_bytes = _make_wav_bytes(sample_rate=16000, duration_s=0.25)
    path = tmp_path / f"audio_wav_{payload_format}.rowpack"

    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder()
            .dataset_name("audio_wav")
            .row_field("files", "file[]", "WAV audio attachments"),
        payload_format=payload_format,
        block_codec="lzav_hi",
        native_module_dir=NATIVE_DIR,
        overwrite=True,
    ) as builder:
        builder.append_file_row(
            files=[{
                "bytes": audio_bytes,
                "name": "speech.wav",
                "mime_type": "audio/wav",
                "sample_rate": 16000,
                "channels": 1,
                "duration_s": 0.25,
            }],
            extra={"sensors": {"mic_0": {"role": "primary"}}},
            name="audio_row_0",
        )

    with RowPackReader(path, native_module_dir=NATIVE_DIR) as reader:
        row = reader.read_row(0)
        files = row["files"]
        assert len(files) == 1
        attachment = files[0]
        assert attachment["mime_type"] == "audio/wav"
        assert attachment["name"] == "speech.wav"
        # Exact byte fidelity: WAV is uncompressed, so RowPack must not mutate it.
        assert bytes(attachment["bytes"]) == audio_bytes
        assert attachment["size"] == len(audio_bytes)
        # Custom metadata fields should survive.
        assert int(attachment["sample_rate"]) == 16000
        assert int(attachment["channels"]) == 1
        # Reopen the recovered bytes through the wave module to verify it is
        # still a valid WAV file.
        with wave.open(io.BytesIO(bytes(attachment["bytes"])), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 16000
            assert wav.getnframes() == int(16000 * 0.25)


def test_rowpack_stores_audio_chunk_as_video_chunk_row(tmp_path):
    """Reuse the ``append_video_chunk_row`` path for an audio stream chunk.

    RowPack has no first-class audio codec; the supported pattern is to attach
    the encoded audio bytes as a file with an ``audio/*`` mime type so the
    ``_rowpack_continuation`` stream metadata is still recorded.
    """
    audio_bytes = _make_wav_bytes(sample_rate=22050, duration_s=0.1, freq=880.0)
    path = tmp_path / "audio_chunked.rowpack"
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder()
            .dataset_name("audio_stream")
            .row_field("files", "file[]", "Streamed audio chunks"),
        payload_format="cista",
        block_codec="lzav_hi",
        native_module_dir=NATIVE_DIR,
        overwrite=True,
    ) as builder:
        for chunk_index in range(2):
            builder.append_video_chunk_row(
                stream="mic_0",
                chunk={
                    "bytes": audio_bytes,
                    "name": f"mic_0_chunk_{chunk_index:06d}.wav",
                },
                chunk_index=chunk_index,
                codec="wav",
                mime_type="audio/wav",
                start_timestamp_ns=chunk_index * 100_000_000,
                end_timestamp_ns=(chunk_index + 1) * 100_000_000,
                frame_count=int(22050 * 0.1),
                fps=22050.0,
            )

    with RowPackReader(path, native_module_dir=NATIVE_DIR) as reader:
        assert len(reader) == 2
        for chunk_index in range(2):
            row = reader.read_row(chunk_index)
            cont = row["_rowpack_continuation"]
            assert cont["kind"] == "video_chunk"
            assert cont["stream"] == "mic_0"
            assert cont["chunk_index"] == chunk_index
            payload = row["files"][0]
            assert payload["mime_type"] == "audio/wav"
            assert payload["codec"] == "wav"
            assert bytes(payload["bytes"]) == audio_bytes


def test_rowpack_stores_raw_pcm_and_arbitrary_audio_mime_types(tmp_path):
    """Raw PCM and synthetic Opus/FLAC bytes survive a round-trip unchanged."""
    pcm_bytes = _make_raw_pcm_bytes(sample_rate=16000, duration_s=0.05)
    # Synthetic Opus / FLAC payloads: just deterministic blobs with the
    # canonical file magics so downstream tools can identify them.
    opus_bytes = b"OggS" + bytes(range(256)) * 4
    flac_bytes = b"fLaC" + bytes(range(64)) * 8

    path = tmp_path / "audio_variants.rowpack"
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder()
            .dataset_name("audio_variants")
            .row_field("files", "file[]", "Multi-codec audio attachments"),
        payload_format="cista",
        block_codec="lzav_hi",
        native_module_dir=NATIVE_DIR,
        overwrite=True,
    ) as builder:
        builder.append_file_row(
            files=[
                {"bytes": pcm_bytes, "name": "pcm.raw", "mime_type": "audio/L16",
                 "sample_rate": 16000, "channels": 1, "bits_per_sample": 16},
                {"bytes": opus_bytes, "name": "speech.opus", "mime_type": "audio/opus",
                 "sample_rate": 48000, "channels": 1},
                {"bytes": flac_bytes, "name": "speech.flac", "mime_type": "audio/flac",
                 "sample_rate": 44100, "channels": 2},
            ],
            name="multi_audio",
        )

    with RowPackReader(path, native_module_dir=NATIVE_DIR) as reader:
        row = reader.read_row(0)
        files = row["files"]
        assert [f["mime_type"] for f in files] == ["audio/L16", "audio/opus", "audio/flac"]
        assert bytes(files[0]["bytes"]) == pcm_bytes
        assert bytes(files[1]["bytes"]) == opus_bytes
        assert bytes(files[2]["bytes"]) == flac_bytes
        assert int(files[0]["sample_rate"]) == 16000
        assert int(files[2]["channels"]) == 2


# ---------------------------------------------------------------------------
# nanoVLM integration: AVIF single-frame as an "encoded" image payload
# ---------------------------------------------------------------------------


def test_nanovlm_loader_decodes_avif_image_payload(tmp_path):
    """Single-frame AVIF stored as an ``encoded`` image must reach nanoVLM.

    This validates the data adapter in ``data/rowpack_datasets.py``: when a
    CISTA payload contains an image with ``storage == "encoded"`` and AVIF
    bytes, the loader must produce a PIL image that nanoVLM can consume.
    """
    _require_libavif()
    native = _native()
    width, height = 64, 48
    frame = _gradient_frame(width, height, phase=2)
    avif_bytes = bytes(
        native.avif_encode_rgb_sequence(
            [frame], height, width, 1.0, 90, 6, 1, "yuv420"
        )
    )

    path = tmp_path / "avif_image.rowpack"
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder().dataset_name("avif_image"),
        payload_format="cista",
        block_codec="lzav_hi",
        native_module_dir=NATIVE_DIR,
        overwrite=True,
    ) as builder:
        encoded_image = builder.encode_image(
            {"bytes": avif_bytes, "height": height, "width": width, "channels": 3},
            codec="encoded",
        )
        # The "encoded" path keeps the AVIF bytes verbatim and tags storage.
        assert encoded_image["storage"] == "encoded"
        assert encoded_image["bytes"] == avif_bytes

        builder.append_vqa_row(
            turns=[
                {"role": "user", "modality": "text", "data": "Describe the image."},
                {"role": "assistant", "modality": "text", "data": "A smooth color gradient."},
            ],
            images=[encoded_image],
            extra={"timestamp_ns": 0},
            name="avif_vqa_0",
        )

    # Read the row through the same code path nanoVLM uses.
    from rowpack import RowPackRows

    rows = list(
        RowPackRows([str(path)], native_module_dir=NATIVE_DIR)
    )
    assert len(rows) == 1
    image_payload = rows[0]["images"][0]
    assert image_payload.get("storage") in {"encoded", None}
    assert bytes(image_payload["bytes"]) == avif_bytes

    # Now exercise the nanoVLM adapter that converts payloads to PIL/tensor.
    from data.rowpack_datasets import (
        is_qoi_payload,
        is_raw_rgb_payload,
    )

    # AVIF payloads should NOT be classified as QOI or raw-RGB so that the
    # generic "encoded" path is taken.
    assert not is_qoi_payload(image_payload)
    assert not is_raw_rgb_payload(image_payload)

    # The generic encoded path inside nanoVLM hands the bytes to PIL via
    # ``image_payload_to_pil`` in benchmarks/mm_infographic_vqa_baseline.py.
    # We exercise the equivalent decode here using the project's helper.
    from rowpack.video import LibAvifVideoDecoder
    decoded = LibAvifVideoDecoder(native_module_dir=NATIVE_DIR).decode(
        {"bytes": bytes(image_payload["bytes"])}
    )
    assert decoded["frame_count"] == 1
    out = bytes(decoded["frames"][0])
    assert len(out) == width * height * 3
    assert _psnr(frame, out) >= 35.0


def _have_pil_avif() -> bool:
    try:
        from PIL import features
        return bool(features.check("avif"))
    except Exception:
        return False


@pytest.mark.skipif(not _have_pil_avif(), reason="PIL build has no AVIF support")
def test_pil_decodes_avif_image_payload_via_image_payload_to_pil():
    """``data.rowpack_datasets.image_payload_to_pil`` must handle AVIF bytes."""
    _require_libavif()
    native = _native()
    width, height = 48, 32
    frame = _gradient_frame(width, height, phase=5)
    avif_bytes = bytes(
        native.avif_encode_rgb_sequence(
            [frame], height, width, 1.0, 90, 6, 1, "yuv420"
        )
    )

    from data.rowpack_datasets import image_payload_to_pil

    pil = image_payload_to_pil({"bytes": avif_bytes})
    assert pil.mode == "RGB"
    assert pil.size == (width, height)
    out = np.asarray(pil).tobytes()
    assert _psnr(frame, out) >= 30.0


def test_rowpack_native_vqa_dataset_yields_training_sample_for_avif(tmp_path):
    """Full nanoVLM data path: AVIF image → ``RowPackNativeVQADataset`` sample.

    Uses ``native_decode_images=False`` so the AVIF bytes are passed through to
    the Python PIL-based decoder. This is the practical wiring for AVIF inside
    nanoVLM today: the native CISTA decoder only handles raw / QOI / stb-image
    formats, while PIL gives us first-class AVIF support.
    """
    _require_libavif()
    if not _have_pil_avif():
        pytest.skip("PIL build has no AVIF support")

    # Heavy imports happen lazily so unrelated test runs don't pay for them.
    try:
        from data.processors import get_image_processor, get_tokenizer
    except Exception as exc:  # pragma: no cover - missing torchvision etc.
        pytest.skip(f"nanoVLM data pipeline unavailable: {exc}")
    from data.rowpack_datasets import RowPackNativeVQADataset
    from rowpack import NativeCistaVQARows

    native = _native()
    width, height = 48, 32
    frame = _gradient_frame(width, height, phase=7)
    avif_bytes = bytes(
        native.avif_encode_rgb_sequence(
            [frame], height, width, 1.0, 90, 6, 1, "yuv420"
        )
    )

    path = tmp_path / "avif_vqa.rowpack"
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder().dataset_name("avif_nanovlm"),
        payload_format="cista",
        block_codec="lzav_hi",
        native_module_dir=NATIVE_DIR,
        overwrite=True,
    ) as builder:
        encoded_image = builder.encode_image(
            {"bytes": avif_bytes, "height": height, "width": width, "channels": 3},
            codec="encoded",
        )
        builder.append_vqa_row(
            turns=[
                {"role": "user", "modality": "text", "data": "What colors do you see?"},
                {"role": "assistant", "modality": "text", "data": "Red, green and blue gradients."},
            ],
            images=[encoded_image],
            extra={"timestamp_ns": 0},
            name="avif_vqa_0",
        )

    # Build the same tokenizer + image processor combination used by train.py.
    from models.config import VLMConfig
    cfg = VLMConfig()
    try:
        tokenizer = get_tokenizer(
            cfg.lm_tokenizer,
            extra_special_tokens=cfg.vlm_extra_tokens,
        )
    except Exception as exc:  # pragma: no cover - offline / missing model
        pytest.skip(f"Tokenizer {cfg.lm_tokenizer!r} unavailable: {exc}")

    # Use a smaller image size to keep the test fast; the processor's behaviour
    # is identical at any resolution and we are not training a real model.
    image_processor = get_image_processor(max_img_size=128, splitted_image_size=64)

    rows = NativeCistaVQARows(
        [str(path)],
        native_module_dir=NATIVE_DIR,
        native_decode_images=False,  # let PIL handle the AVIF bytes
    )
    dataset = RowPackNativeVQADataset(
        rows,
        tokenizer,
        image_processor,
        mp_image_token_length=cfg.mp_image_token_length,
        max_images=1,
    )

    sample = next(iter(dataset))
    assert sample is not None
    assert "images" in sample and len(sample["images"]) == 1
    # ``RowPackNativeVQADataset`` stores per-image processor outputs (tensors
    # of shape ``(num_patches+1, 3, p, p)`` — global patch + split tiles).
    image_tensor = sample["images"][0]
    assert image_tensor.ndim == 4 and image_tensor.shape[1] == 3
    assert image_tensor.shape[0] >= 1
    assert sample["input_ids"].numel() > 0
    assert sample["input_ids"].shape == sample["attention_mask"].shape
    assert sample["input_ids"].shape == sample["labels"].shape
    # The user message should have been prefixed with the image token sequence.
    image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
    assert (sample["input_ids"] == image_token_id).any()
