from __future__ import annotations

import pytest
import os
import io
import importlib.util
from pathlib import Path

from rowpack import (
    DocumentIndexBuilder,
    MetadataBuilder,
    NativeCistaVQARows,
    RowPackBlockDataset,
    RowPackDatasetBuilder,
    RowPackLoaderState,
    RowPackReader,
    RowPackRows,
    RowPackWriter,
)
from PIL import Image
from rowpack.convert_jsonl import convert_jsonl_to_rowpack
from rowpack.convert_jsonl_parallel import convert_jsonl_to_rowpack_parallel
from rowpack.convert_parquet import convert_parquet_to_rowpack


def sample_row(idx: int) -> dict:
    return {
        "data": [
            {"role": "user", "modality": "text", "data": f"question {idx}"},
            {"role": "assistant", "modality": "text", "data": f"answer {idx}"},
        ],
        "images": [{"bytes": f"image-bytes-{idx}".encode("utf-8"), "path": None}],
        "source_id": idx,
    }


def rich_row() -> dict:
    return {
        "data": [
            {"role": "user", "modality": "text", "data": "What is in the chart?"},
            {"role": "assistant", "modality": "text", "data": "Revenue rose by 12%."},
            {"role": "user", "modality": "text", "data": "ありがとう"},
        ],
        "images": [
            {"bytes": bytes(range(256)), "path": None},
            {"bytes": b"\x00rowpack\xffimage\x10payload", "path": None},
        ],
        "source_id": 42,
        "metadata": {
            "document": "infographic-042",
            "scores": [1, 2, 3.5],
            "flags": {"train": True, "valid": False},
        },
    }


def multi_turn_row() -> dict:
    return {
        "data": [
            {"role": "user", "modality": "text", "data": " first question "},
            {"role": "user", "modality": "text", "data": "follow up"},
            {"role": "assistant", "modality": "text", "data": " first answer "},
            {"role": "system", "modality": "text", "data": "ignored"},
            {"role": "user", "modality": "image", "data": "ignored image turn"},
            {"role": "user", "modality": "text", "data": "second question"},
            {"role": "assistant", "modality": "text", "data": "second answer"},
        ],
        "images": [{"bytes": b"direct-image", "path": None}],
        "source_id": 99,
    }


def strip_rowpack_meta(row: dict) -> dict:
    row = dict(row)
    row.pop("_rowpack", None)
    return row


def test_rowpack_roundtrip(tmp_path):
    path = tmp_path / "sample.rowpack"
    with RowPackWriter(path, rows_per_block=2, metadata={"dataset": "unit"}, overwrite=True) as writer:
        writer.append_row(sample_row(0), name="first", aliases=["old_first"])
        writer.append_row(sample_row(1), name="second")
        writer.append_row(sample_row(2), name="third")

    with RowPackReader(path) as reader:
        assert len(reader) == 3
        assert reader.metadata["dataset"] == "unit"
        assert len(reader.blocks) == 2
        assert reader.read_row(1)["data"][0]["data"] == "question 1"
        assert reader.read_row(2)["images"][0]["bytes"] == b"image-bytes-2"
        assert reader.read_window(1, 8)[-1]["source_id"] == 2

        assert reader.row_id_for_name("first") == 0
        with pytest.warns(UserWarning):
            assert reader.row_id_for_name("old_first") == 0
        with pytest.raises(KeyError, match="first"):
            reader.row_id_for_name("frist")


def test_rowpack_document_search_index_roundtrip(tmp_path):
    path = tmp_path / "books.rowpack"
    index = DocumentIndexBuilder()
    index.observe(0, "book_a", labels=["A Practical Guide"], aliases=["old_book_a"])
    index.observe(1, "book_a", labels=["A Practical Guide"])
    index.observe(2, "book_b", labels=["Second Notes"])
    metadata = (
        MetadataBuilder()
        .dataset_name("book_corpus")
        .search_index("documents", index.finish(), schema=index.metadata_schema(key_column="book_id", label_columns=["title"]))
    )

    with RowPackDatasetBuilder(path, metadata=metadata, payload_format="json", block_codec="none", overwrite=True) as builder:
        builder.append_row({"book_id": "book_a", "text": "chapter 1"})
        builder.append_row({"book_id": "book_a", "text": "chapter 2"})
        builder.append_row({"book_id": "book_b", "text": "appendix"})

    with RowPackReader(path) as reader:
        matches = reader.find_index_entries("practical")
        assert matches[0]["key"] == "book_a"
        assert reader.index_entry_for_key("old_book_a")["row_count"] == 2
        rows = reader.read_index_entry("book_a")
        assert [row["text"] for row in rows] == ["chapter 1", "chapter 2"]
        with pytest.raises(KeyError, match="book_a"):
            reader.index_entry_for_key("bok_a")


def test_rowpack_lzav_default_roundtrip_when_native_available(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "sample_lzav.rowpack"
    with RowPackWriter(
        path,
        rows_per_block=2,
        block_codec="lzav_default",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        for idx in range(5):
            writer.append_row(sample_row(idx), name=f"row_{idx}")

    with RowPackReader(path, native_module_dir=native_dir) as reader:
        assert len(reader) == 5
        assert reader.metadata["block_codec"] == "lzav_default"
        assert reader.metadata["observed_compressions"] == ["lzav_default"]
        assert [reader.read_row(idx)["source_id"] for idx in range(5)] == list(range(5))
        assert reader.read_window(1, 3)[-1]["images"][0]["bytes"] == b"image-bytes-3"


def test_rowpack_preserves_nested_values_and_image_bytes(tmp_path):
    path = tmp_path / "rich.rowpack"
    expected = rich_row()

    with RowPackWriter(path, rows_per_block=1, overwrite=True) as writer:
        writer.append_row(expected, name="rich")

    with RowPackReader(path) as reader:
        actual = reader.read_row(0)
        assert actual["_rowpack"]["row_id"] == 0
        assert strip_rowpack_meta(actual) == expected
        assert actual["images"][0]["bytes"] == bytes(range(256))
        assert actual["images"][1]["bytes"] == b"\x00rowpack\xffimage\x10payload"


def test_rowpack_cista_payload_roundtrip_when_native_available(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "rich_cista.rowpack"
    expected = rich_row()

    with RowPackWriter(
        path,
        rows_per_block=1,
        payload_format="cista",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        writer.append_row(expected, name="rich")

    with RowPackReader(path, native_module_dir=native_dir) as reader:
        actual = reader.read_row(0)
        assert actual["_rowpack"]["row_id"] == 0
        assert reader.metadata["payload_format"] == "cista"
        actual_without_meta = strip_rowpack_meta(actual)
        assert actual_without_meta["data"] == expected["data"]
        assert actual_without_meta["source_id"] == expected["source_id"]
        assert actual_without_meta["metadata"] == expected["metadata"]
        assert [image["bytes"] for image in actual_without_meta["images"]] == [
            image["bytes"] for image in expected["images"]
        ]
        assert all(image["storage"] == "encoded" for image in actual_without_meta["images"])


def test_native_cista_vqa_rows_return_pairs_and_images(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "direct_cista.rowpack"
    with RowPackWriter(
        path,
        rows_per_block=1,
        payload_format="cista",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        writer.append_row(multi_turn_row(), name="multi")

    rows = NativeCistaVQARows([str(path)], native_module_dir=native_dir)
    row_id, pairs, images = next(iter(rows))
    assert row_id == 0
    assert pairs == [
        ("first question\nfollow up", "first answer"),
        ("second question", "second answer"),
    ]
    assert images == [
        {
            "bytes": b"direct-image",
            "height": 0,
            "width": 0,
            "channels": 0,
            "storage": "encoded",
        }
    ]


def test_native_cista_vqa_rows_read_lzav_blocks(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "direct_cista_lzav.rowpack"
    with RowPackWriter(
        path,
        rows_per_block=2,
        payload_format="cista",
        block_codec="lzav_default",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        writer.append_row(multi_turn_row(), name="multi_0")
        row = multi_turn_row()
        row["source_id"] = 100
        row["images"] = [{"bytes": b"second-image", "path": None}]
        writer.append_row(row, name="multi_1")

    rows = list(NativeCistaVQARows([str(path)], native_module_dir=native_dir))
    assert [row[0] for row in rows] == [0, 1]
    assert rows[0][1][0] == ("first question\nfollow up", "first answer")
    assert rows[1][2][0]["bytes"] == b"second-image"


def test_native_cista_vqa_rows_return_raw_rgb_metadata(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "raw_cista.rowpack"
    row = rich_row()
    row["images"] = [{
        "bytes": bytes(range(12)),
        "height": 2,
        "width": 2,
        "channels": 3,
        "storage": "raw_rgb",
    }]
    with RowPackWriter(
        path,
        rows_per_block=1,
        payload_format="cista",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        writer.append_row(row, name="raw")

    row_id, _pairs, images = next(iter(NativeCistaVQARows([str(path)], native_module_dir=native_dir)))
    assert row_id == 0
    assert images[0]["bytes"] == bytes(range(12))
    assert images[0]["height"] == 2
    assert images[0]["width"] == 2
    assert images[0]["channels"] == 3
    assert images[0]["storage"] == "raw_rgb"


def test_native_cista_vqa_rows_decode_qoi_lossless(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "qoi_cista.rowpack"
    raw_rgb = bytes([
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        255, 255, 255,
    ])
    row = rich_row()
    row["images"] = [{
        "bytes": raw_rgb,
        "height": 2,
        "width": 2,
        "channels": 3,
        "storage": "qoi_lossless",
    }]
    with RowPackWriter(
        path,
        rows_per_block=1,
        payload_format="cista",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        writer.append_row(row, name="qoi")

    row_id, _pairs, images = next(iter(NativeCistaVQARows([str(path)], native_module_dir=native_dir)))
    assert row_id == 0
    assert images[0]["bytes"] == raw_rgb
    assert images[0]["height"] == 2
    assert images[0]["width"] == 2
    assert images[0]["channels"] == 3
    assert images[0]["storage"] == "raw_rgb"


def test_native_cista_vqa_rows_decode_encoded_jpeg_with_stb(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), (120, 40, 200)).save(buffer, format="JPEG", quality=90)

    path = tmp_path / "jpeg_cista.rowpack"
    row = rich_row()
    row["images"] = [{"bytes": buffer.getvalue(), "path": None}]
    with RowPackWriter(
        path,
        rows_per_block=1,
        payload_format="cista",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        writer.append_row(row, name="jpeg")

    row_id, _pairs, images = next(iter(NativeCistaVQARows([str(path)], native_module_dir=native_dir)))
    assert row_id == 0
    assert images[0]["height"] == 2
    assert images[0]["width"] == 3
    assert images[0]["channels"] == 3
    assert images[0]["storage"] == "raw_rgb"
    assert len(images[0]["bytes"]) == 2 * 3 * 3


def test_native_cista_vqa_rows_can_return_encoded_images_without_decode(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), (120, 40, 200)).save(buffer, format="JPEG", quality=90)
    encoded = buffer.getvalue()

    path = tmp_path / "jpeg_cista_lazy.rowpack"
    row = rich_row()
    row["images"] = [{"bytes": encoded, "path": None}]
    with RowPackWriter(
        path,
        rows_per_block=1,
        payload_format="cista",
        native_module_dir=native_dir,
        overwrite=True,
    ) as writer:
        writer.append_row(row, name="jpeg")

    row_id, _pairs, images = next(iter(NativeCistaVQARows([str(path)], native_module_dir=native_dir, native_decode_images=False)))
    assert row_id == 0
    assert images[0]["bytes"] == encoded
    assert images[0]["height"] == 0
    assert images[0]["width"] == 0
    assert images[0]["channels"] == 0
    assert images[0]["storage"] == "encoded"


def test_rowpack_rows_sequential_and_random_block(tmp_path):
    path = tmp_path / "sample.rowpack"
    with RowPackWriter(path, rows_per_block=2, overwrite=True) as writer:
        for idx in range(5):
            writer.append_row(sample_row(idx), name=f"row_{idx}")

    rows = RowPackRows([str(path)], max_rows=3)
    assert len(rows) == 3
    assert [row["source_id"] for row in rows] == [0, 1, 2]

    random_a = RowPackRows(
        [str(path)],
        max_rows=4,
        read_pattern="random_block",
        read_block_size=2,
        seed=123,
    )
    random_b = RowPackRows(
        [str(path)],
        max_rows=4,
        read_pattern="random_block",
        read_block_size=2,
        seed=123,
    )
    assert [row["source_id"] for row in random_a] == [row["source_id"] for row in random_b]


def test_rowpack_block_dataset_reads_list_file_sequential_blocks(tmp_path):
    path_a = tmp_path / "a.rowpack"
    path_b = tmp_path / "b.rowpack"
    with RowPackWriter(path_a, rows_per_block=2, overwrite=True) as writer:
        for idx in range(4):
            writer.append_row(sample_row(idx), name=f"a_{idx}")
    with RowPackWriter(path_b, rows_per_block=2, overwrite=True) as writer:
        for idx in range(3):
            writer.append_row(sample_row(100 + idx), name=f"b_{idx}")

    list_path = tmp_path / "rowpacks.txt"
    list_path.write_text(f"{path_a.name}\n{path_b.name}\n", encoding="utf-8")

    dataset = RowPackBlockDataset(
        list_path,
        mode="sequential",
        state=RowPackLoaderState(file_index=0, block_index=1),
        max_rows=5,
    )

    assert dataset.state_dict() == {"file_index": 0, "block_index": 1, "seed": 0}
    assert [row["source_id"] for row in dataset] == [2, 3, 100, 101, 102]


def test_rowpack_block_dataset_shuffle_is_reproducible(tmp_path):
    paths = []
    for file_idx in range(2):
        path = tmp_path / f"{file_idx}.rowpack"
        with RowPackWriter(path, rows_per_block=2, overwrite=True) as writer:
            for row_idx in range(6):
                writer.append_row(sample_row(file_idx * 100 + row_idx), name=f"{file_idx}_{row_idx}")
        paths.append(path)

    state = RowPackLoaderState(file_index=7, block_index=11, seed=1234)
    dataset_a = RowPackBlockDataset(paths=paths, mode="shuffle", state=state, max_rows=8)
    dataset_b = RowPackBlockDataset(paths=paths, mode="shuffle", state=state.as_dict(), max_rows=8)

    values_a = [row["source_id"] for row in dataset_a]
    values_b = [row["source_id"] for row in dataset_b]
    assert values_a == values_b
    assert len(values_a) == 8


def test_metadata_builder_and_authoring_json_roundtrip(tmp_path):
    metadata = (
        MetadataBuilder()
        .dataset_name("robot_smoke")
        .description("unit-test capture")
        .row_field("timestamp_ns", "int64", "capture timestamp")
        .sensor("front_camera", "sensor_msgs.msg:Image", topic="/camera/front", frame_id="camera")
        .calibration("front_camera", fx=100.0, fy=101.0)
        .compression(block_codec="none", rows_per_block=2)
        .image_codec("encoded")
    )

    path = tmp_path / "authored_json.rowpack"
    with RowPackDatasetBuilder(
        path,
        metadata=metadata,
        rows_per_block=2,
        payload_format="json",
        block_codec="none",
        image_codec="encoded",
        overwrite=True,
    ) as builder:
        builder.append_sensor_row(
            {"imu": {"linear_acceleration": [1.0, 2.0, 3.0]}},
            images=[{"bytes": b"encoded-image", "height": 0, "width": 0, "channels": 0, "storage": "encoded"}],
            timestamp_ns=123,
            name="frame_0",
            aliases=["old_frame_0"],
        )

    with RowPackReader(path) as reader:
        assert reader.metadata["dataset_name"] == "robot_smoke"
        assert reader.metadata["row_schema"][0]["name"] == "timestamp_ns"
        assert reader.metadata["sensors"][0]["topic"] == "/camera/front"
        assert reader.read_row(0)["timestamp_ns"] == 123
        assert reader.read_row(0)["images"][0]["bytes"] == b"encoded-image"
        with pytest.warns(UserWarning):
            assert reader.row_id_for_name("old_frame_0") == 0


def test_rowpack_file_payload_roundtrip_json(tmp_path):
    path = tmp_path / "files.rowpack"
    payload = b"\x00rowpack arbitrary bytes\xff"

    with RowPackWriter(path, rows_per_block=2, metadata={"dataset": "files"}, overwrite=True) as writer:
        writer.append_row(
            {
                "source_id": "file_row",
                "files": [
                    {
                        "bytes": payload,
                        "name": "payload.bin",
                        "mime_type": "application/octet-stream",
                        "role": "attachment",
                    }
                ],
            }
        )

    with RowPackReader(path) as reader:
        row = reader.read_row(0)
        assert row["images"] == []
        assert row["files"][0]["bytes"] == payload
        assert row["files"][0]["name"] == "payload.bin"
        assert row["files"][0]["size"] == len(payload)


def test_rowpack_sample_avif_video_chunk_roundtrip(tmp_path):
    sample = Path("rowpack/examples/sampleavif.avif")
    if not sample.exists():
        pytest.skip("sample AVIF fixture is not present")

    path = tmp_path / "sample_avif.rowpack"
    sample_bytes = sample.read_bytes()
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder().dataset_name("sample_avif"),
        payload_format="json",
        block_codec="none",
        overwrite=True,
    ) as builder:
        builder.append_video_chunk_row(
            stream="front_camera",
            chunk={"bytes": sample_bytes, "name": sample.name},
            chunk_index=0,
            codec="avif",
            mime_type="image/avif",
            start_timestamp_ns=0,
            end_timestamp_ns=15_000_000_000,
            frame_count=0,
        )

    with RowPackReader(path) as reader:
        row = reader.read_row(0)
        assert row["_rowpack_continuation"]["kind"] == "video_chunk"
        assert row["files"][0]["bytes"] == sample_bytes
        assert row["files"][0]["mime_type"] == "image/avif"
        assert row["files"][0]["codec"] == "avif"
        assert row["files"][0]["stream"] == "front_camera"


def test_authoring_default_cista_lzav_and_jpeg_encode_when_native_available(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "authored_native.rowpack"
    raw_rgb = bytes([
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        255, 255, 255,
    ])
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder().dataset_name("native_authoring"),
        rows_per_block=2,
        native_module_dir=native_dir,
        overwrite=True,
    ) as builder:
        image = builder.encode_image(raw_rgb, codec="jpeg_lossy", height=2, width=2, channels=3, jpeg_quality=85)
        assert image["bytes"].startswith(b"\xff\xd8")
        builder.append_vqa_row(
            turns=[
                {"role": "user", "modality": "text", "data": "question"},
                {"role": "assistant", "modality": "text", "data": "answer"},
            ],
            images=[image],
            extra={"timestamp_ns": 456},
            name="vqa_0",
        )

    with RowPackReader(path, native_module_dir=native_dir) as reader:
        row = reader.read_row(0)
        assert reader.metadata["block_codec"] == "lzav_hi"
        assert reader.metadata["payload_format"] == "cista"
        assert row["timestamp_ns"] == 456
        assert row["images"][0]["bytes"].startswith(b"\xff\xd8")


def test_rowpack_file_payload_roundtrip_cista_when_native_available(tmp_path):
    native_dir = os.environ.get("ROWPACK_NATIVE_DIR")
    if not native_dir:
        pytest.skip("ROWPACK_NATIVE_DIR is not set")

    path = tmp_path / "files_cista.rowpack"
    payload = b"cista-file-payload"
    with RowPackDatasetBuilder(
        path,
        metadata=MetadataBuilder().dataset_name("cista_files"),
        rows_per_block=2,
        payload_format="cista",
        block_codec="none",
        native_module_dir=native_dir,
        overwrite=True,
    ) as builder:
        builder.append_file_row(
            [{"bytes": payload, "name": "payload.dat", "mime_type": "application/octet-stream"}],
            extra={"source_id": 7},
            name="file_0",
        )

    with RowPackReader(path, native_module_dir=native_dir) as reader:
        row = reader.read_row(0)
        assert row["source_id"] == 7
        assert row["images"] == []
        assert row["files"][0]["bytes"] == payload
        assert row["files"][0]["name"] == "payload.dat"


def test_generic_parquet_converter_roundtrip_when_pyarrow_available(tmp_path):
    if importlib.util.find_spec("pyarrow") is None:
        pytest.skip("pyarrow is not installed")

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = tmp_path / "generic.parquet"
    table = pa.table(
        {
            "id": ["row_0", "row_1"],
            "question": ["what is this?", "what color?"],
            "score": [1.25, 2.5],
            "image": [
                {"bytes": b"encoded-image-0", "path": None},
                {"bytes": b"encoded-image-1", "path": None},
            ],
        }
    )
    pq.write_table(table, parquet_path)

    rowpack_path = tmp_path / "generic.rowpack"
    rows = convert_parquet_to_rowpack(
        [parquet_path],
        output=rowpack_path,
        image_columns={"image"},
        name_column="id",
        index_column="id",
        index_label_columns=["question"],
        overwrite=True,
    )

    assert rows == 2
    with RowPackReader(rowpack_path) as reader:
        assert len(reader) == 2
        assert reader.metadata["source_format"] == "parquet"
        assert reader.row_id_for_name("row_1") == 1
        assert reader.find_index_entries("color")[0]["key"] == "row_1"
        row = reader.read_row(0)
        assert row["id"] == "row_0"
        assert row["question"] == "what is this?"
        assert row["score"] == 1.25
        assert row["images"][0]["bytes"] == b"encoded-image-0"


def test_jsonl_converter_lamini_docs_shape_roundtrip(tmp_path):
    jsonl_path = tmp_path / "lamini_docs_sample.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                '{"question":"How can I evaluate generated text?","answer":"Use perplexity, BLEU, and human evaluation."}',
                '{"question":"Can I fine tune a model?","answer":"Yes, prepare instruction response pairs and train carefully."}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rowpack_path = tmp_path / "lamini_docs_sample.rowpack"
    rows = convert_jsonl_to_rowpack(
        [jsonl_path],
        output=rowpack_path,
        dataset_name="lamini_docs_sample",
        overwrite=True,
    )

    assert rows == 2
    with RowPackReader(rowpack_path) as reader:
        assert len(reader) == 2
        assert reader.metadata["source_format"] == "jsonl"
        assert reader.metadata["dataset_name"] == "lamini_docs_sample"
        row = reader.read_row(0)
        assert row["question"] == "How can I evaluate generated text?"
        assert "perplexity" in row["answer"]


def test_jsonl_converter_splits_long_columns_into_continuation_rows(tmp_path):
    jsonl_path = tmp_path / "long_docs.jsonl"
    long_answer = " ".join(f"token{i}" for i in range(18))
    jsonl_path.write_text(
        '{"id":"doc_0","title":"Short title","answer":"' + long_answer + '"}\n',
        encoding="utf-8",
    )

    rowpack_path = tmp_path / "long_docs.rowpack"
    rows = convert_jsonl_to_rowpack(
        [jsonl_path],
        output=rowpack_path,
        name_column="id",
        index_column="id",
        index_label_columns=["title"],
        split_columns={"answer"},
        split_max_chars=32,
        overwrite=True,
    )

    assert rows > 1
    with RowPackReader(rowpack_path) as reader:
        assert len(reader) == rows
        assert reader.metadata["split_policy"]["columns"] == ["answer"]
        first = reader.read_row(0)
        second = reader.read_row(1)
        assert first["id"] == "doc_0"
        assert first["title"] == "Short title"
        assert first["_rowpack_split"]["part_index"] == 0
        assert second["_rowpack_split"]["is_continuation"] is True
        assert "title" not in second
        assert "answer" in second
        assert reader.row_id_for_name("doc_0") == 0
        assert reader.row_id_for_name("doc_0::part_0001") == 1
        indexed_rows = reader.read_index_entry("Short title")
        assert len(indexed_rows) == rows
        assert indexed_rows[0]["_rowpack_split"]["part_index"] == 0


def test_parallel_jsonl_converter_splits_and_indexes_sample_books(tmp_path):
    sample_path = Path("rowpack/examples/sample_books.jsonl")
    rowpack_path = tmp_path / "sample_books_parallel.rowpack"

    metrics = convert_jsonl_to_rowpack_parallel(
        [sample_path],
        output=rowpack_path,
        columns=["meta", "text"],
        split_columns={"text"},
        split_max_chars=180,
        split_overlap_chars=20,
        index_column="meta.short_book_title",
        index_label_columns=["meta.url"],
        rows_per_block=4,
        payload_format="json",
        block_codec="none",
        workers=1,
        overwrite=True,
    )

    assert metrics["input_records"] == 3
    assert metrics["rows_written"] == 8
    assert metrics["blocks_written"] == 2
    with RowPackReader(rowpack_path) as reader:
        assert len(reader) == 8
        assert reader.metadata["converter"] == "rowpack.convert_jsonl_parallel"
        assert reader.metadata["split_policy"]["max_chars"] == 180
        assert reader.metadata["search_index_config"]["key_column"] == "meta.short_book_title"
        entries = {entry["key"]: entry for entry in reader.search_index()}
        assert entries["Tiny Field Guide"]["row_count"] == 2
        assert entries["Long Example"]["row_count"] == 4
        rows = reader.read_index_entry("Long Example")
        assert rows[0]["meta"]["short_book_title"] == "Long Example"
        assert rows[1]["_rowpack_split"]["is_continuation"] is True
