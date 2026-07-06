import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw, ImageOps

from rowpack import NativeCistaVQARows


def parse_args():
    parser = argparse.ArgumentParser(description="Render native RowPack decoded image previews.")
    parser.add_argument(
        "--encoded-rowpack",
        default="results/mm_infographic_vqa_rowpack_raw_rgb_128/variants/rowpack_cista_encoded_128.rowpack",
        help="CISTA RowPack file with original encoded image payloads decoded through STB.",
    )
    parser.add_argument(
        "--qoi-rowpack",
        default="results/mm_infographic_vqa_rowpack_raw_rgb_128/variants/rowpack_cista_qoi_lossless_128.rowpack",
        help="CISTA RowPack file with qoi_lossless image payloads decoded through QOI.",
    )
    parser.add_argument("--rowpack-native-dir", default="rowpack_build_py/Release")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=260)
    parser.add_argument(
        "--output",
        default="results/mm_infographic_vqa_rowpack_decode_preview/native_decode_preview.png",
    )
    return parser.parse_args()


def raw_image_to_pil(image: dict) -> Image.Image:
    height = int(image.get("height") or 0)
    width = int(image.get("width") or 0)
    channels = int(image.get("channels") or 0)
    data = image.get("bytes") or b""
    if height <= 0 or width <= 0:
        raise ValueError("native image did not include decoded height/width metadata")
    if channels == 3:
        return Image.frombytes("RGB", (width, height), data)
    if channels == 4:
        return Image.frombytes("RGBA", (width, height), data).convert("RGB")
    raise ValueError(f"unsupported decoded channel count {channels}")


def load_native_images(path: str, native_dir: str, rows: int) -> list[tuple[int, Image.Image, dict]]:
    out = []
    source = NativeCistaVQARows([path], max_rows=rows, native_module_dir=native_dir)
    for row_id, _pairs, images in source:
        if not images:
            continue
        image = dict(images[0])
        out.append((int(row_id), raw_image_to_pil(image), image))
    return out


def draw_cell(
    canvas: Image.Image,
    image: Image.Image,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    subtitle: str,
):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + width, y + height), outline=(210, 210, 210), width=1)
    draw.text((x + 8, y + 6), title, fill=(20, 20, 20))
    draw.text((x + 8, y + 24), subtitle, fill=(80, 80, 80))
    thumb_area = (width - 16, height - 54)
    thumb = ImageOps.contain(image.convert("RGB"), thumb_area, Image.Resampling.LANCZOS)
    thumb_x = x + (width - thumb.width) // 2
    thumb_y = y + 46 + (thumb_area[1] - thumb.height) // 2
    canvas.paste(thumb, (thumb_x, thumb_y))


def make_contact_sheet(encoded_rows, qoi_rows, output: Path, thumb_width: int, thumb_height: int):
    row_count = min(len(encoded_rows), len(qoi_rows))
    if row_count == 0:
        raise RuntimeError("No decoded images available to render")

    margin = 18
    gap = 14
    header_h = 42
    columns = 2
    width = margin * 2 + columns * thumb_width + gap
    height = margin * 2 + header_h + row_count * thumb_height + (row_count - 1) * gap
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "RowPack native decode preview: STB encoded vs QOI lossless", fill=(10, 10, 10))

    y = margin + header_h
    for encoded, qoi in zip(encoded_rows[:row_count], qoi_rows[:row_count]):
        encoded_row_id, encoded_image, encoded_meta = encoded
        qoi_row_id, qoi_image, qoi_meta = qoi
        draw_cell(
            canvas,
            encoded_image,
            x=margin,
            y=y,
            width=thumb_width,
            height=thumb_height,
            title=f"row {encoded_row_id}: encoded -> STB",
            subtitle=f"{encoded_image.width}x{encoded_image.height}, {encoded_meta.get('storage')}",
        )
        draw_cell(
            canvas,
            qoi_image,
            x=margin + thumb_width + gap,
            y=y,
            width=thumb_width,
            height=thumb_height,
            title=f"row {qoi_row_id}: QOI -> native",
            subtitle=f"{qoi_image.width}x{qoi_image.height}, {qoi_meta.get('storage')}",
        )
        y += thumb_height + gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    args = parse_args()
    encoded_rows = load_native_images(args.encoded_rowpack, args.rowpack_native_dir, args.rows)
    qoi_rows = load_native_images(args.qoi_rowpack, args.rowpack_native_dir, args.rows)
    output = Path(args.output)
    make_contact_sheet(encoded_rows, qoi_rows, output, args.thumb_width, args.thumb_height)
    print(output)


if __name__ == "__main__":
    main()
