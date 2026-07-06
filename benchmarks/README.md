# Dataset Loader Benchmarks

This folder contains the benchmark tooling we use to compare dataset storage
formats, image encodings, compression settings, and loader behavior for VLM
training. The main target dataset is:

```python
from datasets import load_dataset

ds = load_dataset("nimapourjafar/mm_infographic_vqa")
```

The benchmark uses a tiny randomly initialized nanoVLM config. It is meant to
exercise the dataset adapter, row/window access pattern, image decode, image
preprocessing, tokenization, collation, forward pass, backward pass, and
optimizer step. It is not meant to train a useful model.

## Native RowPack Build

Build the optional native RowPack module before running RowPack benchmarks:

```bash
cmake -S rowpack -B rowpack_build_py
cmake --build rowpack_build_py --config Release
```

The benchmark scripts accept `--rowpack-native-dir rowpack_build_py/Release`.
The loader also searches common local build folders such as `rowpack_build_py`,
`rowpack_build_py/Release`, `rowpack_build`, and `build`.

## Single Smoke Run

Run one short CPU smoke benchmark from the repository root:

```bash
python benchmarks/mm_infographic_vqa_baseline.py \
  --steps 3 \
  --warmup-steps 1 \
  --max-rows 128 \
  --batch-size 1 \
  --num-workers 0
```

The JSON artifact defaults to:

```text
results/mm_infographic_vqa_baseline.json
```

Important fields:

- `dataset_load_time_s`: wall time for dataset construction.
- `storage.dataset_total_rows`: total source rows before optional `--max-rows`.
- `storage.rows_profiled`: rows kept for this benchmark run.
- `storage.extra_size_paths`: measured local file sizes from `--size-path`.
- `training.summary.samples_per_s`: measured training-loop throughput.
- `training.summary.data_time_fraction`: fraction of measured step time spent
  waiting for the next batch.
- `training.summary.compute_time_fraction`: fraction spent in forward/backward.
- `training.summary.optimizer_time_fraction`: fraction spent in optimizer work.

For worker-based runs, `data_time_s` is the time the training loop waits on
`next(dataloader_iter)`. That includes visible loader work and excludes work
already hidden by prefetching.

## Read Patterns

The direct PyArrow and RowPack loaders support explicit access patterns:

- `--read-pattern sequential`: read rows in file order.
- `--read-pattern random_block`: sample a reproducible random start row, then
  read `--read-block-size` neighboring rows from that window.

Random-block access is the most useful training-style stress test here. Fully
random single-row reads are a worst case for most storage formats, while fully
sequential reads are too friendly for shuffled VLM training.

Example:

```bash
python benchmarks/mm_infographic_vqa_baseline.py \
  --loader pyarrow \
  --read-pattern random_block \
  --read-block-size 32 \
  --data-files data/variants/mm_infographic_vqa/uncompressed.parquet \
  --size-path parquet_uncompressed=data/variants/mm_infographic_vqa/uncompressed.parquet \
  --skip-hub-size
```

## Parquet Variant Suite

Prepare reproducible Parquet compression variants:

```bash
python benchmarks/prepare_mm_infographic_vqa_variants.py \
  --output-dir data/variants/mm_infographic_vqa
```

This prepares:

- `upstream`: the original Parquet file from `nimapourjafar/mm_infographic_vqa`.
- `uncompressed`: PyArrow-written Parquet with `compression="NONE"`.
- `snappy`: PyArrow-written Parquet with `compression="SNAPPY"`.
- `zstd`: PyArrow-written Parquet with `compression="ZSTD"`.
- `gzip`: PyArrow-written Parquet with `compression="GZIP"`.
- `brotli`: PyArrow-written Parquet with `compression="BROTLI"`.
- `lz4_raw`: requested for coverage, but may be unsupported by PyArrow.
- `lzo`: requested for coverage, but is often unsupported by PyArrow.

Run the suite and generate tables/charts:

```bash
python benchmarks/run_mm_infographic_vqa_parquet_suite.py \
  --manifest data/variants/mm_infographic_vqa/manifest.json \
  --loader pyarrow \
  --read-pattern random_block \
  --read-block-size 32 \
  --steps 20 \
  --warmup-steps 2 \
  --max-rows 256 \
  --output-dir results/mm_infographic_vqa_parquet_suite_random_block
```

Outputs:

```text
results/mm_infographic_vqa_parquet_suite_random_block/summary.csv
results/mm_infographic_vqa_parquet_suite_random_block/summary.md
results/mm_infographic_vqa_parquet_suite_random_block/charts/
```

## RowPack Variants

Prepare a current RowPack baseline with CISTA row payloads, source JPEG bytes,
and high-ratio LZAV block compression:

```bash
python benchmarks/prepare_mm_infographic_vqa_rowpack.py \
  --data-files data/variants/mm_infographic_vqa/uncompressed.parquet \
  --output-dir data/variants/mm_infographic_vqa_rowpack \
  --variant-name rowpack_cista_lzav_hi \
  --rows-per-block 32 \
  --payload-format cista \
  --image-storage encoded \
  --block-codec lzav_hi \
  --rowpack-native-dir rowpack_build_py/Release \
  --overwrite
```

The converter writes the `.rowpack`, `manifest.json`, `rowpacks.txt`, and a
variant-specific list file. `rowpacks.txt` is a plain UTF-8 list with one
`.rowpack` path per line, relative to the list file directory.

Useful RowPack knobs:

- `--payload-format json`: easier to inspect, useful for debugging.
- `--payload-format cista`: fast native payloads for training.
- `--image-storage encoded`: keep source JPEG/JFIF bytes and decode when read.
- `--image-storage raw_rgb`: store decoded RGB bytes for decode-free read tests.
- `--image-storage qoi_lossless`: store decoded pixels through QOI lossless
  image compression.
- `--block-codec none`: uncompressed row-major layout baseline.
- `--block-codec lzav_default`: faster write, good compression.
- `--block-codec lzav_hi`: stronger compression, recommended publishing default.

Run a RowPack suite:

```bash
python benchmarks/run_mm_infographic_vqa_parquet_suite.py \
  --manifest data/variants/mm_infographic_vqa_rowpack/manifest.json \
  --loader rowpack \
  --rowpack-native-dir rowpack_build_py/Release \
  --read-pattern random_block \
  --read-block-size 32 \
  --steps 20 \
  --warmup-steps 2 \
  --max-rows 256 \
  --batch-size 1 \
  --num-workers 0 \
  --sequence-length 128 \
  --image-size 32 \
  --output-dir results/mm_infographic_vqa_rowpack_random_block
```

For CISTA RowPack files, the suite defaults to the direct native VQA path. Add
`--no-rowpack-direct-vqa` to force the generic RowPack reader for A/B checks.

## Mega Comparison Benchmark

The canonical comparison script builds and benchmarks a mixed matrix of Parquet
and RowPack variants, then plots a combined file-size and throughput chart:

Windows PowerShell:

```powershell
.\benchmarks\run_mega_benchmark.ps1 `
  -ResultsRoot results/mm_infographic_vqa_comparison `
  -DataRoot data/variants/mm_infographic_vqa_comparison `
  -Overwrite
```

Linux/macOS shell:

```bash
bash benchmarks/run_mega_benchmark.sh \
  --results-root results/mm_infographic_vqa_comparison \
  --data-root data/variants/mm_infographic_vqa_comparison \
  --overwrite
```

The wrappers configure and build `rowpack_native`, prepare the source Parquet if
needed, write `environment.json`, run the 15-variant benchmark, and generate
the mega chart. They also create and reuse a repository-local `.venv` by
default, then install the benchmark dependencies from
`benchmarks/mega_benchmark_requirements.txt` before running CMake or Python
benchmark steps. Use `--venv-dir` / `-VenvDir` to choose another local
environment directory, or `--skip-env-setup` / `-SkipEnvSetup` with
`--python` / `-Python` to use an already prepared interpreter directly.

The underlying Python command is:

```bash
python benchmarks/reproduce_mm_infographic_vqa_comparison.py \
  --root results/mm_infographic_vqa_comparison \
  --data-root data/variants/mm_infographic_vqa_comparison \
  --source-parquet data/variants/mm_infographic_vqa/uncompressed.parquet \
  --rowpack-native-dir rowpack_build_py/Release \
  --read-pattern random_block \
  --read-block-size 16 \
  --bench-max-rows 256 \
  --bench-steps 32 \
  --bench-warmup-steps 4 \
  --overwrite
```

By default it prepares 15 variants:

- Parquet with source JPEG bytes, PNG bytes, and raw RGB bytes.
- Parquet container compression: uncompressed, GZIP, and Brotli.
- RowPack with source JPEG bytes, QOI lossless pixels, and raw RGB pixels.
- RowPack payload formats: JSON and CISTA.
- RowPack block compression: `lzav_hi`.

Full-dataset preparation can exceed 20 GiB because the matrix includes raw RGB
and lossless image-storage variants. For a quick smoke run:

Windows PowerShell:

```powershell
.\benchmarks\run_mega_benchmark.ps1 `
  -ResultsRoot results/mm_infographic_vqa_comparison_smoke `
  -DataRoot data/variants/mm_infographic_vqa_comparison_smoke `
  -MaxRows 128 `
  -BenchMaxRows 128 `
  -BenchSteps 5 `
  -BenchWarmupSteps 1 `
  -Overwrite
```

Linux/macOS shell:

```bash
bash benchmarks/run_mega_benchmark.sh \
  --results-root results/mm_infographic_vqa_comparison_smoke \
  --data-root data/variants/mm_infographic_vqa_comparison_smoke \
  --max-rows 128 \
  --bench-max-rows 128 \
  --bench-steps 5 \
  --bench-warmup-steps 1 \
  --overwrite
```

Equivalent Python command:

```bash
python benchmarks/reproduce_mm_infographic_vqa_comparison.py \
  --root results/mm_infographic_vqa_comparison_smoke \
  --data-root data/variants/mm_infographic_vqa_comparison_smoke \
  --source-parquet data/variants/mm_infographic_vqa/uncompressed.parquet \
  --rowpack-native-dir rowpack_build_py/Release \
  --max-rows 128 \
  --bench-max-rows 128 \
  --bench-steps 5 \
  --bench-warmup-steps 1 \
  --overwrite
```

Mega-chart outputs:

```text
results/mm_infographic_vqa_comparison/size_vs_throughput.png
results/mm_infographic_vqa_comparison/size_vs_throughput.csv
results/mm_infographic_vqa_comparison/environment.json
results/mm_infographic_vqa_comparison/runs/*.json
results/mm_infographic_vqa_comparison/reproduce_manifest.json
```

If the variants and run JSONs already exist, regenerate only the chart:

```bash
python benchmarks/reproduce_mm_infographic_vqa_comparison.py \
  --root results/mm_infographic_vqa_comparison \
  --data-root data/variants/mm_infographic_vqa_comparison \
  --rowpack-native-dir rowpack_build_py/Release \
  --skip-prepare \
  --skip-bench
```

The chart itself is rendered by:

```bash
python benchmarks/plot_size_vs_throughput.py --help
```

## Comparing Machines

When comparing Windows, Linux, or different CPUs, inspect:

```text
results/mm_infographic_vqa_comparison/environment.json
```

The wrapper scripts write this file before the benchmark. It records Python,
Pillow/libjpeg, PyTorch, PyArrow, thread counts, CPU metadata, and available
`rowpack_native` symbols.

Important decoder details:

- Default JPEG benchmark paths decode through Pillow, both for Parquet and
  RowPack. Pillow may use different libjpeg/libjpeg-turbo builds on different
  machines, so JPEG throughput can vary by OS, wheel, distro package, and CPU
  SIMD support.
- STB JPEG decode is only used for CISTA RowPack when
  `--rowpack-native-decode-images` is enabled.
- QOI RowPack benchmark paths decode through `rowpack_native.qoi_decode_rgb`.
  Older benchmark code accidentally sent CISTA QOI payloads through the generic
  Pillow path, which made `rowpack_qoi_cista_lzav_hi` look much slower and more
  platform-dependent than intended.
