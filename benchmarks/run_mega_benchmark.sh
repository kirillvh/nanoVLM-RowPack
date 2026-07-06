#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
BUILD_DIR="rowpack_build_py"
SOURCE_PARQUET="data/variants/mm_infographic_vqa/uncompressed.parquet"
BASE_VARIANT_DIR="data/variants/mm_infographic_vqa"
DATA_ROOT="data/variants/mm_infographic_vqa_comparison"
RESULTS_ROOT="results/mm_infographic_vqa_comparison"
MAX_ROWS=""
BENCH_MAX_ROWS="256"
BENCH_STEPS="32"
BENCH_WARMUP_STEPS="4"
READ_BLOCK_SIZE="16"
ROWS_PER_BLOCK="64"
ROW_GROUP_SIZE="64"
OVERWRITE="0"
SKIP_ENV_SETUP="0"
SKIP_BUILD="0"
SKIP_SOURCE_PREPARE="0"
ROWPACK_NATIVE_DECODE_IMAGES="0"

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_mega_benchmark.sh [options]

Options:
  --python PATH                    Base Python executable (default: python3 or $PYTHON)
  --venv-dir DIR                   Repo-local virtualenv directory (default: .venv or $VENV_DIR)
  --skip-env-setup                 Use --python directly; do not create/install a local env
  --build-dir DIR                  CMake build directory (default: rowpack_build_py)
  --source-parquet PATH            Local source Parquet file
  --base-variant-dir DIR           Directory used to prepare source Parquet if missing
  --data-root DIR                  Generated benchmark variants directory
  --results-root DIR               Benchmark outputs directory
  --max-rows N                     Subset variant preparation to N rows
  --bench-max-rows N               Rows scanned by each benchmark run (default: 256)
  --bench-steps N                  Measured training steps per variant (default: 32)
  --bench-warmup-steps N           Warmup steps per variant (default: 4)
  --read-block-size N              Random-block read window (default: 16)
  --rows-per-block N               RowPack compression block rows (default: 64)
  --row-group-size N               Parquet row-group rows (default: 64)
  --overwrite                      Regenerate existing variants/runs
  --skip-build                     Do not configure/build rowpack_native
  --skip-source-prepare            Fail instead of preparing source Parquet if missing
  --rowpack-native-decode-images   Decode CISTA images inside rowpack_native where supported
  -h, --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --skip-env-setup) SKIP_ENV_SETUP="1"; shift ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --source-parquet) SOURCE_PARQUET="$2"; shift 2 ;;
    --base-variant-dir) BASE_VARIANT_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --max-rows) MAX_ROWS="$2"; shift 2 ;;
    --bench-max-rows) BENCH_MAX_ROWS="$2"; shift 2 ;;
    --bench-steps) BENCH_STEPS="$2"; shift 2 ;;
    --bench-warmup-steps) BENCH_WARMUP_STEPS="$2"; shift 2 ;;
    --read-block-size) READ_BLOCK_SIZE="$2"; shift 2 ;;
    --rows-per-block) ROWS_PER_BLOCK="$2"; shift 2 ;;
    --row-group-size) ROW_GROUP_SIZE="$2"; shift 2 ;;
    --overwrite) OVERWRITE="1"; shift ;;
    --skip-build) SKIP_BUILD="1"; shift ;;
    --skip-source-prepare) SKIP_SOURCE_PREPARE="1"; shift ;;
    --rowpack-native-decode-images) ROWPACK_NATIVE_DECODE_IMAGES="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ROWPACK_SOURCE_DIR="$REPO_ROOT/rowpack"

run() {
  echo
  printf '>>>'
  printf ' %q' "$@"
  echo
  "$@"
}

resolve_repo_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$REPO_ROOT/$path"
  fi
}

get_venv_python() {
  local venv_dir="$1"
  local candidate
  for candidate in "$venv_dir/bin/python" "$venv_dir/Scripts/python.exe"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

python_has_mega_modules() {
  local python_exe="$1"
  "$python_exe" - <<'PY'
import importlib
import sys

required = {
    "torch": "torch",
    "numpy": "numpy",
    "torchvision": "torchvision",
    "pillow": "PIL",
    "datasets": "datasets",
    "huggingface-hub": "huggingface_hub",
    "transformers": "transformers",
    "safetensors": "safetensors",
    "einops": "einops",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
}

missing = []
for package, module_name in required.items():
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append(f"{package}: {exc.__class__.__name__}: {exc}")

if missing:
    print("Missing or unusable mega benchmark packages:")
    for item in missing:
        print(f"  - {item}")
    sys.exit(1)
PY
}

bootstrap_python_env() {
  local venv_dir="$1"
  local requirements_file="$REPO_ROOT/benchmarks/mega_benchmark_requirements.txt"
  local venv_python=""

  if [[ ! -f "$requirements_file" ]]; then
    echo "Could not find benchmark requirements: $requirements_file" >&2
    exit 1
  fi

  if ! venv_python="$(get_venv_python "$venv_dir")"; then
    echo "Creating local Python environment at $venv_dir"
    if ! run "$PYTHON" -m venv "$venv_dir"; then
      echo "Failed to create a virtual environment. On Debian/Ubuntu, install python3-venv for this Python." >&2
      exit 1
    fi
    venv_python="$(get_venv_python "$venv_dir")"
  fi

  if [[ -z "$venv_python" ]]; then
    echo "Could not find a Python executable inside $venv_dir." >&2
    exit 1
  fi

  if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
    run "$venv_python" -m ensurepip --upgrade
  fi

  if ! python_has_mega_modules "$venv_python"; then
    run "$venv_python" -m pip install --upgrade pip
    run "$venv_python" -m pip install -r "$requirements_file"
    python_has_mega_modules "$venv_python"
  fi

  BOOTSTRAPPED_PYTHON_EXE="$venv_python"
}

if [[ "$SKIP_ENV_SETUP" == "1" ]]; then
  PYTHON_EXE="$("$PYTHON" -c 'import sys; print(sys.executable)')"
else
  VENV_DIR_ABS="$(resolve_repo_path "$VENV_DIR")"
  bootstrap_python_env "$VENV_DIR_ABS"
  PYTHON_EXE="$BOOTSTRAPPED_PYTHON_EXE"
fi

if [[ -z "$PYTHON_EXE" ]]; then
  echo "Could not resolve Python executable from '$PYTHON'." >&2
  exit 1
fi

echo "Using Python environment: $PYTHON_EXE"

if [[ "$SKIP_BUILD" != "1" ]]; then
  run cmake -S "$ROWPACK_SOURCE_DIR" -B "$BUILD_DIR" \
    -DPython_EXECUTABLE="$PYTHON_EXE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DROWPACK_NANOBIND_DIR="$ROWPACK_SOURCE_DIR/third_party/nanobind" \
    -DROWPACK_CISTA_INCLUDE_DIR="$ROWPACK_SOURCE_DIR/third_party/cista/include" \
    -DROWPACK_QOI_INCLUDE_DIR="$ROWPACK_SOURCE_DIR/third_party/qoi" \
    -DROWPACK_STB_INCLUDE_DIR="$ROWPACK_SOURCE_DIR/third_party/stb" \
    -DROWPACK_LZAV_INCLUDE_DIR="$ROWPACK_SOURCE_DIR/third_party/lzav"
  run cmake --build "$BUILD_DIR" --config Release
fi

NATIVE_FILE="$(find "$BUILD_DIR" -type f \( -name 'rowpack_native*.so' -o -name 'rowpack_native*.pyd' -o -name 'rowpack_native*.dylib' \) | sort | head -n 1)"
if [[ -z "$NATIVE_FILE" ]]; then
  echo "Could not find rowpack_native under $BUILD_DIR after build." >&2
  exit 1
fi
NATIVE_MODULE_DIR="$(dirname "$NATIVE_FILE")"
export ROWPACK_NATIVE_DIR="$NATIVE_MODULE_DIR"
echo "Using rowpack_native from $NATIVE_MODULE_DIR"

if [[ ! -f "$SOURCE_PARQUET" ]]; then
  if [[ "$SKIP_SOURCE_PREPARE" == "1" ]]; then
    echo "Source Parquet not found: $SOURCE_PARQUET" >&2
    exit 1
  fi
  prepare_args=(
    benchmarks/prepare_mm_infographic_vqa_variants.py
    --output-dir "$BASE_VARIANT_DIR"
    --image-encoding source
    --variants uncompressed
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    prepare_args+=(--overwrite)
  fi
  run "$PYTHON_EXE" "${prepare_args[@]}"
fi

mkdir -p "$RESULTS_ROOT"
run "$PYTHON_EXE" benchmarks/capture_benchmark_env.py \
  --native-module-dir "$NATIVE_MODULE_DIR" \
  --output "$RESULTS_ROOT/environment.json"

mega_args=(
  benchmarks/reproduce_mm_infographic_vqa_comparison.py
  --root "$RESULTS_ROOT"
  --data-root "$DATA_ROOT"
  --source-parquet "$SOURCE_PARQUET"
  --rowpack-native-dir "$NATIVE_MODULE_DIR"
  --read-pattern random_block
  --read-block-size "$READ_BLOCK_SIZE"
  --bench-max-rows "$BENCH_MAX_ROWS"
  --bench-steps "$BENCH_STEPS"
  --bench-warmup-steps "$BENCH_WARMUP_STEPS"
  --bench-batch-size 1
  --bench-num-workers 0
  --bench-sequence-length 128
  --bench-image-size 32
  --rows-per-block "$ROWS_PER_BLOCK"
  --row-group-size "$ROW_GROUP_SIZE"
)

if [[ -n "$MAX_ROWS" ]]; then
  mega_args+=(--max-rows "$MAX_ROWS")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  mega_args+=(--overwrite)
fi
if [[ "$ROWPACK_NATIVE_DECODE_IMAGES" == "1" ]]; then
  mega_args+=(--rowpack-native-decode-images)
fi

run "$PYTHON_EXE" "${mega_args[@]}"

echo
echo "Mega benchmark complete."
echo "Chart: $RESULTS_ROOT/size_vs_throughput.png"
echo "CSV:   $RESULTS_ROOT/size_vs_throughput.csv"
echo "Env:   $RESULTS_ROOT/environment.json"
