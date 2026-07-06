import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Capture environment metadata for benchmark comparisons.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-module-dir", default=None)
    return parser.parse_args()


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def command_output(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return None
    text = (result.stdout or result.stderr or "").strip()
    return text or None


def pillow_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        from PIL import Image, features
    except Exception as exc:
        info["error"] = repr(exc)
        return info

    info.update({
        "available": True,
        "version": getattr(Image, "__version__", None),
        "jpg": safe_call(features.check, "jpg"),
        "jpg_version": safe_call(features.version_codec, "jpg"),
        "libjpeg_turbo": safe_call(features.check_feature, "libjpeg_turbo"),
        "libjpeg_turbo_version": safe_call(features.version_feature, "libjpeg_turbo"),
        "webp": safe_call(features.check, "webp"),
        "zlib": safe_call(features.check, "zlib"),
        "supported_modules": safe_call(features.get_supported_modules),
        "supported_codecs": safe_call(features.get_supported_codecs),
        "supported_features": safe_call(features.get_supported_features),
    })
    return info


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import torch
    except Exception as exc:
        info["error"] = repr(exc)
        return info

    info.update({
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "mkl_available": safe_call(torch.backends.mkl.is_available),
        "mkldnn_available": safe_call(torch.backends.mkldnn.is_available),
    })
    return info


def pyarrow_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import pyarrow as pa
    except Exception as exc:
        info["error"] = repr(exc)
        return info

    info.update({
        "available": True,
        "version": pa.__version__,
        "cpu_count": safe_call(pa.cpu_count),
        "io_thread_count": safe_call(pa.io_thread_count),
    })
    return info


def rowpack_native_info(native_module_dir: str | None) -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        from rowpack.native import load_native
        native = load_native(native_module_dir)
    except Exception as exc:
        info["error"] = repr(exc)
        return info

    symbols = sorted(
        name
        for name in dir(native)
        if any(token in name.lower() for token in ["qoi", "jpeg", "lzav", "avif", "cista"])
    )
    info.update({
        "available": True,
        "path": getattr(native, "__file__", None),
        "symbols": symbols,
    })
    if hasattr(native, "avif_runtime_info"):
        info["avif_runtime_info"] = safe_call(native.avif_runtime_info)
    return info


def safe_call(func, *args):
    try:
        return func(*args)
    except Exception as exc:
        return {"error": repr(exc)}


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTHONHASHSEED",
        "ROWPACK_NATIVE_DIR",
    ]
    report = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
        "environment": {key: os.environ.get(key) for key in env_keys},
        "native_module_dir": args.native_module_dir,
        "packages": {
            "numpy": module_version("numpy"),
            "datasets": module_version("datasets"),
            "transformers": module_version("transformers"),
            "tokenizers": module_version("tokenizers"),
            "matplotlib": module_version("matplotlib"),
            "nanobind": module_version("nanobind"),
        },
        "pillow": pillow_info(),
        "torch": torch_info(),
        "pyarrow": pyarrow_info(),
        "rowpack_native": rowpack_native_info(args.native_module_dir),
        "tools": {
            "cmake": command_output(["cmake", "--version"]),
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
        },
    }

    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"environment": str(output)}, indent=2))


if __name__ == "__main__":
    main()
