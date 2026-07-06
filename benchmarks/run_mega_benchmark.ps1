param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv",
    [string]$BuildDir = "rowpack_build_py",
    [string]$SourceParquet = "data/variants/mm_infographic_vqa/uncompressed.parquet",
    [string]$BaseVariantDir = "data/variants/mm_infographic_vqa",
    [string]$DataRoot = "data/variants/mm_infographic_vqa_comparison",
    [string]$ResultsRoot = "results/mm_infographic_vqa_comparison",
    [int]$MaxRows = 0,
    [int]$BenchMaxRows = 256,
    [int]$BenchSteps = 32,
    [int]$BenchWarmupSteps = 4,
    [int]$ReadBlockSize = 16,
    [int]$RowsPerBlock = 64,
    [int]$RowGroupSize = 64,
    [switch]$Overwrite,
    [switch]$SkipEnvSetup,
    [switch]$SkipBuild,
    [switch]$SkipSourcePrepare,
    [switch]$RowPackNativeDecodeImages
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
$RowPackSourceDir = Join-Path $RepoRoot "rowpack"

function Invoke-Step {
    param(
        [string]$File,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host ">>> $File $($Arguments -join ' ')"
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $File $($Arguments -join ' ')"
    }
}

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $RepoRoot $Path)
}

function Get-VenvPython {
    param([string]$Path)
    $candidates = @(
        (Join-Path $Path "Scripts/python.exe"),
        (Join-Path $Path "bin/python")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $null
}

function Test-MegaBenchmarkModules {
    param([string]$PythonExe)
    $moduleCheck = @'
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
'@
    & $PythonExe -c $moduleCheck
    return $LASTEXITCODE -eq 0
}

function Initialize-PythonEnvironment {
    param(
        [string]$BasePython,
        [string]$Path
    )

    $requirementsFile = Join-Path $RepoRoot "benchmarks/mega_benchmark_requirements.txt"
    if (-not (Test-Path $requirementsFile)) {
        throw "Could not find benchmark requirements: $requirementsFile"
    }

    $venvPython = Get-VenvPython $Path
    if (-not $venvPython) {
        Write-Host "Creating local Python environment at $Path"
        try {
            Invoke-Step $BasePython @("-m", "venv", $Path)
        }
        catch {
            throw "Failed to create a virtual environment with '$BasePython'. Install the Python venv module for that interpreter. $_"
        }
        $venvPython = Get-VenvPython $Path
    }

    if (-not $venvPython) {
        throw "Could not find a Python executable inside $Path."
    }

    & $venvPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Step $venvPython @("-m", "ensurepip", "--upgrade")
    }

    if (-not (Test-MegaBenchmarkModules $venvPython)) {
        Invoke-Step $venvPython @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Step $venvPython @("-m", "pip", "install", "-r", $requirementsFile)
        if (-not (Test-MegaBenchmarkModules $venvPython)) {
            throw "The local Python environment is still missing benchmark dependencies after installation."
        }
    }

    return $venvPython
}

function Find-RowPackNativeDir {
    param([string]$SearchRoot)
    $native = Get-ChildItem -Path $SearchRoot -Recurse -File -Include "rowpack_native*.pyd", "rowpack_native*.so", "rowpack_native*.dylib" |
        Sort-Object FullName |
        Select-Object -First 1
    if ($null -eq $native) {
        throw "Could not find rowpack_native under $SearchRoot after build."
    }
    return $native.Directory.FullName
}

if ($SkipEnvSetup) {
    $PythonExe = (& $Python -c "import sys; print(sys.executable)").Trim()
}
else {
    $ResolvedVenvDir = Resolve-RepoPath $VenvDir
    $PythonExe = Initialize-PythonEnvironment $Python $ResolvedVenvDir
}

if (-not $PythonExe) {
    throw "Could not resolve Python executable from '$Python'."
}
Write-Host "Using Python environment: $PythonExe"

if (-not $SkipBuild) {
    Invoke-Step "cmake" @(
        "-S", $RowPackSourceDir,
        "-B", $BuildDir,
        "-DPython_EXECUTABLE=$PythonExe",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DROWPACK_NANOBIND_DIR=$(Join-Path $RowPackSourceDir 'third_party/nanobind')",
        "-DROWPACK_CISTA_INCLUDE_DIR=$(Join-Path $RowPackSourceDir 'third_party/cista/include')",
        "-DROWPACK_QOI_INCLUDE_DIR=$(Join-Path $RowPackSourceDir 'third_party/qoi')",
        "-DROWPACK_STB_INCLUDE_DIR=$(Join-Path $RowPackSourceDir 'third_party/stb')",
        "-DROWPACK_LZAV_INCLUDE_DIR=$(Join-Path $RowPackSourceDir 'third_party/lzav')"
    )
    Invoke-Step "cmake" @("--build", $BuildDir, "--config", "Release")
}

$NativeModuleDir = Find-RowPackNativeDir $BuildDir
$env:ROWPACK_NATIVE_DIR = $NativeModuleDir
Write-Host "Using rowpack_native from $NativeModuleDir"

if (-not (Test-Path $SourceParquet)) {
    if ($SkipSourcePrepare) {
        throw "Source Parquet not found: $SourceParquet"
    }
    $prepareArgs = @(
        "benchmarks/prepare_mm_infographic_vqa_variants.py",
        "--output-dir", $BaseVariantDir,
        "--image-encoding", "source",
        "--variants", "uncompressed"
    )
    if ($Overwrite) {
        $prepareArgs += "--overwrite"
    }
    Invoke-Step $PythonExe $prepareArgs
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
Invoke-Step $PythonExe @(
    "benchmarks/capture_benchmark_env.py",
    "--native-module-dir", $NativeModuleDir,
    "--output", (Join-Path $ResultsRoot "environment.json")
)

$megaArgs = @(
    "benchmarks/reproduce_mm_infographic_vqa_comparison.py",
    "--root", $ResultsRoot,
    "--data-root", $DataRoot,
    "--source-parquet", $SourceParquet,
    "--rowpack-native-dir", $NativeModuleDir,
    "--read-pattern", "random_block",
    "--read-block-size", "$ReadBlockSize",
    "--bench-max-rows", "$BenchMaxRows",
    "--bench-steps", "$BenchSteps",
    "--bench-warmup-steps", "$BenchWarmupSteps",
    "--bench-batch-size", "1",
    "--bench-num-workers", "0",
    "--bench-sequence-length", "128",
    "--bench-image-size", "32",
    "--rows-per-block", "$RowsPerBlock",
    "--row-group-size", "$RowGroupSize"
)

if ($MaxRows -gt 0) {
    $megaArgs += @("--max-rows", "$MaxRows")
}
if ($Overwrite) {
    $megaArgs += "--overwrite"
}
if ($RowPackNativeDecodeImages) {
    $megaArgs += "--rowpack-native-decode-images"
}

Invoke-Step $PythonExe $megaArgs

Write-Host ""
Write-Host "Mega benchmark complete."
Write-Host "Chart: $ResultsRoot/size_vs_throughput.png"
Write-Host "CSV:   $ResultsRoot/size_vs_throughput.csv"
Write-Host "Env:   $ResultsRoot/environment.json"
