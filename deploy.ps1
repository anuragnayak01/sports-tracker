# deploy.ps1 - Deploy Sports Tracker to Hugging Face Spaces
# Run: powershell -ExecutionPolicy Bypass -File deploy.ps1

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Sports Tracker - Hugging Face Deploy"   -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  FREE - No credit card needed."           -ForegroundColor Green
Write-Host ""

# ── Find python in venv or system ────────────────────────────
$PYTHON = $null
$candidates = @(
    (Join-Path $PSScriptRoot "venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"),
    "python"
)
foreach ($c in $candidates) {
    try {
        $ver = & $c --version 2>&1
        if ($LASTEXITCODE -eq 0) { $PYTHON = $c; break }
    } catch {}
}
if (-not $PYTHON) {
    Write-Host "ERROR: Python not found." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
Write-Host "[OK] Python: $PYTHON" -ForegroundColor Green

# ── Install huggingface_hub if needed ────────────────────────
Write-Host "[..] Checking huggingface_hub..." -ForegroundColor Yellow
& $PYTHON -c "import huggingface_hub" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "     Installing huggingface_hub..." -ForegroundColor Yellow
    & $PYTHON -m pip install huggingface_hub --quiet
}
Write-Host "[OK] huggingface_hub ready." -ForegroundColor Green
Write-Host ""

# ── Collect inputs ───────────────────────────────────────────
Write-Host "Step 1 - Hugging Face token" -ForegroundColor Yellow
Write-Host "       Get it from: https://huggingface.co/settings/tokens"
Write-Host "       Create token with WRITE permission"
$HF_TOKEN = Read-Host "   Paste token"

Write-Host ""
Write-Host "Step 2 - Space name" -ForegroundColor Yellow
$SPACE_INPUT = Read-Host "   Enter space name (Enter = sports-tracker)"
if ([string]::IsNullOrWhiteSpace($SPACE_INPUT)) { $SPACE_NAME = "sports-tracker" }
else { $SPACE_NAME = $SPACE_INPUT }

# ── Write the Python deploy script and run it ────────────────
$PYSCRIPT = Join-Path $env:TEMP "hf_deploy_$PID.py"

@"
import sys, os, shutil, textwrap
from pathlib import Path
from huggingface_hub import HfApi, login

TOKEN      = r"""$HF_TOKEN"""
SPACE_NAME = r"""$SPACE_NAME"""
SRC        = r"""$PSScriptRoot"""

print("\nLogging in to Hugging Face...")
login(token=TOKEN, add_to_git_credential=False)

api  = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{SPACE_NAME}"

print(f"Creating Space: {repo_id}")
try:
    api.create_repo(
        repo_id   = repo_id,
        repo_type = "space",
        space_sdk = "docker",
        private   = False,
        exist_ok  = True,
    )
    print("Space ready.")
except Exception as e:
    print(f"Space creation note: {e}")

# Files/folders to skip
SKIP = {".git","venv",".venv","__pycache__","models",
        "hf_deploy_tmp","deploy.bat","deploy.sh",
        "fly.toml",".fly",".hfexclude"}

# Collect files to upload
files = []
src_path = Path(SRC)
for item in src_path.rglob("*"):
    # Skip ignored roots
    parts = item.relative_to(src_path).parts
    if any(p in SKIP for p in parts): continue
    if item.suffix in (".pyc",): continue
    if item.is_file():
        rel = str(item.relative_to(src_path)).replace("\\", "/")
        files.append((str(item), rel))

# Write Dockerfile for port 7860 (HF requirement)
dockerfile = textwrap.dedent("""
    FROM python:3.10-slim
    RUN apt-get update && apt-get install -y \\
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \\
        curl wget && rm -rf /var/lib/apt/lists/*
    RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \\
        -o /usr/local/bin/yt-dlp && chmod a+rx /usr/local/bin/yt-dlp
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    RUN mkdir -p models .streamlit
    EXPOSE 7860
    CMD ["streamlit", "run", "app.py", \\
         "--server.port=7860", \\
         "--server.address=0.0.0.0", \\
         "--server.headless=true", \\
         "--server.fileWatcherType=none"]
""").strip()

readme = textwrap.dedent(f"""
    ---
    title: Sports Multi Object Tracker
    colorFrom: blue
    colorTo: green
    sdk: docker
    pinned: false
    license: mit
    ---

    # Sports Multi-Object Tracker
    Multi-object detection and persistent ID tracking using YOLOv8 + BoT-SORT.
    Upload a video or paste a YouTube URL.
""").strip()

import tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp())

# Write Dockerfile and README to tmp
(tmp / "Dockerfile").write_text(dockerfile, encoding="utf-8")
(tmp / "README.md").write_text(readme, encoding="utf-8")
files.append((str(tmp / "Dockerfile"), "Dockerfile"))
files.append((str(tmp / "README.md"), "README.md"))

print(f"\nUploading {len(files)} files to {repo_id}...")
for i, (local, remote) in enumerate(files, 1):
    print(f"  [{i}/{len(files)}] {remote}")
    api.upload_file(
        path_or_fileobj = local,
        path_in_repo    = remote,
        repo_id         = repo_id,
        repo_type       = "space",
        token           = TOKEN,
    )

shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 46)
print("  DEPLOYED SUCCESSFULLY!")
print("=" * 46)
print()
print(f"  URL: https://huggingface.co/spaces/{repo_id}")
print()
print("  Build takes 3-5 min. Then share the URL.")
print("  Runs 24/7. No credit card. Completely free.")
print()
"@ | Set-Content -Path $PYSCRIPT -Encoding UTF8

Write-Host ""
Write-Host "Deploying via Python huggingface_hub..." -ForegroundColor Yellow
Write-Host ""

& $PYTHON $PYSCRIPT

Remove-Item $PYSCRIPT -ErrorAction SilentlyContinue

Write-Host ""
Read-Host "Press Enter to exit"
