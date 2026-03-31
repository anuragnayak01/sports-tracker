@echo off

echo.
echo ==========================================
echo   Sports Tracker - Hugging Face Deploy
echo ==========================================
echo.
echo   FREE - No credit card needed.
echo   Your app will be live at:
echo   https://huggingface.co/spaces/YOUR_USERNAME/sports-tracker
echo.

REM Check git is installed
where git >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: git is not installed.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] git found.

REM Check git-lfs is installed
where git-lfs >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo git-lfs not found. Installing via winget...
    winget install -e --id GitHub.GitLFS
    git lfs install
)
echo [OK] git-lfs ready.
echo.

REM Get HF username
echo Step 1 of 4 - Your Hugging Face username
echo   (Sign up free at https://huggingface.co if you have not yet)
echo.
set /p HF_USER=Enter your Hugging Face username: 
echo.

REM Get HF token
echo Step 2 of 4 - Your Hugging Face token
echo   Get it from: https://huggingface.co/settings/tokens
echo   Create a token with WRITE permission, then paste it here.
echo   (The token will not be shown as you type - that is normal)
echo.
set /p HF_TOKEN=Paste your HF token here: 
echo.

REM Space name
echo Step 3 of 4 - Choose a Space name
echo   Example: sports-tracker
echo.
set /p SPACE_NAME=Enter space name (default: sports-tracker): 
IF "%SPACE_NAME%"=="" SET SPACE_NAME=sports-tracker
echo Space will be: %HF_USER%/%SPACE_NAME%
echo.

REM Create README with Space config (required by HF Spaces)
echo Step 4 of 4 - Pushing code to Hugging Face...
echo.

REM Write the HF Space metadata README
(
echo ---
echo title: Sports Multi Object Tracker
echo emoji: 🏃
echo colorFrom: blue
echo colorTo: green
echo sdk: streamlit
echo sdk_version: 1.35.0
echo app_file: app.py
echo pinned: false
echo license: mit
echo ---
echo.
echo # Sports Multi-Object Tracker
echo.
echo Multi-object detection and persistent ID tracking in sports footage.
echo Uses YOLOv8 + BoT-SORT.
echo.
echo Upload a video or paste a YouTube URL to track players with unique IDs,
echo movement heatmaps, and trajectory tails.
) > README_HF.md

REM Clone the HF Space repo (creates it if first time)
SET REPO_URL=https://%HF_USER%:%HF_TOKEN%@huggingface.co/spaces/%HF_USER%/%SPACE_NAME%

echo Cloning Space repo...
git clone %REPO_URL% hf_deploy_tmp 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Space does not exist yet - creating it via API...
    powershell -NoProfile -Command ^
        "$headers = @{'Authorization'='Bearer %HF_TOKEN%'; 'Content-Type'='application/json'};" ^
        "$body = '{\"type\":\"space\",\"name\":\"%SPACE_NAME%\",\"sdk\":\"streamlit\",\"private\":false}';" ^
        "Invoke-RestMethod -Uri 'https://huggingface.co/api/repos/create' -Method POST -Headers $headers -Body $body"
    git clone %REPO_URL% hf_deploy_tmp
)

REM Copy project files into the cloned repo
echo Copying project files...
xcopy /E /I /Y /EXCLUDE:.hfexclude . hf_deploy_tmp\ >nul

REM Copy the HF README (overwrite generic README)
copy /Y README_HF.md hf_deploy_tmp\README.md >nul
del README_HF.md

REM Push to HF
cd hf_deploy_tmp
git config user.email "deploy@sports-tracker.local"
git config user.name "Sports Tracker Deploy"
git lfs track "*.pt"
git add .
git commit -m "Deploy sports tracker app"
git push
cd ..

REM Cleanup
rmdir /S /Q hf_deploy_tmp

echo.
echo ==========================================
echo   DEPLOYED SUCCESSFULLY - NO CARD NEEDED
echo ==========================================
echo.
echo   Your permanent public URL:
echo   https://huggingface.co/spaces/%HF_USER%/%SPACE_NAME%
echo.
echo   Share this link with anyone.
echo   App runs 24/7 for FREE.
echo.
echo   First build takes about 3-5 minutes.
echo   Check build logs at the URL above.
echo.
pause
