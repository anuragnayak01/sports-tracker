#!/usr/bin/env bash
# deploy.sh — One-shot Fly.io deployment script
# ================================================
# Run this from inside your sports-tracker folder:
#     bash deploy.sh
#
# What it does:
#   1. Checks flyctl is installed (installs if missing)
#   2. Logs you in to Fly.io
#   3. Asks for your desired app name
#   4. Updates fly.toml with your app name
#   5. Launches and deploys the app
#   6. Prints your public URL

set -e  # Exit on any error

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Sports Tracker — Fly.io Deploy Script ${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ── Step 1: Check / install flyctl ───────────────────────────
if ! command -v flyctl &> /dev/null; then
    echo -e "${YELLOW}flyctl not found. Installing...${NC}"
    curl -L https://fly.io/install.sh | sh
    export PATH="$HOME/.fly/bin:$PATH"
    echo -e "${GREEN}✅ flyctl installed${NC}"
else
    echo -e "${GREEN}✅ flyctl found: $(flyctl version)${NC}"
fi

echo ""

# ── Step 2: Login ─────────────────────────────────────────────
echo -e "${CYAN}Step 1/4 — Logging in to Fly.io...${NC}"
echo "      (A browser window will open once for authentication)"
flyctl auth login
echo -e "${GREEN}✅ Logged in${NC}"
echo ""

# ── Step 3: Get app name from user ────────────────────────────
echo -e "${CYAN}Step 2/4 — Choose your app name${NC}"
echo "      Must be globally unique on Fly.io (e.g. sports-tracker-john)"
echo ""
read -p "  Enter app name: " APP_NAME

# Validate — only lowercase letters, numbers, hyphens
if [[ ! "$APP_NAME" =~ ^[a-z0-9-]+$ ]]; then
    echo "❌ App name must only contain lowercase letters, numbers, and hyphens"
    exit 1
fi

# Update fly.toml with the chosen app name
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/^app = .*/app = \"$APP_NAME\"/" fly.toml
else
    sed -i "s/^app = .*/app = \"$APP_NAME\"/" fly.toml
fi

echo -e "${GREEN}✅ App name set to: $APP_NAME${NC}"
echo ""

# ── Step 4: Launch + Deploy ───────────────────────────────────
echo -e "${CYAN}Step 3/4 — Creating app on Fly.io...${NC}"
flyctl apps create "$APP_NAME" || echo "  (App may already exist — continuing)"
echo ""

echo -e "${CYAN}Step 4/4 — Deploying (building Docker image)...${NC}"
echo "      This takes 3-5 minutes on first deploy."
echo ""
flyctl deploy --app "$APP_NAME"

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✅ DEPLOYED SUCCESSFULLY!             ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  🌐 Your public URL:"
echo -e "  ${CYAN}https://${APP_NAME}.fly.dev${NC}"
echo ""
echo -e "  Share this link with anyone — it works 24/7"
echo -e "  even when your laptop is off."
echo ""
echo -e "  Useful commands:"
echo -e "    flyctl logs --app $APP_NAME     # View live logs"
echo -e "    flyctl status --app $APP_NAME   # Check health"
echo -e "    flyctl deploy --app $APP_NAME   # Redeploy after changes"
echo ""
