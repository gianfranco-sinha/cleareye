#!/usr/bin/env bash
# Deploy ClearEye to the production server.
# Usage: CLEAREYE_SERVER=pi@your-host bash deploy/deploy.sh
#        (or export CLEAREYE_SERVER=pi@your-host beforehand)
set -euo pipefail

SERVER="${CLEAREYE_SERVER:?CLEAREYE_SERVER is not set. Export it first: export CLEAREYE_SERVER=pi@your-host}"
REMOTE_DIR="/home/pi/cleareye"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== ClearEye deployment to $SERVER ==="

# 1. Sync project files (exclude dev artifacts)
echo "[1/6] Syncing project files..."
rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'venv' \
  --exclude '.venv' \
  --exclude '.idea' \
  --exclude '.env' \
  --exclude '.pytest_cache' \
  --exclude 'trained_models' \
  "$PROJECT_DIR/" "$SERVER:$REMOTE_DIR/"

# 2. Set up venv + install deps on server
echo "[2/6] Installing Python dependencies..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
cd /home/pi/cleareye

# Create venv if missing
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
REMOTE

# 3. Create InfluxDB database if enabled
echo "[3/6] Ensuring InfluxDB database exists..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
if command -v influx &>/dev/null; then
    influx -execute "CREATE DATABASE water_quality" 2>/dev/null || true
    echo "InfluxDB database 'water_quality' ready"
else
    echo "InfluxDB not installed — skipping database creation"
fi
REMOTE

# 4. Install systemd service
echo "[4/6] Installing systemd service..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
sudo cp /home/pi/cleareye/deploy/cleareye.service /etc/systemd/system/cleareye.service
sudo systemctl daemon-reload
sudo systemctl enable cleareye
sudo systemctl restart cleareye
sleep 2
sudo systemctl status cleareye --no-pager || true
REMOTE

# 5. Install nginx config
echo "[5/6] Configuring nginx reverse proxy..."
ssh "$SERVER" bash -s <<'REMOTE'
set -euo pipefail
sudo cp /home/pi/cleareye/deploy/nginx-cleareye.conf /etc/nginx/sites-available/cleareye
sudo ln -sf /etc/nginx/sites-available/cleareye /etc/nginx/sites-enabled/cleareye
sudo nginx -t
sudo systemctl reload nginx
REMOTE

# 6. Verify
echo "[6/6] Verifying deployment..."
sleep 3
ssh "$SERVER" "curl -sf http://127.0.0.1:8002/cleareye/health | python3 -m json.tool" \
    || echo "Service may still be starting..."

echo ""
echo "=== Deployment complete ==="
echo "  Service:  https://enviro-sensors.uk/cleareye/"
echo "  Docs:     https://enviro-sensors.uk/cleareye/docs"
echo "  Health:   https://enviro-sensors.uk/cleareye/health"
echo "  Predict:  POST https://enviro-sensors.uk/cleareye/predict"
echo "  Logs:     ssh $SERVER 'journalctl -u cleareye -f'"
