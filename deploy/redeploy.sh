#!/usr/bin/env bash

set -euo pipefail

APP_NAME="${APP_NAME:-consilium}"
APP_ROOT="${APP_ROOT:-/home/consilium/consilium}"
BACKEND_DIR="${BACKEND_DIR:-$APP_ROOT/backend}"
VENV_DIR="${VENV_DIR:-$BACKEND_DIR/.venv}"
SERVICE_NAME="${SERVICE_NAME:-consilium.service}"

cd "$BACKEND_DIR"
"$VENV_DIR/bin/pip" install "$BACKEND_DIR"
sudo systemctl restart "$SERVICE_NAME"
echo "$APP_NAME backend redeployed"