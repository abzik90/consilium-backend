#!/usr/bin/env bash

set -euo pipefail

APP_NAME="consilium"
APP_USER="${APP_USER:-ubuntu}"
APP_GROUP="${APP_GROUP:-$APP_USER}"
APP_ROOT="${APP_ROOT:-/home/ubuntu/consilium}"
BACKEND_DIR="${BACKEND_DIR:-$APP_ROOT/backend}"
FRONTEND_DIR="${FRONTEND_DIR:-$APP_ROOT/frontend}"
VENV_DIR="${VENV_DIR:-$BACKEND_DIR/.venv}"
ENV_FILE="${ENV_FILE:-$BACKEND_DIR/.env}"
SERVICE_NAME="${SERVICE_NAME:-consilium.service}"
DOMAIN="${DOMAIN:-_}"
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8000}"
NGINX_SITE_NAME="${NGINX_SITE_NAME:-consilium}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo."
  exit 1
fi

apt-get update
apt-get install -y python3 python3-venv python3-pip nginx rsync ufw

mkdir -p "$BACKEND_DIR" "$FRONTEND_DIR"
chown -R "$APP_USER:$APP_GROUP" "$APP_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  sudo -u "$APP_USER" python3 -m venv "$VENV_DIR"
fi

sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install "$BACKEND_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<EOF
DATABASE_URL=sqlite:///$BACKEND_DIR/consilium.db
OPENROUTER_CONSILIUM_KEY=${OPENROUTER_CONSILIUM_KEY:-}
SECRET_KEY=change-me-in-production
STORAGE_BASE_URL=https://$DOMAIN/files
UPLOAD_DIR=$BACKEND_DIR/uploads
CHROMA_DIR=$BACKEND_DIR/uploads/chroma_db
CHUNKS_DIR=$BACKEND_DIR/uploads/knowledge/chunks
FRONTEND_DIR=$FRONTEND_DIR
APP_HOST=$APP_HOST
APP_PORT=$APP_PORT
APP_RELOAD=false
EOF
  chown "$APP_USER:$APP_GROUP" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
fi

sed \
  -e "s|__APP_USER__|$APP_USER|g" \
  -e "s|__APP_GROUP__|$APP_GROUP|g" \
  -e "s|__BACKEND_DIR__|$BACKEND_DIR|g" \
  -e "s|__ENV_FILE__|$ENV_FILE|g" \
  -e "s|__VENV_DIR__|$VENV_DIR|g" \
  -e "s|__APP_HOST__|$APP_HOST|g" \
  -e "s|__APP_PORT__|$APP_PORT|g" \
  "$SCRIPT_DIR/consilium.service" > "/etc/systemd/system/$SERVICE_NAME"

sed \
  -e "s|__DOMAIN__|$DOMAIN|g" \
  -e "s|__APP_HOST__|$APP_HOST|g" \
  -e "s|__APP_PORT__|$APP_PORT|g" \
  "$SCRIPT_DIR/nginx.consilium.conf" > "/etc/nginx/sites-available/$NGINX_SITE_NAME"

ln -sf "/etc/nginx/sites-available/$NGINX_SITE_NAME" "/etc/nginx/sites-enabled/$NGINX_SITE_NAME"
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
systemctl restart nginx

ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw --force enable

echo "Production setup complete."
echo "Remember to update $ENV_FILE with real secrets and then restart $SERVICE_NAME."