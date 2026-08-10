#!/usr/bin/env bash
#
# install-minio.sh — Single-node MinIO installer for Ubuntu (20.04 / 22.04 / 24.04)
#
# Installs the MinIO server + mc client, creates a dedicated service account,
# writes a systemd unit, and exposes the S3 API and Console on all interfaces.
#
# Usage:  sudo ./install-minio.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MINIO_ROOT_USER="minioadmin"
MINIO_ROOT_PASSWORD='datafederation_hooray!'

MINIO_DATA_DIR="/mnt/minio/data"      # object storage volume
MINIO_API_PORT="9000"                 # S3 API endpoint
MINIO_CONSOLE_PORT="9001"             # web console
MINIO_BIND_ADDR="0.0.0.0"             # listen on all interfaces
MINIO_SERVICE_USER="minio-user"

# Release to install.
#   "latest"  -> current release (community build has a minimal Console UI)
#   pin a tag -> e.g. RELEASE.2025-04-22T22-12-26Z for the full browser Console
MINIO_VERSION="latest"

MINIO_BIN="/usr/local/bin/minio"
MC_BIN="/usr/local/bin/mc"
ENV_FILE="/etc/default/minio"
UNIT_FILE="/etc/systemd/system/minio.service"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run this script as root (sudo $0)"

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
log "Installing prerequisites"
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
APT_OPTS=(-o DPkg::Lock::Timeout=120 -o Acquire::ForceIPv4=true
          -o Acquire::http::Timeout=15 -o Acquire::Retries=3)
apt-get "${APT_OPTS[@]}" install -y curl ca-certificates

# ---------------------------------------------------------------------------
# 2. Architecture detection
# ---------------------------------------------------------------------------
case "$(uname -m)" in
  x86_64)  ARCH="amd64" ;;
  aarch64) ARCH="arm64" ;;
  *)       die "Unsupported architecture: $(uname -m)" ;;
esac
log "Detected architecture: ${ARCH}"

BASE_URL="https://dl.min.io"
if [[ "$MINIO_VERSION" == "latest" ]]; then
  MINIO_URL="${BASE_URL}/server/minio/release/linux-${ARCH}/minio"
else
  MINIO_URL="${BASE_URL}/server/minio/release/linux-${ARCH}/archive/minio.${MINIO_VERSION}"
fi
MC_URL="${BASE_URL}/client/mc/release/linux-${ARCH}/mc"

# ---------------------------------------------------------------------------
# 3. Binaries
# ---------------------------------------------------------------------------
log "Downloading MinIO server (${MINIO_VERSION})"
systemctl stop minio 2>/dev/null || true
curl -fsSL --retry 3 -o "${MINIO_BIN}.tmp" "$MINIO_URL" \
  || die "Download failed: $MINIO_URL"
chmod 755 "${MINIO_BIN}.tmp"
mv -f "${MINIO_BIN}.tmp" "$MINIO_BIN"

log "Downloading mc client"
curl -fsSL --retry 3 -o "${MC_BIN}.tmp" "$MC_URL" || die "mc download failed"
chmod 755 "${MC_BIN}.tmp"
mv -f "${MC_BIN}.tmp" "$MC_BIN"

"$MINIO_BIN" --version | head -n1

# ---------------------------------------------------------------------------
# 4. Service account + data directory
# ---------------------------------------------------------------------------
if ! getent group "$MINIO_SERVICE_USER" >/dev/null; then
  groupadd -r "$MINIO_SERVICE_USER"
fi
if ! id -u "$MINIO_SERVICE_USER" >/dev/null 2>&1; then
  useradd -M -r -g "$MINIO_SERVICE_USER" -s /sbin/nologin "$MINIO_SERVICE_USER"
  log "Created service account: ${MINIO_SERVICE_USER}"
fi

log "Preparing data directory: ${MINIO_DATA_DIR}"
mkdir -p "$MINIO_DATA_DIR"
chown -R "${MINIO_SERVICE_USER}:${MINIO_SERVICE_USER}" "$MINIO_DATA_DIR"
chmod 750 "$MINIO_DATA_DIR"

# ---------------------------------------------------------------------------
# 5. Environment file
# ---------------------------------------------------------------------------
log "Writing ${ENV_FILE}"
cat > "$ENV_FILE" <<EOF
# MinIO configuration — managed by install-minio.sh

MINIO_VOLUMES="${MINIO_DATA_DIR}"

MINIO_ROOT_USER="${MINIO_ROOT_USER}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD}"

# API on :${MINIO_API_PORT}, Console on :${MINIO_CONSOLE_PORT}, all interfaces
MINIO_OPTS="--address ${MINIO_BIND_ADDR}:${MINIO_API_PORT} --console-address ${MINIO_BIND_ADDR}:${MINIO_CONSOLE_PORT}"

# Uncomment and set to the externally reachable hostname if behind a proxy/NAT
#MINIO_SERVER_URL="http://your.host.example:${MINIO_API_PORT}"
#MINIO_BROWSER_REDIRECT_URL="http://your.host.example:${MINIO_CONSOLE_PORT}"
EOF
chown root:"$MINIO_SERVICE_USER" "$ENV_FILE"
chmod 640 "$ENV_FILE"

# ---------------------------------------------------------------------------
# 6. systemd unit
# ---------------------------------------------------------------------------
log "Writing ${UNIT_FILE}"
cat > "$UNIT_FILE" <<EOF
[Unit]
Description=MinIO Object Storage
Documentation=https://min.io/docs/minio/linux/index.html
Wants=network-online.target
After=network-online.target
AssertFileIsExecutable=${MINIO_BIN}

[Service]
Type=notify
User=${MINIO_SERVICE_USER}
Group=${MINIO_SERVICE_USER}
WorkingDirectory=/usr/local
EnvironmentFile=${ENV_FILE}
ExecStartPre=/bin/bash -c "if [ -z \"\${MINIO_VOLUMES}\" ]; then echo 'MINIO_VOLUMES not set in ${ENV_FILE}'; exit 1; fi"
ExecStart=${MINIO_BIN} server \$MINIO_OPTS \$MINIO_VOLUMES
Restart=always
RestartSec=5s
LimitNOFILE=1048576
TasksMax=infinity
TimeoutStopSec=infinity
SendSIGKILL=no
OOMScoreAdjust=-1000

[Install]
WantedBy=multi-user.target
EOF

# ---------------------------------------------------------------------------
# 7. Firewall (only if ufw is active)
# ---------------------------------------------------------------------------
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q "Status: active"; then
  log "Opening ports ${MINIO_API_PORT} and ${MINIO_CONSOLE_PORT} in ufw"
  ufw allow "${MINIO_API_PORT}/tcp"     >/dev/null
  ufw allow "${MINIO_CONSOLE_PORT}/tcp" >/dev/null
else
  warn "ufw not active — skipping firewall rules (check any cloud security groups)"
fi

# ---------------------------------------------------------------------------
# 8. Start
# ---------------------------------------------------------------------------
log "Starting MinIO"
systemctl daemon-reload
systemctl enable --now minio >/dev/null

for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:${MINIO_API_PORT}/minio/health/live" >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 1
done
[[ "${READY:-0}" == "1" ]] || {
  journalctl -u minio -n 40 --no-pager
  die "MinIO did not become healthy — see the log above"
}

# ---------------------------------------------------------------------------
# 9. mc alias for local admin use
# ---------------------------------------------------------------------------
log "Configuring mc alias 'local'"
"$MC_BIN" alias set local "http://127.0.0.1:${MINIO_API_PORT}" \
  "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat <<EOF

────────────────────────────────────────────────────────────
 MinIO is running
────────────────────────────────────────────────────────────
 S3 API      : http://${IP:-<host>}:${MINIO_API_PORT}
 Console     : http://${IP:-<host>}:${MINIO_CONSOLE_PORT}
 Root user   : ${MINIO_ROOT_USER}
 Root passwd : ${MINIO_ROOT_PASSWORD}
 Data dir    : ${MINIO_DATA_DIR}

 Service     : systemctl status minio
 Logs        : journalctl -u minio -f
 CLI         : mc ls local        (alias already configured for root)
────────────────────────────────────────────────────────────
EOF
