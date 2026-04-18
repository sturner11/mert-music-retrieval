#!/usr/bin/env bash
set -euo pipefail

# Sync local experiment outputs to Google Drive via rclone.
#
# Defaults:
#   remote: gdrive
#   drive root folder: mert-music-retrieval
#
# Usage:
#   bash scripts/sync_to_drive.sh
#   bash scripts/sync_to_drive.sh --dry-run
#   DRIVE_REMOTE=myremote DRIVE_ROOT=my-folder bash scripts/sync_to_drive.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${DRIVE_REMOTE:-gdrive}"
ROOT_FOLDER="${DRIVE_ROOT:-mert-music-retrieval}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

RCLONE_FLAGS=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  RCLONE_FLAGS+=(--dry-run)
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Drive remote: ${REMOTE}"
echo "Drive root folder: ${ROOT_FOLDER}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Mode: dry-run"
fi

mkdir_remote_dir() {
  local subdir="$1"
  rclone mkdir "${REMOTE}:${ROOT_FOLDER}/${subdir}"
}

sync_dir() {
  local local_dir="$1"
  local remote_subdir="$2"
  echo "Syncing ${local_dir} -> ${REMOTE}:${ROOT_FOLDER}/${remote_subdir}"
  rclone sync "${PROJECT_ROOT}/${local_dir}" "${REMOTE}:${ROOT_FOLDER}/${remote_subdir}" "${RCLONE_FLAGS[@]}"
}

mkdir_remote_dir "notebooks"
mkdir_remote_dir "artifacts"

sync_dir "notebooks" "notebooks"
sync_dir "artifacts" "artifacts"

echo "Drive sync complete."
