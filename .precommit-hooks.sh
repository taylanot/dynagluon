#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${REPO_ROOT}/.precommit-image"

log() { echo "[precommit-hooks] $*"; }
die() { echo "[precommit-hooks] ERROR: $*" >&2; exit 1; }

command -v podman &>/dev/null        || die "podman is not installed or not in PATH."
[[ -f "${DOCKERFILE}" ]]             || die "Dockerfile not found: ${DOCKERFILE}."

FULL_IMAGE="$(grep -i '^LABEL image\.name=' "${DOCKERFILE}" | head -1 \
    | sed 's/^LABEL image\.name=//I; s/"//g; s/'"'"'//g; s/[[:space:]]*$//')"
[[ -n "${FULL_IMAGE}" ]] || die "No 'LABEL image.name=...' found in ${DOCKERFILE}."
log "Image: '${FULL_IMAGE}'"

REBUILD=false; EXTRA_ARGS=()
for arg in "$@"; do
    [[ "${arg}" == "--rebuild" ]] && REBUILD=true || EXTRA_ARGS+=("${arg}")
done

if [[ "${REBUILD}" == true ]]; then
    log "--rebuild: removing '${FULL_IMAGE}'..."
    podman rmi --force "${FULL_IMAGE}" 2>/dev/null || true
fi

if ! podman image exists "${FULL_IMAGE}" 2>/dev/null; then
    log "Building '${FULL_IMAGE}'..."
    podman build --tag "${FULL_IMAGE}" --file "${DOCKERFILE}" "${REPO_ROOT}"
fi

log "Running hooks..."
podman run --rm --tty \
    --volume "${REPO_ROOT}:/workspace:z" --workdir /workspace \
    "${FULL_IMAGE}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
