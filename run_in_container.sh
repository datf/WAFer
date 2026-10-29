#!/usr/bin/env bash
# run_in_container.sh - Helper script to run commands inside the containerized AI engine
set -euo pipefail

# Choose container runtime: podman if available, else docker
if command -v podman >/dev/null 2>&1; then
    RUNTIME="podman"
elif command -v docker >/dev/null 2>&1; then
    RUNTIME="docker"
else
    echo "Error: Neither podman nor docker found in PATH." >&2
    exit 1
fi

IMAGE_TAG="wafer-engine:dev"
CMD="${*:-pytest tests/ -v}"

# Build image if it does not exist
if ! $RUNTIME image exists "$IMAGE_TAG" 2>/dev/null; then
    echo "Building container image $IMAGE_TAG..."
    $RUNTIME build -t "$IMAGE_TAG" -f docker/Dockerfile.engine .
fi

echo "Running in $RUNTIME container ($IMAGE_TAG): $CMD"
$RUNTIME run --rm -e PYTHONPATH=/app -v "$(pwd):/app:Z" -w /app "$IMAGE_TAG" bash -c "$CMD"
