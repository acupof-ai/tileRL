#!/usr/bin/env bash
# Build, deploy and tail the tilerl GPU pod.
# Usage: scripts/pod.sh {build|load|apply|logs|forward|all}
set -euo pipefail

IMAGE="${IMAGE:-tilerl:latest}"
POD="${POD:-tilerl}"
PORT="${PORT:-8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

build() {
  docker build -t "$IMAGE" "$ROOT"
}

load() {
  if kind get clusters 2>/dev/null | grep -q .; then
    kind load docker-image "$IMAGE"
  elif command -v minikube >/dev/null && minikube status >/dev/null 2>&1; then
    minikube image load "$IMAGE"
  else
    echo "no local kind/minikube cluster found; push $IMAGE to a registry and" >&2
    echo "update k8s/pod.yaml image: instead" >&2
    exit 1
  fi
}

apply() {
  kubectl apply -f "$ROOT/k8s/pod.yaml"
}

logs() {
  kubectl logs -f "$POD"
}

forward() {
  kubectl port-forward "pod/$POD" "$PORT:8000"
}

case "${1:-}" in
  build)   build ;;
  load)    load ;;
  apply)  apply ;;
  logs)   logs ;;
  forward) forward ;;
  all)    build && load && apply && logs ;;
  *)      echo "usage: $0 {build|load|apply|logs|forward|all}" >&2; exit 2 ;;
esac
