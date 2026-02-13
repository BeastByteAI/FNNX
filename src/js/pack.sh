#!/bin/bash
set -e

cd "$(dirname "$0")"

pnpm build

mkdir -p local-packages

for pkg in packages/*/; do
  (cd "$pkg" && pnpm pack)
  mv "$pkg"/*.tgz local-packages/
done
