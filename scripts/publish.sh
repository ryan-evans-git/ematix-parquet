#!/usr/bin/env bash
#
# Topological crates.io publish for the ematix-parquet workspace.
#
# The workspace has 6 publishable member crates with inter-dependencies, so
# they must be published to crates.io in dependency order — each crate only
# after every sibling it depends on is already live on the registry (otherwise
# the downstream crate's verification build can't resolve the new version).
#
# Dependency graph (derived from crates/*/Cargo.toml):
#   ematix-parquet-format   — no sibling deps          ┐ level 0
#   ematix-parquet-crypto   — no sibling deps          ┘
#   ematix-parquet-io       — needs: format              level 1
#   ematix-parquet-codec    — needs: crypto, format, io  level 2
#   ematix-parquet-async    — needs: codec, format, io ┐ level 3
#   ematix-iceberg          — needs: codec             ┘
#
# PREREQUISITES (NOT done by this script — publishing is irreversible):
#   1. `cargo login <crates.io-token>`   (no token is configured by default)
#   2. clean release commit checked out, workspace.package version bumped
#   3. green: `cargo test --workspace` + `cargo publish -p ematix-parquet-format --dry-run`
#
# Usage:
#   scripts/publish.sh --dry-run   # validate packaging+manifest for all 6 crates
#                                  #   (cargo package --no-verify; no cross-dep
#                                  #   verify build, so no false errors)
#   scripts/publish.sh             # REAL publish, in order, waiting between levels
#
# After a successful publish:
#   git tag v$VER && git push origin v$VER     # tag the release
#   # then in ../ematix-flow bump `ematix-parquet-* = "0.17"` and verify the
#   # Cargo.lock actually resolves 0.17.0 (the [patch.crates-io] version-match trap).
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"
WAIT="${PUBLISH_WAIT_SEC:-30}"

VER=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
echo "=== ematix-parquet workspace publish — v$VER  (mode: ${DRY:-REAL}) ==="

levels=(
  "ematix-parquet-format ematix-parquet-crypto"
  "ematix-parquet-io"
  "ematix-parquet-codec"
  "ematix-parquet-async ematix-iceberg"
)

if [[ -n "$DRY" ]]; then
  # Only the leaf crates (no sibling deps) can be validated pre-publish. A
  # downstream crate's `version = "$VER"` dep on a sibling can't be resolved
  # from the registry until that sibling is actually published, so cargo can't
  # even prepare its package ("failed to prepare local package for uploading").
  # Those validate during the ordered real publish, once each level is live.
  # Workspace-wide compile/tests: run `cargo test --workspace`.
  echo "dry-run: validating leaf crates (${levels[0]}) only."
  echo "  downstream crates can't be packaged until their deps are live —"
  echo "  they validate during the real ordered publish below."
  for crate in ${levels[0]}; do
    echo "==> cargo package --no-verify -p $crate"
    cargo package --no-verify -p "$crate"
  done
  echo "=== dry-run OK: leaf crates package cleanly ==="
  exit 0
fi

for i in "${!levels[@]}"; do
  echo "--- level $i: ${levels[$i]} ---"
  for crate in ${levels[$i]}; do
    echo "==> cargo publish -p $crate"
    cargo publish -p "$crate"
  done
  # Wait for the registry index to propagate so the next level can resolve the
  # versions just published. Skip after the final level.
  if [[ "$i" -lt $(( ${#levels[@]} - 1 )) ]]; then
    echo "--- waiting ${WAIT}s for crates.io index propagation ---"
    sleep "$WAIT"
  fi
done

echo "=== published ematix-parquet v$VER ==="
echo "next: git tag v$VER  (push is your call) ; bump ../ematix-flow ematix-parquet-* pin to \"$VER\""
