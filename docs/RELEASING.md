# Releasing ematix-parquet

The release flow is **PR → merge to main → tag main → workflow
publishes to crates.io**. Tags on non-main commits are blocked by
ruleset; direct pushes to main are blocked by branch protection.
Every release is reproducible from `main` at the tagged commit.

## Cutting a release

1. **Land all release-bound work on a feature branch.** Open PRs
   into `main` for each meaningful unit; merge them as they go.
   The branch name convention is `claude/<descriptor>` for agent
   branches and `<your-handle>/<descriptor>` for human ones.

2. **Open the release PR.** On a feature branch, bump the workspace
   version in `Cargo.toml` (workspace.package.version) and the
   inter-crate `version = "X.Y"` pins in
   `crates/ematix-parquet-{io,codec,async}/Cargo.toml`. Update
   `docs/plans/CURRENT.md` and the README `## Status` block.
   Open a PR titled `release: vX.Y.Z — <theme>`.

3. **Wait for CI.** The required check is `build-test` (build +
   test on macos-14 for NEON coverage). `rustfmt` and `clippy`
   run informationally for now.

4. **Merge to main.** Use a **merge commit** (preserves the per-
   commit messages in the release branch) or **squash** if the
   branch is just a version bump. Linear history is required —
   no merge bubbles, so rebase first if main has moved.

5. **Tag main.** From a clean local `main`:

   ```sh
   git checkout main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z — <one-line theme>"
   git push origin vX.Y.Z
   ```

   The `release.yml` workflow fires on the tag push and publishes
   the five crates to crates.io in dependency order
   (`format → io → crypto → codec → async`), with a 45 s sleep
   between publishes so crates.io's index propagates. `crypto`
   slots before `codec` because codec optionally depends on it via
   the `encryption` feature.

6. **Verify.** After ~5 min check the workflow run and confirm
   all five crates show the new version on crates.io. If a
   publish fails midway, the workflow's order means the failed
   crate and everything downstream of it didn't ship — fix the
   issue, bump the patch version, and re-cut.

## Branch protection on `main`

Configured via repository ruleset (`Settings → Rules → Rulesets`):

- **Restrict creations / updates / deletions on direct push** —
  no `git push origin main` from a workstation.
- **Require pull request before merging** — every change to main
  goes through a PR.
- **Require status checks: `build-test`** — CI must be green.
- **Require linear history** — no merge bubbles in main's log.
- **Allow force-pushes**: false.
- **Allow deletions**: false.
- **Bypass list**: repo admin (so you can fix things in an
  emergency without locking yourself out).
- **Required approvers**: 0 (single-maintainer setup; flip to 1
  when a second human / agent is reliably reviewing).

## Tag protection for `v*`

A separate ruleset on tags matching `v*`:

- **Restrict creations** — only repo admins can create matching
  tags. (Anyone with push access can technically tag any commit
  in git's data model, so this is a process gate, not cryptographic
  enforcement; with a single-admin repo the practical effect is
  "tags only get created by you, after the release PR merges.")
- **Restrict deletions** — `v*` tags are immutable once published.

## What if the release workflow fails?

The workflow has three sequential publish steps. If
`ematix-parquet-format` publishes but `ematix-parquet-io` fails,
the workspace ends up with two crates at the new version on
crates.io and one at the old one — recovery is to bump the patch
(e.g. v0.2.0 → v0.2.1), commit, PR, merge, re-tag. Crates.io
versions are immutable; you can `cargo yank` but not delete.

## Hot-fix releases

Same flow. Branch off main, fix the bug, bump patch, PR, merge,
tag. The version-pinning convention (`version = "0.2"` in
inter-crate deps) means a v0.2.x patch is auto-pulled by anyone
on the v0.2 line.

## Pre-release tags (alpha / beta / rc)

Use semver pre-release identifiers: `v0.3.0-rc.1`,
`v0.3.0-beta.2`, etc. The `release.yml` workflow's tag matcher is
`v*`, so pre-release tags publish too — `cargo publish` accepts
pre-release versions and downstream consumers won't pull them
unless they explicitly opt in (`= "0.3.0-rc.1"`).
