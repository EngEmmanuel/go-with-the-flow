#!/usr/bin/env bash
set -euo pipefail

# Trap errors and report the last step
trap 'echo "❌ Error during: ${STEP}"; exit 1' ERR

SUBMODULE_PATH="rectified-flow-pytorch"
COMMIT_MSG="chore: bump ${SUBMODULE_PATH} to upstream"

echo "🔄 Starting submodule update workflow…"

# 1️⃣ Ensure .gitmodules is synced
STEP="Syncing .gitmodules"
git submodule sync "${SUBMODULE_PATH}"

# 2️⃣ Fetch & fast-forward the tracked branch in one go
STEP="Updating submodule via --remote"
git submodule update --remote "${SUBMODULE_PATH}"

# 3️⃣ Stage the updated pointer
STEP="Staging updated submodule pointer"
git add "${SUBMODULE_PATH}"

# 4️⃣ Commit if there’s any change
STEP="Committing pointer bump"
if git diff --cached --quiet; then
  echo "ℹ️  No pointer changes to commit."
else
  git commit -m "${COMMIT_MSG}"
  echo "✅  Committed: ${COMMIT_MSG}"
fi

echo "🎉  Submodule update complete!"