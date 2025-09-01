#!/usr/bin/env bash
set -euo pipefail

# Trap errors and report the last step
trap 'echo "❌ Error during: ${STEP}"; exit 1' ERR

# List of submodules to update
SUBMODULES=("rectified-flow-pytorch" "stylegan-v")

echo "🔄 Starting submodule update workflow…"

for SUBMODULE_PATH in "${SUBMODULES[@]}"; do
  COMMIT_MSG="chore: bump ${SUBMODULE_PATH} to upstream"

  echo "📦 Processing submodule: ${SUBMODULE_PATH}"

  # 1️⃣ Ensure .gitmodules is synced
  STEP="Syncing .gitmodules for ${SUBMODULE_PATH}"
  git submodule sync "${SUBMODULE_PATH}"

  # 2️⃣ Fetch & fast-forward the tracked branch in one go
  STEP="Updating ${SUBMODULE_PATH} via --remote"
  git submodule update --remote "${SUBMODULE_PATH}"

  # 3️⃣ Stage the updated pointer
  STEP="Staging updated pointer for ${SUBMODULE_PATH}"
  git add "${SUBMODULE_PATH}"

  # 4️⃣ Commit if there’s any change
  STEP="Committing pointer bump for ${SUBMODULE_PATH}"
  if git diff --cached --quiet "${SUBMODULE_PATH}"; then
    echo "ℹ️  No pointer changes to commit for ${SUBMODULE_PATH}."
  else
    git commit -m "${COMMIT_MSG}"
    echo "✅  Committed: ${COMMIT_MSG}"
  fi

  echo "✅ Finished ${SUBMODULE_PATH}"
  echo "-----------------------------------"
done

echo "🎉  All submodules updated successfully!"
