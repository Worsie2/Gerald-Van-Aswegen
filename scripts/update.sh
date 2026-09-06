#!/usr/bin/env bash
# Update DSAI on macOS or Linux, then check the result.
#
#     ./scripts/update.sh
#
# Stop the running app first (Ctrl+C in its terminal). Streamlit keeps imported
# modules in memory, so a server left running serves the old code no matter how
# successful the pull was — that is the most common reason an update "does not
# work".

set -euo pipefail
branch="claude/ai-data-analytics-platform-pc429g"

cd "$(dirname "$0")/.."
echo "Working in $(pwd)"

if [ ! -f pyproject.toml ]; then
  echo "This is not the project folder — pyproject.toml is not here." >&2
  exit 1
fi

if command -v lsof >/dev/null && lsof -i :8501 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Something is already serving port 8501. Stop it (Ctrl+C in its terminal) before"
  echo "starting again, or the app you open will still be the old version."
fi

if [ -f .venv/bin/activate ]; then
  echo "Activating .venv"
  # shellcheck disable=SC1091
  . .venv/bin/activate
else
  echo "No .venv found — installing into whichever Python is on PATH."
fi

stashed=""
if [ -n "$(git status --porcelain)" ]; then
  echo "You have local changes. Stashing them so the pull can proceed."
  git stash push -m "dsai update $(date -Iseconds)"
  stashed=1
fi

echo "Pulling $branch"
git pull origin "$branch"

if [ -n "$stashed" ]; then
  echo "Restoring your local changes"
  git stash pop
fi

echo "Installing dependencies"
pip install -e ".[full]" --quiet

echo
dsai doctor
echo
echo "Start the app with:  dsai app"
