#!/usr/bin/env bash
# Snapshot the Firestore skill library + bandit state (spec 11, "Firestore snapshot").
#
# The demo's headline number depends on what is in `skills`: retrieval picks a prior
# kernel to build on, and the UCB1 bandit's arm statistics decide which one leads.
# Replaying a recorded demo against a library that has drifted since is not a replay.
# Take a snapshot immediately before recording, and again before any run you intend
# to be able to reproduce.
#
# Usage:
#   bash scripts/export_firestore.sh                 # export to the default bucket
#   BUCKET=gs://my-bucket bash scripts/export_firestore.sh
#
# Restore (destructive — overwrites live documents):
#   gcloud firestore import gs://gpuyantra-backups/snapshot-YYYYmmdd-HHMMSS \
#     --project=gpuyantra --database='(default)'
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-gpuyantra}"
DATABASE="${FIRESTORE_DATABASE:-(default)}"
BUCKET="${BUCKET:-gs://gpuyantra-backups}"
DESTINATION="${BUCKET}/snapshot-$(date +%Y%m%d-%H%M%S)"

# The export is asynchronous and bills against the project; fail loudly rather than
# silently writing a snapshot nobody can find.
if ! gcloud storage ls "${BUCKET}" >/dev/null 2>&1; then
  echo "error: bucket ${BUCKET} is not reachable." >&2
  echo "Create it once with:" >&2
  echo "  gcloud storage buckets create ${BUCKET} --project=${PROJECT} --location=us-central1" >&2
  exit 1
fi

echo "Exporting ${PROJECT} database '${DATABASE}' -> ${DESTINATION}"
gcloud firestore export "${DESTINATION}" \
  --project="${PROJECT}" \
  --database="${DATABASE}"

echo
echo "Snapshot written to ${DESTINATION}"
echo "Restore with:"
echo "  gcloud firestore import ${DESTINATION} --project=${PROJECT} --database='${DATABASE}'"
