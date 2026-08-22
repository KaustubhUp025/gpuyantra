#!/usr/bin/env bash
# Create the composite vector index for the `skills` collection (spec 6.2).
#
# Run ONCE at setup. The build takes several minutes — NEVER run this on demo day.
# The equality pre-filter fields (op_family, hardware) MUST precede the vector field.
# Firestore vector search supports equality pre-filters only — no inequalities.
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-gpuyantra}"
DATABASE="${FIRESTORE_DATABASE:-(default)}"

gcloud firestore indexes composite create \
  --project="${PROJECT}" \
  --collection-group=skills \
  --query-scope=COLLECTION \
  --field-config=field-path=op_family,order=ASCENDING \
  --field-config=field-path=hardware,order=ASCENDING \
  --field-config='vector-config={"dimension":"768","flat":"{}"},field-path=embedding' \
  --database="${DATABASE}"

echo "Index creation submitted. Poll with:"
echo "  gcloud firestore indexes composite list --project=${PROJECT} --database='${DATABASE}'"
