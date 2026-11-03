#!/usr/bin/env bash
# Rebuild the full Thought Anchors trace viewer and publish it at https://robots.ox.ac.uk/~be/anchors/.
# The site data lives in ~/anchors_site on robots (outside WWW, which the robots repo's deploy action
# rebuilds from scratch); robots_publish.sh hard-links it into WWW/anchors on every site deploy, and
# this script re-links it immediately after an update.
#
# Usage: bash bash_scripts/deploy_anchors_viewer.sh "Name=<run_dir>" ...
set -euo pipefail
SITE=${SITE_DIR:-/data/math-lasr/shil6499/ta_viewer_site}
uv run python -m utils.heuristic.ta_viewer_site --runs "$@" --out "$SITE"
rsync -az --delete "$SITE/" robots:anchors_site/
ssh -o BatchMode=yes robots 'rm -rf ~/WWW/anchors && cp -al ~/anchors_site ~/WWW/anchors && chmod -R 777 ~/WWW/anchors'
curl -sf -o /dev/null https://robots.ox.ac.uk/~be/anchors/data/index.json && echo "Live: https://robots.ox.ac.uk/~be/anchors/"
