#!/bin/zsh
# Commits (and pushes, if a remote is set) each benchmark run as soon as it finishes, i.e. when its
# results/<tag>_summary.csv appears. In-progress runs (no summary yet) are never committed.
cd "$(dirname "$0")/.." || exit 1
log() { echo "$(date '+%F %T') $*"; }
while true; do
  for s in results/*_summary.csv(N); do
    git ls-files --error-unmatch "$s" >/dev/null 2>&1 && continue          # already committed
    tag=${${s:t}%_summary.csv}
    files=("$s" "results/$tag.jsonl"); [[ -f results/compare_0.8B_vs_2B.csv ]] && true
    git add -- $files && git commit -q -m "Add results: $tag

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && log "committed $tag"
    if git remote get-url origin >/dev/null 2>&1; then git push -q origin HEAD && log "pushed $tag" || log "PUSH FAILED $tag"; else log "no git remote set - committed locally only"; fi
  done
  sleep 120
done
