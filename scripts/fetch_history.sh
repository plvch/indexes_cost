#!/bin/bash
# Fetch the SEC DERA history used by the published analysis. Extracts only the
# tables the panel needs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/raw"
cd "$ROOT/data/raw"
UA="${SEC_USER_AGENT:?Set SEC_USER_AGENT to your name and email before running}"

FEE_TAGS='ExpensesOverAssets|NetExpensesOverAssets|ManagementFeesOverAssets|OtherExpensesOverAssets|FeeWaiverOrReimbursementOverAssets'

fetch() { # $1=url $2=dirname $3...=tables to extract
  local url=$1 dir=$2; shift 2
  [ -d "$dir" ] && { echo "skip $dir"; return 0; }
  curl -sf -A "$UA" -o "$dir.zip" "$url" || { rm -f "$dir.zip"; return 1; }
  mkdir -p "$dir" && unzip -oq "$dir.zip" -d "$dir" "$@"
  rm -f "$dir.zip"
  # keep only fee-table rows of num.tsv (the rest is never read)
  if [ -f "$dir/num.tsv" ]; then
    awk -F'\t' -v tags="^($FEE_TAGS)$" 'NR==1 || $2 ~ tags' "$dir/num.tsv" \
      > "$dir/num.f" && mv "$dir/num.f" "$dir/num.tsv"
  fi
  echo "got $dir"
}

# --- Risk/Return: 2010q4..2026q1
RRBASE="https://www.sec.gov/files/dera/data/mutual-fund-prospectus-risk/return-summary-data-sets"
QS="2010q4"
for y in $(seq 2011 2025); do for q in q1 q2 q3 q4; do QS="$QS $y$q"; done; done
QS="$QS 2026q1"
for t in $QS; do
  fetch "$RRBASE/${t}_rr1.zip" "${t}_rr1" num.tsv sub.tsv || echo "MISS ${t}_rr1"
  sleep 0.5
done

# --- N-CEN: 2018q3..2026q1 (with _0 filename fallback)
NCBASE="https://www.sec.gov/files/dera/data/form-n-cen-data-sets"
NQS="2018q3 2018q4"
for y in $(seq 2019 2025); do for q in q1 q2 q3 q4; do NQS="$NQS $y$q"; done; done
NQS="$NQS 2026q1"
for t in $NQS; do
  fetch "$NCBASE/${t}_ncen.zip" "${t}_ncen" FUND_REPORTED_INFO.tsv REGISTRANT.tsv \
    || fetch "$NCBASE/${t}_ncen_0.zip" "${t}_ncen" FUND_REPORTED_INFO.tsv REGISTRANT.tsv \
    || echo "MISS ${t}_ncen"
  sleep 0.5
done
echo "DONE"
