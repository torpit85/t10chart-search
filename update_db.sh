#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-t10.sqlite}"
IMPORTER="${IMPORTER:-import_zip_to_sqlite.py}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <charts.zip> [--git]"
  echo "  <charts.zip> : path to your exported zip"
  echo "  --git        : also git add/commit/push the updated DB"
  exit 1
fi

ZIP="$1"
DO_GIT=0
if [[ "${2:-}" == "--git" ]]; then
  DO_GIT=1
fi

if [[ ! -f "$ZIP" ]]; then
  echo "ERROR: zip not found: $ZIP" >&2
  exit 1
fi
if [[ ! -f "$DB" ]]; then
  echo "ERROR: DB not found: $DB" >&2
  exit 1
fi
if [[ ! -f "$IMPORTER" ]]; then
  echo "ERROR: importer not found: $IMPORTER" >&2
  exit 1
fi

ts="$(date +%F_%H%M%S)"
mkdir -p backups logs

BACKUP="backups/${DB%.sqlite}.bak.${ts}.sqlite"
LOG="logs/update_${ts}.log"

echo "[update] ZIP=$ZIP" | tee "$LOG"
echo "[update] DB =$DB"  | tee -a "$LOG"
echo "[update] backup -> $BACKUP" | tee -a "$LOG"

cp -a "$DB" "$BACKUP"

echo "[update] running importer..." | tee -a "$LOG"
python3 -u "$IMPORTER" "$ZIP" "$DB" 2>&1 | tee -a "$LOG"

echo "[update] backfilling last_week (NEW/RE/pos)..." | tee -a "$LOG"
sqlite3 "$DB" <<'SQL'
.bail on
UPDATE t10_entry AS cur
SET last_week =
  COALESCE(
    (SELECT CAST(prev.pos AS TEXT)
     FROM t10_entry AS prev
     WHERE prev.show_id = cur.show_id
       AND date(prev.week_ending) = date(cur.week_ending, '-7 day')
     LIMIT 1),
    CASE
      WHEN EXISTS (
        SELECT 1
        FROM t10_entry AS earlier
        WHERE earlier.show_id = cur.show_id
          AND date(earlier.week_ending) < date(cur.week_ending)
      )
      THEN 'RE'
      ELSE 'NEW'
    END
  )
WHERE (cur.last_week IS NULL OR cur.last_week = '');
SQL

missing="$(sqlite3 "$DB" "SELECT COUNT(*) FROM t10_entry WHERE last_week IS NULL OR last_week='';")"
echo "[check] missing last_week rows: $missing" | tee -a "$LOG"
if [[ "$missing" != "0" ]]; then
  echo "ERROR: last_week still missing for $missing rows. See $LOG" >&2
  exit 1
fi

bad_imprints="$(sqlite3 "$DB" "SELECT COUNT(*) FROM t10_entry WHERE imprint_1 LIKE '%/%' OR imprint_2 LIKE '%/%' OR imprint_1 LIKE '%|%' OR imprint_2 LIKE '%|%';")"
echo "[check] rows with '/' or '|' in imprints: $bad_imprints" | tee -a "$LOG"
if [[ "$bad_imprints" != "0" ]]; then
  echo "WARNING: Some imprints still contain '/' or '|'. (Importer *should* normalize these.) See $LOG" | tee -a "$LOG"
fi

if [[ "$DO_GIT" == "1" ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[git] committing DB + pushing..." | tee -a "$LOG"
    git add "$DB" | tee -a "$LOG"
    git commit -m "Update DB from $(basename "$ZIP")" | tee -a "$LOG" || true
    git push | tee -a "$LOG"
  else
    echo "WARNING: not a git repo; skipping --git" | tee -a "$LOG"
  fi
fi

echo "[done] OK. Log: $LOG" | tee -a "$LOG"
