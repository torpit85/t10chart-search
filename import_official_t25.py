#!/usr/bin/env python3
"""
Import official Monthly T-25 chart rows into the T-10 Chart app database.

Input CSV expected from the extraction workflow:
  t25_monthly_charts_extracted_final.csv

The importer creates a dedicated `official_t25_entry` table instead of forcing
these official charts into `monthly_chart`, because official charts can contain
real ties, such as 2003-05 with duplicate #1 and intentionally skipped #2.

Usage:
  python3 import_official_t25.py --db t10.sqlite --csv t25_monthly_charts_extracted_final.csv
  python3 import_official_t25.py --db t10.sqlite --csv t25_monthly_charts_extracted_final.csv --dry-run

By default, months present in the CSV are replaced in `official_t25_entry`.
The script resolves imported show titles against `show` and `show_alias`, then
creates missing shows/aliases unless --no-create-missing-shows is used.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REQUIRED_COLUMNS = {
    "year",
    "month",
    "month_num",
    "position",
    "canonical_points",
    "show",
    "source_raw_points",
    "source_file",
    "source_sheet",
    "source_row",
    "source_col",
    "extraction_method",
}

KNOWN_OK_SHORT = {
    "1998-01": "Known-correct short early chart: 24 positions.",
    "2000-08": "Known-correct short early chart: 20 positions.",
}

KNOWN_OK_TIES = {
    "2003-05": "Intentional #1 tie; #2 skipped.",
}


@dataclass(frozen=True)
class ImportRow:
    csv_line: int
    year: int
    month: str
    month_num: int
    row_pos: int
    rank: int
    points: int
    raw_title: str
    source_raw_points: Optional[float]
    source_file: str
    source_sheet: str
    source_row: Optional[int]
    source_col: Optional[int]
    extraction_method: str


def normalize_title(value: str) -> str:
    """Normalize titles for fuzzy-ish deterministic matching."""
    s = (value or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # Normalize common punctuation and symbols before stripping.
    s = s.replace("&", " and ")
    s = s.replace("+", " plus ")
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_int(value: object, *, field: str, csv_line: int) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"CSV line {csv_line}: missing integer field {field}")
    try:
        return int(float(text))
    except Exception as exc:
        raise ValueError(f"CSV line {csv_line}: invalid integer for {field}: {value!r}") from exc


def parse_float_or_none(value: object) -> Optional[float]:
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def read_import_rows(csv_path: Path) -> list[ImportRow]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise SystemExit(f"CSV is missing required columns: {', '.join(missing)}")

        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        csv_line_by_id: dict[int, int] = {}
        for idx, rec in enumerate(reader, start=2):
            month = (rec.get("month") or "").strip()
            if not re.fullmatch(r"\d{4}-\d{2}", month):
                raise ValueError(f"CSV line {idx}: invalid month {month!r}; expected YYYY-MM")
            grouped[month].append(rec)
            csv_line_by_id[id(rec)] = idx

    rows: list[ImportRow] = []
    for month in sorted(grouped):
        month_rows = grouped[month]
        # Preserve CSV/extraction order inside rank/points sorting. This gives every row a
        # physical chart row even when displayed ranks contain ties.
        def sort_key(rec: dict[str, str]) -> tuple[int, int, int, int]:
            points = parse_int(rec.get("canonical_points"), field="canonical_points", csv_line=csv_line_by_id[id(rec)])
            rank = parse_int(rec.get("position"), field="position", csv_line=csv_line_by_id[id(rec)])
            source_row = parse_int(rec.get("source_row"), field="source_row", csv_line=csv_line_by_id[id(rec)])
            return (-points, rank, source_row, csv_line_by_id[id(rec)])

        month_rows = sorted(month_rows, key=sort_key)
        for row_pos, rec in enumerate(month_rows, start=1):
            csv_line = csv_line_by_id[id(rec)]
            raw_title = (rec.get("show") or "").strip()
            if not raw_title:
                raise ValueError(f"CSV line {csv_line}: missing show title")
            rows.append(
                ImportRow(
                    csv_line=csv_line,
                    year=parse_int(rec.get("year"), field="year", csv_line=csv_line),
                    month=month,
                    month_num=parse_int(rec.get("month_num"), field="month_num", csv_line=csv_line),
                    row_pos=row_pos,
                    rank=parse_int(rec.get("position"), field="position", csv_line=csv_line),
                    points=parse_int(rec.get("canonical_points"), field="canonical_points", csv_line=csv_line),
                    raw_title=raw_title,
                    source_raw_points=parse_float_or_none(rec.get("source_raw_points")),
                    source_file=(rec.get("source_file") or "").strip(),
                    source_sheet=(rec.get("source_sheet") or "").strip(),
                    source_row=parse_int(rec.get("source_row"), field="source_row", csv_line=csv_line),
                    source_col=parse_int(rec.get("source_col"), field="source_col", csv_line=csv_line),
                    extraction_method=(rec.get("extraction_method") or "").strip(),
                )
            )
    return rows


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS official_t25_entry (
          id INTEGER PRIMARY KEY,
          month TEXT NOT NULL,
          year INTEGER NOT NULL,
          month_num INTEGER NOT NULL,

          -- row_pos = physical row/order in the monthly chart.
          -- rank = displayed chart rank, allowing ties (e.g. 1, 1, 3...).
          row_pos INTEGER NOT NULL,
          rank INTEGER NOT NULL,

          points INTEGER NOT NULL,
          show_id INTEGER NOT NULL,
          raw_title TEXT NOT NULL,

          source_raw_points REAL,
          source_file TEXT,
          source_sheet TEXT,
          source_row INTEGER,
          source_col INTEGER,
          extraction_method TEXT,
          created_at TEXT NOT NULL DEFAULT (datetime('now')),

          FOREIGN KEY(show_id) REFERENCES show(show_id) ON DELETE CASCADE,
          UNIQUE(month, row_pos),
          UNIQUE(month, show_id)
        );

        CREATE INDEX IF NOT EXISTS idx_official_t25_entry_month_rank
          ON official_t25_entry(month, rank, row_pos);

        CREATE INDEX IF NOT EXISTS idx_official_t25_entry_show_month
          ON official_t25_entry(show_id, month);

        CREATE INDEX IF NOT EXISTS idx_official_t25_entry_year_points
          ON official_t25_entry(year, points);

        CREATE VIEW IF NOT EXISTS v_official_t25_year_end AS
        SELECT
          e.year,
          e.show_id,
          s.canonical_title,
          SUM(e.points) AS year_end_points,
          COUNT(*) AS months_on_chart,
          MIN(e.rank) AS peak_monthly_rank,
          MIN(e.month) AS first_month,
          MAX(e.month) AS last_month
        FROM official_t25_entry e
        JOIN show s ON s.show_id = e.show_id
        GROUP BY e.year, e.show_id, s.canonical_title;

        CREATE VIEW IF NOT EXISTS v_official_t25_show_summary AS
        SELECT
          e.show_id,
          s.canonical_title,
          COUNT(*) AS monthly_appearances,
          SUM(e.points) AS total_official_t25_points,
          MIN(e.rank) AS best_monthly_rank,
          MIN(e.month) AS first_month,
          MAX(e.month) AS last_month
        FROM official_t25_entry e
        JOIN show s ON s.show_id = e.show_id
        GROUP BY e.show_id, s.canonical_title;
        """
    )


def load_show_maps(con: sqlite3.Connection) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    exact_show: dict[str, int] = {}
    norm_show: dict[str, int] = {}
    exact_alias: dict[str, int] = {}
    norm_alias: dict[str, int] = {}

    for show_id, title in con.execute("SELECT show_id, canonical_title FROM show"):
        title = str(title or "").strip()
        if not title:
            continue
        exact_show[title.casefold()] = int(show_id)
        norm = normalize_title(title)
        if norm and norm not in norm_show:
            norm_show[norm] = int(show_id)

    for alias_title, show_id, alias_norm in con.execute("SELECT alias_title, show_id, alias_norm FROM show_alias"):
        title = str(alias_title or "").strip()
        if not title:
            continue
        exact_alias[title.casefold()] = int(show_id)
        norm = str(alias_norm or "").strip() or normalize_title(title)
        if norm and norm not in norm_alias:
            norm_alias[norm] = int(show_id)

    return exact_show, norm_show, exact_alias, norm_alias


def get_or_create_show(
    con: sqlite3.Connection,
    raw_title: str,
    *,
    create_missing: bool,
    maps: tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]],
    stats: Counter,
) -> int:
    exact_show, norm_show, exact_alias, norm_alias = maps
    key = raw_title.casefold()
    norm = normalize_title(raw_title)

    if key in exact_show:
        stats["matched_exact_show"] += 1
        return exact_show[key]
    if key in exact_alias:
        stats["matched_exact_alias"] += 1
        return exact_alias[key]
    if norm in norm_show:
        stats["matched_norm_show"] += 1
        return norm_show[norm]
    if norm in norm_alias:
        stats["matched_norm_alias"] += 1
        return norm_alias[norm]

    if not create_missing:
        raise KeyError(f"No show/alias match for {raw_title!r}")

    cur = con.execute("INSERT INTO show (canonical_title) VALUES (?)", (raw_title,))
    show_id = int(cur.lastrowid)
    exact_show[key] = show_id
    if norm:
        norm_show[norm] = show_id
    stats["created_show"] += 1
    return show_id


def ensure_alias(con: sqlite3.Connection, raw_title: str, show_id: int, stats: Counter) -> None:
    raw_title = raw_title.strip()
    if not raw_title:
        return

    row = con.execute("SELECT show_id FROM show_alias WHERE alias_title = ?", (raw_title,)).fetchone()
    if row:
        if int(row[0]) != int(show_id):
            stats["alias_conflict"] += 1
        return

    # Do not add an alias if the raw title is already exactly the canonical title.
    canonical = con.execute("SELECT canonical_title FROM show WHERE show_id = ?", (show_id,)).fetchone()
    if canonical and str(canonical[0]).strip() == raw_title:
        return

    alias_norm = normalize_title(raw_title)
    # Some existing DBs enforce UNIQUE(alias_norm). If a normalized alias already exists,
    # do not add a duplicate spelling variant; the match logic already used normalized
    # aliases before creating/importing this row.
    if alias_norm:
        norm_row = con.execute("SELECT show_id FROM show_alias WHERE alias_norm = ?", (alias_norm,)).fetchone()
        if norm_row:
            if int(norm_row[0]) != int(show_id):
                stats["alias_conflict"] += 1
            else:
                stats["alias_norm_already_exists"] += 1
            return

    con.execute(
        "INSERT INTO show_alias (alias_title, show_id, alias_norm) VALUES (?, ?, ?)",
        (raw_title, show_id, alias_norm),
    )
    stats["created_alias"] += 1


def delete_months(con: sqlite3.Connection, months: Iterable[str]) -> int:
    months = sorted(set(months))
    total = 0
    for month in months:
        cur = con.execute("DELETE FROM official_t25_entry WHERE month = ?", (month,))
        total += int(cur.rowcount or 0)
    return total


def validate_rows(rows: list[ImportRow]) -> list[dict[str, object]]:
    by_month: dict[str, list[ImportRow]] = defaultdict(list)
    for r in rows:
        by_month[r.month].append(r)

    report: list[dict[str, object]] = []
    for month in sorted(by_month):
        group = by_month[month]
        ranks = [r.rank for r in group]
        points = [r.points for r in group]
        duplicate_ranks = sorted(k for k, v in Counter(ranks).items() if v > 1)
        max_rank = max(ranks) if ranks else 0
        expected = set(range(1, max_rank + 1))
        missing_ranks = sorted(expected - set(ranks))
        status = "OK"
        notes: list[str] = []
        if len(group) != 25:
            notes.append(f"{len(group)} rows")
            if month not in KNOWN_OK_SHORT:
                status = "REVIEW"
        if duplicate_ranks or missing_ranks:
            notes.append(f"duplicate ranks {duplicate_ranks or '-'}; missing ranks {missing_ranks or '-'}")
            if month not in KNOWN_OK_TIES:
                status = "REVIEW"
        if month in KNOWN_OK_SHORT:
            notes.append(KNOWN_OK_SHORT[month])
        if month in KNOWN_OK_TIES:
            notes.append(KNOWN_OK_TIES[month])
        report.append(
            {
                "month": month,
                "rows": len(group),
                "unique_ranks": len(set(ranks)),
                "min_rank": min(ranks) if ranks else None,
                "max_rank": max_rank if ranks else None,
                "min_points": min(points) if points else None,
                "max_points": max(points) if points else None,
                "duplicate_ranks": ";".join(map(str, duplicate_ranks)),
                "missing_ranks": ";".join(map(str, missing_ranks)),
                "status": status,
                "notes": " | ".join(notes),
            }
        )
    return report


def write_validation_csv(path: Path, report: list[dict[str, object]]) -> None:
    if not report:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)


def run_import(args: argparse.Namespace) -> int:
    db_path = Path(args.db).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = read_import_rows(csv_path)
    if not rows:
        raise SystemExit("No import rows found.")

    months = sorted({r.month for r in rows})
    validation = validate_rows(rows)
    review_months = [r for r in validation if r["status"] != "OK"]

    if args.validation_csv:
        write_validation_csv(Path(args.validation_csv), validation)

    print(f"Read {len(rows):,} official T-25 rows from {csv_path.name}")
    print(f"Months: {len(months):,} ({months[0]} through {months[-1]})")
    print(f"Validation: {len(validation) - len(review_months):,} OK, {len(review_months):,} REVIEW")
    if review_months:
        print("Review months:")
        for r in review_months[:25]:
            print(f"  {r['month']}: {r['notes']}")
        if len(review_months) > 25:
            print(f"  ... {len(review_months) - 25} more")

    if args.dry_run:
        print("Dry run only; database was not changed.")
        return 0

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    stats: Counter = Counter()

    try:
        con.execute("BEGIN")
        ensure_schema(con)
        show_maps = load_show_maps(con)

        deleted = delete_months(con, months) if args.replace else 0
        stats["deleted_existing_rows"] = deleted

        insert_sql = """
            INSERT INTO official_t25_entry (
              month, year, month_num, row_pos, rank, points,
              show_id, raw_title, source_raw_points, source_file, source_sheet,
              source_row, source_col, extraction_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for r in rows:
            show_id = get_or_create_show(
                con,
                r.raw_title,
                create_missing=not args.no_create_missing_shows,
                maps=show_maps,
                stats=stats,
            )
            if not args.no_create_aliases:
                ensure_alias(con, r.raw_title, show_id, stats)
            con.execute(
                insert_sql,
                (
                    r.month,
                    r.year,
                    r.month_num,
                    r.row_pos,
                    r.rank,
                    r.points,
                    show_id,
                    r.raw_title,
                    r.source_raw_points,
                    r.source_file,
                    r.source_sheet,
                    r.source_row,
                    r.source_col,
                    r.extraction_method,
                ),
            )
            stats["inserted_rows"] += 1

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print("Import complete.")
    print(f"Deleted existing official rows for imported months: {stats['deleted_existing_rows']:,}")
    print(f"Inserted official T-25 rows: {stats['inserted_rows']:,}")
    print(
        "Show matching: "
        f"exact show {stats['matched_exact_show']:,}, "
        f"exact alias {stats['matched_exact_alias']:,}, "
        f"normalized show {stats['matched_norm_show']:,}, "
        f"normalized alias {stats['matched_norm_alias']:,}, "
        f"created shows {stats['created_show']:,}, "
        f"created aliases {stats['created_alias']:,}"
    )
    if stats["alias_conflict"]:
        print(f"Warning: alias conflicts encountered: {stats['alias_conflict']:,}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Import official Monthly T-25 charts into t10.sqlite")
    parser.add_argument("--db", required=True, help="Path to t10.sqlite")
    parser.add_argument("--csv", required=True, help="Path to t25_monthly_charts_extracted_final.csv")
    parser.add_argument("--validation-csv", help="Optional path to write month-level validation report")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write to SQLite")
    parser.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace official_t25_entry rows for months present in the CSV (default: true)",
    )
    parser.add_argument(
        "--no-create-missing-shows",
        action="store_true",
        help="Fail if an imported title cannot be matched to show/show_alias",
    )
    parser.add_argument(
        "--no-create-aliases",
        action="store_true",
        help="Do not add imported raw titles to show_alias",
    )
    return run_import(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
