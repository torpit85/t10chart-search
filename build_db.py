#!/usr/bin/env python3
"""
Builds/refreshes the SQLite database from the provided ZIP of Excel files.
- Drops 'Total Weeks on Chart' (derived instead)
- Normalizes show titles into show/show_alias for merge-friendly management.
"""
from __future__ import annotations
import argparse, sqlite3, zipfile, re
from pathlib import Path
import pandas as pd

COL_MAP = {
    "Week #": "week_number",
    "Week Ending": "week_ending",
    "This Week": "rank",
    "Last Week": "last_week",
    "Show": "raw_title",
    "Imprint 1": "imprint_1",
    "Imprint 2": "imprint_2",
    "Amount Grossed (in millions)": "gross_millions",
}

def norm_title(t: str) -> str:
    # gentle normalization (keeps punctuation); used only for default canonical title
    if t is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(t):
            return ""
    except Exception:
        pass
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    return t

def ensure_show(cur: sqlite3.Cursor, raw_title: str) -> int:
    raw_title = norm_title(raw_title)
    # If alias already exists, return mapped show_id
    row = cur.execute("SELECT show_id FROM show_alias WHERE alias_title = ?", (raw_title,)).fetchone()
    if row:
        return int(row[0])
    # Otherwise create canonical show (default canonical == raw) if not exists
    cur.execute("INSERT OR IGNORE INTO show(canonical_title) VALUES (?)", (raw_title,))
    show_id = cur.execute("SELECT show_id FROM show WHERE canonical_title = ?", (raw_title,)).fetchone()[0]
    # Create alias
    cur.execute("INSERT OR IGNORE INTO show_alias(alias_title, show_id) VALUES (?, ?)", (raw_title, show_id))
    return int(show_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to 'T-10 Chart ... .zip'")
    ap.add_argument("--db", default="t10.sqlite", help="Output SQLite path")
    ap.add_argument("--schema", default="schema.sql", help="Schema SQL path")
    ap.add_argument("--rebuild", action="store_true", help="Delete existing db and rebuild")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    db_path = Path(args.db)
    schema_path = Path(args.schema)

    if args.rebuild and db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")
    cur = con.cursor()

    cur.executescript(schema_path.read_text(encoding="utf-8"))

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not names:
            raise SystemExit("No Excel files found in ZIP.")

        total_rows = 0
        for name in sorted(names):
            with z.open(name) as f:
                df = pd.read_excel(f)

            # Rename and select expected columns (ignore Total Weeks on Chart entirely)
            df = df.rename(columns=COL_MAP)
            missing = [c for c in ["week_number","week_ending","rank","raw_title"] if c not in df.columns]
            if missing:
                raise SystemExit(f"{name}: missing required columns: {missing}")

            # Coerce
            df["week_number"] = df["week_number"].astype(int)
            df["rank"] = df["rank"].astype(int)
            df["week_ending"] = pd.to_datetime(df["week_ending"]).dt.strftime("%Y-%m-%d")
            if "gross_millions" in df.columns:
                df["gross_millions"] = pd.to_numeric(df["gross_millions"], errors="coerce")
            else:
                df["gross_millions"] = None

            for col in ["last_week","imprint_1","imprint_2","raw_title"]:
                if col in df.columns:
                    df[col] = df[col].astype("string").fillna(pd.NA)

            # Insert row-by-row to maintain show normalization
            for r in df.itertuples(index=False):
                raw_title = norm_title(getattr(r, "raw_title"))
                show_id = ensure_show(cur, raw_title)

                last_week = getattr(r, "last_week", None)
                last_week = None if last_week is None or pd.isna(last_week) else str(last_week).strip()

                imprint_1 = getattr(r, "imprint_1", None)
                imprint_1 = None if imprint_1 is None or pd.isna(imprint_1) or str(imprint_1).strip()=="" else str(imprint_1).strip()

                imprint_2 = getattr(r, "imprint_2", None)
                imprint_2 = None if imprint_2 is None or pd.isna(imprint_2) or str(imprint_2).strip()=="" else str(imprint_2).strip()

                gross = getattr(r, "gross_millions", None)
                gross = None if gross is None or (isinstance(gross, float) and pd.isna(gross)) else float(gross)

                week_number = int(getattr(r, "week_number"))
                week_ending = str(getattr(r, "week_ending"))
                rank = int(getattr(r, "rank"))

                cur.execute("""
                    INSERT INTO t10_entry(
                        week_number, week_ending, rank, last_week,
                        show_id, raw_title, imprint_1, imprint_2, gross_millions, source_file
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(week_ending, rank) DO UPDATE SET
                        week_number=excluded.week_number,
                        last_week=excluded.last_week,
                        show_id=excluded.show_id,
                        raw_title=excluded.raw_title,
                        imprint_1=excluded.imprint_1,
                        imprint_2=excluded.imprint_2,
                        gross_millions=excluded.gross_millions,
                        source_file=excluded.source_file
                """, (week_number, week_ending, rank, last_week, show_id, raw_title, imprint_1, imprint_2, gross, name))

                total_rows += 1

            con.commit()
            print(f"Loaded {name}: {len(df)} rows")

    # Quick sanity counts
    shows = cur.execute("SELECT COUNT(*) FROM show").fetchone()[0]
    entries = cur.execute("SELECT COUNT(*) FROM t10_entry").fetchone()[0]
    con.commit()
    con.close()

    print(f"Done. Entries: {entries}, Shows: {shows} -> {db_path}")

if __name__ == "__main__":
    main()
