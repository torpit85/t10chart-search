#!/usr/bin/env python3
from __future__ import annotations

import re
import sqlite3
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional, Iterable

import pandas as pd

VERSION = "2025-12-28 v4 (ties supported via pos; upsert on week_ending,pos; stable ordering)"

# -------------------------------------------------------------------
# Week-number anchor:
# 2019-06-08 is Week 1039, count in steps of 7 days from there.
# -------------------------------------------------------------------
BASE_DATE = date(2019, 6, 8)
BASE_WEEK = 1039


def weeknum_from_date(d: date) -> int:
    delta_days = (d - BASE_DATE).days
    if delta_days % 7 != 0:
        print(f"WARNING: {d.isoformat()} is {delta_days} days from base (not multiple of 7).")
    return BASE_WEEK + (delta_days // 7)


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _clean_text(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    return s.replace("\r", " ").replace("\n", " ").strip() or None


def _to_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None


def _to_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x).strip()))
        except Exception:
            return None


def split_imprint_pipe(s: Any) -> tuple[Optional[str], Optional[str]]:
    """
    Split 'Imprint|Network' into (imprint_1, imprint_2).
    If no pipe exists, imprint_2 is None.
    """
    t = _clean_text(s)
    if not t:
        return None, None
    if "|" in t:
        a, b = t.split("|", 1)
        return _clean_text(a), _clean_text(b)
    return t, None


def chunked(seq: list[Any], n: int) -> Iterable[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def parse_year_from_filename(name: str) -> Optional[int]:
    # "1st quarter 24.xls" -> 2024
    m = re.search(r"\b(\d{2})\b(?=\.(xlsx|xls)$)", name, re.I)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy <= 68 else 1900 + yy
    m = re.search(r"(\d{4})", name)
    return int(m.group(1)) if m else None


def parse_quarter_from_filename(name: str) -> Optional[int]:
    m = re.search(r"(\d)(st|nd|rd|th)\s+quarter", name, re.I)
    return int(m.group(1)) if m else None


def compute_sheet_date(year_base: int, quarter: int, sheetname: str) -> Optional[date]:
    m = re.match(r"(\d{2})-(\d{2})$", sheetname.strip())
    if not m:
        return None
    mm, dd = int(m.group(1)), int(m.group(2))
    yy = year_base

    # handle year rollover
    if quarter == 4 and mm in (1, 2, 3):
        yy = year_base + 1
    if quarter == 1 and mm in (10, 11, 12):
        yy = year_base - 1

    try:
        return date(yy, mm, dd)
    except ValueError:
        print(f"WARNING: Invalid sheet date '{sheetname}' for {year_base} Q{quarter}")
        return None


# -------------------------------------------------------------------
# LibreOffice conversion
# -------------------------------------------------------------------
def libreoffice_convert_xls_to_xlsx(xls_path: Path, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="lo_profile_"))
    cmd = [
        "soffice",
        "--headless",
        f"-env:UserInstallation=file://{profile}",
        "--convert-to",
        "xlsx",
        "--outdir",
        str(outdir),
        str(xls_path),
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("LibreOffice conversion failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Failed to convert {xls_path} to xlsx.")
    xlsx_path = outdir / (xls_path.stem + ".xlsx")
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Expected converted file not found: {xlsx_path}")
    return xlsx_path


# -------------------------------------------------------------------
# Parsers
#   Output columns (no 'pos' yet):
#     week_number, week_ending, rank, last_week, raw_title, imprint_1, imprint_2, gross_millions, _seq
#   We compute pos later globally per week_ending to guarantee uniqueness + keep ties.
# -------------------------------------------------------------------
def parse_weeknum_workbook(xlsx_path: Path, seq_base: int) -> tuple[pd.DataFrame, int]:
    """
    Week 1–1100 reformatted sheets with explicit columns:
    Week # | Week Ending | This Week | Last Week | Show | Imprint 1 | Imprint 2 | Total Weeks on Chart | Amount Grossed (in millions)
    """
    xl = pd.ExcelFile(xlsx_path)
    frames: list[pd.DataFrame] = []
    seq = seq_base

    for sh in xl.sheet_names:
        raw = pd.read_excel(xlsx_path, sheet_name=sh, header=None)

        header_row = None
        for i in range(min(150, len(raw))):
            row_vals = [str(x).strip().lower() for x in raw.iloc[i].tolist() if pd.notna(x)]
            if ("week #" in row_vals) and ("week ending" in row_vals) and ("this week" in row_vals) and ("show" in row_vals):
                header_row = i
                break
        if header_row is None:
            continue

        df = pd.read_excel(xlsx_path, sheet_name=sh, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        def pick_col(needles: list[str]) -> Optional[str]:
            cols_lower = {c.lower(): c for c in df.columns}
            for needle in needles:
                for k, orig in cols_lower.items():
                    if needle in k:
                        return orig
            return None

        c_weeknum = pick_col(["week #", "week#"])
        c_weekend = pick_col(["week ending"])
        c_rank = pick_col(["this week"])
        c_last = pick_col(["last week"])
        c_show = pick_col(["show"])
        c_imp1 = pick_col(["imprint 1", "imprint1"])
        c_imp2 = pick_col(["imprint 2", "imprint2"])
        c_gross = pick_col(["amount grossed", "grossed", "gross (in millions)", "gross"])

        if any(x is None for x in (c_weeknum, c_weekend, c_rank, c_show)):
            continue

        out = pd.DataFrame()
        out["week_number"] = pd.to_numeric(df[c_weeknum], errors="coerce")
        out["week_ending"] = pd.to_datetime(df[c_weekend], errors="coerce").dt.strftime("%Y-%m-%d")
        out["rank"] = pd.to_numeric(df[c_rank], errors="coerce")
        out["last_week"] = df[c_last].astype("string").str.strip() if c_last else None
        out["raw_title"] = df[c_show].astype("string").str.strip()
        out["imprint_1"] = df[c_imp1].astype("string").str.strip() if c_imp1 else None
        out["imprint_2"] = df[c_imp2].astype("string").str.strip() if c_imp2 else None
        out["gross_millions"] = pd.to_numeric(df[c_gross], errors="coerce") if c_gross else None

        out = out.dropna(subset=["week_number", "week_ending", "rank", "raw_title"])
        out = out[(out["rank"] >= 1) & (out["rank"] <= 50)]

        # stable input order marker (preserves ties within same rank)
        n = len(out)
        out["_seq"] = range(seq, seq + n)
        seq += n

        frames.append(out)

    if frames:
        return pd.concat(frames, ignore_index=True), seq
    return pd.DataFrame(), seq


def find_quarter_header_row(df: pd.DataFrame) -> Optional[int]:
    for i in range(min(150, len(df))):
        c1 = df.iloc[i, 1] if df.shape[1] > 1 else None
        c3 = df.iloc[i, 3] if df.shape[1] > 3 else None
        if isinstance(c3, str) and c3.strip().lower() == "show":
            if isinstance(c1, str) and "this" in c1.lower():
                return i
    return None


def parse_quarter_workbook(xlsx_path: Path, seq_base: int) -> tuple[pd.DataFrame, int]:
    """
    2019–2024 quarter workbooks:
    - each sheet is a week, sheet name is MM-DD
    - filename includes quarter+year
    """
    fname = xlsx_path.name
    year_base = parse_year_from_filename(fname)
    quarter = parse_quarter_from_filename(fname)
    if not year_base or not quarter:
        return pd.DataFrame(), seq_base

    xl = pd.ExcelFile(xlsx_path)
    rows: list[dict[str, Any]] = []
    seq = seq_base

    for sh in xl.sheet_names:
        d = compute_sheet_date(year_base, quarter, sh)
        if not d:
            continue

        df = pd.read_excel(xlsx_path, sheet_name=sh, header=None)
        hdr = find_quarter_header_row(df)
        if hdr is None:
            continue

        block = df.iloc[hdr + 1 : hdr + 1 + 250].copy()

        for _, r in block.iterrows():
            rank_int = _to_int(r.iloc[1] if len(r) > 1 else None)
            if rank_int is None:
                continue

            raw_title = _clean_text(r.iloc[3] if len(r) > 3 else None)
            if not raw_title:
                continue

            last_week = _clean_text(r.iloc[2] if len(r) > 2 else None)

            imprint_cell = r.iloc[4] if len(r) > 4 else None
            gross_cell = r.iloc[5] if len(r) > 5 else None

            imprint_as_float = _to_float(imprint_cell)
            gross_as_float = _to_float(gross_cell)

            # swapped-column detection (like your 2019-06-08)
            if (imprint_as_float is not None) and (gross_as_float is None):
                gross_m = imprint_as_float
                im1, im2 = split_imprint_pipe(gross_cell)
            else:
                im1, im2 = split_imprint_pipe(imprint_cell)
                gross_m = gross_as_float

            rows.append(
                dict(
                    week_ending=d.isoformat(),
                    week_number=weeknum_from_date(d),
                    rank=rank_int,
                    last_week=last_week,
                    raw_title=raw_title,
                    imprint_1=im1,
                    imprint_2=im2,
                    gross_millions=gross_m,
                    _seq=seq,
                )
            )
            seq += 1

    return (pd.DataFrame(rows) if rows else pd.DataFrame()), seq


def parse_2025_workbook(xlsx_path: Path, seq_base: int) -> tuple[pd.DataFrame, int]:
    """
    2025-style workbook: one long sheet with a header row containing 'Date'
    in column index 1, then rows include:
      Date, Rank, Show, Imprint, Gross
    """
    df = pd.read_excel(xlsx_path, sheet_name=0, header=None)

    hdr = None
    for i in range(min(250, len(df))):
        v = df.iloc[i, 1] if df.shape[1] > 1 else None
        if isinstance(v, str) and v.strip().lower() == "date":
            hdr = i
            break
    if hdr is None:
        return pd.DataFrame(), seq_base

    rows: list[dict[str, Any]] = []
    seq = seq_base

    for _, r in df.iloc[hdr + 1 :].iterrows():
        dval = r.iloc[1] if len(r) > 1 else None
        if dval is None or (isinstance(dval, float) and pd.isna(dval)):
            continue

        dd_parsed = pd.to_datetime(dval, errors="coerce")
        if pd.isna(dd_parsed):
            continue
        dd = dd_parsed.date()

        rank_int = _to_int(r.iloc[2] if len(r) > 2 else None)
        if rank_int is None:
            continue

        title = _clean_text(r.iloc[3] if len(r) > 3 else None)
        if not title:
            continue

        im1, im2 = split_imprint_pipe(r.iloc[4] if len(r) > 4 else None)
        gross_m = _to_float(r.iloc[5] if len(r) > 5 else None)

        rows.append(
            dict(
                week_ending=dd.isoformat(),
                week_number=weeknum_from_date(dd),
                rank=rank_int,
                last_week=None,
                raw_title=title,
                imprint_1=im1,
                imprint_2=im2,
                gross_millions=gross_m,
                _seq=seq,
            )
        )
        seq += 1

    return (pd.DataFrame(rows) if rows else pd.DataFrame()), seq


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------
def get_con(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def ensure_pos_schema(con: sqlite3.Connection) -> None:
    """
    Safety check so the importer fails fast with a clear message if the DB wasn't migrated.
    """
    cols = [r[1] for r in con.execute("PRAGMA table_info(t10_entry);").fetchall()]
    if "pos" not in cols:
        raise RuntimeError(
            "Database schema is missing t10_entry.pos. Run the ties migration first "
            "(add pos + UNIQUE(week_ending,pos) + rebuild FTS)."
        )

    # Best-effort check that conflict target exists
    idxs = con.execute("PRAGMA index_list(t10_entry);").fetchall()
    # index_list: (seq, name, unique, origin, partial) in newer sqlite
    unique_indexes = [r[1] for r in idxs if int(r[2]) == 1]
    if not unique_indexes:
        print("WARNING: Could not verify UNIQUE index for (week_ending,pos). Continuing anyway.")


def get_or_create_show_id(con: sqlite3.Connection, raw_title: str) -> int:
    cur = con.cursor()
    row = cur.execute("SELECT show_id FROM show_alias WHERE alias_title = ?", (raw_title,)).fetchone()
    if row:
        return int(row[0])

    row = cur.execute("SELECT show_id FROM show WHERE canonical_title = ?", (raw_title,)).fetchone()
    if row:
        return int(row[0])

    cur.execute("INSERT INTO show(canonical_title) VALUES (?)", (raw_title,))
    return int(cur.lastrowid)


def upsert_entry(con: sqlite3.Connection, rec: dict[str, Any]) -> None:
    """
    UPSERT keyed to UNIQUE(week_ending, pos) to support ties.
    """
    raw_title = _clean_text(rec.get("raw_title")) or ""
    show_id = get_or_create_show_id(con, raw_title)

    gross = rec.get("gross_millions")
    gross_val = None if gross is None or (isinstance(gross, float) and pd.isna(gross)) else float(gross)

    params = (
        rec.get("week_ending"),
        int(rec.get("week_number")),
        int(rec.get("rank")),
        int(rec.get("pos")),
        _clean_text(rec.get("last_week")),
        show_id,
        raw_title,
        _clean_text(rec.get("imprint_1")),
        _clean_text(rec.get("imprint_2")),
        gross_val,
    )

    con.execute(
        """
        INSERT INTO t10_entry(
            week_ending, week_number, rank, pos, last_week,
            show_id, raw_title, imprint_1, imprint_2, gross_millions
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(week_ending, pos) DO UPDATE SET
            week_number    = excluded.week_number,
            rank           = excluded.rank,
            last_week      = excluded.last_week,
            show_id        = excluded.show_id,
            raw_title      = excluded.raw_title,
            imprint_1      = excluded.imprint_1,
            imprint_2      = excluded.imprint_2,
            gross_millions = excluded.gross_millions
        """,
        params,
    )


@dataclass
class ImportSummary:
    parsed_rows: int
    unique_weeks: int
    max_rank: int
    max_pos: int


def _collect_spreadsheets_from_zip(zip_path: Path, workdir: Path) -> list[Path]:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(workdir)

    xlsx_out = workdir / "xlsx"
    xlsx_out.mkdir(exist_ok=True)

    to_parse: list[Path] = []

    for p in workdir.rglob("*.xls"):
        to_parse.append(libreoffice_convert_xls_to_xlsx(p, xlsx_out))

    for p in workdir.rglob("*.xlsx"):
        if p.parent == xlsx_out:
            continue
        to_parse.append(p)

    seen = set()
    unique: list[Path] = []
    for p in to_parse:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def import_zip_file_to_db(zip_path: Path, db_path: Path) -> ImportSummary:
    print(f"[importer] {VERSION}")
    print(f"ZIP: {zip_path}")
    print(f"DB : {db_path}")
    print("IMPORTANT: Backup your DB before importing.\n")

    with tempfile.TemporaryDirectory(prefix="t10_zip_") as td_str:
        td = Path(td_str)
        files = _collect_spreadsheets_from_zip(zip_path, td)

        print(f"Found {len(files)} spreadsheet(s) to parse.")
        frames: list[pd.DataFrame] = []
        seq = 1

        for p in files:
            name = p.name.lower()

            if "quarter" in name:
                dfp, seq = parse_quarter_workbook(p, seq)
                if not dfp.empty:
                    print(f"  parsed quarter: {p.name} -> {len(dfp)} rows")
                    frames.append(dfp)
                continue

            # Always try weeknum first (most reliable)
            dfw, seq = parse_weeknum_workbook(p, seq)
            if dfw is not None and not dfw.empty:
                print(f"  parsed weeknum: {p.name} -> {len(dfw)} rows")
                frames.append(dfw)
                continue

            dfp, seq = parse_2025_workbook(p, seq)
            if not dfp.empty:
                print(f"  parsed 2025   : {p.name} -> {len(dfp)} rows")
                frames.append(dfp)

        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            raise RuntimeError("No rows parsed from any workbook. Check formats / headers.")

        df = pd.concat(frames, ignore_index=True)

        # Normalize required columns
        df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["week_number", "rank", "week_ending", "raw_title"])

        df["week_number"] = df["week_number"].astype(int)
        df["rank"] = df["rank"].astype(int)
        df["week_ending"] = df["week_ending"].astype(str).str.strip()
        df["raw_title"] = df["raw_title"].astype(str).str.strip()
        df["imprint_1"] = df.get("imprint_1")
        df["imprint_2"] = df.get("imprint_2")

        # Remove exact duplicate rows (but DO NOT collapse ties)
        df = df.drop_duplicates(
            subset=["week_ending", "rank", "raw_title", "imprint_1", "imprint_2", "gross_millions"],
            keep="first",
        )

        # Assign pos per week_ending using stable ordering:
        # rank ascending, then input order (_seq) to preserve ties within rank.
        df = df.sort_values(["week_ending", "rank", "_seq"], ascending=[True, True, True])
        df["pos"] = df.groupby("week_ending").cumcount() + 1

        summary = ImportSummary(
            parsed_rows=int(len(df)),
            unique_weeks=int(df["week_ending"].nunique()),
            max_rank=int(df["rank"].max()),
            max_pos=int(df["pos"].max()),
        )

        print("\nParsed rows :", summary.parsed_rows)
        print("Unique weeks:", summary.unique_weeks)
        print("Max rank    :", summary.max_rank)
        print("Max pos     :", summary.max_pos)

        week_endings = sorted(set(df["week_ending"].tolist()))

        con = get_con(db_path)
        try:
            ensure_pos_schema(con)
            con.execute("BEGIN;")

            # Delete existing rows for these weeks
            deleted = 0
            for chunk in chunked(week_endings, 900):
                placeholders = ",".join("?" for _ in chunk)
                con.execute(f"DELETE FROM t10_entry WHERE week_ending IN ({placeholders})", chunk)
                deleted += len(chunk)
            print(f"Deleted existing rows for {deleted} week_ending value(s).")

            # Upsert rows
            for rec in df.to_dict(orient="records"):
                upsert_entry(con, rec)

            con.commit()
            print("✅ Import complete.")
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

        return summary


def main(zip_path_str: str, db_path_str: str) -> None:
    zip_path = Path(zip_path_str).expanduser().resolve()
    db_path = Path(db_path_str).expanduser().resolve()

    summary = import_zip_file_to_db(zip_path, db_path)

    print("\nSummary:")
    print(f"  Parsed rows : {summary.parsed_rows}")
    print(f"  Unique weeks: {summary.unique_weeks}")
    print(f"  Max rank    : {summary.max_rank}")
    print(f"  Max pos     : {summary.max_pos}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 import_zip_to_sqlite.py <zipfile.zip> <t10.sqlite>")
        raise SystemExit(2)

    main(sys.argv[1], sys.argv[2])

