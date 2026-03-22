#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3
import math
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Callable

from datetime import date, timedelta

import numpy as np
import pandas as pd
import altair as alt

import streamlit as st
import matplotlib.pyplot as plt

from charts import chart_top_gross_weeks


# ----------------------------
# Configuration
# ----------------------------
APP_TITLE = "T-10 Chart Search Engine"
DB_PATH = Path(__file__).with_name("t10.sqlite")

# Gross tracking starts the week ending March 17, 2001
GROSS_TRACKING_START = date(2001, 3, 17)

# For streaks when week_number is missing/spotty
CONSECUTIVE_DAY_TOLERANCE = (6, 8)  # inclusive


# ----------------------------
# Utilities
# ----------------------------
def _as_date_str(series: pd.Series) -> pd.Series:
    """Return a YYYY-MM-DD string series without times."""
    s = series.copy()
    dt = pd.to_datetime(s, errors="coerce")
    out = dt.dt.strftime("%Y-%m-%d")
    fallback = s.astype("string").fillna(pd.NA).str.strip()
    return out.fillna(fallback)


# FTS5 sanitization: user input -> safe MATCH query (prevents errors on punctuation like '!').
_FTS_TOKEN_RE = re.compile(r"[0-9A-Za-z]+(?:'[0-9A-Za-z]+)?")

def fts5_safe_query(raw: str | None) -> str:
    """Convert arbitrary user text into a safe SQLite FTS5 query.

    - Extracts alphanumeric tokens (keeps apostrophes inside words)
    - Quotes each token to avoid operator interpretation
    - Joins tokens with AND so all terms are required
    - Returns '' if no usable tokens
    """
    s = (raw or "").strip()
    if not s:
        return ""
    tokens = _FTS_TOKEN_RE.findall(s)
    if not tokens:
        return ""
    return " AND ".join([f'"{t}"' for t in tokens])


def get_con() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}.")
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON;")
    return con

@st.cache_data(show_spinner=False)
def sql_df(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    con = get_con()
    try:
        df = pd.read_sql(sql, con, params=params)
    finally:
        con.close()
    return df

def sql_exec(sql: str, params: tuple[Any, ...] = ()) -> None:
    con = get_con()
    try:
        cur = con.cursor()
        cur.execute("BEGIN;")
        cur.execute(sql, params)
        con.commit()
    finally:
        con.close()

def sql_execmany(sql: str, rows: Iterable[tuple[Any, ...]]) -> None:
    con = get_con()
    try:
        cur = con.cursor()
        cur.execute("BEGIN;")
        cur.executemany(sql, rows)
        con.commit()
    finally:
        con.close()


# ----------------------------
# SMPS (Share–Momentum Point System) helpers
# ----------------------------
SMPS_METHOD_VERSION = "SMPS_v1"
SMPS_START_MONTH = "2001-04"  # first monthly chart in the grossing era
SMPS_OFFICIAL_DUAL_END_MONTH = "2025-01"  # Apr 2001 → Jan 2025: Official + SMPS


def _ym_from_year_month(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _ym_add(ym: str, delta_months: int) -> str:
    """Add delta_months to a YYYY-MM string."""
    y, m = ym.split("-")
    yi = int(y)
    mi = int(m)
    idx = yi * 12 + (mi - 1) + int(delta_months)
    ny = idx // 12
    nm = (idx % 12) + 1
    return _ym_from_year_month(ny, nm)


def _percentile_inc(values: Iterable[float], p: float, *, ignore_nonpositive: bool = True) -> Optional[float]:
    """Excel-like PERCENTILE.INC with linear interpolation (no numpy needed)."""
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if ignore_nonpositive and fv <= 0:
            continue
        vals.append(fv)
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    k = float(p) / 100.0
    r = 1.0 + (n - 1) * k  # 1-based
    lo = int(np.floor(r))
    hi = int(np.ceil(r))
    frac = r - lo
    lo_i = lo - 1
    hi_i = hi - 1
    if hi_i == lo_i:
        return vals[lo_i]
    return vals[lo_i] + (vals[hi_i] - vals[lo_i]) * frac


def _p10_inc(values: Iterable[float], *, ignore_nonpositive: bool = True) -> float:
    v = _percentile_inc(values, 10.0, ignore_nonpositive=ignore_nonpositive)
    return max(float(v) if v is not None else 1.0, 1.0)


def _chart_month_series(week_ending_dt: pd.Series) -> pd.Series:
    """Vectorized chart-month mapping for weekly week_ending dates (cutoff day 28).

    Rule:
      - if day <= 28: chart_month = next month
      - else: chart_month = month after next

    Returns YYYY-MM strings.
    """
    dt = pd.to_datetime(week_ending_dt, errors="coerce")
    y = dt.dt.year.astype("Int64")
    m = dt.dt.month.astype("Int64")
    d = dt.dt.day.astype("Int64")

    add = np.where(d <= 28, 1, 2)
    idx = (y.astype("int64") * 12) + (m.astype("int64") - 1) + add
    ny = (idx // 12).astype(int)
    nm = (idx % 12 + 1).astype(int)

    out = pd.Series([_ym_from_year_month(a, b) for a, b in zip(ny, nm)], index=week_ending_dt.index)
    return out


def _apply_chart_month_logic(
    df: pd.DataFrame,
    *,
    week_dt_col: str = "week_ending_dt",
    week_str_col: str = "week_ending",
) -> pd.DataFrame:
    """Apply the shared chart-month logic used by monthly grossing views."""
    if df is None or df.empty:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if isinstance(out, pd.DataFrame):
            for col in ("month", "month_ord"):
                if col not in out.columns:
                    out[col] = pd.Series(dtype="object")
        return out

    out = df.copy()
    if week_dt_col not in out.columns:
        if week_str_col not in out.columns:
            raise KeyError(f"Expected '{week_dt_col}' or '{week_str_col}' in dataframe.")
        out[week_dt_col] = pd.to_datetime(out[week_str_col], errors="coerce")

    out = out.dropna(subset=[week_dt_col]).copy()
    out[week_str_col] = _as_date_str(out[week_str_col]) if week_str_col in out.columns else _as_date_str(out[week_dt_col])
    out["month"] = _chart_month_series(out[week_dt_col])

    # Special rule: April 2001 uses only Mar 17 + Mar 24, 2001 weeks.
    out = out[
        ~(
            (out["month"] == "2001-04")
            & (~out[week_str_col].isin(["2001-03-17", "2001-03-24"]))
        )
    ].copy()

    y = out["month"].str.slice(0, 4).astype(int)
    m = out["month"].str.slice(5, 7).astype(int)
    out["month_ord"] = y * 12 + (m - 1)
    return out



def _ensure_smps_schema() -> None:
    """Create SMPS tables if missing."""
    con = get_con()
    try:
        cur = con.cursor()
        cur.execute("BEGIN;")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS monthly_chart (
              month TEXT NOT NULL,
              chart_type TEXT NOT NULL,
              method_version TEXT,
              position INTEGER NOT NULL,
              show_id INTEGER NOT NULL,
              month_gross_millions REAL,
              points_total REAL,
              points_share REAL,
              points_breakout REAL,
              points_heat REAL,
              points_carryover REAL,
              inactive_streak INTEGER,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              PRIMARY KEY (month, chart_type, method_version, position),
              UNIQUE (month, chart_type, method_version, show_id)
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS show_month (
              month TEXT NOT NULL,
              show_id INTEGER NOT NULL,
              method_version TEXT NOT NULL DEFAULT 'SMPS_v1',
              month_gross_millions REAL NOT NULL,
              weeks_in_month INTEGER NOT NULL,
              first2_avg_millions REAL,
              last2_avg_millions REAL,
              prev_month_gross_millions REAL,
              inactive_streak INTEGER,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              PRIMARY KEY (month, show_id, method_version)
            );
            """
        )

        con.commit()
    finally:
        con.close()

@st.cache_data(show_spinner=False)
def load_lists() -> tuple[pd.DataFrame, pd.DataFrame]:
    shows = sql_df("SELECT show_id, canonical_title FROM show ORDER BY canonical_title")
    companies = sql_df("""
        SELECT DISTINCT COALESCE(imprint_1,'(Unknown)') AS company
        FROM t10_entry
        ORDER BY company
    """)
    return shows, companies


# ----------------------------
# Data fetchers
# ----------------------------
@dataclass(frozen=True)
class FilterSpec:
    date_min: str | None
    date_max: str | None
    rank_min: int
    rank_max: int

def build_where(filters: FilterSpec, table_alias: str = "e") -> tuple[str, list[Any]]:
    where = [f"{table_alias}.rank BETWEEN ? AND ?"]
    params: list[Any] = [filters.rank_min, filters.rank_max]
    if filters.date_min:
        where.append(f"{table_alias}.week_ending >= ?")
        params.append(filters.date_min)
    if filters.date_max:
        where.append(f"{table_alias}.week_ending <= ?")
        params.append(filters.date_max)
    return " AND ".join(where), params

def fetch_entries(filters: FilterSpec, fts_query: str | None = None, limit: int = 1000, week_min: int | None = None, week_max: int | None = None) -> pd.DataFrame:
    where, params = build_where(filters, "e")
    params2 = list(params)

    # Week number filter (optional)
    if week_min is not None and week_max is not None:
        where = f"{where} AND e.week_number BETWEEN ? AND ?"
        params2.extend([int(week_min), int(week_max)])
    elif week_min is not None:
        where = f"{where} AND e.week_number = ?"
        params2.append(int(week_min))
    safe_fts = fts5_safe_query(fts_query)

    if safe_fts:
        sql = f"""
        SELECT
          e.show_id,
          e.week_ending,
          e.week_number,
          e.rank,
          e.pos,
          e.last_week,
          s.canonical_title,
          e.raw_title,
          e.imprint_1,
          e.imprint_2,
          e.gross_millions AS base_gross_millions,
          COALESCE(gb.bonus_millions, 0) AS bonus_millions,
          (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions
        FROM t10_fts f
        JOIN t10_entry e ON e.id = f.rowid
        LEFT JOIN (
          SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
          FROM gross_bonus
          GROUP BY show_id, week_ending
        ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
        JOIN show s ON s.show_id = e.show_id
        WHERE t10_fts MATCH ?
          AND {where}
        ORDER BY e.week_ending DESC, e.rank ASC, e.pos ASC
        LIMIT ?
        """
        params2 = [safe_fts] + params2 + [int(limit)]
    else:
        sql = f"""
        SELECT
          e.week_ending,
          e.week_number,
          e.rank,
          e.pos,
          e.last_week,
          s.canonical_title,
          e.raw_title,
          e.imprint_1,
          e.imprint_2,
          e.gross_millions AS base_gross_millions,
          COALESCE(gb.bonus_millions, 0) AS bonus_millions,
          (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions
        FROM t10_entry e
        LEFT JOIN (
          SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
          FROM gross_bonus
          GROUP BY show_id, week_ending
        ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
        JOIN show s ON s.show_id = e.show_id
        WHERE {where}
        ORDER BY e.week_ending DESC, e.rank ASC, e.pos ASC
        LIMIT ?
        """
        params2 = params2 + [int(limit)]

    df = sql_df(sql, tuple(params2))
    if not df.empty:
        df["week_ending"] = _as_date_str(df["week_ending"])
    return df

def fetch_show_entries(show_id: int, filters: FilterSpec) -> pd.DataFrame:
    where, params = build_where(filters, "e")
    sql = f"""
    SELECT
      e.week_ending,
      e.week_number,
      e.rank,
      e.pos,
      e.last_week,
      e.raw_title,
      e.imprint_1,
      e.imprint_2,
      e.gross_millions AS base_gross_millions,
      COALESCE(gb.bonus_millions, 0) AS bonus_millions,
      (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions
    FROM t10_entry e
    LEFT JOIN (
      SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
      FROM gross_bonus
      GROUP BY show_id, week_ending
    ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
    WHERE e.show_id = ?
      AND {where}
    ORDER BY e.week_number ASC, e.rank ASC, e.pos ASC
    """
    df = sql_df(sql, tuple([show_id] + params))
    if not df.empty:
        df["week_ending"] = _as_date_str(df["week_ending"])
    return df

def fetch_show_stats(show_id: int) -> pd.DataFrame:
    """Show-level summary stats.

    Notes:
      - weeks_on_chart / peak_rank / first/last appearance are based on actual chart rows (t10_entry).
      - total_gross_millions includes *all* gross bonuses from gross_bonus, even if a bonus lands on a week
        where the show is not on the chart (no t10_entry row for that week).
      - avg_gross_millions is computed as total_gross_millions / weeks_on_chart (when weeks_on_chart > 0).
    """
    return sql_df(
        """
        WITH
          chart AS (
            SELECT date(week_ending) AS we, rank
            FROM t10_entry
            WHERE show_id = ?
          ),
          base AS (
            SELECT COALESCE(SUM(COALESCE(gross_millions, 0.0)), 0.0) AS base_gross
            FROM t10_entry
            WHERE show_id = ?
          ),
          bon AS (
            SELECT COALESCE(SUM(COALESCE(bonus_millions, 0.0)), 0.0) AS bonus_gross
            FROM gross_bonus
            WHERE show_id = ?
          )
        SELECT
          (SELECT COUNT(DISTINCT we) FROM chart) AS weeks_on_chart,
          (SELECT MIN(rank) FROM chart) AS peak_rank,
          (SELECT MIN(we) FROM chart) AS first_appearance,
          (SELECT MAX(we) FROM chart) AS last_appearance,
          ((SELECT base_gross FROM base) + (SELECT bonus_gross FROM bon)) AS total_gross_millions,
          CASE
            WHEN (SELECT COUNT(DISTINCT we) FROM chart) > 0
            THEN ((SELECT base_gross FROM base) + (SELECT bonus_gross FROM bon)) * 1.0
                 / (SELECT COUNT(DISTINCT we) FROM chart)
            ELSE NULL
          END AS avg_gross_millions,
          (SELECT AVG(rank) FROM t10_entry WHERE show_id = ?) AS avg_rank
        """,
        (show_id, show_id, show_id, show_id),
    )


def fetch_show_weekly_ledger(show_id: int) -> pd.DataFrame:
    """Weekly ledger for a show that includes bonus-only weeks.

    This is a *time series* view (not just chart appearances): it unions t10_entry gross rows
    with gross_bonus rows and collapses to one row per week_ending.
    """
    return sql_df(
        """
        WITH combined AS (
          SELECT
            date(week_ending) AS week_ending,
            COALESCE(gross_millions, 0.0) AS base_gross_millions,
            0.0 AS bonus_millions
          FROM t10_entry
          WHERE show_id = ?

          UNION ALL

          SELECT
            date(week_ending) AS week_ending,
            0.0 AS base_gross_millions,
            COALESCE(bonus_millions, 0.0) AS bonus_millions
          FROM gross_bonus
          WHERE show_id = ?
        )
        SELECT
          week_ending,
          SUM(base_gross_millions) AS base_gross_millions,
          SUM(bonus_millions) AS bonus_millions,
          SUM(base_gross_millions + bonus_millions) AS gross_millions
        FROM combined
        GROUP BY week_ending
        ORDER BY date(week_ending) ASC;
        """,
        (show_id, show_id),
    )



def fetch_company_entries(company: str, filters: FilterSpec, imprint_col: str = "imprint_1", limit: int = 2000) -> pd.DataFrame:
    if imprint_col not in ("imprint_1", "imprint_2"):
        raise ValueError("imprint_col must be 'imprint_1' or 'imprint_2'")
    where, params = build_where(filters, "e")
    sql = f"""
    SELECT
      e.week_ending,
      e.week_number,
      e.rank,
      e.pos,
      s.canonical_title,
      e.raw_title,
      e.imprint_1,
      e.imprint_2,
      e.gross_millions AS base_gross_millions,
      COALESCE(gb.bonus_millions, 0) AS bonus_millions,
      (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions
    FROM t10_entry e
    LEFT JOIN (
      SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
      FROM gross_bonus
      GROUP BY show_id, week_ending
    ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
    JOIN show s ON s.show_id = e.show_id
    WHERE COALESCE(e.{imprint_col},'(Unknown)') = ?
      AND {where}
    ORDER BY e.week_ending DESC, e.rank ASC, e.pos ASC
    LIMIT ?
    """
    df = sql_df(sql, tuple([company] + params + [int(limit)]))
    if not df.empty:
        df["week_ending"] = _as_date_str(df["week_ending"])
    return df

@st.cache_data(show_spinner=False)
def fetch_company_list(imprint_col: str) -> list[str]:
    """Distinct company names from the chosen imprint column (with '(Unknown)' for blanks)."""
    if imprint_col not in ("imprint_1", "imprint_2"):
        raise ValueError("imprint_col must be 'imprint_1' or 'imprint_2'")
    df = sql_df(
        f"""
        SELECT DISTINCT COALESCE(NULLIF(TRIM({imprint_col}), ''), '(Unknown)') AS company
        FROM t10_entry
        WHERE week_ending IS NOT NULL
        ORDER BY company
        """
    )
    return df["company"].tolist() if (df is not None and not df.empty and "company" in df.columns) else ["(Unknown)"]


def fetch_company_stats(company: str, filters: FilterSpec, imprint_col: str = "imprint_1") -> pd.DataFrame:
    """Company summary stats (entries/unique shows/total+avg gross), computed on the fly."""
    if imprint_col not in ("imprint_1", "imprint_2"):
        raise ValueError("imprint_col must be 'imprint_1' or 'imprint_2'")

    where, params = build_where(filters, "e")
    sql = f"""
    WITH bonus_by_row AS (
      SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
      FROM gross_bonus
      GROUP BY show_id, week_ending
    )
    SELECT
      COUNT(*) AS entries,
      COUNT(DISTINCT e.show_id) AS unique_shows,
      SUM(e.gross_millions + COALESCE(b.bonus_millions, 0)) AS total_gross_millions,
      AVG(e.gross_millions + COALESCE(b.bonus_millions, 0)) AS avg_gross_millions
    FROM t10_entry e
    LEFT JOIN bonus_by_row b ON b.show_id = e.show_id AND b.week_ending = e.week_ending
    WHERE COALESCE(e.{imprint_col}, '(Unknown)') = ?
      AND {where}
    """
    return sql_df(sql, tuple([company] + params))

# ----------------------------
# Plot helpers (matplotlib only)
# ----------------------------
def plot_line_dates(x_dates: pd.Series, y: pd.Series, xlabel: str, ylabel: str, invert_y: bool = False):
    fig = plt.figure()
    plt.plot(pd.to_datetime(x_dates), y)
    if invert_y:
        plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_barh(labels: pd.Series, values: pd.Series, xlabel: str, ylabel: str):
    fig = plt.figure()
    plt.barh(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_hist(values: pd.Series, bins: int, xlabel: str, ylabel: str):
    fig = plt.figure()
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_scatter(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str):
    fig = plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------
# New: Streaks + Holidays helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def _fetch_chart_week_sequence() -> pd.DataFrame:
    """Distinct chart weeks in chronological order for consecutive-run logic."""
    df = sql_df(
        """
        SELECT DISTINCT
          date(week_ending) AS week_ending,
          week_number
        FROM t10_entry
        WHERE week_ending IS NOT NULL
        ORDER BY date(week_ending) ASC, week_number ASC
        """
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["week_ending", "week_number"])
    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    return df.drop_duplicates(subset=["week_ending", "week_number"]).reset_index(drop=True)


def _consecutive_by_week_number(wn: pd.Series) -> pd.Series:
    """True where the current row is the next available chart week by week_number."""
    seq = _fetch_chart_week_sequence()
    pos_map: dict[int, int] = {}
    if not seq.empty and seq["week_number"].notna().any():
        valid = seq.dropna(subset=["week_number"]).drop_duplicates(subset=["week_number"]).reset_index(drop=True)
        pos_map = {int(v): i for i, v in enumerate(valid["week_number"].astype(int).tolist())}

    vals = pd.to_numeric(wn, errors="coerce")
    pos = vals.map(lambda v: pos_map.get(int(v)) if pd.notna(v) and int(v) in pos_map else np.nan)
    return pos.diff().eq(1)


def _consecutive_by_date(week_ending_str: pd.Series) -> pd.Series:
    """True where the current row is the next available chart week by date."""
    seq = _fetch_chart_week_sequence()
    pos_map: dict[str, int] = {}
    if not seq.empty:
        valid = seq.dropna(subset=["week_ending"]).drop_duplicates(subset=["week_ending"]).reset_index(drop=True)
        pos_map = {str(v): i for i, v in enumerate(valid["week_ending"].tolist())}

    weeks = _as_date_str(pd.Series(week_ending_str, copy=True))
    pos = weeks.map(lambda v: pos_map.get(str(v)) if pd.notna(v) and str(v) in pos_map else np.nan)
    out = pos.diff().eq(1)

    # Fallback for any dates not found in the chart-week map.
    missing = pos.isna()
    if bool(missing.any()):
        dt = pd.to_datetime(weeks, errors="coerce")
        dd = dt.diff().dt.days
        lo, hi = CONSECUTIVE_DAY_TOLERANCE
        fallback = dd.between(lo, hi)
        out = out.where(~missing, fallback)
    return out

def compute_longest_streaks(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Compute longest consecutive-week streak per (show_id, rank).
    Uses week_number when available; otherwise falls back to ~7-day date spacing.
    Returns: show_id, canonical_title, rank, streak_len, start_week_ending, end_week_ending
    """
    if rows.empty:
        return pd.DataFrame(columns=[
            "show_id", "canonical_title", "rank", "streak_len", "start_week_ending", "end_week_ending"
        ])

    df = rows.copy()
    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")

    out_rows = []
    for (sid, rnk), g in df.groupby(["show_id", "rank"], dropna=False):
        g = g.sort_values(["week_number", "week_ending"]).reset_index(drop=True)
        title = g["canonical_title"].iloc[0] if "canonical_title" in g.columns else None

        # Decide consecutive logic
        if g["week_number"].notna().all():
            cont = _consecutive_by_week_number(g["week_number"])
        else:
            cont = _consecutive_by_date(g["week_ending"])

        # Walk streaks
        best_len = 0
        best_start = None
        best_end = None

        cur_len = 1
        cur_start = g.loc[0, "week_ending"]
        cur_end = g.loc[0, "week_ending"]

        for i in range(1, len(g)):
            if bool(cont.iloc[i]):
                cur_len += 1
                cur_end = g.loc[i, "week_ending"]
            else:
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                    best_end = cur_end
                cur_len = 1
                cur_start = g.loc[i, "week_ending"]
                cur_end = g.loc[i, "week_ending"]

        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
            best_end = cur_end

        out_rows.append({
            "show_id": int(sid) if pd.notna(sid) else sid,
            "canonical_title": title,
            "rank": int(rnk) if pd.notna(rnk) else rnk,
            "streak_len": int(best_len),
            "start_week_ending": best_start,
            "end_week_ending": best_end,
        })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["rank", "streak_len", "canonical_title"], ascending=[True, False, True]).reset_index(drop=True)
    return out


def compute_longest_charted_runs(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute longest consecutive charted run per show across filtered chart weeks."""
    if rows.empty:
        return pd.DataFrame(columns=[
            "show_id", "canonical_title", "run_len", "start_week_ending", "end_week_ending"
        ])

    df = rows.copy()
    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")

    out_rows = []
    for sid, g in df.groupby("show_id", dropna=False):
        g = g.sort_values(["week_number", "week_ending", "rank", "pos"], ascending=[True, True, True, True]).copy()
        if g.empty:
            continue

        # De-duplicate to one row per chart week for this show.
        g = g.drop_duplicates(subset=["week_ending"], keep="first").reset_index(drop=True)
        title = g["canonical_title"].iloc[0] if "canonical_title" in g.columns else None

        if g["week_number"].notna().all():
            cont = _consecutive_by_week_number(g["week_number"])
        else:
            cont = _consecutive_by_date(g["week_ending"])

        best_len = 1
        best_start = g.loc[0, "week_ending"]
        best_end = g.loc[0, "week_ending"]

        cur_len = 1
        cur_start = g.loc[0, "week_ending"]
        cur_end = g.loc[0, "week_ending"]

        for i in range(1, len(g)):
            if bool(cont.iloc[i]):
                cur_len += 1
                cur_end = g.loc[i, "week_ending"]
            else:
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                    best_end = cur_end
                cur_len = 1
                cur_start = g.loc[i, "week_ending"]
                cur_end = g.loc[i, "week_ending"]

        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
            best_end = cur_end

        out_rows.append({
            "show_id": int(sid) if pd.notna(sid) else sid,
            "canonical_title": title,
            "run_len": int(best_len),
            "start_week_ending": best_start,
            "end_week_ending": best_end,
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "show_id", "canonical_title", "run_len", "start_week_ending", "end_week_ending"
        ])
    out = out.sort_values(["run_len", "end_week_ending", "canonical_title"], ascending=[False, False, True]).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def fetch_week_endings_distinct() -> list[date]:
    df = sql_df("SELECT DISTINCT week_ending FROM t10_entry ORDER BY week_ending")
    if df.empty:
        return []
    df["week_ending"] = _as_date_str(df["week_ending"])
    dt = pd.to_datetime(df["week_ending"], errors="coerce")
    d = dt.dt.date.dropna().tolist()
    return sorted(d)

def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    d = d + timedelta(days=shift)
    return d + timedelta(weeks=n - 1)

def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        d = date(year, 12, 31)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d

def easter_date(year: int) -> date:
    # Anonymous Gregorian algorithm (computus)
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)

HOLIDAYS: dict[str, Callable[[int], date]] = {
    "New Year's Day (Jan 1)": lambda y: date(y, 1, 1),
    "Martin Luther King Jr. Day (3rd Mon in Jan)": lambda y: nth_weekday_of_month(y, 1, 0, 3),
    "Valentine's Day (Feb 14)": lambda y: date(y, 2, 14),
    "Presidents Day (3rd Mon in Feb)": lambda y: nth_weekday_of_month(y, 2, 0, 3),
    "Easter (variable)": easter_date,
    "Memorial Day (last Mon in May)": lambda y: last_weekday_of_month(y, 5, 0),
    "Independence Day (Jul 4)": lambda y: date(y, 7, 4),
    "Labor Day (1st Mon in Sep)": lambda y: nth_weekday_of_month(y, 9, 0, 1),
    "Halloween (Oct 31)": lambda y: date(y, 10, 31),
    "Thanksgiving (4th Thu in Nov)": lambda y: nth_weekday_of_month(y, 11, 3, 4),
    "Christmas Day (Dec 25)": lambda y: date(y, 12, 25),
}

def holiday_week_ending_for_date(all_week_endings: list[date], holiday_dt: date, holiday_name: str) -> Optional[date]:
    """
    Choose which chart week_ending to use for a given holiday.

    Rules (week_ending dates are assumed to be Saturdays in your data):
    - Fixed-date holidays (New Year's, Valentine's, Independence Day, Halloween, Christmas):
        * If the holiday is Sun/Mon/Tue     -> use the previous weekend (Saturday before)
        * If the holiday is Wed/Thu/Fri/Sat -> use the following weekend (Saturday on/after)
      Example: Independence Day (07-04) on Thursday -> use week_ending 07-06.
    - Thanksgiving: use the following week_ending (Saturday on/after Thanksgiving).
      Example: Thanksgiving 11-23 -> use 11-25.
    - Weekend/Monday holidays (Easter, Memorial Day, Labor Day, MLK Day, Presidents Day):
      use the weekend the holiday is part of (Saturday before).
      Example: Easter 04-17 -> use 04-16.

    IMPORTANT: If the computed week_ending is not present in the database for that year
    (missing chart) or hasn't been reached yet (future relative to max week_ending),
    return None so the Holidays tab can show blanks/dashes.
    """
    if not all_week_endings:
        return None

    weeks_set = set(all_week_endings)

    def saturday_before(d: date) -> date:
        """Saturday on/before the given date."""
        # weekday: Mon=0..Sun=6, Saturday=5
        return d - timedelta(days=(d.weekday() - 5) % 7)

    def saturday_on_or_after(d: date) -> date:
        """Saturday on/after the given date."""
        return d + timedelta(days=(5 - d.weekday()) % 7)

    def keep_if_present(we: date) -> Optional[date]:
        return we if we in weeks_set else None

    name = (holiday_name or "").strip()

    # Thanksgiving: always the following week ending
    if name.startswith("Thanksgiving"):
        return keep_if_present(saturday_on_or_after(holiday_dt))

    # Weekend/Monday-style holidays: Saturday before
    if (
        name.startswith("Easter")
        or name.startswith("Memorial Day")
        or name.startswith("Labor Day")
        or name.startswith("Martin Luther King")
        or name.startswith("Presidents Day")
    ):
        return keep_if_present(saturday_before(holiday_dt))

    # Fixed-date holidays: previous vs following weekend depends on weekday
    fixed = (
        name.startswith("New Year's Day")
        or name.startswith("Valentine's Day")
        or name.startswith("Independence Day")
        or name.startswith("Halloween")
        or name.startswith("Christmas Day")
    )
    if fixed:
        wd = holiday_dt.weekday()  # Mon=0 ... Sun=6
        if wd in (6, 0, 1):  # Sun/Mon/Tue
            return keep_if_present(saturday_before(holiday_dt))
        else:  # Wed/Thu/Fri/Sat
            return keep_if_present(saturday_on_or_after(holiday_dt))

    # Default: previous chart as-of holiday date
    return keep_if_present(saturday_before(holiday_dt))







# ----------------------------
# New tab: Monthly T-25 (SMPS)
# ----------------------------


# ----------------------------
# New tab: Gross Races
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_gross_races_base(db_path: str, db_mtime: float) -> pd.DataFrame:
    """Weekly gross (including annual+quarter bonuses) per show. db_mtime busts cache on DB updates."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            WITH combined AS (
              -- Base weekly gross rows
              SELECT
                date(e.week_ending) AS week_ending,
                e.show_id AS show_id,
                COALESCE(e.gross_millions, 0.0) AS base_gross_millions,
                0.0 AS bonus_millions
              FROM t10_entry e
              WHERE e.gross_millions IS NOT NULL

              UNION ALL

              -- Bonus rows (include even if show wasn't on chart that week)
              SELECT
                date(gb.week_ending) AS week_ending,
                gb.show_id AS show_id,
                0.0 AS base_gross_millions,
                COALESCE(gb.bonus_millions, 0.0) AS bonus_millions
              FROM gross_bonus gb
              WHERE gb.bonus_type IN ('annual', 'quarter')
            )
            SELECT
              c.week_ending,
              c.show_id,
              s.canonical_title AS canonical_title,
              MAX(e.rank) AS rank,
              SUM(c.base_gross_millions) AS base_gross_millions,
              SUM(c.bonus_millions) AS bonus_millions,
              SUM(c.base_gross_millions + c.bonus_millions) AS gross_millions
            FROM combined c
            JOIN show s ON s.show_id = c.show_id
            LEFT JOIN t10_entry e ON e.show_id = c.show_id AND date(e.week_ending) = c.week_ending
            GROUP BY c.week_ending, c.show_id, s.canonical_title
            ORDER BY c.show_id, c.week_ending
            """,
            con,
        )
    finally:
        con.close()

    if df.empty:
        return df

    return df


@st.cache_data(show_spinner=False)
def _load_smps_weekly_base(db_path: str, db_mtime: float, include_bonuses: bool) -> pd.DataFrame:
    """Weekly gross base for SMPS.

    If include_bonuses is True, use the same base as Gross Races (weekly gross plus
    annual/quarter bonuses via gross_bonus). If False, use only t10_entry weekly gross
    rows (no gross_bonus union).
    """
    if include_bonuses:
        return _load_gross_races_base(db_path, db_mtime)

    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
              date(e.week_ending) AS week_ending,
              e.show_id AS show_id,
              s.canonical_title AS canonical_title,
              SUM(COALESCE(e.gross_millions, 0.0)) AS base_gross_millions,
              0.0 AS bonus_millions,
              SUM(COALESCE(e.gross_millions, 0.0)) AS gross_millions
            FROM t10_entry e
            JOIN show s ON s.show_id = e.show_id
            WHERE e.gross_millions IS NOT NULL
            GROUP BY date(e.week_ending), e.show_id, s.canonical_title
            ORDER BY e.show_id, date(e.week_ending)
            """,
            con,
        )
    finally:
        con.close()
    return df


@st.cache_data(show_spinner=False)
def _compute_show_month_metrics(db_path: str, db_mtime: float, include_bonuses: bool) -> pd.DataFrame:
    """Per-show monthly aggregates for SMPS.

    Uses a weekly gross base that can optionally include gross bonuses.
    """
    base = _load_smps_weekly_base(db_path, db_mtime, include_bonuses)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "month_ord",
                "show_id",
                "canonical_title",
                "imprint_1",
                "imprint_2",
                "month_gross_millions",
                "weeks_in_month",
                "first2_avg_millions",
                "last2_avg_millions",
                "prev_month_gross_millions",
            ]
        )

    meta = _load_show_meta_for_gross_races(db_path, db_mtime)
    if not meta.empty:
        base = base.merge(meta[["show_id", "imprint_1", "imprint_2"]], on="show_id", how="left")
    else:
        base["imprint_1"] = ""
        base["imprint_2"] = ""

    base["imprint_1"] = base["imprint_1"].fillna("")
    base["imprint_2"] = base["imprint_2"].fillna("")

    # Ensure we have a datetime column for week ending
    if "week_ending_dt" not in base.columns:
        if "week_ending" in base.columns:
            base["week_ending_dt"] = pd.to_datetime(base["week_ending"], errors="coerce")
        else:
            raise KeyError("Expected 'week_ending' or 'week_ending_dt' in base dataframe.")

    base = base.dropna(subset=["week_ending_dt"]).copy()

    base = base[base["week_ending_dt"].dt.date >= GROSS_TRACKING_START].copy()

    base = _apply_chart_month_logic(base, week_dt_col="week_ending_dt", week_str_col="week_ending")
    base["gross_millions"] = pd.to_numeric(base["gross_millions"], errors="coerce").fillna(0.0)

    # Aggregate per show/month
    def _agg_one(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("week_ending_dt")
        arr = g["gross_millions"].to_numpy(dtype=float)
        n = len(arr)
        if n == 0:
            f2 = 0.0
            l2 = 0.0
        elif n == 1:
            f2 = float(arr[0])
            l2 = float(arr[0])
        else:
            f2 = float(arr[:2].mean())
            l2 = float(arr[-2:].mean())
        return pd.Series(
            {
                "month_gross_millions": float(arr.sum()),
                "weeks_in_month": int(n),
                "first2_avg_millions": f2,
                "last2_avg_millions": l2,
            }
        )

    agg = (
        base.groupby(["month", "month_ord", "show_id", "canonical_title", "imprint_1", "imprint_2"], as_index=False)
        .apply(_agg_one, include_groups=False)
        .reset_index(drop=True)
    )

    # Prev-month gross per show
    agg = agg.sort_values(["show_id", "month_ord"]).reset_index(drop=True)
    agg["prev_month_gross_millions"] = agg.groupby("show_id")["month_gross_millions"].shift(1).fillna(0.0)

    return agg


@st.cache_data(show_spinner=False)
def _compute_smps_history(db_path: str, db_mtime: float, include_bonuses: bool) -> dict[str, pd.DataFrame]:
    """Compute SMPS_v1 charts for every chart-month (Apr 2001 → latest)."""
    metrics = _compute_show_month_metrics(db_path, db_mtime, include_bonuses)
    if metrics.empty:
        return {}

    # Available months from the data
    months = sorted(metrics["month"].unique().tolist())
    # Ensure we start at the first SMPS month
    if SMPS_START_MONTH in months:
        start_idx = months.index(SMPS_START_MONTH)
        months = months[start_idx:]

    # Index by month for fast access
    by_month: dict[str, pd.DataFrame] = {m: metrics[metrics["month"] == m].copy() for m in months}

    # Dynamic Share cap per month:
    # For each chart-month, compute the highest single-show month_gross_millions on record
    # *up to and including* that month. No rounding.
    # Share uses an 80% normalization: month_gross >= 0.80 * Cap(m) yields the full 50 Share points.
    #
    # This makes older eras comparable by scaling Share against the historical ceiling
    # that existed at the time.
    month_max: dict[str, float] = {}
    for mm in months:
        mm_df = by_month.get(mm)
        if mm_df is None or mm_df.empty:
            month_max[mm] = 0.0
        else:
            month_max[mm] = float(pd.to_numeric(mm_df["month_gross_millions"], errors="coerce").fillna(0.0).max())

    share_cap_by_month: dict[str, float] = {}
    running_record = 0.0
    for mm in months:
        running_record = max(running_record, float(month_max.get(mm, 0.0)))
        # No rounding: use the running record as-is (keep a safe, non-zero minimum).
        cap = running_record if running_record > 0 else 100.0
        share_cap_by_month[mm] = float(cap)


    # Meta lookups (loaded once):
    # - Titles come from the show table so carryover-only candidates always have names.
    # - Imprints come from the same helper used by Gross Races.
    con = sqlite3.connect(db_path)
    try:
        show_titles = pd.read_sql_query("SELECT show_id, canonical_title FROM show", con)
    finally:
        con.close()

    show_titles = (
        show_titles.drop_duplicates('show_id')
        if not show_titles.empty
        else pd.DataFrame(columns=['show_id', 'canonical_title'])
    )

    _imp = _load_show_meta_for_gross_races(db_path, db_mtime)
    if _imp is None or _imp.empty:
        imprint_meta = pd.DataFrame(columns=['show_id', 'imprint_1', 'imprint_2'])
    else:
        cols = [c for c in ['show_id', 'imprint_1', 'imprint_2'] if c in _imp.columns]
        imprint_meta = _imp[cols].drop_duplicates('show_id').copy()
        if 'imprint_1' not in imprint_meta.columns:
            imprint_meta['imprint_1'] = ''
        if 'imprint_2' not in imprint_meta.columns:
            imprint_meta['imprint_2'] = ''

    inactive: dict[int, int] = {}  # show_id -> consecutive zero-gross months (candidate-only)
    prev_chart: Optional[pd.DataFrame] = None

    out: dict[str, pd.DataFrame] = {}

    for m in months:
        mdf = by_month.get(m)
        if mdf is None:
            continue

        # Dicts for quick lookup
        gross = {int(r.show_id): float(r.month_gross_millions) for r in mdf.itertuples(index=False)}
        f2 = {int(r.show_id): float(r.first2_avg_millions) for r in mdf.itertuples(index=False)}
        l2 = {int(r.show_id): float(r.last2_avg_millions) for r in mdf.itertuples(index=False)}
        prevg = {int(r.show_id): float(r.prev_month_gross_millions) for r in mdf.itertuples(index=False)}

        grossing_ids = {sid for sid, g in gross.items() if g > 0}
        prev_ids: set[int] = set(prev_chart["show_id"].astype(int).tolist()) if prev_chart is not None else set()
        candidates = set(grossing_ids) | set(prev_ids)

        # Update inactive streaks for candidates
        for sid in list(candidates):
            g = gross.get(sid, 0.0)
            if g > 0:
                inactive[sid] = 0
            else:
                inactive[sid] = int(inactive.get(sid, 0)) + 1

        # Zombie rule: ineligible if 4+ consecutive zero-gross months
        def is_zombie(sid: int) -> bool:
            return (gross.get(sid, 0.0) <= 0) and (inactive.get(sid, 0) >= 4)

        candidates = {sid for sid in candidates if not is_zombie(sid)}

        # Month total (no longer used for Share scoring; kept for reference/other diagnostics)
        total_month_gross = sum(gross.get(sid, 0.0) for sid in grossing_ids)
        total_month_gross = float(total_month_gross) if total_month_gross > 0 else 0.0

        # Floors for breakout + heat
        F = _p10_inc([prevg.get(sid, 0.0) for sid in candidates], ignore_nonpositive=True)
        FW = _p10_inc([f2.get(sid, 0.0) for sid in candidates], ignore_nonpositive=True)

        rows = []
        prev_pts = {}
        if prev_chart is not None and not prev_chart.empty:
            prev_pts = {int(r.show_id): float(r.points_total) for r in prev_chart.itertuples(index=False)}

        for sid in candidates:
            mg = float(gross.get(sid, 0.0))
            pg = float(prevg.get(sid, 0.0))
            # Share (ratio-based, top-heavy)
            # Dynamic denom per month: Cap(m) = running record (through month m) of the highest
            # single-show month gross (no rounding).
            #
            # New regime: keep the same raw share ratio basis r = clamp(MG/Cap(m), 0..1), but
            # grant full Share points at 80% of Cap(m) by rescaling r -> r_scaled = clamp(r/0.80, 0..1).
            #
            # Displayed "Share Ratio" uses the raw ratio r (MG/Cap).
            share_cap = float(share_cap_by_month.get(m, 100.0))
            share_raw = (mg / share_cap) if (mg > 0 and share_cap > 0) else 0.0  # r = MG/Cap(m)
            if share_raw < 0.0:
                share_raw = 0.0
            elif share_raw > 1.0:
                share_raw = 1.0

            share_scaled = share_raw / 0.80
            if share_scaled < 0.0:
                share_scaled = 0.0
            elif share_scaled > 1.0:
                share_scaled = 1.0

            # Nonlinear top-heavy ramp (power exponent 1.5)
            pts_share = 50.0 * (share_scaled ** 1.5)

            # Momentum components only apply when the show grossed this month
            if mg > 0:
                breakout_raw = (mg - pg) / max(pg, F)
                f2v = float(f2.get(sid, 0.0))
                l2v = float(l2.get(sid, 0.0))
                heat_raw = (l2v - f2v) / max(f2v, FW)
            else:
                breakout_raw = 0.0
                heat_raw = 0.0

            rows.append(
                {
                    "show_id": int(sid),
                    "month_gross_millions": mg,
                    "prev_month_gross_millions": pg,
                    "share_raw": share_raw,
                    "breakout_raw": breakout_raw,
                    "heat_raw": heat_raw,
                    "inactive_streak": int(inactive.get(sid, 0)),
                    "points_share": pts_share,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            out[m] = df
            prev_chart = df
            continue

        # Rank-based momentum scoring (0..1) among *grossing* shows only
        grossing_mask = df["month_gross_millions"] > 0
        mom_df = df[grossing_mask].copy()

        def _rank01(s: pd.Series) -> pd.Series:
            if len(s) <= 1:
                return pd.Series([1.0] * len(s), index=s.index)
            r = s.rank(method="average", ascending=True)
            return (r - 1.0) / (len(s) - 1.0)

        if not mom_df.empty:
            mom_df["breakout_score"] = _rank01(mom_df["breakout_raw"])
            mom_df["heat_score"] = _rank01(mom_df["heat_raw"])
            mom_df["points_breakout"] = 30.0 * mom_df["breakout_score"]
            mom_df["points_heat"] = 20.0 * mom_df["heat_score"]

            df = df.merge(
                mom_df[["show_id", "points_breakout", "points_heat"]],
                on="show_id",
                how="left",
            )
        else:
            df["points_breakout"] = 0.0
            df["points_heat"] = 0.0

        df["points_breakout"] = pd.to_numeric(df.get("points_breakout"), errors="coerce").fillna(0.0)
        df["points_heat"] = pd.to_numeric(df.get("points_heat"), errors="coerce").fillna(0.0)


        # --- Debut/re-entry guardrails (reduce #1 debuts unless the month is truly monstrous) ---
        # If a show was not on last month's SMPS Top 25, taper its Breakout points unless it
        # commands a big enough Share of the month. This keeps true mega-launches eligible for #1,
        # but prevents "auto-#1" Breakout wins caused by pg≈0/denominator effects.
        #
        # Taper rule (nonlinear power ramp, exponent 1.25):
        #   share_raw <= 0.40 -> 0% of Breakout points  (<= 40% of Cap(m))
        #   share_raw >= 0.75 -> 100% of Breakout points (>= 75% of Cap(m))
        #   between -> ramp in [0..1] then apply power exponent 1.25
        s0, s1 = 0.40, 0.75
        denom = (s1 - s0) if (s1 - s0) != 0 else 1.0
        df["is_debut_or_reentry"] = ~df["show_id"].astype(int).isin(list(prev_ids))
        _u = ((df["share_raw"] - s0) / denom).clip(lower=0.0, upper=1.0)
        df["debut_breakout_factor"] = _u ** 1.25
        _mask_debut = df["is_debut_or_reentry"] & (df["month_gross_millions"] > 0)
        df.loc[_mask_debut, "points_breakout"] = df.loc[_mask_debut, "points_breakout"] * df.loc[_mask_debut, "debut_breakout_factor"]

        # Continuity bonus: active incumbents get a small inertia bump to reduce leapfrogging by re-entries.
        # (Only applies when the show grosses this month.)
        df["points_continuity"] = 0.0
        if prev_pts:
            _mask_inc = (df["month_gross_millions"] > 0) & df["show_id"].astype(int).isin(list(prev_pts.keys()))
            df.loc[_mask_inc, "points_continuity"] = df.loc[_mask_inc, "show_id"].astype(int).map(prev_pts).fillna(0.0) * 0.10

        # Carryover for 0-gross months (prev SMPS Top 25 only), decays by inactive streak
        def _carry(sid: int, mg: float) -> float:
            if mg > 0:
                return 0.0
            if sid not in prev_pts:
                return 0.0
            k = int(inactive.get(sid, 0))
            if k < 1 or k > 3:
                return 0.0
            return float(prev_pts[sid]) * 0.30 * (0.55 ** k)

        df["points_carryover"] = df.apply(lambda r: _carry(int(r["show_id"]), float(r["month_gross_millions"])), axis=1)

        df["points_total"] = (
            df["points_share"]
            + df["points_breakout"]
            + df["points_heat"]
            + df["points_carryover"]
            + df.get("points_continuity", 0.0)
        )

        # Attach titles/imprints
        # Use show table titles so carryover-only candidates do not show up as (Unknown).
        df = df.merge(show_titles, on="show_id", how="left", suffixes=("", "_show"))
        if "canonical_title_show" in df.columns:
            if "canonical_title" in df.columns:
                df["canonical_title"] = df["canonical_title"].fillna(df["canonical_title_show"])
            else:
                df = df.rename(columns={"canonical_title_show": "canonical_title"})
            df = df.drop(columns=["canonical_title_show"], errors="ignore")

        # Imprints: fill from imprint_meta for any show_ids that are missing them this month
        if imprint_meta is not None and not imprint_meta.empty:
            df = df.merge(imprint_meta, on="show_id", how="left", suffixes=("", "_imp"))
            for col in ("imprint_1", "imprint_2"):
                imp_col = f"{col}_imp"
                if imp_col in df.columns:
                    if col in df.columns:
                        df[col] = df[col].fillna(df[imp_col])
                    else:
                        df[col] = df[imp_col]
                    df = df.drop(columns=[imp_col], errors="ignore")

        if "canonical_title" not in df.columns:
            df["canonical_title"] = pd.NA
        if "imprint_1" not in df.columns:
            df["imprint_1"] = ""
        if "imprint_2" not in df.columns:
            df["imprint_2"] = ""
        df["canonical_title"] = df["canonical_title"].fillna("(Unknown)")
        df["imprint_1"] = df["imprint_1"].fillna("")
        df["imprint_2"] = df["imprint_2"].fillna("")

        # Tie-breaks: total_pts, breakout_raw, heat_raw, month_gross, title
        df = df.sort_values(
            ["points_total", "breakout_raw", "heat_raw", "month_gross_millions", "canonical_title"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)

        df.insert(0, "position", np.arange(1, len(df) + 1))
        chart = df.head(25).copy()

        # Keep only needed columns for chart storage
        chart["month"] = m
        chart["chart_type"] = "SMPS"
        chart["method_version"] = SMPS_METHOD_VERSION

        out[m] = chart
        prev_chart = chart[[
            "show_id",
            "points_total",
        ]].copy()

    return out


def _write_smps_to_db(month: str, chart_df: pd.DataFrame) -> None:
    """Persist one month of SMPS chart results into SQLite."""
    if chart_df.empty:
        return

    _ensure_smps_schema()

    con = get_con()
    try:
        cur = con.cursor()
        cur.execute("BEGIN;")
        cur.execute(
            "DELETE FROM monthly_chart WHERE month = ? AND chart_type = 'SMPS' AND method_version = ?;",
            (month, SMPS_METHOD_VERSION),
        )

        rows = []
        for r in chart_df.itertuples(index=False):
            rows.append(
                (
                    str(r.month),
                    "SMPS",
                    SMPS_METHOD_VERSION,
                    int(r.position),
                    int(r.show_id),
                    float(getattr(r, "month_gross_millions", 0.0)),
                    float(getattr(r, "points_total", 0.0)),
                    float(getattr(r, "points_share", 0.0)),
                    float(getattr(r, "points_breakout", 0.0)),
                    float(getattr(r, "points_heat", 0.0)),
                    float(getattr(r, "points_carryover", 0.0)),
                    int(getattr(r, "inactive_streak", 0)),
                )
            )

        cur.executemany(
            """
            INSERT INTO monthly_chart(
              month, chart_type, method_version, position, show_id,
              month_gross_millions,
              points_total, points_share, points_breakout, points_heat, points_carryover,
              inactive_streak
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);
            """,
            rows,
        )
        con.commit()
    finally:
        con.close()


def tab_monthly_smps_t25():
    st.subheader("Monthly T-25 (SMPS)")
    st.caption(
        "SMPS_v1 = Share (50; ratio-based to the running record monthly max (no rounding) with exponent 1.4) + Breakout (30) + Heat (20) + carryover (0-gross months only) + continuity bonus (active incumbents). "
        "Floors use PERCENTILE.INC (10th percentile). Zombie rule: 4 consecutive 0-gross chart-months => ineligible."
    )

    include_bonuses = st.checkbox(
        "Include gross bonuses in SMPS month gross (and Share cap)",
        value=False,
        key="smps_include_bonuses",
        help="If checked, SMPS month gross includes annual/quarter bonuses from gross_bonus (like Gross Races). If unchecked, SMPS uses only the weekly gross in t10_entry.",
    )

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}.")
        return

    db_mtime = DB_PATH.stat().st_mtime
    hist = _compute_smps_history(str(DB_PATH), db_mtime, include_bonuses)
    if not hist:
        st.info("No SMPS history could be computed from the current DB.")
        return

    months = sorted(hist.keys())

    # Default month: latest
    pick = st.selectbox("Chart month", options=months, index=len(months) - 1, key="smps_month_pick")
    chart = hist.get(pick)
    if chart is None or chart.empty:
        st.info("No chart rows for that month.")
        return

    # Display
    disp = chart.copy()

    # Add last-month position + total appearances (Months on Chart) + NEW/RE status (SMPS)
    pick_idx = months.index(pick)
    prev_pos_map = {}
    if pick_idx > 0:
        prev_m = months[pick_idx - 1]
        prev_chart = hist.get(prev_m)
        if prev_chart is not None and not prev_chart.empty:
            prev_pos_map = dict(zip(prev_chart["show_id"].astype(int), prev_chart["position"].astype(int)))

    from collections import Counter
    cnt_before = Counter()
    for mm in months[:pick_idx]:
        cdf = hist.get(mm)
        if cdf is None or cdf.empty:
            continue
        cnt_before.update(cdf["show_id"].astype(int).tolist())

    disp["Last Mo Pos"] = disp["show_id"].astype(int).apply(lambda sid: prev_pos_map.get(int(sid)))
    disp["Months on Chart"] = disp["show_id"].astype(int).map(cnt_before).fillna(0).astype(int) + 1
    # Cast to object before mixing numeric positions with string flags (NEW/RE)
    disp["Last Mo Pos"] = disp["Last Mo Pos"].astype(object)

    # NEW/RE flags folded into Last Mo Pos: if not on last month, NEW if first-ever appearance; else RE
    _lastmo_missing = disp["Last Mo Pos"].isna()
    _seen_before = disp["show_id"].astype(int).map(cnt_before).fillna(0).astype(int) > 0
    disp.loc[_lastmo_missing & (~_seen_before), "Last Mo Pos"] = "NEW"
    disp.loc[_lastmo_missing & (_seen_before), "Last Mo Pos"] = "RE"

    # Remove any .0 decimals from numeric last-month positions (keep NEW/RE as-is)
    def _fmt_last_mo_pos(v):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return v
        try:
            return str(int(v))
        except Exception:
            return v

    disp["Last Mo Pos"] = disp["Last Mo Pos"].apply(_fmt_last_mo_pos)

    disp["Share Ratio"] = disp["share_raw"].round(3)
    disp["Month Gross"] = disp["month_gross_millions"].round(2)
    disp["Pts Share"] = disp["points_share"].round(2)
    disp["Pts Breakout"] = disp["points_breakout"].round(2)
    disp["Pts Heat"] = disp["points_heat"].round(2)
    disp["Pts Carryover"] = disp["points_carryover"].round(2)
    disp["Pts Total"] = disp["points_total"].round(2)

    show_cols = [
        "position",
        "canonical_title",
        "Last Mo Pos",
        "Months on Chart",
        "imprint_1",
        "imprint_2",
        "Month Gross",
        "Share Ratio",
        "breakout_raw",
        "heat_raw",
        "Pts Share",
        "Pts Breakout",
        "Pts Heat",
        "Pts Carryover",
        "Pts Total",
    ]

    st.dataframe(
        disp[show_cols].rename(
            columns={
                "position": "Pos",
                "canonical_title": "Show",
                "imprint_1": "Imprint 1",
                "imprint_2": "Imprint 2",
                "breakout_raw": "BreakoutRaw",
                "heat_raw": "HeatRaw",
            }
        ),
        width='stretch',
        hide_index=True,
    )



    # Export: download this month's SMPS T-25 as .xlsx
    # (Column order mirrors your legacy .xls layout.)
    try:
        export_disp = disp.copy()

        # Peak Position = best (lowest) SMPS monthly rank so far (through the selected month)
        _pos_frames = []
        for _mm in months[: pick_idx + 1]:
            _cdf = hist.get(_mm)
            if _cdf is None or _cdf.empty:
                continue
            if 'show_id' in _cdf.columns and 'position' in _cdf.columns:
                _pos_frames.append(_cdf[['show_id', 'position']].copy())
        if _pos_frames:
            _peak_map = pd.concat(_pos_frames, ignore_index=True).groupby('show_id')['position'].min()
        else:
            _peak_map = pd.Series(dtype='float')

        export_disp['This Month'] = export_disp['position'].astype(int)
        export_disp['Last Month'] = export_disp['Last Mo Pos']
        export_disp['Peak Position'] = export_disp['show_id'].map(_peak_map).fillna(export_disp['position']).astype(int)

        export_disp['Total Points'] = pd.to_numeric(export_disp.get('points_total'), errors='coerce').fillna(0.0)
        export_disp['Pts Continuity'] = pd.to_numeric(export_disp.get('points_continuity'), errors='coerce').fillna(0.0)

        export_df = pd.DataFrame({
            'This Month': export_disp['This Month'],
            'Last Month': export_disp['Last Month'],
            'Months on Chart': export_disp['Months on Chart'],
            'Show': export_disp['canonical_title'],
            'Imprint 1': export_disp['imprint_1'],
            'Imprint 2': export_disp['imprint_2'],
            'Peak Position': export_disp['Peak Position'],
            'Total Points': export_disp['Total Points'].round(2),
            '': [''] * len(export_disp),
            'Share Ratio': export_disp['Share Ratio'],
            'BreakoutRaw': export_disp['breakout_raw'],
            'HeatRaw': export_disp['heat_raw'],
            'Pts Share': export_disp['Pts Share'],
            'Pts Breakout': export_disp['Pts Breakout'],
            'Pts Heat': export_disp['Pts Heat'],
            'Pts Carryover': export_disp['Pts Carryover'],
            'Pts Continuity': export_disp['Pts Continuity'].round(2),
        })

        export_df = export_df[[
            'This Month', 'Last Month', 'Months on Chart', 'Show',
            'Imprint 1', 'Imprint 2', 'Peak Position', 'Total Points',
            '',
            'Share Ratio', 'BreakoutRaw', 'HeatRaw',
            'Pts Share', 'Pts Breakout', 'Pts Heat', 'Pts Carryover', 'Pts Continuity'
        ]]

        def _smps_month_xlsx_bytes(df_out: pd.DataFrame) -> bytes:
            try:
                import openpyxl  # type: ignore  # noqa: F401
                from openpyxl.styles import Font, Alignment
            except Exception:
                return b''

            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine='openpyxl') as writer:
                df_out.to_excel(writer, index=False, sheet_name='Monthly T-25')
                ws = writer.sheets['Monthly T-25']
                ws.freeze_panes = 'A2'

                header_font = Font(bold=True)
                for cell in ws[1]:
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center', vertical='center')

                # Basic auto-fit (capped) + keep the spacer column narrow
                for col_cells in ws.columns:
                    col_letter = col_cells[0].column_letter
                    header_val = col_cells[0].value
                    if header_val == '':
                        ws.column_dimensions[col_letter].width = 3
                        continue
                    max_len = 0
                    for c in col_cells[: min(len(col_cells), 200)]:
                        v = c.value
                        if v is None:
                            continue
                        max_len = max(max_len, len(str(v)))
                    ws.column_dimensions[col_letter].width = min(max(8, max_len + 2), 44)

            return bio.getvalue()

        xlsx_bytes = _smps_month_xlsx_bytes(export_df)
        if xlsx_bytes:
            st.download_button(
                'Download .xlsx',
                data=xlsx_bytes,
                file_name=f"SMPS_Monthly_T25_{pick}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='smps_monthly_xlsx_dl',
            )
        else:
            st.info('Excel export requires `openpyxl`. Install it in your venv with: `pip install openpyxl`')

    except Exception as _e:
        st.error(f"XLSX export failed: {_e}")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Write this SMPS month to DB", key="smps_write_one"):
            _write_smps_to_db(pick, chart)
            st.success(f"Saved SMPS {pick} to monthly_chart.")

    with c2:
        with st.expander("Backfill: write ALL SMPS months to DB (SMPS_v1)"):
            st.warning("This writes/overwrites SMPS_v1 rows in monthly_chart for every month in history.")
            if st.button("Run full SMPS backfill", key="smps_write_all"):
                prog = st.progress(0)
                for i, m2 in enumerate(months, start=1):
                    _write_smps_to_db(m2, hist[m2])
                    prog.progress(int(i * 100 / max(1, len(months))))
                st.success("Backfill complete.")

# ----------------------------
# New tab: Year-End (SMPS) — Top 35
# ----------------------------
def tab_year_end_smps_t35():
    st.subheader("Year-End (Weekly Points)")
    st.caption(
        "Year-end rankings are calculated by summing weekly points across the selected year. "
        "Grossing era: the #1 show of each week gets 100 points; all other shows get (their weekly gross ÷ #1 gross) × 100. "
        "Pre-grossing era: inverse points by rank (100, 90, 80, ...). Returns the Top 35."
    )

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}.")
        return

    years_df = sql_df(
        """
        SELECT DISTINCT CAST(substr(week_ending, 1, 4) AS INTEGER) AS year
        FROM t10_entry
        WHERE week_ending IS NOT NULL AND TRIM(week_ending) <> ''
        ORDER BY year DESC
        """
    )
    years = years_df["year"].dropna().astype(int).tolist() if not years_df.empty else []
    if not years:
        st.info("No weeks found in the database.")
        return

    year = int(st.selectbox("Year", options=years, index=0, key="year_end_weekly_points_year"))

    db_mtime = DB_PATH.stat().st_mtime

    @st.cache_data(show_spinner=False)
    def _load_year_end_weekly_base(db_path: str, db_mtime: float) -> pd.DataFrame:
        """Weekly base rows for year-end charts (includes bonuses where present).

        - Uses t10_entry ranks for all weeks.
        - Grossing era (>= GROSS_TRACKING_START): uses (base + bonus) as weekly gross.
        - Pre-grossing era: gross is not required; points come from inverse rank.
        """
        con = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(
                """
                WITH bonus AS (
                  SELECT show_id, week_ending, SUM(COALESCE(bonus_millions, 0.0)) AS bonus_millions
                  FROM gross_bonus
                  GROUP BY show_id, week_ending
                )
                SELECT
                  date(e.week_ending) AS week_ending,
                  e.show_id AS show_id,
                  s.canonical_title AS canonical_title,
                  COALESCE(NULLIF(TRIM(e.imprint_1), ''), '') AS imprint_1,
                  COALESCE(NULLIF(TRIM(e.imprint_2), ''), '') AS imprint_2,
                  e.rank AS rank,
                  COALESCE(e.gross_millions, 0.0) AS base_gross_millions,
                  COALESCE(b.bonus_millions, 0.0) AS bonus_millions,
                  (COALESCE(e.gross_millions, 0.0) + COALESCE(b.bonus_millions, 0.0)) AS gross_millions
                FROM t10_entry e
                JOIN show s ON s.show_id = e.show_id
                LEFT JOIN bonus b ON b.show_id = e.show_id AND b.week_ending = e.week_ending
                WHERE e.rank BETWEEN 1 AND 10
                ORDER BY date(e.week_ending) ASC, e.rank ASC
                """,
                con,
            )
        finally:
            con.close()

        if df.empty:
            return df

        df["week_ending"] = _as_date_str(df["week_ending"])
        df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
        df = df.dropna(subset=["week_ending_dt"]).copy()

        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["gross_millions"] = pd.to_numeric(df["gross_millions"], errors="coerce").fillna(0.0)

        df["year"] = df["week_ending_dt"].dt.year.astype(int)

        # Determine grossing era vs pre-grossing era
        pre_mask = df["week_ending_dt"].dt.date < GROSS_TRACKING_START

        # Top-1 weekly gross (grossing era only)
        top1 = (
            df.loc[~pre_mask & df["rank"].eq(1)]
            .groupby("week_ending", as_index=False)["gross_millions"]
            .max()
            .rename(columns={"gross_millions": "top1_gross_millions"})
        )
        df = df.merge(top1, on="week_ending", how="left")
        df["top1_gross_millions"] = pd.to_numeric(df["top1_gross_millions"], errors="coerce").fillna(0.0)

        # Weekly points
        df["week_points"] = 0.0

        # Pre-grossing era: inverse rank points (100, 90, 80, ...)
        df.loc[pre_mask, "week_points"] = (110.0 - 10.0 * df.loc[pre_mask, "rank"]).clip(lower=0.0, upper=100.0)

        # Grossing era: #1 = 100, others = gross/top1 * 100
        post_mask = ~pre_mask
        df.loc[post_mask & df["rank"].eq(1), "week_points"] = 100.0

        other = post_mask & (~df["rank"].eq(1))
        denom = df.loc[other, "top1_gross_millions"]
        numer = df.loc[other, "gross_millions"]
        df.loc[other, "week_points"] = np.where(denom > 0, (numer / denom) * 100.0, 0.0)

        df["week_points"] = pd.to_numeric(df["week_points"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)

        return df

    base = _load_year_end_weekly_base(str(DB_PATH), db_mtime)
    if base.empty:
        st.info("No chart rows found for year-end computation.")
        return

    base_y = base[base["year"].eq(int(year))].copy()
    if base_y.empty:
        st.info(f"No chart weeks found for {year}.")
        return

    group_cols = ["show_id", "canonical_title", "imprint_1", "imprint_2"]
    agg = (
        base_y.groupby(group_cols, as_index=False)
        .agg(
            weeks_charted=("week_ending", "nunique"),
            best_rank=("rank", "min"),
            total_gross_millions=("gross_millions", "sum"),
            points_total=("week_points", "sum"),
        )
    )

    agg["points_total"] = pd.to_numeric(agg["points_total"], errors="coerce").fillna(0.0)
    agg["total_gross_millions"] = pd.to_numeric(agg["total_gross_millions"], errors="coerce").fillna(0.0)

    out = (
        agg.sort_values(
            ["points_total", "total_gross_millions", "canonical_title"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    out.insert(0, "position", np.arange(1, len(out) + 1))
    out = out.head(35).copy()

    st.dataframe(out, width='stretch', hide_index=True)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"year_end_weekly_points_top35_{year}.csv",
        mime="text/csv",
        key="year_end_weekly_points_download",
    )

# ----------------------------
# New tab: Grossing Milestones
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_milestone_base(db_path: str, db_mtime: float) -> pd.DataFrame:
    """Load minimal data for milestone calculations. db_mtime busts cache on DB updates."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            WITH
            e_weekly AS (
              SELECT show_id, week_ending, SUM(gross_millions) AS gross_millions
              FROM t10_entry
              GROUP BY show_id, week_ending
            ),
            gb AS (
              SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
              FROM gross_bonus
              GROUP BY show_id, week_ending
            ),
            weeks AS (
              SELECT show_id, week_ending FROM e_weekly
              UNION
              SELECT show_id, week_ending FROM gb
            )
            SELECT
              w.week_ending,
              w.show_id,
              s.canonical_title AS canonical_title,
              COALESCE(e.gross_millions, 0) AS base_gross_millions,
              COALESCE(gb.bonus_millions, 0) AS bonus_millions,
              (COALESCE(e.gross_millions, 0) + COALESCE(gb.bonus_millions, 0)) AS gross_millions
            FROM weeks w
            LEFT JOIN e_weekly e
              ON e.show_id = w.show_id AND e.week_ending = w.week_ending
            LEFT JOIN gb
              ON gb.show_id = w.show_id AND gb.week_ending = w.week_ending
            JOIN show s ON s.show_id = w.show_id
            """,
            con,
        )
    finally:
        con.close()

    if df.empty:
        return df

    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["week_ending_dt", "show_id", "canonical_title"]).copy()
    df["gross_millions"] = pd.to_numeric(df["gross_millions"], errors="coerce").fillna(0.0)

    # If ties/duplicates ever create multiple rows for a show/week, collapse to weekly sum first
    df = (
        df.groupby(["show_id", "canonical_title", "week_ending"], as_index=False)["gross_millions"]
        .sum()
        .sort_values(["show_id", "week_ending"])
        .reset_index(drop=True)
    )
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    return df


def _compute_milestones_1k(df_base: pd.DataFrame, step: int = 1000) -> pd.DataFrame:
    """
    Compute first week each show reaches each step milestone in cumulative gross_millions.
    Returns one row per (show, milestone).
    """
    if df_base.empty:
        return pd.DataFrame(
            columns=[
                "canonical_title",
                "show_id",
                "milestone",
                "week_ending",
                "cumulative_gross_millions",
                "week_gross_millions",
            ]
        )

    work = df_base.copy()
    work = work.sort_values(["show_id", "week_ending_dt"]).reset_index(drop=True)
    work["cumulative_gross_millions"] = work.groupby("show_id")["gross_millions"].cumsum()

    out_rows: list[dict[str, Any]] = []
    for sid, g in work.groupby("show_id", sort=False):
        g = g.sort_values("week_ending_dt")
        title = g["canonical_title"].iloc[0]
        cum = g["cumulative_gross_millions"].to_numpy()

        max_cum = float(cum.max()) if len(cum) else 0.0
        if max_cum < step:
            continue

        top = int(max_cum // step) * step
        milestones = range(step, top + step, step)

        for m in milestones:
            idxs = (cum >= m).nonzero()[0]
            if len(idxs) == 0:
                continue
            i = int(idxs[0])
            row = g.iloc[i]
            out_rows.append(
                {
                    "canonical_title": title,
                    "show_id": int(sid),
                    "milestone": int(m),
                    "week_ending": str(row["week_ending"]),
                    "cumulative_gross_millions": float(row["cumulative_gross_millions"]),
                    "week_gross_millions": float(row["gross_millions"]),
                }
            )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    out["week_ending_dt"] = pd.to_datetime(out["week_ending"], errors="coerce")
    out = out.sort_values(["canonical_title", "milestone"]).reset_index(drop=True)
    return out


def tab_grossing_milestones():
    st.subheader("Grossing milestones")
    st.caption("Milestones are based on cumulative sum of gross_millions over time (no currency formatting).")

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}.")
        return

    # Cache-buster so this tab refreshes after DB updates/redeploys
    db_mtime = DB_PATH.stat().st_mtime
    base = _load_milestone_base(str(DB_PATH), db_mtime)

    df_m = _compute_milestones_1k(base, step=1000)
    if df_m.empty:
        st.info("No shows have reached 1,000 gross_millions yet.")
        return

    def fmt_int(x: int) -> str:
        return f"{int(x):,}"

    # -------------------------
    # Section 1: Show → milestones (1k increments)
    # -------------------------
    st.markdown("### Show → milestones (1k increments)")
    shows = sorted(df_m["canonical_title"].unique().tolist())
    pick_show = st.selectbox("Show", shows, key="ms_show_pick")

    one = df_m[df_m["canonical_title"] == pick_show].copy()
    one = one.sort_values("milestone").reset_index(drop=True)

    st.dataframe(
        one[["milestone", "week_ending", "cumulative_gross_millions", "week_gross_millions"]],
        width='stretch',
        hide_index=True,
    )

    if not one.empty:
        pick_m = st.selectbox(
            "Jump to a milestone for this show",
            options=one["milestone"].tolist(),
            format_func=fmt_int,
            key="ms_show_jump",
        )
        r = one[one["milestone"] == pick_m].iloc[0]
        st.success(
            f"**{pick_show}** first reached **{fmt_int(pick_m)}** on **{r['week_ending']}** "
            f"(cumulative: **{r['cumulative_gross_millions']:.1f}**, that week: **{r['week_gross_millions']:.1f}**)."
        )

    st.divider()

    # -------------------------
    # Section 2: Big milestone → shows (10k/20k/30k club)
    # -------------------------
    st.markdown("### Big milestone → shows (10k club)")
    big_milestones = sorted([m for m in df_m["milestone"].unique().tolist() if int(m) % 10000 == 0])
    if not big_milestones:
        st.info("No shows have reached 10,000 gross_millions yet.")
        return

    pick_big = st.selectbox(
        "Big milestone",
        options=big_milestones,
        format_func=fmt_int,
        key="ms_big_pick",
    )

    hit = df_m[df_m["milestone"] == pick_big].copy()
    hit["week_ending_dt"] = pd.to_datetime(hit["week_ending"], errors="coerce")
    hit = hit.sort_values(["week_ending_dt", "canonical_title"]).reset_index(drop=True)

    st.caption(f"Shows that first reached {fmt_int(pick_big)} gross_millions (earliest first).")
    st.dataframe(
        hit[["canonical_title", "week_ending", "cumulative_gross_millions", "week_gross_millions"]],
        width='stretch',
        hide_index=True,
    )
    st.write(f"Count: **{len(hit):,}**")
# ----------------------------
# New tab: Grossing Trends
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_grossing_trends_base(db_path: str, db_mtime: float) -> pd.DataFrame:
    """Load weekly chart rows needed for trend analysis.
    - Base gross from t10_entry.gross_millions
    - Optional bonus from gross_bonus (annual/quarter) is included as a separate column
    db_mtime busts cache when DB changes.
    """
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            WITH bonus AS (
              SELECT
                date(week_ending) AS week_ending,
                show_id,
                SUM(COALESCE(bonus_millions, 0.0)) AS bonus_millions
              FROM gross_bonus
              WHERE bonus_type IN ('annual', 'quarter')
              GROUP BY 1, 2
            )
            SELECT
              date(e.week_ending) AS week_ending,
              e.show_id AS show_id,
              e.rank AS rank,
              s.canonical_title AS canonical_title,
              COALESCE(e.gross_millions, 0.0) AS base_gross_millions,
              COALESCE(b.bonus_millions, 0.0) AS bonus_millions,
              (COALESCE(e.gross_millions, 0.0) + COALESCE(b.bonus_millions, 0.0)) AS gross_millions,
              COALESCE(e.imprint_1, '(Unknown)') AS imprint_1,
              COALESCE(e.imprint_2, '') AS imprint_2
            FROM t10_entry e
            JOIN show s
              ON s.show_id = e.show_id
            LEFT JOIN bonus b
              ON b.week_ending = date(e.week_ending)
             AND b.show_id = e.show_id
            """,
            con,
        )
    finally:
        con.close()

    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["imprint_1"] = df["imprint_1"].fillna("(Unknown)").astype(str).str.strip()
    df["imprint_2"] = df["imprint_2"].fillna("").astype(str).str.strip()
    return df


def tab_grossing_trends():
    st.subheader("Grossing Trends")
    st.caption(
        "Seasonality, rank/gross structure, volatility, momentum, longevity, concentration/turnover, imprint trends, and anomalies. "
        "Use the filters once, and everything below updates."
    )

    db_mtime = DB_PATH.stat().st_mtime
    base = _load_grossing_trends_base(str(DB_PATH), db_mtime)

    # Default: grossing era only
    base = base[base["week_ending_dt"].notna()].copy()

    # --- Filters ---
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.2])

        min_dt = base["week_ending_dt"].min().date()
        max_dt = base["week_ending_dt"].max().date()

        with c1:
            date_range = st.date_input(
                "Date range",
                value=(max(min_dt, GROSS_TRACKING_START), max_dt),
                min_value=min_dt,
                max_value=max_dt,
                key="trends_date_range",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                d0, d1 = date_range
            else:
                d0, d1 = max(min_dt, GROSS_TRACKING_START), max_dt

        with c2:
            grossing_era_only = st.checkbox(
                f"Grossing era only (≥ {GROSS_TRACKING_START.isoformat()})",
                value=True,
                key="trends_grossing_era_only",
            )
            include_bonuses = st.checkbox("Include annual/quarter bonuses", value=False, key="trends_include_bonuses")

        with c3:
            rank_scope = st.selectbox(
                "Rank scope (for totals & shares)",
                ["Top 10", "Top 5", "Top 3", "#1 only"],
                index=0,
                key="trends_rank_scope",
            )
            min_weeks = st.number_input("Min chart weeks for show-level stats", min_value=1, max_value=200, value=5, step=1, key="trends_min_weeks")

        with c4:
            # Imprint/company filter (matches imprint_1 or imprint_2)
            imprints = sorted(set(base["imprint_1"].dropna().astype(str)) | set(base["imprint_2"].dropna().astype(str)))
            imprints = [i for i in imprints if i and i != "nan"]
            selected_imprints = st.multiselect("Filter by imprint/company (optional)", imprints, default=[], key="trends_imprints")

    df = base.copy()
    if grossing_era_only:
        df = df[df["week_ending_dt"].dt.date >= GROSS_TRACKING_START].copy()

    df = df[(df["week_ending_dt"].dt.date >= d0) & (df["week_ending_dt"].dt.date <= d1)].copy()

    if selected_imprints:
        df = df[df["imprint_1"].isin(selected_imprints) | df["imprint_2"].isin(selected_imprints)].copy()

    gross_col = "gross_millions" if include_bonuses else "base_gross_millions"
    df["gross_use"] = pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)

    # rank scope helper
    rank_max = 10
    if rank_scope == "Top 5":
        rank_max = 5
    elif rank_scope == "Top 3":
        rank_max = 3
    elif rank_scope == "#1 only":
        rank_max = 1

    df_top = df[df["rank"].between(1, rank_max)].copy()
    df_top10 = df[df["rank"].between(1, 10)].copy()  # used in several sections regardless of rank_scope
    df_n1 = df[df["rank"].eq(1)].copy()

    # A stable, chart-ordered list of weeks
    chart_weeks = pd.Series(sorted(df["week_ending_dt"].dropna().unique()))
    if chart_weeks.empty:
        st.info("No rows match the current filters.")
        return

    through_week = st.selectbox(
        "Through week ending (applies to Momentum & some tables)",
        [d.date() for d in chart_weeks],
        index=len(chart_weeks) - 1,
        key="trends_through_week",
    )
    df_through = df[df["week_ending_dt"].dt.date <= through_week].copy()
    df_top_through = df_through[df_through["rank"].between(1, rank_max)].copy()
    df_top10_through = df_through[df_through["rank"].between(1, 10)].copy()
    df_n1_through = df_through[df_through["rank"].eq(1)].copy()

    trends_section = st.selectbox(
        "Grossing Trends section",
        ["Overview", "Momentum", "Longevity", "Market Structure", "Imprints/Companies", "Anomalies & Eras"],
        index=0,
        key="grossing_trends_section",
    )

    # ----------------------------
    # Overview
    # ----------------------------
    if trends_section == "Overview":
        st.markdown("### Seasonality")

        weekly_total = df_top10.groupby("week_ending_dt", as_index=True)["gross_use"].sum().sort_index()
        if weekly_total.empty:
            st.info("No gross data in the selected range.")
        else:
            season_df = weekly_total.reset_index().rename(columns={"gross_use": "total_gross"})
            season_df["month"] = season_df["week_ending_dt"].dt.month
            season_df["week_of_year"] = season_df["week_ending_dt"].dt.isocalendar().week.astype(int)

            month_stats = (
                season_df.groupby("month")["total_gross"]
                .agg(avg="mean", median="median", n="count")
                .reset_index()
            )
            month_stats["month_name"] = pd.to_datetime(month_stats["month"], format="%m").dt.strftime("%B")

            fig = plt.figure()
            plt.bar(month_stats["month_name"], month_stats["avg"])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Average total gross (Top 10)")
            plt.title("Average total gross by month-of-year")
            st.pyplot(fig, clear_figure=True)

            woy_stats = season_df.groupby("week_of_year")["total_gross"].mean().reset_index()
            fig = plt.figure()
            plt.plot(woy_stats["week_of_year"], woy_stats["total_gross"])
            plt.xlabel("ISO week of year")
            plt.ylabel("Average total gross (Top 10)")
            plt.title("Average total gross by week-of-year")
            st.pyplot(fig, clear_figure=True)

            with st.expander("Seasonality table"):
                show_tbl = month_stats[["month_name", "avg", "median", "n"]].rename(columns={"month_name": "Month", "avg": "Avg", "median": "Median", "n": "Weeks"})
                st.dataframe(show_tbl, width='stretch')

        st.markdown("### Position decay curves (gross by rank)")
        if df_top10.empty:
            st.info("No ranked rows in the selected range.")
        else:
            pos_stats = (
                df_top10.groupby("rank")["gross_use"]
                .agg(median="median", avg="mean", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), n="count")
                .reset_index()
                .sort_values("rank")
            )

            fig = plt.figure()
            plt.plot(pos_stats["rank"], pos_stats["median"], marker="o")
            plt.xlabel("Rank")
            plt.ylabel("Median weekly gross")
            plt.title("Median gross by rank (Top 10)")
            st.pyplot(fig, clear_figure=True)

            with st.expander("Position decay table"):
                st.dataframe(pos_stats, width='stretch')

        st.markdown("### Volatility index")
        if not weekly_total.empty:
            roll_w = st.slider("Rolling window (weeks)", min_value=4, max_value=26, value=13, step=1, key="trends_vol_window")
            roll_mean = weekly_total.rolling(roll_w, min_periods=max(2, roll_w // 2)).mean()
            roll_std = weekly_total.rolling(roll_w, min_periods=max(2, roll_w // 2)).std()
            vol_score = (roll_std / roll_mean.replace(0, np.nan)).fillna(0.0)

            fig = plt.figure()
            plt.plot(weekly_total.index, weekly_total.values, label="Weekly total")
            plt.plot(roll_mean.index, roll_mean.values, label=f"{roll_w}w avg")
            plt.legend()
            plt.title("Weekly total gross (Top 10)")
            st.pyplot(fig, clear_figure=True)

            fig = plt.figure()
            plt.plot(roll_std.index, roll_std.values)
            plt.title(f"Rolling {roll_w}-week std dev (volatility)")
            plt.ylabel("Std dev")
            st.pyplot(fig, clear_figure=True)

            fig = plt.figure()
            plt.plot(vol_score.index, vol_score.values)
            plt.title(f"Volatility score (std/mean) over {roll_w} weeks")
            st.pyplot(fig, clear_figure=True)

        st.markdown("### #1 premium (#1 vs #2)")
        premium_tbl = _build_number_one_premium_table(df, "gross_use")
        if premium_tbl.empty:
            st.info("Not enough data for #1 vs #2.")
        else:
            fig = plt.figure()
            plt.plot(premium_tbl["week_ending_dt"], premium_tbl["premium_diff"])
            plt.title("#1 premium (difference: #1 − #2)")
            st.pyplot(fig, clear_figure=True)

            fig = plt.figure()
            plt.plot(premium_tbl["week_ending_dt"], premium_tbl["premium_ratio"])
            plt.title("#1 premium (ratio: #1 ÷ #2)")
            st.pyplot(fig, clear_figure=True)

    # ----------------------------
    # Momentum
    # ----------------------------
    if trends_section == "Momentum":
        st.markdown("### Momentum")
        st.caption("Biggest week-over-week gains/declines, 4-week moves, hot slopes, and rebounders. Uses consecutive chart appearances (date gaps ignored).")

        show_week = (
            df_top10_through.groupby(["show_id", "canonical_title", "week_ending_dt"], as_index=False)["gross_use"]
            .sum()
            .sort_values(["show_id", "week_ending_dt"])
        )
        if show_week.empty:
            st.info("No show/week data for Momentum under current filters.")
        else:
            # Per-show deltas across consecutive chart appearances
            show_week["prev_gross"] = show_week.groupby("show_id")["gross_use"].shift(1)
            show_week["abs_change"] = show_week["gross_use"] - show_week["prev_gross"]
            show_week["pct_change"] = np.where(
                show_week["prev_gross"].fillna(0.0) > 0,
                show_week["abs_change"] / show_week["prev_gross"].replace(0, np.nan),
                np.nan,
            )

            min_prev = st.number_input("Min previous-week gross for % change leaderboard", min_value=0.0, value=1.0, step=0.5, key="trends_min_prev_gross")
            topn = st.slider("Top N", min_value=5, max_value=50, value=20, step=5, key="trends_mom_topn")

            wo = show_week[show_week["prev_gross"] >= float(min_prev)].copy()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Biggest week-over-week % gains")
                tbl = wo.sort_values("pct_change", ascending=False).head(topn)[
                    ["canonical_title", "week_ending_dt", "prev_gross", "gross_use", "pct_change"]
                ].copy()
                tbl["week_ending_dt"] = tbl["week_ending_dt"].dt.date
                st.dataframe(tbl, width='stretch')

                st.markdown("#### Biggest week-over-week $ gains")
                tbl = show_week.sort_values("abs_change", ascending=False).head(topn)[
                    ["canonical_title", "week_ending_dt", "prev_gross", "gross_use", "abs_change"]
                ].copy()
                tbl["week_ending_dt"] = tbl["week_ending_dt"].dt.date
                st.dataframe(tbl, width='stretch')

            with c2:
                st.markdown("#### Biggest week-over-week drops")
                tbl = show_week.sort_values("abs_change", ascending=True).head(topn)[
                    ["canonical_title", "week_ending_dt", "prev_gross", "gross_use", "abs_change"]
                ].copy()
                tbl["week_ending_dt"] = tbl["week_ending_dt"].dt.date
                st.dataframe(tbl, width='stretch')

                st.markdown("#### Biggest 4-appearance moves (net change)")
                show_week["gross_4_ago"] = show_week.groupby("show_id")["gross_use"].shift(4)
                show_week["net_4"] = show_week["gross_use"] - show_week["gross_4_ago"]
                tbl = show_week.dropna(subset=["gross_4_ago"]).sort_values("net_4", ascending=False).head(topn)[
                    ["canonical_title", "week_ending_dt", "gross_4_ago", "gross_use", "net_4"]
                ].copy()
                tbl["week_ending_dt"] = tbl["week_ending_dt"].dt.date
                st.dataframe(tbl, width='stretch')

            st.markdown("#### Hot shows right now (slope over last N appearances)")
            window_n = st.slider("N appearances", min_value=3, max_value=20, value=6, step=1, key="trends_hot_n")
            slopes = []
            for (sid, title), g in show_week.groupby(["show_id", "canonical_title"]):
                if len(g) < window_n:
                    continue
                gg = g.tail(window_n).copy()
                y = gg["gross_use"].to_numpy()
                x = np.arange(len(y), dtype=float)
                # simple slope
                try:
                    slope = np.polyfit(x, y, 1)[0]
                except Exception:
                    continue
                slopes.append({"show_id": sid, "canonical_title": title, "slope": slope, "last_week": gg["week_ending_dt"].iloc[-1].date()})
            slope_df = pd.DataFrame(slopes).sort_values("slope", ascending=False).head(topn)
            st.dataframe(slope_df, width='stretch')

            st.markdown("#### Rebounders (big drop then recovery next appearance)")
            drop_thr = st.number_input("Drop threshold (absolute)", min_value=0.0, value=5.0, step=0.5, key="trends_drop_thr")
            rebound_thr = st.number_input("Rebound threshold (absolute)", min_value=0.0, value=5.0, step=0.5, key="trends_rebound_thr")

            rebounds = []
            for (sid, title), g in show_week.groupby(["show_id", "canonical_title"]):
                g = g.sort_values("week_ending_dt").reset_index(drop=True)
                for i in range(1, len(g)):
                    if pd.isna(g.loc[i, "abs_change"]) or pd.isna(g.loc[i-1, "abs_change"]):
                        continue
                    # drop happened at i (from i-1 -> i)
                    if g.loc[i, "abs_change"] <= -float(drop_thr):
                        # recovery at next step
                        if i + 1 < len(g) and g.loc[i + 1, "abs_change"] >= float(rebound_thr):
                            rebounds.append({
                                "canonical_title": title,
                                "drop_week": g.loc[i, "week_ending_dt"].date(),
                                "drop_change": g.loc[i, "abs_change"],
                                "rebound_week": g.loc[i + 1, "week_ending_dt"].date(),
                                "rebound_change": g.loc[i + 1, "abs_change"],
                            })
            reb_df = pd.DataFrame(rebounds).sort_values(["drop_week", "canonical_title"], ascending=[False, True]).head(topn)
            st.dataframe(reb_df, width='stretch')

    # ----------------------------
    # Longevity
    # ----------------------------
    if trends_section == "Longevity":
        st.markdown("### Longevity & lifecycle")
        st.caption("How shows typically rise/fall, time-to-peak, and half-life (time to 50% of peak).")

        show_week = (
            df_top10.groupby(["show_id", "canonical_title", "week_ending_dt"], as_index=False)["gross_use"]
            .sum()
            .sort_values(["show_id", "week_ending_dt"])
        )
        if show_week.empty:
            st.info("No show/week data under current filters.")
        else:
            # chart age
            show_week["chart_age"] = show_week.groupby("show_id").cumcount() + 1

            # filter to shows with >= min_weeks
            counts = show_week.groupby("show_id")["chart_age"].max()
            keep_ids = counts[counts >= int(min_weeks)].index
            sw = show_week[show_week["show_id"].isin(keep_ids)].copy()
            if sw.empty:
                st.info("No shows meet the 'min weeks' filter.")
            else:
                # Lifecycle curve (median + IQR by chart_age)
                life = (
                    sw.groupby("chart_age")["gross_use"]
                    .agg(median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), n="count")
                    .reset_index()
                )
                fig = plt.figure()
                plt.plot(life["chart_age"], life["median"])
                plt.fill_between(life["chart_age"], life["p25"], life["p75"], alpha=0.2)
                plt.xlabel("Chart week age")
                plt.ylabel("Gross")
                plt.title("Typical show lifecycle (median with IQR)")
                st.pyplot(fig, clear_figure=True)

                # Time to peak
                peak_rows = []
                for (sid, title), g in sw.groupby(["show_id", "canonical_title"]):
                    g = g.sort_values("chart_age")
                    peak_idx = g["gross_use"].idxmax()
                    row = g.loc[peak_idx]
                    peak_rows.append({"canonical_title": title, "peak_age": int(row["chart_age"]), "peak_gross": float(row["gross_use"]), "peak_week": row["week_ending_dt"].date()})
                peaks = pd.DataFrame(peak_rows)
                fig = plt.figure()
                bins = range(1, int(peaks["peak_age"].max()) + 2)
                plt.hist(peaks["peak_age"], bins=bins)
                plt.xlabel("Chart age at peak")
                plt.ylabel("Count of shows")
                plt.title("Time-to-peak distribution")
                st.pyplot(fig, clear_figure=True)

                with st.expander("Peak table (top 200 by peak gross)"):
                    st.dataframe(peaks.sort_values("peak_gross", ascending=False).head(200), width='stretch')

                # Half-life
                half_rows = []
                for (sid, title), g in sw.groupby(["show_id", "canonical_title"]):
                    g = g.sort_values("chart_age")
                    peak_g = g["gross_use"].max()
                    if peak_g <= 0:
                        continue
                    half = 0.5 * peak_g
                    # after (and including) peak, first time <= half
                    peak_age = int(g.loc[g["gross_use"].idxmax(), "chart_age"])
                    g_after = g[g["chart_age"] >= peak_age].copy()
                    hit = g_after[g_after["gross_use"] <= half]
                    if hit.empty:
                        continue
                    first = hit.iloc[0]
                    half_rows.append({"canonical_title": title, "peak_age": peak_age, "half_age": int(first["chart_age"]), "weeks_to_half": int(first["chart_age"]) - peak_age})
                half_df = pd.DataFrame(half_rows)
                if not half_df.empty:
                    fig = plt.figure()
                    plt.hist(half_df["weeks_to_half"], bins=range(0, int(half_df["weeks_to_half"].max()) + 2))
                    plt.xlabel("Weeks to fall to 50% of peak (after peak)")
                    plt.ylabel("Count of shows")
                    plt.title("Half-life distribution")
                    st.pyplot(fig, clear_figure=True)

                    with st.expander("Half-life table (top 200 slowest to decay)"):
                        st.dataframe(half_df.sort_values("weeks_to_half", ascending=False).head(200), width='stretch')

    # ----------------------------
    # Market Structure
    # ----------------------------
    if trends_section == "Context":
        st.markdown("### Concentration / dominance")
        wk_total = df_top10.groupby("week_ending_dt")["gross_use"].sum().sort_index()
        if wk_total.empty:
            st.info("No weekly totals.")
        else:
            wk_n1 = df_top10[df_top10["rank"].eq(1)].groupby("week_ending_dt")["gross_use"].sum()
            wk_top3 = df_top10[df_top10["rank"].le(3)].groupby("week_ending_dt")["gross_use"].sum()
            share1 = (wk_n1 / wk_total.replace(0, np.nan)).fillna(0.0)
            share3 = (wk_top3 / wk_total.replace(0, np.nan)).fillna(0.0)

            fig = plt.figure()
            plt.plot(share1.index, share1.values, label="Top #1 share")
            plt.plot(share3.index, share3.values, label="Top 3 share")
            plt.legend()
            plt.title("Share of total gross captured by #1 and Top 3")
            st.pyplot(fig, clear_figure=True)

            # HHI concentration across shows each week
            hhi_rows = []
            for wk, g in df_top10.groupby("week_ending_dt"):
                tot = g["gross_use"].sum()
                if tot <= 0:
                    continue
                shares = (g.groupby("show_id")["gross_use"].sum() / tot).to_numpy()
                hhi = float(np.sum(shares ** 2))
                hhi_rows.append({"week_ending_dt": wk, "hhi": hhi})
            hhi_df = pd.DataFrame(hhi_rows).sort_values("week_ending_dt")
            if not hhi_df.empty:
                fig = plt.figure()
                plt.plot(hhi_df["week_ending_dt"], hhi_df["hhi"])
                plt.title("HHI-style concentration index (higher = more dominated)")
                st.pyplot(fig, clear_figure=True)

        st.markdown("### Turnover")
        # New shows per week (first appearance in the FULL df, then filtered to current range)
        full_sw = df_top10.groupby(["show_id"])["week_ending_dt"].min()
        df_first = full_sw.reset_index().rename(columns={"week_ending_dt": "first_week"})
        # Join back to get title for display
        titles = df_top10.drop_duplicates("show_id")[["show_id", "canonical_title"]]
        df_first = df_first.merge(titles, on="show_id", how="left")

        new_counts = df_first.groupby("first_week").size().reset_index(name="new_shows").sort_values("first_week")
        # only show within filter range
        new_counts = new_counts[(new_counts["first_week"].dt.date >= d0) & (new_counts["first_week"].dt.date <= d1)].copy()

        fig = plt.figure()
        plt.plot(new_counts["first_week"], new_counts["new_shows"])
        plt.title("New shows entering Top 10 (first-ever appearance) per week")
        st.pyplot(fig, clear_figure=True)

        with st.expander("Newest shows (top 200)"):
            newest = df_first.sort_values("first_week", ascending=False).head(200)
            newest["first_week"] = newest["first_week"].dt.date
            st.dataframe(newest[["canonical_title", "first_week"]], width='stretch')

    # ----------------------------
    # Imprints / Companies
    # ----------------------------
    if trends_section == "Anomalies":
        st.markdown("### Imprints / Companies")
        st.caption("Share, momentum, and hit rate. By default, assigns gross to imprint_1 to avoid double-counting.")

        mode = st.selectbox("Attribution mode", ["imprint_1 (no double count)", "imprint_2 (secondary only)", "split imprint_1 + imprint_2"], index=0, key="trends_imprint_mode")
        topn = st.slider("Top N imprints to chart", min_value=3, max_value=15, value=8, step=1, key="trends_imprint_topn")

        rows = []
        for _, r in df_top10.iterrows():
            g = float(r["gross_use"])
            i1 = (r.get("imprint_1") or "(Unknown)").strip() if isinstance(r.get("imprint_1"), str) else "(Unknown)"
            i2 = (r.get("imprint_2") or "").strip() if isinstance(r.get("imprint_2"), str) else ""
            wk = r["week_ending_dt"]
            sid = r["show_id"]
            title = r["canonical_title"]

            if mode.startswith("imprint_1"):
                rows.append({"week_ending_dt": wk, "imprint": i1, "gross": g, "show_id": sid, "canonical_title": title})
            elif mode.startswith("imprint_2"):
                rows.append({"week_ending_dt": wk, "imprint": i2 or "(None)", "gross": g, "show_id": sid, "canonical_title": title})
            else:
                if i2:
                    rows.append({"week_ending_dt": wk, "imprint": i1, "gross": g * 0.5, "show_id": sid, "canonical_title": title})
                    rows.append({"week_ending_dt": wk, "imprint": i2, "gross": g * 0.5, "show_id": sid, "canonical_title": title})
                else:
                    rows.append({"week_ending_dt": wk, "imprint": i1, "gross": g, "show_id": sid, "canonical_title": title})

        imp = pd.DataFrame(rows)
        if imp.empty:
            st.info("No imprint-attributed rows.")
        else:
            totals = imp.groupby("imprint")["gross"].sum().sort_values(ascending=False)
            top_imps = list(totals.head(int(topn)).index)
            imp["bucket"] = np.where(imp["imprint"].isin(top_imps), imp["imprint"], "Other")

            ts = imp.groupby(["week_ending_dt", "bucket"])["gross"].sum().reset_index()
            # pivot for plotting
            piv = ts.pivot(index="week_ending_dt", columns="bucket", values="gross").fillna(0.0).sort_index()

            fig = plt.figure()
            for col in piv.columns:
                plt.plot(piv.index, piv[col], label=str(col))
            plt.legend(loc="upper left", ncol=2)
            plt.title("Weekly gross by imprint (Top N + Other)")
            st.pyplot(fig, clear_figure=True)

            share_tbl = (totals / totals.sum()).reset_index()
            share_tbl.columns = ["Imprint", "Share"]
            share_tbl["TotalGross"] = totals.reset_index(drop=True)
            st.dataframe(share_tbl.head(50), width='stretch')

            st.markdown("#### Imprint momentum (last 13 weeks vs prior 13)")
            w = 13
            weekly_imp = imp.groupby(["week_ending_dt", "imprint"])["gross"].sum().reset_index()
            all_weeks = sorted(weekly_imp["week_ending_dt"].unique())
            if len(all_weeks) >= 2 * w:
                end = all_weeks[-1]
                last_start = all_weeks[-w]
                prev_start = all_weeks[-2*w]
                last = weekly_imp[(weekly_imp["week_ending_dt"] >= last_start) & (weekly_imp["week_ending_dt"] <= end)].groupby("imprint")["gross"].sum()
                prev = weekly_imp[(weekly_imp["week_ending_dt"] >= prev_start) & (weekly_imp["week_ending_dt"] < last_start)].groupby("imprint")["gross"].sum()
                mom = pd.DataFrame({"last13": last, "prev13": prev}).fillna(0.0)
                mom["change"] = mom["last13"] - mom["prev13"]
                mom = mom.sort_values("change", ascending=False).reset_index()
                st.dataframe(mom.head(50), width='stretch')
            else:
                st.info("Not enough weeks for 13+13 momentum window.")

            st.markdown("#### Hit rate")
            # For hit rate, use imprint assignment on weekly #1 and top3
            top3 = df[df["rank"].le(3)].copy()
            top1 = df[df["rank"].eq(1)].copy()

            def _assign_imprint(dfx: pd.DataFrame) -> pd.DataFrame:
                out = []
                for _, rr in dfx.iterrows():
                    i1 = (rr.get("imprint_1") or "(Unknown)").strip() if isinstance(rr.get("imprint_1"), str) else "(Unknown)"
                    i2 = (rr.get("imprint_2") or "").strip() if isinstance(rr.get("imprint_2"), str) else ""
                    if mode.startswith("imprint_1"):
                        out.append({"imprint": i1, "show_id": rr["show_id"]})
                    elif mode.startswith("imprint_2"):
                        out.append({"imprint": i2 or "(None)", "show_id": rr["show_id"]})
                    else:
                        if i2:
                            out.append({"imprint": i1, "show_id": rr["show_id"]})
                            out.append({"imprint": i2, "show_id": rr["show_id"]})
                        else:
                            out.append({"imprint": i1, "show_id": rr["show_id"]})
                return pd.DataFrame(out)

            all_imp = _assign_imprint(df_top10)
            t3_imp = _assign_imprint(top3)
            t1_imp = _assign_imprint(top1)

            hits = pd.DataFrame({
                "entries": all_imp.groupby("imprint").size(),
                "top3_entries": t3_imp.groupby("imprint").size(),
                "n1_entries": t1_imp.groupby("imprint").size(),
            }).fillna(0.0)
            hits["top3_rate"] = hits["top3_entries"] / hits["entries"].replace(0, np.nan)
            hits["n1_rate"] = hits["n1_entries"] / hits["entries"].replace(0, np.nan)
            hits = hits.fillna(0.0).sort_values("entries", ascending=False).reset_index()
            st.dataframe(hits.head(50), width='stretch')

    # ----------------------------
    # Anomalies & Eras
    # ----------------------------
    if trends_section == "Anomalies & Eras":
        st.markdown("### Outlier weeks")
        weekly_total = df_top10.groupby("week_ending_dt")["gross_use"].sum().sort_index()
        if weekly_total.empty:
            st.info("No weekly totals.")
        else:
            w = st.slider("Rolling window for outliers (weeks)", min_value=4, max_value=26, value=13, step=1, key="trends_outlier_window")
            zthr = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=2.5, step=0.1, key="trends_zthr")

            mu = weekly_total.rolling(w, min_periods=max(2, w//2)).mean()
            sd = weekly_total.rolling(w, min_periods=max(2, w//2)).std().replace(0, np.nan)
            z = ((weekly_total - mu) / sd).replace([np.inf, -np.inf], np.nan)

            out = pd.DataFrame({"week": weekly_total.index, "total_gross": weekly_total.values, "z": z.values}).dropna(subset=["z"])
            out = out[np.abs(out["z"]) >= float(zthr)].sort_values("z", ascending=False)

            fig = plt.figure()
            plt.plot(weekly_total.index, weekly_total.values)
            plt.title("Weekly total gross (Top 10)")
            st.pyplot(fig, clear_figure=True)

            if out.empty:
                st.info("No outliers at the current threshold/window.")
            else:
                # add top contributors (top 3 shows that week)
                contrib = []
                for wk in out["week"].head(50):
                    g = df_top10[df_top10["week_ending_dt"].eq(wk)].sort_values("gross_use", ascending=False)
                    tops = g.head(3)[["canonical_title", "gross_use"]].to_records(index=False)
                    contrib.append(", ".join([f"{t} ({v:.2f})" for t, v in tops]))
                out_disp = out.head(50).copy()
                out_disp["week"] = out_disp["week"].dt.date
                out_disp["top_contributors"] = contrib + [""] * max(0, len(out_disp) - len(contrib))
                st.dataframe(out_disp, width='stretch')

        st.markdown("### Era detection (heuristic change points)")
        st.caption("Detects large sustained shifts in average total gross using a lookback/lookahead window. This is a heuristic, not a statistical guarantee.")

        weekly_total = df_top10.groupby("week_ending_dt")["gross_use"].sum().sort_index()
        if len(weekly_total) < 60:
            st.info("Need at least ~60 weeks in the selected range for era detection.")
        else:
            look = st.slider("Lookback/lookahead window (weeks)", min_value=8, max_value=52, value=26, step=1, key="trends_era_look")
            thr = st.slider("Shift threshold (%)", min_value=5, max_value=50, value=15, step=1, key="trends_era_thr")

            vals = weekly_total.values
            idx = weekly_total.index.to_list()
            boundaries = []
            last_b = 0
            for i in range(look, len(vals) - look):
                if i - last_b < look:
                    continue
                back = np.mean(vals[i - look : i])
                fwd = np.mean(vals[i : i + look])
                if back <= 0:
                    continue
                pct = abs(fwd - back) / back * 100.0
                if pct >= float(thr):
                    boundaries.append(i)
                    last_b = i

            # Build eras
            cuts = [0] + boundaries + [len(vals)]
            eras = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a < max(6, look // 2):
                    continue
                seg = vals[a:b]
                eras.append({
                    "start_week": idx[a].date(),
                    "end_week": idx[b-1].date(),
                    "weeks": b - a,
                    "avg_total": float(np.mean(seg)),
                    "median_total": float(np.median(seg)),
                    "volatility": float(np.std(seg)),
                })
            eras_df = pd.DataFrame(eras)

            fig = plt.figure()
            plt.plot(idx, vals)
            for bi in boundaries:
                plt.axvline(idx[bi], linestyle="--")
            plt.title("Weekly total gross with detected era boundaries")
            st.pyplot(fig, clear_figure=True)

            if eras_df.empty:
                st.info("No eras detected at current settings.")
            else:
                st.dataframe(eras_df, width='stretch')

def _show_years(show_id: int) -> list[str]:
    df = sql_df(
        "SELECT DISTINCT strftime('%Y', week_ending) AS y FROM t10_entry WHERE show_id=? ORDER BY y",
        (int(show_id),),
    )
    if df is None or df.empty:
        return []
    ys = [str(y) for y in df["y"].dropna().astype(str).tolist()]
    return ys


@st.cache_data(show_spinner=False)
def _fetch_show_trends_rows(show_id: int, year: str | None) -> pd.DataFrame:
    """Base rows for a single show (chart weeks only)."""
    params: list[Any] = [int(show_id)]
    year_clause = ""
    if year:
        year_clause = " AND strftime('%Y', e.week_ending) = ?"
        params.append(str(year))

    df = sql_df(
        f"""
        SELECT
          e.week_number,
          date(e.week_ending) AS week_ending,
          e.rank,
          e.pos,
          e.last_week,
          s.canonical_title,
          e.imprint_1,
          e.imprint_2,
          COALESCE(e.gross_millions, 0) AS base_gross_millions,
          COALESCE(gb.bonus_millions, 0) AS bonus_millions,
          (COALESCE(e.gross_millions, 0) + COALESCE(gb.bonus_millions, 0)) AS gross_millions
        FROM t10_entry e
        JOIN show s ON s.show_id = e.show_id
        LEFT JOIN (
          SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
          FROM gross_bonus
          GROUP BY show_id, week_ending
        ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
        WHERE e.show_id = ?
        {year_clause}
        ORDER BY e.week_number ASC, date(e.week_ending) ASC, e.rank ASC, e.pos ASC
        """,
        tuple(params),
    )
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "week_number",
                "week_ending",
                "rank",
                "pos",
                "last_week",
                "canonical_title",
                "imprint_1",
                "imprint_2",
                "base_gross_millions",
                "bonus_millions",
                "gross_millions",
            ]
        )
    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["base_gross_millions"] = pd.to_numeric(df["base_gross_millions"], errors="coerce").fillna(0.0)
    df["bonus_millions"] = pd.to_numeric(df["bonus_millions"], errors="coerce").fillna(0.0)
    df["gross_millions"] = pd.to_numeric(df["gross_millions"], errors="coerce").fillna(0.0)
    return df


@st.cache_data(show_spinner=False)
def _fetch_top10_totals_by_week(year: str | None) -> pd.DataFrame:
    """Top-10 total gross by week (includes bonuses)."""
    params: list[Any] = []
    year_clause = ""
    if year:
        year_clause = "WHERE strftime('%Y', e.week_ending) = ?"
        params.append(str(year))

    df = sql_df(
        f"""
        WITH bonus_by_row AS (
          SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
          FROM gross_bonus
          GROUP BY show_id, week_ending
        )
        SELECT
          date(e.week_ending) AS week_ending,
          SUM(COALESCE(e.gross_millions,0) + COALESCE(b.bonus_millions,0)) AS top10_gross_millions
        FROM t10_entry e
        LEFT JOIN bonus_by_row b ON b.show_id = e.show_id AND b.week_ending = e.week_ending
        {year_clause}
        AND e.rank BETWEEN 1 AND 10
        GROUP BY date(e.week_ending)
        ORDER BY date(e.week_ending) ASC
        """,
        tuple(params),
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["week_ending", "top10_gross_millions"])
    df["week_ending"] = _as_date_str(df["week_ending"])
    df["top10_gross_millions"] = pd.to_numeric(df["top10_gross_millions"], errors="coerce").fillna(0.0)
    return df


def _longest_run_masked(df: pd.DataFrame, mask: pd.Series) -> dict[str, Any] | None:
    """Return longest consecutive run where mask==True; tie-friendly if mask defines membership."""
    if df.empty or mask is None or len(df) == 0:
        return None

    g = df.sort_values(["week_number", "week_ending_dt"]).reset_index(drop=True).copy()
    mask = mask.reset_index(drop=True).fillna(False)

    # consecutive logic
    if g["week_number"].notna().all():
        cont = _consecutive_by_week_number(g["week_number"])
    else:
        cont = _consecutive_by_date(g["week_ending"])

    best = {"len": 0, "start": None, "end": None}
    cur_len = 0
    cur_start = None

    for i in range(len(g)):
        if bool(mask.iloc[i]):
            if cur_len == 0:
                cur_len = 1
                cur_start = g.loc[i, "week_ending"]
            else:
                if bool(cont.iloc[i]):
                    cur_len += 1
                else:
                    # break run; start new
                    cur_len = 1
                    cur_start = g.loc[i, "week_ending"]
            cur_end = g.loc[i, "week_ending"]
            if cur_len > best["len"]:
                best = {"len": int(cur_len), "start": cur_start, "end": cur_end}
        else:
            cur_len = 0
            cur_start = None

    return best if best["len"] > 0 else None


def tab_show_trends():
    st.markdown("### Show Trends")
    st.caption("Grossing + rank trends for a single show over time. (Altair charts; ties supported.)")

    # Shrink Show Trends KPI typography so the 6-metric row fits on smaller screens/window widths.
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            line-height: 1.05 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
            line-height: 1.0 !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-size: 0.75rem !important;
            line-height: 1.0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    shows_df, _ = load_lists()
    if shows_df is None or shows_df.empty:
        st.info("No shows found.")
        return

    # Global controls
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])

    with c1:
        title = st.selectbox(
            "Show",
            options=shows_df["canonical_title"].astype(str).tolist(),
            index=0,
            key="show_trends_title",
        )
        show_id = int(shows_df.loc[shows_df["canonical_title"] == title, "show_id"].iloc[0])

    years = _show_years(show_id)
    with c2:
        year = st.selectbox("Year", ["All"] + years, index=0, key="show_trends_year")
        year_val = None if year == "All" else year

    with c3:
        include_bonuses = st.checkbox("Include bonuses", value=True, key="show_trends_bonus")

    with c4:
        smoothing = st.selectbox("Smoothing", ["None", "4-week MA", "8-week MA"], index=0, key="show_trends_smooth")

    df = _fetch_show_trends_rows(show_id, year_val)
    if df.empty:
        st.info("No chart rows for this selection.")
        return

    df["gross_use"] = df["gross_millions"] if include_bonuses else df["base_gross_millions"]

    # Smoothing
    win = 0
    if smoothing.startswith("4"):
        win = 4
    elif smoothing.startswith("8"):
        win = 8
    if win > 0:
        df["gross_ma"] = df["gross_use"].rolling(win, min_periods=1).mean()
    else:
        df["gross_ma"] = np.nan

    trends_section = st.selectbox(
        "Show Trends section",
        ["Trend", "Momentum", "Runs & peaks", "Context", "Anomalies"],
        index=0,
        key="show_trends_section",
    )

    # ----------------------------
    # Trend
    # ----------------------------
    if trends_section == "Trend":
        # KPIs
        peak_idx = int(df["gross_use"].idxmax())
        peak_week = df.loc[peak_idx, "week_ending"]
        peak_value = float(df.loc[peak_idx, "gross_use"])

        # Total gross should match Show Details / Gross Races when bonuses are included:
        # include bonus-only weeks from gross_bonus (weeks with no t10_entry row).
        if include_bonuses:
            _ledger = fetch_show_weekly_ledger(show_id)
            if _ledger is not None and not _ledger.empty:
                _ledger = _ledger.copy()
                _ledger["week_ending"] = _as_date_str(_ledger["week_ending"])
                if year_val is not None:
                    _ledger = _ledger[_ledger["week_ending"].astype(str).str[:4] == str(year_val)].copy()
                _ledger["gross_millions"] = pd.to_numeric(_ledger["gross_millions"], errors="coerce").fillna(0.0)
                total_gross = float(_ledger["gross_millions"].sum())
            else:
                total_gross = float(df["gross_use"].sum())
        else:
            total_gross = float(df["gross_use"].sum())

        weeks_charted = int(df["week_ending"].nunique())
        avg_gross = (total_gross / weeks_charted) if weeks_charted else 0.0
        n1_weeks = int((df["rank"] == 1).sum())
        top3_weeks = int((df["rank"] <= 3).sum())

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Peak week", f"{peak_week}")
        k2.metric("Peak gross", f"{peak_value:,.1f}")
        k3.metric("Total gross", f"{total_gross:,.1f}")
        k4.metric("Avg gross", f"{avg_gross:,.1f}")
        k5.metric("Weeks charted", f"{weeks_charted}")
        k6.metric("#1 / Top-3 weeks", f"{n1_weeks} / {top3_weeks}")

        # Line chart (gross over time + optional MA)
        plot_df = df.dropna(subset=["week_ending_dt"]).copy()
        base_line = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=alt.X("week_ending_dt:T", title="Week ending"),
                y=alt.Y("gross_use:Q", title="Gross (millions)"),
                tooltip=[
                    alt.Tooltip("week_ending:N", title="Week"),
                    alt.Tooltip("week_number:Q", title="Week #"),
                    alt.Tooltip("rank:Q", title="Rank"),
                    alt.Tooltip("pos:Q", title="Pos"),
                    alt.Tooltip("gross_use:Q", title="Gross", format=",.1f"),
                ],
            )
        )
        layers = [base_line]
        if win > 0:
            ma_line = (
                alt.Chart(plot_df)
                .mark_line(strokeDash=[4, 4])
                .encode(
                    x="week_ending_dt:T",
                    y=alt.Y("gross_ma:Q", title="Gross (millions)"),
                    tooltip=[
                        alt.Tooltip("week_ending:N", title="Week"),
                        alt.Tooltip("gross_ma:Q", title=f"{win}-wk MA", format=",.1f"),
                    ],
                )
            )
            layers.append(ma_line)

        st.altair_chart(alt.layer(*layers).interactive(), width='stretch')

        # Weekly detail table
        cols = [
            "week_number",
            "week_ending",
            "rank",
            "pos",
            "last_week",
            "canonical_title",
            "imprint_1",
            "imprint_2",
            "base_gross_millions",
            "bonus_millions",
            "gross_use",
        ]
        out = df[cols].copy()
        out = out.rename(columns={"gross_use": "gross_millions (selected)"})
        st.dataframe(out, width='stretch', hide_index=True)

    # ----------------------------
    # Momentum
    # ----------------------------
    if trends_section == "Momentum":
        g = df.sort_values(["week_number", "week_ending_dt"]).reset_index(drop=True).copy()
        g["prev_gross"] = g["gross_use"].shift(1)
        g["delta"] = g["gross_use"] - g["prev_gross"]
        g["delta_pct"] = np.where(g["prev_gross"] > 0, g["delta"] / g["prev_gross"], np.nan)

        def _top(df_in: pd.DataFrame, col: str, n: int = 10, asc: bool = False) -> pd.DataFrame:
            d2 = df_in.dropna(subset=[col]).copy()
            if d2.empty:
                return d2
            return d2.sort_values(col, ascending=asc).head(n)

        st.markdown("### Biggest week-over-week % gains")
        st.dataframe(
            _top(g[g["delta_pct"] > 0], "delta_pct", 10, asc=False)[
                ["week_number", "week_ending", "rank", "pos", "gross_use", "prev_gross", "delta_pct", "delta"]
            ],
            width='stretch',
            hide_index=True,
        )

        st.markdown("### Biggest week-over-week % drops")
        st.dataframe(
            _top(g[g["delta_pct"] < 0], "delta_pct", 10, asc=True)[
                ["week_number", "week_ending", "rank", "pos", "gross_use", "prev_gross", "delta_pct", "delta"]
            ],
            width='stretch',
            hide_index=True,
        )

        st.markdown("### Biggest absolute gains")
        st.dataframe(
            _top(g[g["delta"] > 0], "delta", 10, asc=False)[
                ["week_number", "week_ending", "rank", "pos", "gross_use", "prev_gross", "delta", "delta_pct"]
            ],
            width='stretch',
            hide_index=True,
        )

        st.markdown("### Biggest absolute drops")
        st.dataframe(
            _top(g[g["delta"] < 0], "delta", 10, asc=True)[
                ["week_number", "week_ending", "rank", "pos", "gross_use", "prev_gross", "delta", "delta_pct"]
            ],
            width='stretch',
            hide_index=True,
        )

        # Streaks: longest up/down streak (by delta sign)
        up_run = _longest_run_masked(g, g["delta"] > 0)
        down_run = _longest_run_masked(g, g["delta"] < 0)

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("### Longest up-streak")
            if up_run:
                st.write(f"{up_run['len']} weeks ({up_run['start']} → {up_run['end']})")
            else:
                st.write("—")
        with s2:
            st.markdown("### Longest down-streak")
            if down_run:
                st.write(f"{down_run['len']} weeks ({down_run['start']} → {down_run['end']})")
            else:
                st.write("—")

    # ----------------------------
    # Runs & peaks
    # ----------------------------
    if trends_section == "Runs & peaks":
        # Longest consecutive charted run (all rows)
        run_all = _longest_run_masked(df, pd.Series([True] * len(df)))
        if run_all:
            st.markdown("### Longest consecutive charted run")
            st.write(f"{run_all['len']} weeks ({run_all['start']} → {run_all['end']})")

        # Longest consecutive #1 run (tie-friendly: rank==1, pos irrelevant)
        run_n1 = _longest_run_masked(df, df["rank"] == 1)
        st.markdown("### Longest consecutive #1 run (tie-friendly)")
        if run_n1:
            st.write(f"{run_n1['len']} weeks ({run_n1['start']} → {run_n1['end']})")
        else:
            st.write("—")

        st.markdown("### Top 10 peak weeks (by gross)")
        top10 = df.sort_values("gross_use", ascending=False).head(10)[
            ["week_number", "week_ending", "rank", "pos", "base_gross_millions", "bonus_millions", "gross_use"]
        ].copy()
        st.dataframe(top10.rename(columns={"gross_use": "gross_millions (selected)"}), width='stretch', hide_index=True)

        # Aggregates
        g = df.dropna(subset=["week_ending_dt"]).copy()
        g["year"] = g["week_ending_dt"].dt.year

        st.markdown("### Top months")
        m = _apply_chart_month_logic(g, week_dt_col="week_ending_dt", week_str_col="week_ending")
        top_months = (
            m.groupby("month", as_index=False)["gross_use"].sum()
            .sort_values("gross_use", ascending=False)
            .head(12)
            .rename(columns={"gross_use": "gross_millions (selected)"})
        )
        st.dataframe(top_months, width='stretch', hide_index=True)

        st.markdown("### Top years")
        top_years = (
            g.groupby("year", as_index=False)["gross_use"].sum()
            .sort_values("gross_use", ascending=False)
            .rename(columns={"gross_use": "gross_millions (selected)"})
        )
        st.dataframe(top_years, width='stretch', hide_index=True)

    # ----------------------------
    # Context
    # ----------------------------
    if trends_section == "Context":
        # Share of Top 10 each week
        top10_tot = _fetch_top10_totals_by_week(year_val)
        ctx = df.merge(top10_tot, on="week_ending", how="left")
        ctx["top10_gross_millions"] = pd.to_numeric(ctx["top10_gross_millions"], errors="coerce").fillna(0.0)

        if include_bonuses:
            ctx["share_top10"] = np.where(ctx["top10_gross_millions"] > 0, ctx["gross_millions"] / ctx["top10_gross_millions"], np.nan)
        else:
            # If bonuses are excluded for the show, compare to Top-10 base only
            base_top10 = sql_df(
                f"""
                SELECT date(week_ending) AS week_ending, SUM(COALESCE(gross_millions,0)) AS top10_base_gross
                FROM t10_entry
                WHERE rank BETWEEN 1 AND 10
                {("AND strftime('%Y', week_ending) = ?" if year_val else "")}
                GROUP BY date(week_ending)
                ORDER BY date(week_ending) ASC
                """,
                tuple([year_val] if year_val else []),
            )
            if base_top10 is None or base_top10.empty:
                base_top10 = pd.DataFrame(columns=["week_ending", "top10_base_gross"])
            base_top10["week_ending"] = _as_date_str(base_top10["week_ending"])
            base_top10["top10_base_gross"] = pd.to_numeric(base_top10["top10_base_gross"], errors="coerce").fillna(0.0)
            ctx = ctx.drop(columns=["top10_gross_millions"]).merge(base_top10, on="week_ending", how="left")
            ctx["top10_base_gross"] = pd.to_numeric(ctx["top10_base_gross"], errors="coerce").fillna(0.0)
            ctx["share_top10"] = np.where(ctx["top10_base_gross"] > 0, ctx["base_gross_millions"] / ctx["top10_base_gross"], np.nan)

        st.markdown("### Share of Top 10 each week")
        share_plot = ctx.dropna(subset=["week_ending_dt"]).copy()
        share_plot["share_top10"] = pd.to_numeric(share_plot["share_top10"], errors="coerce")
        st.altair_chart(
            alt.Chart(share_plot)
            .mark_line()
            .encode(
                x=alt.X("week_ending_dt:T", title="Week ending"),
                y=alt.Y("share_top10:Q", title="Share of Top 10"),
                tooltip=[
                    alt.Tooltip("week_ending:N", title="Week"),
                    alt.Tooltip("share_top10:Q", title="Share", format=".3f"),
                ],
            )
            .interactive(),
            width='stretch',
        )

        st.markdown("### Rank vs gross")
        scatter = df.dropna(subset=["rank"]).copy()
        scatter["rank"] = pd.to_numeric(scatter["rank"], errors="coerce")
        scatter["gross_use"] = pd.to_numeric(scatter["gross_use"], errors="coerce")
        st.altair_chart(
            alt.Chart(scatter)
            .mark_circle()
            .encode(
                x=alt.X("gross_use:Q", title="Gross (millions)"),
                y=alt.Y("rank:Q", title="Rank", scale=alt.Scale(reverse=True)),
                tooltip=[
                    alt.Tooltip("week_ending:N", title="Week"),
                    alt.Tooltip("rank:Q", title="Rank"),
                    alt.Tooltip("gross_use:Q", title="Gross", format=",.1f"),
                ],
            )
            .interactive(),
            width='stretch',
        )


    # ----------------------------
    # Anomalies (show-level z-scores)
    # ----------------------------
    if trends_section == "Anomalies":
        st.markdown("### Anomalies (z-scores)")
        st.caption("Detect weeks where this show over- or under-performed versus its own recent history (rolling window).")

        method = st.selectbox(
            "Method",
            options=["Rolling z-score (mean/std)", "Rolling robust z-score (median/MAD)"],
            index=1,
            key="show_trends_anom_method",
        )
        window = st.slider("Rolling window (weeks)", 4, 52, 26, key="show_trends_anom_window")
        threshold = st.slider("Outlier threshold (|z|)", 1.0, 5.0, 2.5, 0.1, key="show_trends_anom_threshold")

        s = df[["week_ending", "rank", "gross_use"]].copy()
        s["week_dt"] = pd.to_datetime(s["week_ending"], errors="coerce")
        s["gross_use"] = pd.to_numeric(s["gross_use"], errors="coerce")

        minp = max(6, int(window // 2))

        if method.startswith("Rolling z-score"):
            mu = s["gross_use"].rolling(window, min_periods=minp).mean()
            sig = s["gross_use"].rolling(window, min_periods=minp).std(ddof=0)
            s["z"] = (s["gross_use"] - mu) / sig.replace(0, np.nan)
        else:
            med = s["gross_use"].rolling(window, min_periods=minp).median()

            def _mad(arr):
                arr = np.asarray(arr, dtype=float)
                arr = arr[~np.isnan(arr)]
                if arr.size == 0:
                    return np.nan
                m = np.median(arr)
                return np.median(np.abs(arr - m))

            mad = s["gross_use"].rolling(window, min_periods=minp).apply(_mad, raw=True)
            s["z"] = 0.6745 * (s["gross_use"] - med) / mad.replace(0, np.nan)

        s["is_outlier"] = s["z"].abs() >= float(threshold)

        base = alt.Chart(s.dropna(subset=["week_dt", "gross_use"])).encode(
            x=alt.X("week_dt:T", title="Week ending"),
            tooltip=[
                alt.Tooltip("week_ending:N", title="Week"),
                alt.Tooltip("rank:Q", title="Rank"),
                alt.Tooltip("gross_use:Q", title="Gross", format=",.1f"),
                alt.Tooltip("z:Q", title="z", format=".2f"),
            ],
        )

        line = base.mark_line().encode(y=alt.Y("gross_use:Q", title="Gross (millions)"))
        pts = (
            alt.Chart(s[s["is_outlier"]].dropna(subset=["week_dt", "gross_use"]))
            .mark_point(size=60)
            .encode(x=alt.X("week_dt:T"), y=alt.Y("gross_use:Q"))
        )

        st.altair_chart((line + pts).interactive(), width='stretch')

        out = s[s["is_outlier"]].copy()
        out["abs_z"] = out["z"].abs()
        out = out.sort_values("abs_z", ascending=False).drop(columns=["abs_z", "week_dt"])
        st.dataframe(out, width='stretch', hide_index=True)
def tab_search():
    st.subheader("Search")
    with st.sidebar:
        st.header("Search filters")
        fts = st.text_input("Full-text search (FTS)", placeholder="e.g. Nickelodeon AND (school OR kids)")
        date_min = st.text_input("Start date (YYYY-MM-DD)", value="", key="gt_date_min")
        date_max = st.text_input("End date (YYYY-MM-DD)", value="", key="gt_date_max")
        week_num = st.text_input("Week # (e.g. 2500 or 2400-2600)", value="", key="gt_week_num")
        rank_min, rank_max = st.slider("Rank range", 1, 17, (1, 10))
        limit = st.slider("Max results", 50, 10000, 1000, step=50)

    filters = FilterSpec(
        date_min=date_min.strip() or None,
        date_max=date_max.strip() or None,
        rank_min=int(rank_min),
        rank_max=int(rank_max),
    )
        # Parse Week # filter (optional)
    wk_min: int | None = None
    wk_max: int | None = None
    wk_txt = (week_num or "").strip()
    if wk_txt:
        try:
            if "-" in wk_txt:
                a, b = [p.strip() for p in wk_txt.split("-", 1)]
                wk_min = int(a)
                wk_max = int(b)
                if wk_min > wk_max:
                    wk_min, wk_max = wk_max, wk_min
            else:
                wk_min = int(wk_txt)
                wk_max = int(wk_txt)
        except Exception:
            st.warning("Week # must be an integer or a range like 2400-2600.")
            wk_min = None
            wk_max = None

    df = fetch_entries(filters, fts_query=fts, limit=int(limit), week_min=wk_min, week_max=wk_max)

    st.write(f"Results: **{len(df)}**")
    st.dataframe(df, width='stretch')

    if not df.empty:
        gross = df["gross_millions"].dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique dates", int(df["week_ending"].nunique()))
        c2.metric("Unique shows", int(df["canonical_title"].nunique()))
        c3.metric("Avg rank", float(df["rank"].mean()))
        c4.metric("Rows with gross", int(len(gross)))
        if len(gross):
            st.write({
                "sum gross (M)": float(gross.sum()),
                "avg gross (M)": float(gross.mean()),
                "min gross (M)": float(gross.min()),
                "max gross (M)": float(gross.max()),
            })


def tab_show_detail():
    st.subheader("Show detail")
    shows, _ = load_lists()

    colA, colB = st.columns([2, 3])
    with colA:
        show_name = st.selectbox("Show (canonical)", shows["canonical_title"].tolist())
    with colB:
        st.caption("Tip: use Admin → Alias/Merge if you have slight title variants that should be unified.")

    show_id = int(shows.loc[shows["canonical_title"] == show_name, "show_id"].iloc[0])

    with st.sidebar:
        st.header("Show detail filters")
        date_min = st.text_input("Start date (YYYY-MM-DD) ", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)  ", value="")
        rank_min, rank_max = st.slider("Rank range (show)", 1, 17, (1, 10))

    filters = FilterSpec(date_min.strip() or None, date_max.strip() or None, int(rank_min), int(rank_max))

    stats = fetch_show_stats(show_id)
    if not stats.empty:
        s = stats.iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)

        weeks_on = s.get("weeks_on_chart")
        peak_rank = s.get("peak_rank")
        first_app = s.get("first_appearance")
        last_app = s.get("last_appearance")

        c1.metric("Weeks on chart", 0 if pd.isna(weeks_on) else int(weeks_on))
        c2.metric("Peak rank", "—" if pd.isna(peak_rank) else int(peak_rank))
        c3.metric("First appearance", "—" if pd.isna(first_app) else str(first_app))
        c4.metric("Last appearance", "—" if pd.isna(last_app) else str(last_app))

        total_gross = s.get("total_gross_millions")
        avg_gross = s.get("avg_gross_millions")
        avg_rank = s.get("avg_rank")

        st.write({
            "Total gross (M)": 0.0 if pd.isna(total_gross) else float(total_gross),
            "Avg gross (M)": None if pd.isna(avg_gross) else float(avg_gross),
            "Avg rank": None if pd.isna(avg_rank) else float(avg_rank),
        })


    with st.expander("Weekly ledger (includes bonus-only weeks)"):
        led = fetch_show_weekly_ledger(show_id)
        if led.empty:
            st.caption("No ledger rows found for this show.")
        else:
            st.dataframe(led, width='stretch', hide_index=True)

    with st.expander("All-Time Gross Races rank history (every chart week since debut/era start)"):
            load_rank_history = st.checkbox(
                "Load all-time rank history",
                value=False,
                key=f"show_rankhist_load_{show_id}",
                help="This is one of the heaviest Show Detail calculations, especially on mobile.",
            )
            if not load_rank_history:
                st.caption("Turn on 'Load all-time rank history' to compute this section.")
            else:
                # We compute the show's all-time rank as-of *every* chart week in the grossing era,
            # because other shows can keep grossing after this show disappears from the weekly chart.
                hist_mode = st.radio(
                    "History view",
                    ["All weeks", "Rank changes only"],
                    index=0,
                    horizontal=True,
                    key=f"show_rankhist_mode_{show_id}",
                )
                db_mtime = DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0
                base = _load_gross_races_base(str(DB_PATH), db_mtime)
    
                if base.empty:
                    st.caption("No gross races base rows found (cannot compute all-time ranks).")
                else:
                    weeks = pd.to_datetime(base["week_ending"], errors="coerce").dropna().dt.normalize()
    
                    # Start the history at the show's debut (first chart week) if it debuted after the grossing era began.
                    # If it debuted before the grossing era, start at the grossing era start (2001-03-17).
                    debut_df = sql_df(
                        "SELECT MIN(date(week_ending)) AS debut FROM t10_entry WHERE show_id = ?;",
                        (int(show_id),),
                    )
                    debut_str = None
                    if debut_df is not None and not debut_df.empty:
                        debut_str = debut_df.loc[0, "debut"]
                    debut_ts = pd.to_datetime(debut_str, errors="coerce")
                    start_ts = pd.Timestamp(GROSS_TRACKING_START)
                    if debut_ts is not None and not pd.isna(debut_ts):
                        start_ts = max(start_ts, debut_ts.normalize())
    
                    weeks = weeks[weeks >= start_ts]
                    weeks = sorted(pd.unique(weeks).tolist())
    
                    # Apply the same date window as the Show detail filters (if set)
                    tmin = pd.to_datetime(filters.date_min, errors="coerce") if filters.date_min else None
                    tmax = pd.to_datetime(filters.date_max, errors="coerce") if filters.date_max else None
                    if tmin is not None and not pd.isna(tmin):
                        weeks = [w for w in weeks if w >= tmin.normalize()]
                    if tmax is not None and not pd.isna(tmax):
                        weeks = [w for w in weeks if w <= tmax.normalize()]
    
                    rt = _alltime_rank_table_for_show_weeks(str(DB_PATH), db_mtime, show_id, weeks)
    
                    if rt.empty:
                        st.caption("No all-time rank rows could be computed for the selected weeks.")
                    else:
                        rt2 = rt.copy()
                        rt2["rank"] = pd.to_numeric(rt2["rank"], errors="coerce").astype("Int64")
                        rt2["rank_change"] = pd.to_numeric(rt2["rank_change"], errors="coerce").astype("Int64")
                        rt2["total_gross_millions"] = pd.to_numeric(rt2["total_gross_millions"], errors="coerce")
                        if hist_mode == "Rank changes only":
                            # Keep the first row, then only weeks where the rank changed vs the prior week.
                            ch = pd.to_numeric(rt2["rank_change"], errors="coerce").fillna(0)
                            keep = (ch != 0)
                            if len(rt2) > 0:
                                keep.iloc[0] = True
                            rt2 = rt2.loc[keep].copy()
                        st.dataframe(rt2, width='stretch', hide_index=True)

    df = fetch_show_entries(show_id, filters)
    st.dataframe(df, width='stretch')

    if df.empty:
        st.info("No rows match your filters for this show.")
        return

    st.markdown("### Rank trajectory")
    plot_line_dates(df["week_ending"], df["rank"].astype(float), "Week Ending", "Rank", invert_y=True)

    st.markdown("### Gross over time")
    dg = df.dropna(subset=["gross_millions"]).copy()
    if dg.empty:
        st.info("No gross values for this show (within current filters).")
    else:
        plot_line_dates(dg["week_ending"], dg["gross_millions"].astype(float), "Week Ending", "Gross (Millions)")

    st.markdown("### Rank vs Gross (scatter)")
    if dg.empty:
        st.info("Need gross values to compute rank vs gross scatter.")
    else:
        plot_scatter(dg["rank"].astype(float), dg["gross_millions"].astype(float), "Rank", "Gross (Millions)")


def tab_compare_two_shows():
    st.subheader("Compare two shows")

    shows, _ = load_lists()
    titles = shows["canonical_title"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Show A", titles, index=0)
    with c2:
        b = st.selectbox("Show B", titles, index=1 if len(titles) > 1 else 0)

    with st.sidebar:
        st.header("Compare filters")
        date_min = st.text_input("Start date (YYYY-MM-DD)   ", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)    ", value="")
        rank_min, rank_max = st.slider("Rank range (compare)", 1, 17, (1, 10))
        align_mode = st.selectbox("Alignment", ["Calendar (Week Ending)", "Relative (weeks since first appearance)"])

    filters = FilterSpec(date_min.strip() or None, date_max.strip() or None, int(rank_min), int(rank_max))

    aid = int(shows.loc[shows["canonical_title"] == a, "show_id"].iloc[0])
    bid = int(shows.loc[shows["canonical_title"] == b, "show_id"].iloc[0])

    dfa = fetch_show_entries(aid, filters)
    dfb = fetch_show_entries(bid, filters)

    if dfa.empty or dfb.empty:
        st.warning("One (or both) shows have no rows in the selected filter range.")
        st.write({"Show A rows": int(len(dfa)), "Show B rows": int(len(dfb))})
        return

    def summarize(df: pd.DataFrame) -> dict[str, Any]:
        gross = df["gross_millions"].dropna()
        return {
            "rows": int(len(df)),
            "unique_dates": int(df["week_ending"].nunique()),
            "peak_rank": int(df["rank"].min()),
            "avg_rank": float(df["rank"].mean()),
            "gross_rows": int(len(gross)),
            "total_gross_M": float(gross.sum()) if len(gross) else 0.0,
            "avg_gross_M": float(gross.mean()) if len(gross) else None,
        }

    sa = summarize(dfa)
    sb = summarize(dfb)
    st.markdown("### Summary")
    s1, s2 = st.columns(2)
    with s1:
        st.write(f"**{a}**")
        st.write(sa)
    with s2:
        st.write(f"**{b}**")
        st.write(sb)

    st.markdown("### Overlap weeks")
    # Ties can create multiple rows per week; overlap is still meaningful on week_ending.
    overlap = pd.merge(
        dfa[["week_ending", "rank", "gross_millions"]],
        dfb[["week_ending", "rank", "gross_millions"]],
        on="week_ending",
        how="inner",
        suffixes=("_A", "_B")
    )
    st.write(f"Overlap weeks (same Week Ending): **{overlap['week_ending'].nunique()}**")
    if not overlap.empty:
        st.dataframe(overlap, width='stretch')

    st.markdown("### Rank comparison")
    if align_mode.startswith("Calendar"):
        fig = plt.figure()
        plt.plot(pd.to_datetime(dfa["week_ending"]), dfa["rank"].astype(float), label="A")
        plt.plot(pd.to_datetime(dfb["week_ending"]), dfb["rank"].astype(float), label="B")
        plt.gca().invert_yaxis()
        plt.xlabel("Week Ending")
        plt.ylabel("Rank")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        dfa2 = dfa.copy()
        dfb2 = dfb.copy()
        dfa2["t"] = np.arange(len(dfa2))
        dfb2["t"] = np.arange(len(dfb2))
        fig = plt.figure()
        plt.plot(dfa2["t"], dfa2["rank"].astype(float), label="A")
        plt.plot(dfb2["t"], dfb2["rank"].astype(float), label="B")
        plt.gca().invert_yaxis()
        plt.xlabel("Weeks since first appearance (within filters)")
        plt.ylabel("Rank")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Gross comparison")
    ga = dfa.dropna(subset=["gross_millions"]).copy()
    gb = dfb.dropna(subset=["gross_millions"]).copy()
    if ga.empty or gb.empty:
        st.info("At least one show has no gross values within the selected filters.")
    else:
        if align_mode.startswith("Calendar"):
            fig = plt.figure()
            plt.plot(pd.to_datetime(ga["week_ending"]), ga["gross_millions"].astype(float), label="A")
            plt.plot(pd.to_datetime(gb["week_ending"]), gb["gross_millions"].astype(float), label="B")
            plt.xlabel("Week Ending")
            plt.ylabel("Gross (Millions)")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            ga2 = ga.copy()
            gb2 = gb.copy()
            ga2["t"] = np.arange(len(ga2))
            gb2["t"] = np.arange(len(gb2))
            fig = plt.figure()
            plt.plot(ga2["t"], ga2["gross_millions"].astype(float), label="A")
            plt.plot(gb2["t"], gb2["gross_millions"].astype(float), label="B")
            plt.xlabel("Weeks since first gross row (within filters)")
            plt.ylabel("Gross (Millions)")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def tab_companies():
    company_mode = st.radio("Company field", ["Imprint 1", "Imprint 2"], horizontal=True, index=0)
    imprint_col = "imprint_1" if company_mode == "Imprint 1" else "imprint_2"

    st.subheader(f"Company view ({company_mode})")

    if company_mode == "Imprint 1":
        _, companies = load_lists()
        company_list = companies["company"].tolist()
    else:
        company_list = fetch_company_list(imprint_col)

    company = st.selectbox(f"Company ({company_mode})", company_list)

    with st.sidebar:
        st.header("Company filters")
        date_min = st.text_input("Start date (YYYY-MM-DD)    ", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)     ", value="")
        rank_min, rank_max = st.slider("Rank range (company)", 1, 17, (1, 10))

    filters = FilterSpec(date_min.strip() or None, date_max.strip() or None, int(rank_min), int(rank_max))

    stat = fetch_company_stats(company, filters, imprint_col=imprint_col)
    if not stat.empty:
        s = stat.iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entries", int(s.get("entries", 0) or 0))
        c2.metric("Unique shows", int(s.get("unique_shows", 0) or 0))
        c3.metric("Total gross (M)", float(s.get("total_gross_millions", 0.0) or 0.0))
        av = s.get("avg_gross_millions")
        c4.metric("Avg gross (M)", None if pd.isna(av) else float(av))

    df = fetch_company_entries(company, filters, imprint_col=imprint_col)
    st.dataframe(df, width='stretch')


def tab_analytics():
    st.subheader("Analytics (grossing + movement)")
    st.caption("All metrics are computed from stored rows.")

    with st.sidebar:
        st.header("Analytics filters")
        date_min = st.text_input("Start date (YYYY-MM-DD)     ", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)      ", value="")
        rank_min, rank_max = st.slider("Rank range (analytics)", 1, 17, (1, 10))
        top_n = st.slider("Top N", 5, 50, 15)

    filters = FilterSpec(date_min.strip() or None, date_max.strip() or None, int(rank_min), int(rank_max))

    where, params = build_where(filters, "e")
    df = sql_df(
        f"""
        SELECT
          e.show_id,
          e.week_ending,
          e.week_number,
          e.rank,
          e.pos,
          e.last_week,
          e.gross_millions AS base_gross_millions,
          COALESCE(gb.bonus_millions, 0) AS bonus_millions,
          (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions,
          COALESCE(e.imprint_1,'(Unknown)') AS imprint_1,
          COALESCE(e.imprint_2,'') AS imprint_2,
          COALESCE(e.imprint_1,'(Unknown)') AS company,
          s.canonical_title
        FROM t10_entry e
        LEFT JOIN (
          SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
          FROM gross_bonus
          GROUP BY show_id, week_ending
        ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
        JOIN show s ON s.show_id = e.show_id
        WHERE {where}
        ORDER BY e.week_ending ASC, e.rank ASC, e.pos ASC
        """,
        tuple(params),
    )

    if df.empty:
        st.info("No rows match your filters.")
        return

    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["year"] = df["week_ending_dt"].dt.year
    df["month"] = df["week_ending_dt"].dt.month
    # ISO week-of-year (1–53)
    df["week_of_year"] = df["week_ending_dt"].dt.isocalendar().week.astype("Int64")

    dg = df.dropna(subset=["gross_millions"]).copy()
    for col in ["gross_millions", "base_gross_millions", "bonus_millions"]:
        if col in dg.columns:
            dg[col] = pd.to_numeric(dg[col], errors="coerce")

    ignore_bonus = st.checkbox(
        "Ignore gross bonuses in most analytics (Top shows/companies + yearly totals still include bonuses)",
        value=True,
        key="analytics_ignore_bonus",
    )
    gross_col = "base_gross_millions" if ignore_bonus else "gross_millions"
    gross_label = "Base gross (millions)" if gross_col == "base_gross_millions" else "Gross + bonuses (millions)"
    if ignore_bonus:
        st.caption(
            "Bonuses are excluded from most charts/totals below, except Top shows, Top companies, and Yearly gross totals (those always include bonuses)."
        )

    if dg.empty:
        st.warning("No gross values in the selected range.")
        return

    # -------------------------
    # Reusable weekly series
    # -------------------------
    weekly = (
        dg.groupby("week_ending", as_index=False)[gross_col]
        .sum()
        .rename(columns={gross_col: "gross_millions"})
        .sort_values("week_ending")
    )

    counts = (
        dg[dg[gross_col].fillna(0.0) > 0.0]
        .groupby("week_ending", as_index=False)["show_id"]
        .nunique()
        .rename(columns={"show_id": "num_shows"})
    )

    medians = (
        dg[dg[gross_col].fillna(0.0) > 0.0]
        .groupby("week_ending", as_index=False)[gross_col]
        .median()
        .rename(columns={gross_col: "weekly_median_millions"})
    )

    wa_ts = weekly.merge(counts, on="week_ending", how="left")
    wa_ts = wa_ts.merge(medians, on="week_ending", how="left")
    wa_ts["num_shows"] = wa_ts["num_shows"].fillna(0).astype(int)
    wa_ts["weekly_avg_millions"] = np.where(
        wa_ts["num_shows"] > 0,
        wa_ts["gross_millions"] / wa_ts["num_shows"],
        0.0,
    )
    wa_ts["weekly_median_millions"] = wa_ts["weekly_median_millions"].fillna(0.0).astype(float)
    wa_ts["week_ending_dt"] = pd.to_datetime(wa_ts["week_ending"], errors="coerce")
    wa_ts["year"] = wa_ts["week_ending_dt"].dt.year
    wa_ts["month"] = wa_ts["week_ending_dt"].dt.month
    wa_ts["week_of_year"] = wa_ts["week_ending_dt"].dt.isocalendar().week.astype("Int64")
    wa_ts = wa_ts.sort_values("week_ending")

    premium_ts = _build_number_one_premium_table(dg, gross_col)

    # -------------------------
    # Analytics sub-tabs
    # -------------------------
    analytics_section = st.selectbox(
        "Analytics section",
        ["Overview", "Heatmaps", "Premiums", "Distribution", "Outliers"],
        index=0,
        key="analytics_section",
    )

    # -------------------------
    # Overview
    # -------------------------
    if analytics_section == "Overview":
        st.markdown("### Total gross over time (weekly sum)")
        plot_line_dates(weekly["week_ending"], weekly["gross_millions"], "Week Ending", "Total Gross (Millions)")

        if st.checkbox("Show Top Gross Weeks", key="analytics_show_top_gross_weeks"):
            chart_top_gross_weeks(weekly, n=int(top_n))

        st.markdown("### Rolling average total gross")
        win = st.slider("Rolling window (weeks)", 2, 52, 13, key="analytics_roll_total_window")
        w2 = weekly.copy()
        w2["roll"] = w2["gross_millions"].rolling(win, min_periods=max(1, win // 3)).mean()
        plot_line_dates(w2["week_ending"], w2["roll"], "Week Ending", f"{win}-week avg gross (Millions)")

        st.markdown("### Weekly average gross")
        use_ma = st.checkbox("Show moving average instead of raw weekly average", value=False, key="analytics_wa_use_ma")
        ma_win = st.slider("Moving average window (weeks)", 2, 52, 13, key="analytics_wa_ma_window")
        wa_plot = wa_ts.copy()
        wa_plot["ma"] = wa_plot["weekly_avg_millions"].rolling(ma_win, min_periods=max(1, ma_win // 3)).mean()
        ycol = "ma" if use_ma else "weekly_avg_millions"
        ylabel = f"{ma_win}-week moving avg (Millions)" if use_ma else "Weekly avg gross (Millions)"
        plot_line_dates(wa_plot["week_ending"], wa_plot[ycol], "Week Ending", ylabel)

        if st.checkbox("Show Top Weekly Averages", key="analytics_show_top_weekly_avgs"):
            wa = wa_ts.sort_values("weekly_avg_millions", ascending=False)
            st.dataframe(
                wa[["week_ending", "gross_millions", "num_shows", "weekly_avg_millions"]].head(int(top_n)),
                width='stretch',
            )

        st.markdown("### Weekly median gross")
        use_med_ma = st.checkbox(
            "Show moving average instead of raw weekly median",
            value=False,
            key="analytics_wm_use_ma",
        )
        med_ma_win = st.slider("Median moving average window (weeks)", 2, 52, 13, key="analytics_wm_ma_window")
        wm_plot = wa_ts.copy()
        wm_plot["ma"] = wm_plot["weekly_median_millions"].rolling(
            med_ma_win, min_periods=max(1, med_ma_win // 3)
        ).mean()
        med_ycol = "ma" if use_med_ma else "weekly_median_millions"
        med_ylabel = (
            f"{med_ma_win}-week moving median avg (Millions)" if use_med_ma else "Weekly median gross (Millions)"
        )
        plot_line_dates(wm_plot["week_ending"], wm_plot[med_ycol], "Week Ending", med_ylabel)

        if st.checkbox("Show Top Weekly Medians", key="analytics_show_top_weekly_medians"):
            wm = wa_ts.sort_values("weekly_median_millions", ascending=False)
            st.dataframe(
                wm[["week_ending", "gross_millions", "num_shows", "weekly_avg_millions", "weekly_median_millions"]].head(int(top_n)),
                width='stretch',
            )

        st.markdown("### Rank vs Gross (scatter)")
        plot_scatter(dg["rank"].astype(float), dg[gross_col].astype(float), "Rank", gross_label)

        st.markdown("### Top companies by total gross")
        # Combine imprint_1 + imprint_2 (dedupe per row) so companies appearing in either column are counted.
        if ("imprint_1" in dg.columns) or ("imprint_2" in dg.columns):
            c1 = dg["imprint_1"] if "imprint_1" in dg.columns else dg.get("company")
            c1 = c1.fillna("(Unknown)").astype(str)

            if "imprint_2" in dg.columns:
                c2 = dg["imprint_2"].fillna("").astype(str)
            else:
                c2 = pd.Series([""] * len(dg), index=dg.index)

            comp_rows_1 = pd.DataFrame({"company": c1, "gross_millions": dg["gross_millions"]})

            comp_rows_2 = pd.DataFrame({"company": c2, "gross_millions": dg["gross_millions"], "_c1": c1})
            comp_rows_2 = comp_rows_2[
                (comp_rows_2["company"].str.strip() != "") & (comp_rows_2["company"] != comp_rows_2["_c1"])
            ][["company", "gross_millions"]]

            comp_rows = pd.concat([comp_rows_1, comp_rows_2], ignore_index=True)
            comp_rows["company"] = comp_rows["company"].fillna("(Unknown)").replace({"": "(Unknown)"})

            top_comp = comp_rows.groupby("company", as_index=False)["gross_millions"].sum()
        else:
            top_comp = dg.groupby("company", as_index=False)["gross_millions"].sum()

        top_comp = top_comp.sort_values("gross_millions", ascending=False).head(int(top_n))
        st.dataframe(top_comp, width='stretch')
        plot_barh(top_comp["company"][::-1], top_comp["gross_millions"][::-1], "Total Gross (Millions)", "Company")

        st.markdown("### Gross distribution")
        plot_hist(dg[gross_col].astype(float), bins=30, xlabel=gross_label, ylabel="Count")

        st.markdown("### Yearly gross totals")
        yearly = dg.groupby("year", as_index=False)["gross_millions"].sum().sort_values("year")
        st.dataframe(yearly, width='stretch')
        fig = plt.figure()
        plt.plot(yearly["year"], yearly["gross_millions"])
        plt.xlabel("Year")
        plt.ylabel("Total Gross (Millions)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------
    # Heatmaps
    # -------------------------
    if analytics_section == "Heatmaps":
        metric = st.selectbox(
            "Heatmap metric",
            options=["Weekly average gross", "Total gross", "#1 premium difference", "#1 premium ratio"],
            index=0,
            key="analytics_heat_metric",
        )

        st.markdown("### Calendar heatmap (year × week-of-year)")
        if metric == "Weekly average gross":
            hm = wa_ts.dropna(subset=["week_ending_dt"]).copy()
            hm["metric"] = hm["weekly_avg_millions"].astype(float)
        elif metric == "Total gross":
            hm = wa_ts.dropna(subset=["week_ending_dt"]).copy()
            hm["metric"] = hm["gross_millions"].astype(float)
        elif metric == "#1 premium difference":
            hm = premium_ts.dropna(subset=["week_ending_dt"]).copy()
            hm["metric"] = pd.to_numeric(hm["premium_diff"], errors="coerce")
            hm["num_shows"] = 2
            hm["gross_millions"] = pd.to_numeric(hm["n1_gross"], errors="coerce") + pd.to_numeric(hm["n2_gross"], errors="coerce")
        else:
            hm = premium_ts.dropna(subset=["week_ending_dt"]).copy()
            hm["metric"] = pd.to_numeric(hm["premium_ratio"], errors="coerce")
            hm["num_shows"] = 2
            hm["gross_millions"] = pd.to_numeric(hm["n1_gross"], errors="coerce") + pd.to_numeric(hm["n2_gross"], errors="coerce")

        if not hm.empty:
            hm["year"] = pd.to_datetime(hm["week_ending_dt"], errors="coerce").dt.year
            hm["week_of_year"] = pd.to_datetime(hm["week_ending_dt"], errors="coerce").dt.isocalendar().week.astype("Int64")

        # If week_of_year is NA (shouldn't be, but just in case), drop
        hm = hm.dropna(subset=["year", "week_of_year"]).copy()
        hm["year"] = hm["year"].astype(int)
        hm["week_of_year"] = hm["week_of_year"].astype(int)

        if hm.empty:
            st.info("Not enough weekly data for a heatmap in the selected range.")
        else:
            chart = (
                alt.Chart(hm)
                .mark_rect()
                .encode(
                    x=alt.X("week_of_year:O", title="ISO week of year"),
                    y=alt.Y("year:O", title="Year"),
                    color=alt.Color("metric:Q", title=metric),
                    tooltip=[
                        alt.Tooltip("week_ending:N", title="Week ending"),
                        alt.Tooltip("metric:Q", title=metric, format=",.2f"),
                        alt.Tooltip("num_shows:Q", title="# shows"),
                        alt.Tooltip("gross_millions:Q", title="Total gross", format=",.2f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width='stretch')

        st.markdown("### Rank-by-week heatmap (gross by rank over time)")
        rb = dg.dropna(subset=["week_ending_dt", "rank", gross_col]).copy()
        if rb.empty:
            st.info("No ranked gross rows available for the selected range.")
        else:
            rb["gross_use"] = rb[gross_col].astype(float)
            # Aggregate in case of duplicate rank rows (shouldn't happen, but safe)
            rb2 = (
                rb.groupby(["week_ending_dt", "week_ending", "rank"], as_index=False)["gross_use"]
                .sum()
                .sort_values(["week_ending_dt", "rank"])
            )

            chart2 = (
                alt.Chart(rb2)
                .mark_rect()
                .encode(
                    x=alt.X("week_ending_dt:T", title="Week ending"),
                    y=alt.Y("rank:O", title="Rank", sort="ascending"),
                    color=alt.Color("gross_use:Q", title=gross_label),
                    tooltip=[
                        alt.Tooltip("week_ending:N", title="Week ending"),
                        alt.Tooltip("rank:Q", title="Rank"),
                        alt.Tooltip("gross_use:Q", title=gross_label, format=",.2f"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(chart2, width='stretch')

    # -------------------------
    # Premiums
    # -------------------------
    if analytics_section == "Premiums":
        st.markdown("### #1 premium tables (#1 vs #2)")

        if premium_ts.empty:
            st.info("Not enough data for #1 premium tables in the selected filters.")
        else:
            min_available = pd.to_datetime(premium_ts["week_ending_dt"].min()).date()
            max_available = pd.to_datetime(premium_ts["week_ending_dt"].max()).date()

            c1, c2, c3 = st.columns([1.15, 1.15, 1])
            with c1:
                pmin = st.date_input("Start date", value=min_available, min_value=min_available, max_value=max_available, key="analytics_premium_start")
            with c2:
                pmax = st.date_input("End date", value=max_available, min_value=min_available, max_value=max_available, key="analytics_premium_end")
            with c3:
                premium_top_n = st.slider("Top N", 5, 200, min(50, max(5, int(top_n))), key="analytics_premium_topn")

            if pd.to_datetime(pmin) > pd.to_datetime(pmax):
                st.warning("Start date is after end date.")
            else:
                pt = premium_ts[(premium_ts["week_ending_dt"] >= pd.Timestamp(pmin)) & (premium_ts["week_ending_dt"] <= pd.Timestamp(pmax))].copy()
                if pt.empty:
                    st.info("No #1 premium rows found for that date range.")
                else:
                    diff_tbl = pt.sort_values(["premium_diff", "week_ending_dt"], ascending=[False, False]).head(int(premium_top_n)).copy()
                    diff_tbl.insert(0, "Rank", np.arange(1, len(diff_tbl) + 1))
                    diff_tbl = diff_tbl.rename(columns={
                        "week_ending": "Week Ending",
                        "n1_show": "#1 Show",
                        "n1_gross": "#1 Gross",
                        "n2_show": "#2 Show",
                        "n2_gross": "#2 Gross",
                        "premium_diff": "Premium Difference",
                        "premium_ratio": "Premium Ratio",
                    })
                    st.markdown("#### Biggest #1 premium gaps")
                    st.dataframe(diff_tbl[["Rank", "Week Ending", "#1 Show", "#1 Gross", "#2 Show", "#2 Gross", "Premium Difference", "Premium Ratio"]], width='stretch', hide_index=True)

                    ratio_tbl = pt.dropna(subset=["premium_ratio"]).sort_values(["premium_ratio", "week_ending_dt"], ascending=[False, False]).head(int(premium_top_n)).copy()
                    ratio_tbl.insert(0, "Rank", np.arange(1, len(ratio_tbl) + 1))
                    ratio_tbl = ratio_tbl.rename(columns={
                        "week_ending": "Week Ending",
                        "n1_show": "#1 Show",
                        "n1_gross": "#1 Gross",
                        "n2_show": "#2 Show",
                        "n2_gross": "#2 Gross",
                        "premium_diff": "Premium Difference",
                        "premium_ratio": "Premium Ratio",
                    })
                    st.markdown("#### Biggest #1 premium ratios")
                    st.dataframe(ratio_tbl[["Rank", "Week Ending", "#1 Show", "#1 Gross", "#2 Show", "#2 Gross", "Premium Difference", "Premium Ratio"]], width='stretch', hide_index=True)

    # -------------------------
    # Distribution (boxplots)
    # -------------------------
    if analytics_section == "Distribution":
        metric = st.selectbox(
            "Distribution metric",
            options=["Weekly average gross", "Total gross"],
            index=0,
            key="analytics_dist_metric",
        )
        metric_col = "weekly_avg_millions" if metric == "Weekly average gross" else "gross_millions"

        dd = wa_ts.dropna(subset=["week_ending_dt"]).copy()
        dd["metric"] = dd[metric_col].astype(float)
        dd["month_name"] = dd["week_ending_dt"].dt.strftime("%b")
        dd["week_of_year"] = dd["week_of_year"].astype("Int64")

        if dd.empty:
            st.info("No weekly data available for distribution charts.")
        else:
            st.markdown("### Summary distribution")
            chart = (
                alt.Chart(dd)
                .mark_boxplot(extent="min-max")
                .encode(
                    y=alt.Y("metric:Q", title=metric),
                    tooltip=[
                        alt.Tooltip("metric:Q", title=metric, format=",.2f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width='stretch')

            st.markdown("### Seasonality: month-of-year")
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            chart_m = (
                alt.Chart(dd)
                .mark_boxplot()
                .encode(
                    x=alt.X("month_name:O", sort=month_order, title="Month"),
                    y=alt.Y("metric:Q", title=metric),
                    tooltip=[
                        alt.Tooltip("month_name:N", title="Month"),
                        alt.Tooltip("metric:Q", title=metric, format=",.2f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_m, width='stretch')

            st.markdown("### Seasonality: week-of-year")
            # 52 (sometimes 53) boxes can be a lot; allow a min-week filter
            show_woy = st.checkbox("Show week-of-year boxplots", value=False, key="analytics_dist_woy_show")
            if show_woy:
                chart_w = (
                    alt.Chart(dd.dropna(subset=["week_of_year"]))
                    .mark_boxplot(size=6)
                    .encode(
                        x=alt.X("week_of_year:O", title="ISO week of year"),
                        y=alt.Y("metric:Q", title=metric),
                        tooltip=[
                            alt.Tooltip("week_of_year:Q", title="ISO week"),
                            alt.Tooltip("metric:Q", title=metric, format=",.2f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_w, width='stretch')
            else:
                st.caption("Toggle the checkbox above if you want the full week-of-year view (it can be very wide).")

            with st.expander("Summary stats table"):
                stats = (
                    dd["metric"]
                    .agg(["count", "mean", "median", "min", "max", "std"])
                    .to_frame()
                    .reset_index()
                    .rename(columns={"index": "stat", "metric": "value"})
                )
                st.dataframe(stats, width='stretch', hide_index=True)

    # -------------------------
    # Outlier weeks detector
    # -------------------------
    if analytics_section == "Outliers":

        # -------------------------
        # Week-level outliers
        # -------------------------
        metric = st.selectbox(
            "Outlier metric",
            options=["Weekly average gross", "Total gross"],
            index=0,
            key="analytics_out_metric",
        )
        metric_col = "weekly_avg_millions" if metric == "Weekly average gross" else "gross_millions"

        method = st.selectbox(
            "Scoring method",
            options=["Rolling z-score (mean/std)", "Rolling robust z-score (median/MAD)"],
            index=0,
            key="analytics_out_method",
        )

        roll_w = st.slider(
            "Rolling window (weeks)",
            min_value=4,
            max_value=52,
            value=13,
            step=1,
            key="analytics_out_window",
        )
        thr = st.slider(
            "Outlier threshold (absolute z)",
            min_value=1.0,
            max_value=6.0,
            value=2.5,
            step=0.1,
            key="analytics_out_thr",
        )

        o = wa_ts.dropna(subset=["week_ending_dt"]).copy()
        o["x"] = o[metric_col].astype(float)

        if o.empty:
            st.info("No weekly data available for outlier detection.")
        else:
            o = o.sort_values("week_ending_dt").reset_index(drop=True)

            if method.startswith("Rolling z-score"):
                mu = o["x"].rolling(roll_w, min_periods=max(2, roll_w // 2)).mean()
                sd = o["x"].rolling(roll_w, min_periods=max(2, roll_w // 2)).std().replace(0.0, np.nan)
                o["z"] = ((o["x"] - mu) / sd).fillna(0.0)
            else:
                med = o["x"].rolling(roll_w, min_periods=max(2, roll_w // 2)).median()
                mad = (o["x"] - med).abs().rolling(roll_w, min_periods=max(2, roll_w // 2)).median()
                mad = mad.replace(0.0, np.nan)
                # 0.6745 scales MAD to std for normal dist
                o["z"] = (0.6745 * (o["x"] - med) / mad).fillna(0.0)

            o["is_outlier"] = o["z"].abs() >= float(thr)

            # Chart: line + highlighted points
            base = alt.Chart(o).encode(
                x=alt.X("week_ending_dt:T", title="Week ending"),
            )

            line = base.mark_line().encode(
                y=alt.Y("x:Q", title=metric),
                tooltip=[
                    alt.Tooltip("week_ending:N", title="Week ending"),
                    alt.Tooltip("x:Q", title=metric, format=",.2f"),
                    alt.Tooltip("z:Q", title="z", format=",.2f"),
                    alt.Tooltip("num_shows:Q", title="# shows"),
                ],
            )

            pts = (
                base.transform_filter(alt.datum.is_outlier == True)
                .mark_point(size=80)
                .encode(
                    y="x:Q",
                    tooltip=[
                        alt.Tooltip("week_ending:N", title="Week ending"),
                        alt.Tooltip("x:Q", title=metric, format=",.2f"),
                        alt.Tooltip("z:Q", title="z", format=",.2f"),
                        alt.Tooltip("gross_millions:Q", title="Total gross", format=",.2f"),
                        alt.Tooltip("num_shows:Q", title="# shows"),
                    ],
                )
            )

            st.altair_chart((line + pts).properties(height=320), width='stretch')

            outs = o[o["is_outlier"]].copy()
            if outs.empty:
                st.info("No outliers at this threshold.")
            else:
                outs = outs.assign(abs_z=outs["z"].abs()).sort_values("abs_z", ascending=False)
                show_cols = ["week_ending", "x", "z", "gross_millions", "num_shows"]
                tbl = outs[show_cols].rename(columns={"x": metric, "z": "z_score"})
                st.dataframe(tbl.head(int(top_n)), width='stretch')

                pick = st.selectbox(
                    "Drilldown week",
                    options=outs["week_ending"].head(min(50, len(outs))).tolist(),
                    index=0,
                    key="analytics_out_pick_week",
                )
                with st.expander("Top contributors (by show) for selected week"):
                    wk = dg[dg["week_ending"] == pick].copy()
                    if wk.empty:
                        st.info("No show-level rows found for that week under current filters.")
                    else:
                        wk["gross_use"] = wk[gross_col].astype(float)
                        wk = wk.sort_values("gross_use", ascending=False)
                        st.dataframe(
                            wk[["rank", "canonical_title", "imprint_1", "imprint_2", "gross_use"]].head(10),
                            width='stretch',
                            hide_index=True,
                        )

@st.cache_data(show_spinner=False)
def _alltime_rank_table_for_show_weeks(db_path: str, db_mtime: float, show_id: int, weeks: list[pd.Timestamp]) -> pd.DataFrame:
    """All-Time Gross Races rank for a show as-of selected weeks (grossing-era only, includes bonuses)."""
    if not weeks:
        return pd.DataFrame(columns=["rank", "rank_change", "week_ending", "total_gross_millions"])

    base = _load_gross_races_base(db_path, db_mtime).copy()
    if base.empty:
        return pd.DataFrame(columns=["rank", "rank_change", "week_ending", "total_gross_millions"])

    base["week_ending_dt"] = pd.to_datetime(base["week_ending"], errors="coerce")
    base = base[base["week_ending_dt"] >= pd.Timestamp(GROSS_TRACKING_START)].copy()
    base = base.dropna(subset=["week_ending_dt"]).copy()

    pivot = base.pivot_table(
        index="week_ending_dt",
        columns="show_id",
        values="gross_millions",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    cum = pivot.cumsum()

    if show_id not in cum.columns:
        return pd.DataFrame(columns=["rank", "rank_change", "week_ending", "total_gross_millions"])

    ranks = cum.rank(axis=1, method="min", ascending=False)

    week_idx = pd.to_datetime(pd.Series(weeks), errors="coerce").dropna().dt.normalize().tolist()
    week_idx = [w for w in week_idx if w in cum.index]
    if not week_idx:
        return pd.DataFrame(columns=["rank", "rank_change", "week_ending", "total_gross_millions"])

    show_cum = cum.loc[week_idx, show_id].astype(float)
    show_rank = ranks.loc[week_idx, show_id].astype(int)

    out = pd.DataFrame(
        {
            "rank": show_rank.tolist(),
            "week_ending": [w.strftime("%Y-%m-%d") for w in week_idx],
            "total_gross_millions": show_cum.tolist(),
        }
    )
    prev = out["rank"].shift(1)
    out["rank_change"] = (prev - out["rank"]).fillna(0).astype(int)
    out = out[["rank", "rank_change", "week_ending", "total_gross_millions"]]
    return out



@st.cache_data(show_spinner=False)
def _load_show_meta_for_gross_races(db_path: str, db_mtime: float) -> pd.DataFrame:
    """Show-level metadata for gross races (imprints + debut date). db_mtime busts cache on DB updates."""
    con = sqlite3.connect(db_path)
    try:
        raw = pd.read_sql_query(
            """
            SELECT
              e.show_id AS show_id,
              s.canonical_title AS canonical_title,
              date(e.week_ending) AS week_ending,
              NULLIF(TRIM(e.imprint_1), '') AS imprint_1,
              NULLIF(TRIM(e.imprint_2), '') AS imprint_2
            FROM t10_entry e
            JOIN show s ON s.show_id = e.show_id
            WHERE e.week_ending IS NOT NULL
            """,
            con,
        )
    finally:
        con.close()

    if raw.empty:
        return pd.DataFrame(columns=["show_id", "canonical_title", "debut_date", "imprint_1", "imprint_2"])

    raw["week_ending_dt"] = pd.to_datetime(raw["week_ending"], errors="coerce")
    raw = raw.sort_values(["show_id", "week_ending_dt"]).copy()

    def _norm(s: pd.Series) -> pd.Series:
        s2 = s.astype(object)
        s2 = s2.apply(lambda x: x.strip() if isinstance(x, str) else x)
        s2 = s2.replace({"<none>": None, "<None>": None, "None": None, "NONE": None, "": None})
        return s2

    raw["imprint_1"] = _norm(raw["imprint_1"])
    raw["imprint_2"] = _norm(raw["imprint_2"])

    def _last_nonempty(s: pd.Series) -> str:
        s2 = s.dropna().astype(str).str.strip()
        s2 = s2[s2 != ""]
        s2 = s2[s2.str.lower() != "<none>"]
        return s2.iloc[-1] if len(s2) else ""

    meta = (
        raw.groupby(["show_id", "canonical_title"], as_index=False)
        .agg(
            debut_date=("week_ending_dt", "min"),
            imprint_1=("imprint_1", _last_nonempty),
            imprint_2=("imprint_2", _last_nonempty),
        )
    )
    meta["debut_date"] = meta["debut_date"].dt.date
    meta["imprint_1"] = meta["imprint_1"].fillna("")
    meta["imprint_2"] = meta["imprint_2"].fillna("")
    return meta
    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df = df.dropna(subset=["week_ending_dt", "show_id", "canonical_title"]).copy()

    # Ensure numeric
    for col in ("base_gross_millions", "bonus_millions", "gross_millions"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # If anything still duplicates (shouldn't), collapse safely
    df = (
        df.groupby(["show_id", "canonical_title", "week_ending"], as_index=False)[
            ["base_gross_millions", "bonus_millions", "gross_millions"]
        ]
        .sum()
        .sort_values(["show_id", "week_ending"])
        .reset_index(drop=True)
    )
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    return df

def _plot_multi_line(dates: list[pd.Timestamp], series_by_label: dict[str, pd.Series], xlabel: str, ylabel: str):
    fig = plt.figure()
    for label, y in series_by_label.items():
        plt.plot(dates, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _year_cumulative(base: pd.DataFrame, year: int, through_dt: pd.Timestamp) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    ydf = base[base["week_ending_dt"].dt.year == year].copy()
    ydf = ydf[ydf["week_ending_dt"] <= through_dt].copy()
    if ydf.empty:
        return ydf, []

    ydf = ydf.sort_values(["show_id", "week_ending_dt"]).reset_index(drop=True)
    ydf["cum_gross_millions"] = ydf.groupby("show_id")["gross_millions"].cumsum()

    weeks = sorted(ydf["week_ending_dt"].dropna().unique().tolist())
    return ydf, weeks


def _quarter_cumulative(base: pd.DataFrame, year: int, quarter: int, through_week_dt: pd.Timestamp) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    q = int(quarter)
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2

    qdf = base[(base["week_ending_dt"].dt.year == year)].copy()
    qdf = qdf[(qdf["week_ending_dt"].dt.month >= start_month) & (qdf["week_ending_dt"].dt.month <= end_month)].copy()
    qdf = qdf[qdf["week_ending_dt"] <= through_week_dt].copy()

    if qdf.empty:
        return qdf, []

    qdf = qdf.sort_values(["show_id", "week_ending_dt"]).reset_index(drop=True)
    qdf["cum_gross_millions"] = qdf.groupby("show_id")["gross_millions"].cumsum()

    weeks = sorted(qdf["week_ending_dt"].dropna().unique().tolist())
    return qdf, weeks



@st.cache_data(show_spinner=False)
def _load_gross_races_show_leaderboard_rows(db_path: str, db_mtime: float) -> pd.DataFrame:
    """Weekly chart rows (ranked entries only) with annual/quarter bonuses rolled into the week for show leaderboards."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            WITH bonus AS (
              SELECT show_id, week_ending, SUM(COALESCE(bonus_millions, 0.0)) AS bonus_millions
              FROM gross_bonus
              WHERE bonus_type IN ('annual', 'quarter')
              GROUP BY show_id, week_ending
            )
            SELECT
              date(e.week_ending) AS week_ending,
              e.show_id,
              s.canonical_title,
              e.rank,
              COALESCE(e.gross_millions, 0.0) AS base_gross_millions,
              COALESCE(b.bonus_millions, 0.0) AS bonus_millions,
              (COALESCE(e.gross_millions, 0.0) + COALESCE(b.bonus_millions, 0.0)) AS gross_millions
            FROM t10_entry e
            JOIN show s ON s.show_id = e.show_id
            LEFT JOIN bonus b ON b.show_id = e.show_id AND b.week_ending = e.week_ending
            WHERE e.week_ending IS NOT NULL
            ORDER BY date(e.week_ending), e.rank, e.show_id
            """,
            con,
        )
    finally:
        con.close()
    if df.empty:
        return df
    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    for c in ["base_gross_millions", "bonus_millions", "gross_millions"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df




def _build_number_one_premium_table(df: pd.DataFrame, gross_col: str) -> pd.DataFrame:
    """Weekly #1-vs-#2 premium table for the currently filtered dataframe."""
    if df.empty or gross_col not in df.columns:
        return pd.DataFrame()

    top2 = df[df["rank"].between(1, 2)].copy()
    if top2.empty:
        return pd.DataFrame()

    top2[gross_col] = pd.to_numeric(top2[gross_col], errors="coerce")
    top2 = top2.dropna(subset=["week_ending_dt", gross_col]).copy()
    if top2.empty:
        return pd.DataFrame()

    agg = (
        top2.groupby(["week_ending", "week_ending_dt", "rank"], as_index=False)
        .agg(
            canonical_title=("canonical_title", "first"),
            gross_use=(gross_col, "sum"),
        )
    )

    gross_wide = (
        agg.pivot(index=["week_ending", "week_ending_dt"], columns="rank", values="gross_use")
        .rename(columns={1: "n1_gross", 2: "n2_gross"})
        .reset_index()
    )
    if "n1_gross" not in gross_wide.columns or "n2_gross" not in gross_wide.columns:
        return pd.DataFrame()

    names_wide = (
        agg.pivot(index=["week_ending", "week_ending_dt"], columns="rank", values="canonical_title")
        .rename(columns={1: "n1_show", 2: "n2_show"})
        .reset_index()
    )

    out = gross_wide.merge(names_wide, on=["week_ending", "week_ending_dt"], how="left")
    if "n1_show" not in out.columns:
        out["n1_show"] = pd.NA
    if "n2_show" not in out.columns:
        out["n2_show"] = pd.NA

    out["premium_diff"] = pd.to_numeric(out["n1_gross"], errors="coerce") - pd.to_numeric(out["n2_gross"], errors="coerce")
    out["premium_ratio"] = (
        pd.to_numeric(out["n1_gross"], errors="coerce")
        / pd.to_numeric(out["n2_gross"], errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    out = out.sort_values("week_ending_dt").reset_index(drop=True)
    return out[[
        "week_ending", "week_ending_dt", "n1_show", "n1_gross", "n2_show", "n2_gross", "premium_diff", "premium_ratio"
    ]]


def _render_gross_races_all_gross_entries(base: pd.DataFrame, meta: pd.DataFrame, latest_date: date):
    st.markdown("### Top Gross Entries")
    st.caption("Table view of all ranked weekly gross entries in the selected slice, sorted by gross.")

    if base.empty:
        st.info("No gross rows found.")
        return

    min_available = pd.to_datetime(base["week_ending_dt"].min()).date()
    max_available = pd.to_datetime(base["week_ending_dt"].max()).date()

    c1, c2, c3, c4 = st.columns([1.15, 1.15, 1, 1])
    with c1:
        dmin = st.date_input("Start date", value=min_available, min_value=min_available, max_value=max_available, key="gr_entries_start")
    with c2:
        dmax = st.date_input("End date", value=max_available, min_value=min_available, max_value=max_available, key="gr_entries_end")
    with c3:
        top_n = st.slider("Top N", 5, 500, 200, key="gr_entries_topn")
    with c4:
        rank_pick = st.selectbox("Rank", options=list(range(1, 18)), index=0, key="gr_entries_rank")

    include_bon = st.checkbox(
        "Include bonuses",
        value=False,
        key="gr_entries_include_bonuses",
        help="Uses bonus-adjusted gross when checked. Base gross only when unchecked.",
    )

    if pd.to_datetime(dmin) > pd.to_datetime(dmax):
        st.warning("Start date is after end date.")
        return

    gross_col = "gross_millions" if include_bon else "base_gross_millions"
    gross_label = "Gross + bonuses (millions)" if include_bon else "Base gross (millions)"

    rows = base.copy()
    rows = rows[(rows["week_ending_dt"] >= pd.Timestamp(dmin)) & (rows["week_ending_dt"] <= pd.Timestamp(dmax))].copy()
    if "rank" in rows.columns:
        rows["rank"] = pd.to_numeric(rows["rank"], errors="coerce")
        rows = rows[rows["rank"].eq(int(rank_pick))].copy()
    else:
        st.info("Ranked weekly entry data is not available for this table.")
        return

    rows[gross_col] = pd.to_numeric(rows[gross_col], errors="coerce")
    rows = rows[rows[gross_col].fillna(0.0) > 0.0].copy()
    if rows.empty:
        st.info("No gross entries found for the selected filters.")
        return

    if not meta.empty:
        use_meta = meta[["show_id", "imprint_1", "imprint_2"]].drop_duplicates("show_id")
        rows = rows.merge(use_meta, on="show_id", how="left", suffixes=("", "_meta"))
        for col in ("imprint_1", "imprint_2"):
            meta_col = f"{col}_meta"
            if meta_col in rows.columns:
                rows[col] = rows[col].fillna(rows[meta_col])
                rows = rows.drop(columns=[meta_col], errors="ignore")

    rows = rows.sort_values([gross_col, "week_ending_dt", "canonical_title"], ascending=[False, False, True]).head(int(top_n)).copy()
    rows = rows.reset_index(drop=True)
    rows.insert(0, "table_rank", np.arange(1, len(rows) + 1))

    base_cols = ["table_rank", "week_ending", "rank", "canonical_title", "imprint_1", "imprint_2", "base_gross_millions"]
    if include_bon:
        base_cols.extend(["bonus_millions", gross_col])
    show = rows[base_cols].copy()
    show = show.rename(columns={
        "table_rank": "Rank",
        "week_ending": "Week Ending",
        "rank": "Week Rank",
        "canonical_title": "Show",
        "imprint_1": "Imprint 1",
        "imprint_2": "Imprint 2",
        "base_gross_millions": "Base gross (millions)",
        "bonus_millions": "Bonuses",
        gross_col: gross_label,
    })
    st.dataframe(show, width='stretch', hide_index=True)

def _build_gross_races_show_leaderboards(rows: pd.DataFrame, min_weeks: int, top_n: int, include_bonuses_avg_share: bool, share_mode: str = "peak") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, ("Bonus-adjusted gross" if include_bonuses_avg_share else "Base gross")

    df = rows.copy()
    df = df[df["canonical_title"].notna()].copy()
    df["gross_base"] = pd.to_numeric(df["base_gross_millions"], errors="coerce")
    df["gross_adjusted"] = pd.to_numeric(df["gross_millions"], errors="coerce")
    sel_col = "gross_adjusted" if include_bonuses_avg_share else "gross_base"
    basis_label = "Bonus-adjusted gross" if include_bonuses_avg_share else "Base gross"
    df["gross_for_avg_share"] = pd.to_numeric(df[sel_col], errors="coerce")

    # Weekly totals for share in current filtered rank-range/date universe
    wk = (
        df.dropna(subset=["gross_for_avg_share"])
          .groupby("week_ending", as_index=False)["gross_for_avg_share"]
          .sum()
          .rename(columns={"gross_for_avg_share": "weekly_total_for_share"})
    )
    df = df.merge(wk, on="week_ending", how="left")
    df["weekly_share"] = np.where(
        (df["weekly_total_for_share"].notna()) & (df["weekly_total_for_share"] > 0) & (df["gross_for_avg_share"].notna()),
        df["gross_for_avg_share"] / df["weekly_total_for_share"],
        np.nan,
    )

    # Base stats (median leaderboard stays base-only)
    df_base = df.dropna(subset=["gross_base"]).copy()
    g_base = (
        df_base.groupby(["show_id", "canonical_title"], as_index=False)
        .agg(
            weeks=("gross_base", "count"),
            first_week=("week_ending", "min"),
            last_week=("week_ending", "max"),
            avg_gross_base=("gross_base", "mean"),
            median_gross_base=("gross_base", "median"),
            peak_gross_base=("gross_base", "max"),
        )
    )

    df_sel = df.dropna(subset=["gross_for_avg_share"]).copy()
    g_sel = (
        df_sel.groupby(["show_id", "canonical_title"], as_index=False)
        .agg(
            avg_gross_selected=("gross_for_avg_share", "mean"),
            peak_gross_selected=("gross_for_avg_share", "max"),
            avg_share=("weekly_share", "mean"),
            median_share=("weekly_share", "median"),
            peak_share=("weekly_share", "max"),
        )
    )

    show_stats = g_base.merge(g_sel, on=["show_id", "canonical_title"], how="left")

    # Peak share week metadata
    df_share = df_sel[df_sel["weekly_share"].notna()].copy()
    if not df_share.empty:
        idx = df_share.groupby(["show_id", "canonical_title"]) ["weekly_share"].idxmax()
        peak_rows = df_share.loc[idx, [
            "show_id", "canonical_title", "week_ending", "gross_for_avg_share", "weekly_total_for_share", "weekly_share"
        ]].rename(columns={
            "week_ending": "peak_share_week",
            "gross_for_avg_share": "peak_share_show_gross",
            "weekly_total_for_share": "peak_share_weekly_total",
            "weekly_share": "peak_share_confirm",
        })
        show_stats = show_stats.merge(peak_rows, on=["show_id", "canonical_title"], how="left")
    else:
        show_stats["peak_share_week"] = pd.NA
        show_stats["peak_share_show_gross"] = np.nan
        show_stats["peak_share_weekly_total"] = np.nan

    show_stats = show_stats[show_stats["weeks"] >= int(min_weeks)].copy()

    if show_stats.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, basis_label

    avg = (
        show_stats.dropna(subset=["avg_gross_selected"])
        .sort_values(["avg_gross_selected", "weeks", "median_gross_base", "peak_gross_selected", "canonical_title"], ascending=[False, False, False, False, True])
        .head(int(top_n)).copy()
    )
    med = (
        show_stats.dropna(subset=["median_gross_base"])
        .sort_values(["median_gross_base", "avg_gross_base", "weeks", "peak_gross_base", "canonical_title"], ascending=[False, False, False, False, True])
        .head(int(top_n)).copy()
    )
    share_mode_key = str(share_mode or "peak").strip().lower()
    if share_mode_key not in {"peak", "average", "median"}:
        share_mode_key = "peak"
    share_col = {"peak": "peak_share", "average": "avg_share", "median": "median_share"}[share_mode_key]

    shr = (
        show_stats.dropna(subset=[share_col])
        .sort_values([share_col, "avg_share", "weeks", "peak_share_show_gross", "canonical_title"], ascending=[False, False, False, False, True])
        .head(int(top_n)).copy()
    )
    shr["_share_mode"] = share_mode_key
    shr["_share_value"] = pd.to_numeric(shr[share_col], errors="coerce")

    for tbl in (avg, med, shr):
        if not tbl.empty:
            tbl.insert(0, "rank", np.arange(1, len(tbl) + 1))

    return avg, med, shr, basis_label


def _render_gross_races_show_leaderboards(base_race: pd.DataFrame, latest_date: date, db_path: str, db_mtime: float):
    st.markdown("### Show Leaderboards")
    st.caption("By-show all-time leaderboards built from weekly chart rows within the selected filters. Median uses base gross only; bonuses can optionally apply to Average + Share.")

    rows = _load_gross_races_show_leaderboard_rows(db_path, db_mtime)
    if rows.empty:
        st.info("No chart rows found for show leaderboards.")
        return

    rows = rows[rows["week_ending_dt"] >= pd.Timestamp(GROSS_TRACKING_START)].copy()
    if rows.empty:
        st.info("No leaderboard rows found on/after the gross-tracking start date (2001-03-17).")
        return

    min_available = rows["week_ending_dt"].min().date()
    max_available = rows["week_ending_dt"].max().date()

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    with c1:
        dmin = st.date_input("Start date", value=min_available, min_value=min_available, max_value=max_available, key="gr_lead_start")
    with c2:
        dmax = st.date_input("End date", value=max_available, min_value=min_available, max_value=max_available, key="gr_lead_end")
    with c3:
        top_n = st.slider("Top N", 5, 200, 25, key="gr_lead_topn")
    with c4:
        min_weeks = st.slider("Min weeks", 1, 100, 5, key="gr_lead_minweeks")

    r1, r2, r3 = st.columns([2, 2, 1.4])
    with r1:
        rank_min, rank_max = st.slider("Rank range", 1, 17, (1, 10), key="gr_lead_rank_range")
    with r2:
        include_bon = st.checkbox(
            "Include bonuses (Average + Share)",
            value=False,
            key="gr_lead_include_bonuses",
            help="Applies bonus-adjusted gross to Average and Share leaderboards. Median leaderboard uses base weekly gross only for comparability.",
        )
    with r3:
        share_mode_label = st.selectbox(
            "Share mode",
            options=["Peak", "Average", "Median"],
            index=0,
            key="gr_lead_share_mode",
            help="Chooses which by-show share metric is ranked in the Share leaderboard.",
        )

    if pd.to_datetime(dmin) > pd.to_datetime(dmax):
        st.warning("Start date is after end date.")
        return

    f = rows[(rows["week_ending_dt"] >= pd.to_datetime(dmin)) & (rows["week_ending_dt"] <= pd.to_datetime(dmax))].copy()
    f = f[(f["rank"] >= int(rank_min)) & (f["rank"] <= int(rank_max))].copy()

    if f.empty:
        st.info("No leaderboard rows match the selected filters.")
        return

    avg, med, shr, basis_label = _build_gross_races_show_leaderboards(
        f, min_weeks=int(min_weeks), top_n=int(top_n), include_bonuses_avg_share=bool(include_bon), share_mode=str(share_mode_label).lower()
    )

    st.caption(
        f"Average/Share basis: **{basis_label}** · Median basis: **Base gross** · "
        f"Share denominator = total weekly gross within the current filtered rank-range universe."
    )

    # KPI row (headline leaders)
    k1, k2, k3 = st.columns(3)
    with k1:
        if not avg.empty:
            row = avg.iloc[0]
            st.metric("Top Avg Gross Show", row["canonical_title"], f"{float(row['avg_gross_selected']):,.2f}")
        else:
            st.metric("Top Avg Gross Show", "—", "")
    with k2:
        if not med.empty:
            row = med.iloc[0]
            st.metric("Top Median Gross Show", row["canonical_title"], f"{float(row['median_gross_base']):,.2f}")
        else:
            st.metric("Top Median Gross Show", "—", "")
    share_mode_key = (str(share_mode_label).strip().lower() if 'share_mode_label' in locals() else "peak")
    share_metric_title = {
        "peak": "Top Peak Share Show",
        "average": "Top Avg Share Show",
        "median": "Top Median Share Show",
    }.get(share_mode_key, "Top Peak Share Show")
    with k3:
        if not shr.empty:
            row = shr.iloc[0]
            share_val = pd.to_numeric(row.get("_share_value", row.get("peak_share")), errors="coerce")
            st.metric(share_metric_title, row["canonical_title"], f"{float(share_val):.2%}" if pd.notna(share_val) else "")
        else:
            st.metric(share_metric_title, "—", "")

    def _fmt_datestr(x):
        if pd.isna(x):
            return x
        try:
            return pd.to_datetime(x).date().isoformat()
        except Exception:
            return x

    st.markdown("#### Highest Average Gross (By Show)")
    st.caption("Ranks shows by mean weekly gross across eligible weeks in the current filters.")
    if avg.empty:
        st.info("No shows meet the minimum-week threshold for the Average leaderboard.")
    else:
        ad = avg[["rank", "canonical_title", "weeks", "avg_gross_selected", "median_gross_base", "peak_gross_selected", "first_week", "last_week"]].copy()
        ad = ad.rename(columns={
            "rank": "Rank", "canonical_title": "Show", "weeks": "Weeks",
            "avg_gross_selected": "Avg Gross", "median_gross_base": "Median Gross (Base)",
            "peak_gross_selected": "Peak Week Gross", "first_week": "First Week", "last_week": "Last Week"
        })
        ad["First Week"] = ad["First Week"].map(_fmt_datestr)
        ad["Last Week"] = ad["Last Week"].map(_fmt_datestr)
        st.dataframe(ad, width='stretch', hide_index=True)

    st.markdown("#### Highest Median Gross (By Show)")
    st.caption("Median leaderboard uses base weekly gross only (bonuses excluded) for comparability.")
    if med.empty:
        st.info("No shows meet the minimum-week threshold for the Median leaderboard.")
    else:
        md = med[["rank", "canonical_title", "weeks", "median_gross_base", "avg_gross_base", "peak_gross_base", "first_week", "last_week"]].copy()
        md = md.rename(columns={
            "rank": "Rank", "canonical_title": "Show", "weeks": "Weeks",
            "median_gross_base": "Median Gross", "avg_gross_base": "Avg Gross (Base)",
            "peak_gross_base": "Peak Week Gross", "first_week": "First Week", "last_week": "Last Week"
        })
        md["First Week"] = md["First Week"].map(_fmt_datestr)
        md["Last Week"] = md["Last Week"].map(_fmt_datestr)
        st.dataframe(md, width='stretch', hide_index=True)

    share_mode_key = (str(share_mode_label).strip().lower() if 'share_mode_label' in locals() else "peak")
    share_title = {
        "peak": "Highest Peak Weekly Share (By Show)",
        "average": "Highest Average Weekly Share (By Show)",
        "median": "Highest Median Weekly Share (By Show)",
    }.get(share_mode_key, "Highest Peak Weekly Share (By Show)")
    share_value_col = {
        "peak": "peak_share",
        "average": "avg_share",
        "median": "median_share",
    }.get(share_mode_key, "peak_share")
    share_display_col = {
        "peak": "Peak Share %",
        "average": "Avg Share %",
        "median": "Median Share %",
    }.get(share_mode_key, "Peak Share %")
    st.markdown(f"#### {share_title}")
    st.caption("Share = show gross ÷ total weekly gross in the current filtered rank-range universe (using the selected Average/Share metric basis).")
    if shr.empty:
        st.info("No shows meet the minimum-week threshold for the Share leaderboard.")
    else:
        sd = shr[["rank", "canonical_title", "weeks", share_value_col, "peak_share_week", "peak_share_show_gross", "peak_share_weekly_total", "avg_share", "median_share", "peak_share"]].copy()
        sd = sd.rename(columns={
            "rank": "Rank", "canonical_title": "Show", "weeks": "Weeks",
            share_value_col: share_display_col, "peak_share_week": "Peak Share Week",
            "peak_share_show_gross": "Show Gross (Peak Week)", "peak_share_weekly_total": "Weekly Total (Peak Week)",
            "avg_share": "Avg Share %", "median_share": "Median Share %", "peak_share": "Peak Share %"
        })
        sd["Peak Share Week"] = sd["Peak Share Week"].map(_fmt_datestr)

        # Avoid duplicate column names when selected mode is also one of the context columns
        for dup_name in ["Peak Share %", "Avg Share %", "Median Share %"]:
            if list(sd.columns).count(dup_name) > 1:
                first = True
                new_cols = []
                for c in sd.columns:
                    if c != dup_name:
                        new_cols.append(c)
                    else:
                        if first:
                            new_cols.append(c)
                            first = False
                        else:
                            new_cols.append(f"{c} (Context)")
                sd.columns = new_cols

        st.dataframe(sd, width='stretch', hide_index=True)


def _render_gross_races_race_views(base: pd.DataFrame, meta: pd.DataFrame, latest_dt: pd.Timestamp, latest_date: date):
    # -------------------------
    # 1) All-Time Gross Races Chart (unlimited rank)
    # -------------------------
    st.markdown("### All-Time Gross Races Chart")

    pick_all_dt = st.date_input(
        "As-of date (pick any date to view all-time totals through that date)",
        value=latest_date,
        min_value=GROSS_TRACKING_START,
        max_value=latest_date,
        key="all_time_race_date",
    )
    pick_all_ts = pd.to_datetime(pick_all_dt)

    base_all = base[base["week_ending_dt"] <= pick_all_ts].copy()
    if base_all.empty:
        st.info("No gross rows found through the selected date (gross-tracking era filter applied).")
    else:
        all_time = base_all.groupby(["show_id", "canonical_title"], as_index=False)["gross_millions"].sum()
        all_time = all_time[all_time["gross_millions"] > 0].copy()

        if not meta.empty:
            all_time = all_time.merge(meta[["show_id", "imprint_1", "imprint_2", "debut_date"]], on="show_id", how="left")
        else:
            all_time["imprint_1"] = ""
            all_time["imprint_2"] = ""
            all_time["debut_date"] = pd.NaT

        all_time["imprint_1"] = all_time["imprint_1"].fillna("")
        all_time["imprint_2"] = all_time["imprint_2"].fillna("")

        all_time = all_time.sort_values("gross_millions", ascending=False).reset_index(drop=True)
        all_time.insert(0, "rank", np.arange(1, len(all_time) + 1))

        st.caption(f"Unlimited rank: every show with any gross is included. (Through **{pick_all_dt.isoformat()}**)")
        all_time_disp = all_time[["rank", "canonical_title", "imprint_1", "imprint_2", "debut_date", "gross_millions"]].copy()
        st.dataframe(all_time_disp, width='stretch', hide_index=True)

        with st.expander("Optional: visualize the leaders (bar chart)"):
            top_plot = st.slider("How many shows to display in the bar chart", 5, min(200, int(len(all_time))), min(50, int(len(all_time))))
            top_block = all_time.head(int(top_plot)).copy()
            plot_barh(top_block["canonical_title"][::-1], top_block["gross_millions"][::-1], "Total Gross (Millions)", "Show")

    st.divider()

    # -------------------------
    # 2) Annual Gross Races
    # -------------------------
    st.markdown("### Annual Gross Races")
    st.caption("Cumulative grosses reset at the start of each year.")

    pick_dt = st.date_input(
        "As-of date (pick any date to view that year's race)",
        value=latest_date,
        min_value=GROSS_TRACKING_START,
        max_value=latest_date,
        key="annual_race_date"
    )
    pick_ts = pd.to_datetime(pick_dt)

    ydf, weeks = _year_cumulative(base, int(pick_dt.year), pick_ts)
    if ydf.empty:
        st.info("No gross rows found for that year (through the selected date).")
    else:
        # Leaderboard as-of date
        last = ydf.sort_values(["show_id", "week_ending_dt"]).groupby(["show_id", "canonical_title"], as_index=False).tail(1)
        leaders = last[["show_id", "canonical_title", "cum_gross_millions"]].copy()

        if not meta.empty:
            leaders = leaders.merge(meta[["show_id", "imprint_1", "imprint_2"]], on="show_id", how="left")
        else:
            leaders["imprint_1"] = ""
            leaders["imprint_2"] = ""

        leaders["imprint_1"] = leaders["imprint_1"].fillna("")
        leaders["imprint_2"] = leaders["imprint_2"].fillna("")

        leaders = leaders.sort_values("cum_gross_millions", ascending=False).reset_index(drop=True)
        leaders.insert(0, "rank", np.arange(1, len(leaders) + 1))
        leader_total = float(leaders["cum_gross_millions"].iloc[0]) if not leaders.empty else 0.0
        leaders["grosses_behind_leader"] = leader_total - pd.to_numeric(leaders["cum_gross_millions"], errors="coerce").fillna(0.0)
        leaders = leaders[["rank", "canonical_title", "imprint_1", "imprint_2", "cum_gross_millions", "grosses_behind_leader"]].copy()
        leaders_disp = leaders.copy()
        leaders_disp["grosses_behind_leader"] = leaders_disp["grosses_behind_leader"].map(
            lambda x: "Leader" if pd.notna(x) and abs(float(x)) < 1e-12 else f"-{float(x):,.1f}"
        )
        leaders_styler = leaders_disp.style.format({"cum_gross_millions": "{:,.1f}"}).map(
            lambda v: "color: red;" if isinstance(v, str) and v.startswith("-") else "",
            subset=["grosses_behind_leader"],
        )

        st.caption(f"Leaderboard for **{pick_dt.year}** (through **{pick_dt.isoformat()}**)" )
        st.dataframe(leaders_styler, width='stretch', hide_index=True)

        # Line chart for top K at the selected date
        top_k = st.slider("Shows to plot (annual)", 2, min(50, int(len(leaders))), min(10, int(len(leaders))))
        top_titles = leaders.head(int(top_k))["canonical_title"].tolist()

        piv = ydf[ydf["canonical_title"].isin(top_titles)].copy()
        piv = piv.pivot_table(index="week_ending_dt", columns="canonical_title", values="cum_gross_millions", aggfunc="max").sort_index()
        piv = piv.reindex(pd.to_datetime(weeks)).ffill()

        series_by_label = {c: piv[c] for c in piv.columns}
        _plot_multi_line(list(piv.index), series_by_label, "Week Ending", "Cumulative Gross (Millions)")

    st.divider()

    # -------------------------
    # 3) Quarter Gross Races
    # -------------------------
    st.markdown("### Quarter Gross Races")
    st.caption("Cumulative grosses reset at the start of each quarter.")

    # Default to the current quarter/year based on latest week ending
    cur_year = int(latest_dt.year)
    cur_quarter = int(((latest_dt.month - 1) // 3) + 1)

    q1, q2, q3 = st.columns([1, 2, 2])
    with q1:
        quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=cur_quarter - 1, key="q_race_quarter")
    with q2:
        # Years available for this quarter
        start_month = (int(quarter) - 1) * 3 + 1
        end_month = start_month + 2
        years_avail = (
            base[(base["week_ending_dt"].dt.month >= start_month) & (base["week_ending_dt"].dt.month <= end_month)]["week_ending_dt"]
            .dt.year.dropna().astype(int).unique().tolist()
        )
        years_avail = sorted(set(years_avail))
        if not years_avail:
            years_avail = [cur_year]
        year_pick = st.selectbox("Year", options=years_avail, index=years_avail.index(cur_year) if cur_year in years_avail else len(years_avail) - 1, key="q_race_year")

    # Week dropdown depends on quarter/year selection
    start_month = (int(quarter) - 1) * 3 + 1
    end_month = start_month + 2
    q_weeks = base[(base["week_ending_dt"].dt.year == int(year_pick))].copy()
    q_weeks = q_weeks[(q_weeks["week_ending_dt"].dt.month >= start_month) & (q_weeks["week_ending_dt"].dt.month <= end_month)].copy()
    q_week_list = sorted(pd.to_datetime(q_weeks["week_ending_dt"].dropna().unique()))
    q_week_list = [pd.to_datetime(x).normalize() for x in q_week_list]

    if not q_week_list:
        st.info("No weeks found for that quarter/year.")
        return

    latest_norm = pd.to_datetime(latest_dt).normalize()

    # Default week: latest week if it's in this quarter/year, else last week of that quarter
    default_week_idx = len(q_week_list) - 1
    if (latest_norm.year == int(year_pick)) and (latest_norm in q_week_list):
        default_week_idx = q_week_list.index(latest_norm)

    with q3:
        wk_num = st.selectbox(
            "Week",
            options=list(range(1, len(q_week_list) + 1)),
            index=default_week_idx,
            format_func=lambda i: f"Week {i}",
            key="q_race_week",
        )

    wk_dt = pd.to_datetime(q_week_list[int(wk_num) - 1])
    st.caption(f"Selected week ending: **{wk_dt.date().isoformat()}**")

    qdf, qweeks = _quarter_cumulative(base, int(year_pick), int(quarter), wk_dt)
    if qdf.empty:
        st.info("No gross rows found for that quarter (through the selected week).")
        return

    lastq = qdf.sort_values(["show_id", "week_ending_dt"]).groupby(["show_id", "canonical_title"], as_index=False).tail(1)
    leaders_q = lastq[["show_id", "canonical_title", "cum_gross_millions"]].copy()

    if not meta.empty:
        leaders_q = leaders_q.merge(meta[["show_id", "imprint_1", "imprint_2"]], on="show_id", how="left")
    else:
        leaders_q["imprint_1"] = ""
        leaders_q["imprint_2"] = ""

    leaders_q["imprint_1"] = leaders_q["imprint_1"].fillna("")
    leaders_q["imprint_2"] = leaders_q["imprint_2"].fillna("")

    leaders_q = leaders_q.sort_values("cum_gross_millions", ascending=False).reset_index(drop=True)
    leaders_q.insert(0, "rank", np.arange(1, len(leaders_q) + 1))
    leader_total_q = float(leaders_q["cum_gross_millions"].iloc[0]) if not leaders_q.empty else 0.0
    leaders_q["grosses_behind_leader"] = leader_total_q - pd.to_numeric(leaders_q["cum_gross_millions"], errors="coerce").fillna(0.0)
    leaders_q = leaders_q[["rank", "canonical_title", "imprint_1", "imprint_2", "cum_gross_millions", "grosses_behind_leader"]].copy()
    leaders_q_disp = leaders_q.copy()
    leaders_q_disp["grosses_behind_leader"] = leaders_q_disp["grosses_behind_leader"].map(
        lambda x: "Leader" if pd.notna(x) and abs(float(x)) < 1e-12 else f"-{float(x):,.1f}"
    )
    leaders_q_styler = leaders_q_disp.style.format({"cum_gross_millions": "{:,.1f}"}).map(
        lambda v: "color: red;" if isinstance(v, str) and v.startswith("-") else "",
        subset=["grosses_behind_leader"],
    )

    st.caption(f"Leaderboard for **Q{int(quarter)} {int(year_pick)}** (through **{wk_dt.date().isoformat()}**)" )
    st.dataframe(leaders_q_styler, width='stretch', hide_index=True)

    top_kq = st.slider("Shows to plot (quarter)", 2, min(50, int(len(leaders_q))), min(10, int(len(leaders_q))), key="q_race_topk")
    top_titles_q = leaders_q.head(int(top_kq))["canonical_title"].tolist()

    pivq = qdf[qdf["canonical_title"].isin(top_titles_q)].copy()
    pivq = pivq.pivot_table(index="week_ending_dt", columns="canonical_title", values="cum_gross_millions", aggfunc="max").sort_index()
    pivq = pivq.reindex(pd.to_datetime(qweeks)).ffill()

    series_by_label_q = {c: pivq[c] for c in pivq.columns}
    _plot_multi_line(list(pivq.index), series_by_label_q, "Week Ending", "Cumulative Gross (Millions)")

    st.divider()

    # -------------------------
    # 4) Monthly Gross Races (shared chart-month logic)
    # -------------------------
    st.markdown("### Monthly Gross Races")
    st.caption("Uses the same chart-month logic as the monthly charts and records (cutoff day 28).")

    pick_m_dt = st.date_input(
        "As-of date (pick any date to view that chart-month race)",
        value=latest_date,
        min_value=GROSS_TRACKING_START,
        max_value=latest_date,
        key="gross_races_month_pick",
    )
    through_m_dt = pd.to_datetime(pick_m_dt)

    mbase = _apply_chart_month_logic(base, week_dt_col="week_ending_dt", week_str_col="week_ending")
    pick_month = _apply_chart_month_logic(
        pd.DataFrame({"week_ending": [through_m_dt.date().isoformat()], "week_ending_dt": [through_m_dt]}),
        week_dt_col="week_ending_dt",
        week_str_col="week_ending",
    )["month"].iloc[0]

    mbase = mbase[mbase["month"].eq(pick_month)].copy()
    mbase = mbase[mbase["week_ending_dt"] <= through_m_dt].copy()

    if mbase.empty:
        st.info("No monthly data available for that chart-month/date range (gross-tracking era filter applied).")
    else:
        mdf = mbase.sort_values(["show_id", "week_ending_dt"]).reset_index(drop=True)
        mdf["cum_gross_millions"] = mdf.groupby("show_id")["gross_millions"].cumsum()
        mweeks = sorted(mdf["week_ending_dt"].dropna().unique().tolist())

        lastm = mdf.sort_values(["show_id", "week_ending_dt"]).groupby(["show_id", "canonical_title"], as_index=False).tail(1)
        leaders_m = lastm[["show_id", "canonical_title", "cum_gross_millions"]].copy()

        if not meta.empty:
            leaders_m = leaders_m.merge(meta[["show_id", "imprint_1", "imprint_2"]], on="show_id", how="left")
        else:
            leaders_m["imprint_1"] = ""
            leaders_m["imprint_2"] = ""

        leaders_m["imprint_1"] = leaders_m["imprint_1"].fillna("")
        leaders_m["imprint_2"] = leaders_m["imprint_2"].fillna("")

        leaders_m = leaders_m.sort_values("cum_gross_millions", ascending=False).reset_index(drop=True)
        leaders_m.insert(0, "rank", np.arange(1, len(leaders_m) + 1))
        leaders_m = leaders_m[["rank", "canonical_title", "imprint_1", "imprint_2", "cum_gross_millions"]].copy()

        cycle_start = pd.to_datetime(mdf["week_ending_dt"].min()).date().isoformat()
        cycle_end = pd.to_datetime(mdf["week_ending_dt"].max()).date().isoformat()
        st.caption(
            f"Leaderboard for chart-month **{pick_month}** "
            f"(weeks currently in range: **{cycle_start} → {cycle_end}**, through **{through_m_dt.date().isoformat()}** )"
        )
        st.dataframe(leaders_m, width='stretch', hide_index=True)

        top_km = st.slider(
            "Shows to plot (monthly)",
            2,
            min(50, int(len(leaders_m))),
            min(10, int(len(leaders_m))),
            key="m_race_topk",
        )
        top_titles_m = leaders_m.head(int(top_km))["canonical_title"].tolist()

        pivm = mdf[mdf["canonical_title"].isin(top_titles_m)].copy()
        pivm = pivm.pivot_table(
            index="week_ending_dt",
            columns="canonical_title",
            values="cum_gross_millions",
            aggfunc="max",
        ).sort_index()
        pivm = pivm.reindex(pd.to_datetime(mweeks)).ffill()

        series_by_label_m = {c: pivm[c] for c in pivm.columns}
        _plot_multi_line(list(pivm.index), series_by_label_m, "Week Ending", "Cumulative Gross (Millions)")

    return

    st.divider()
    st.markdown("### Per-show streak breakdown")
    title_pick = st.selectbox("Show (canonical)", shows["canonical_title"].tolist(), key="streak_show_pick")
    show_id = int(shows.loc[shows["canonical_title"] == title_pick, "show_id"].iloc[0])

    show_block = streaks[streaks["show_id"] == show_id].sort_values(["rank"]).copy()
    if show_block.empty:
        st.info("No streak data for this show in the selected filters.")
        return

    st.dataframe(show_block, width='stretch')

    st.markdown("### Quick peek: raw weeks for this show (filtered)")
    # Useful for validating consecutive week_number behavior
    show_rows = rows[rows["show_id"] == show_id].copy()
    show_rows["week_ending"] = _as_date_str(show_rows["week_ending"])
    st.dataframe(show_rows.sort_values(["week_number", "rank", "pos"]), width='stretch')


def tab_gross_races():
    st.subheader("Gross Races")
    st.caption("All-time, annual, quarter, and monthly gross races, plus by-show gross leaderboards.")

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}.")
        return

    db_mtime = DB_PATH.stat().st_mtime
    base = _load_gross_races_base(str(DB_PATH), db_mtime)
    if base.empty:
        st.info("No gross rows found in the database.")
        return

    base = base.copy()
    if "week_ending" not in base.columns:
        st.error("Gross Races data is missing week_ending.")
        return

    base["week_ending"] = _as_date_str(base["week_ending"])
    base["week_ending_dt"] = pd.to_datetime(base["week_ending"], errors="coerce")
    base = base.dropna(subset=["week_ending_dt"]).copy()
    base = base[base["week_ending_dt"].dt.date >= GROSS_TRACKING_START].copy()
    if base.empty:
        st.info("No gross rows found on or after the gross-tracking start date (2001-03-17).")
        return

    latest_dt = pd.to_datetime(base["week_ending_dt"].max())
    latest_date = latest_dt.date()
    meta = _load_show_meta_for_gross_races(str(DB_PATH), db_mtime)

    subtab_race, subtab_lead, subtab_entries = st.tabs(["Race Views", "Show Leaderboards", "Top Gross Entries"])
    with subtab_race:
        _render_gross_races_race_views(base, meta, latest_dt, latest_date)
    with subtab_lead:
        _render_gross_races_show_leaderboards(base, latest_date, str(DB_PATH), db_mtime)
    with subtab_entries:
        _render_gross_races_all_gross_entries(base, meta, latest_date)


# ----------------------------
# New tab: Holidays
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_distinct_imprints() -> list[str]:
    """All distinct non-empty imprints from imprint_1 and imprint_2."""
    df = sql_df(
        """
        WITH imps AS (
          SELECT TRIM(imprint_1) AS imp FROM t10_entry WHERE imprint_1 IS NOT NULL AND TRIM(imprint_1) <> ''
          UNION
          SELECT TRIM(imprint_2) AS imp FROM t10_entry WHERE imprint_2 IS NOT NULL AND TRIM(imprint_2) <> ''
        )
        SELECT imp FROM imps ORDER BY imp
        """
    )
    if df.empty:
        return []
    return df["imp"].astype(str).tolist()

def tab_holidays():
    st.subheader("Holidays: #1 show by year")
    st.caption("Pick a holiday and see the #1 show(s) for the holiday week, by year. (Ties supported.)")

    week_endings = fetch_week_endings_distinct()
    if not week_endings:
        st.info("No week endings found in the database.")
        return

    min_year = min(d.year for d in week_endings)
    max_year = max(d.year for d in week_endings)

    holiday_name = st.selectbox("Holiday", list(HOLIDAYS.keys()))
    maker = HOLIDAYS[holiday_name]

    c1, c2 = st.columns(2)
    with c1:
        year_start = st.number_input("Start year", min_value=min_year, max_value=max_year, value=min_year, step=1)
    with c2:
        year_end = st.number_input("End year", min_value=min_year, max_value=max_year, value=max_year, step=1)

    if year_start > year_end:
        year_start, year_end = year_end, year_start

    rows_out: list[dict[str, Any]] = []
    for y in range(int(year_start), int(year_end) + 1):
        hdt = maker(y)
        we = holiday_week_ending_for_date(week_endings, hdt, holiday_name)
        if we is None:
            # If the computed week_ending isn't present in the DB (future/unreached or missing),
            # keep the year in the table but leave values blank/dashed.
            rows_out.append({
                "year": y,
                "holiday_date": hdt.isoformat(),
                "week_ending": None,
                "#1_show(s)": None,
                "imprint_1": None,
                "imprint_2": None,
                "gross_millions_sum": None,
            })
            continue

        we_str = we.isoformat()

        # date(e.week_ending)=? handles cases where week_ending has a time component
        top = sql_df("""
            SELECT
              s.canonical_title,
              e.pos,
              e.imprint_1,
              e.imprint_2,
              e.gross_millions AS base_gross_millions,
              COALESCE(gb.bonus_millions, 0) AS bonus_millions,
              (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions
            FROM t10_entry e
            LEFT JOIN (
              SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
              FROM gross_bonus
              GROUP BY show_id, week_ending
            ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
            JOIN show s ON s.show_id = e.show_id
            WHERE date(e.week_ending) = ?
              AND e.rank = 1
            ORDER BY e.pos ASC, s.canonical_title ASC
        """, (we_str,))

        if top.empty:
            rows_out.append({
                "year": y,
                "holiday_date": hdt.isoformat(),
                "week_ending": we_str,
                "#1_show(s)": None,
                "imprint_1": None,
                "imprint_2": None,
                "gross_millions_sum": None,
            })
            continue

        # If ties: join titles
        titles = top["canonical_title"].astype(str).tolist()
        im1 = top["imprint_1"].astype("string").fillna("").replace("", pd.NA).dropna().unique().tolist()
        im2 = top["imprint_2"].astype("string").fillna("").replace("", pd.NA).dropna().unique().tolist()

        gross = pd.to_numeric(top["gross_millions"], errors="coerce").dropna()
        gross_sum = float(gross.sum()) if len(gross) else None

        rows_out.append({
            "year": y,
            "holiday_date": hdt.isoformat(),
            "week_ending": we_str,
            "#1_show(s)": " / ".join(titles),
            "imprint_1": " / ".join(im1) if im1 else None,
            "imprint_2": " / ".join(im2) if im2 else None,
            "gross_millions_sum": gross_sum,
        })

    out = pd.DataFrame(rows_out).sort_values("year")

    out_disp = out.copy()
    if not out_disp.empty and "gross_millions_sum" in out_disp.columns:
        out_disp["gross_millions_sum"] = out_disp["gross_millions_sum"].apply(
            lambda v: (f"{float(v):.1f}" if pd.notna(v) else pd.NA)
        )
    st.dataframe(out_disp.fillna("—"), width='stretch')

    miss = out["#1_show(s)"].isna().sum() if not out.empty else 0
    if miss:
        st.warning(
            f"{miss} year(s) had no #1 record for the computed holiday-week. "
            "This usually means your database doesn’t have that week, or rank=1 is missing for that week."
        )

    with st.expander("How the holiday week is chosen"):
        st.write(
            "- Fixed-date holidays (New Year's, Valentine's, Independence Day, Halloween, Christmas):\n"
            "  - Sun/Mon/Tue → previous weekend (Saturday before)\n"
            "  - Wed/Thu/Fri/Sat → current week (Saturday on/after)\n"
            "- Thanksgiving → following week ending (Saturday after)\n"
            "- Easter/Memorial Day/Labor Day/MLK Day/Presidents Day → Saturday before\n"
        )

def tab_admin():
    st.subheader("Admin (Normalize titles: aliases + merges)")
    st.warning("This edits the database. If you're experimenting, copy t10.sqlite first.")
    shows, _ = load_lists()
    titles = shows["canonical_title"].tolist()

    st.markdown("### Add alias (map a raw title string to a canonical show)")
    col1, col2 = st.columns(2)
    with col1:
        canonical = st.selectbox("Canonical show", titles, key="alias_canonical")
    with col2:
        alias = st.text_input("Alias title (exact)", key="alias_title", placeholder="Type the exact variant you want to map")

    if st.button("Add alias mapping"):
        if not alias.strip():
            st.error("Alias title can't be blank.")
        else:
            show_id = int(shows.loc[shows["canonical_title"] == canonical, "show_id"].iloc[0])
            sql_exec("INSERT OR REPLACE INTO show_alias(alias_title, show_id) VALUES (?, ?)", (alias.strip(), show_id))
            sql_df.clear()
            load_lists.clear()
            st.success("Alias saved.")

    st.markdown("### Rename canonical show title")
    r1, r2 = st.columns(2)
    with r1:
        rename_from = st.selectbox("Current canonical title", titles, key="rename_canonical_from")
    with r2:
        rename_to = st.text_input(
            "New canonical title",
            key="rename_canonical_to",
            placeholder="Type the new canonical title",
        )

    if st.button("Rename canonical title"):
        new_title = rename_to.strip()
        if not new_title:
            st.error("New canonical title can't be blank.")
        elif new_title == rename_from:
            st.error("New title matches the current title.")
        elif (shows["canonical_title"].astype(str).str.casefold() == new_title.casefold()).any():
            st.error("That canonical title already exists. Use Merge two canonical shows instead.")
        else:
            show_id = int(shows.loc[shows["canonical_title"] == rename_from, "show_id"].iloc[0])
            con = get_con()
            try:
                cur = con.cursor()
                cur.execute("BEGIN;")
                cur.execute("UPDATE show SET canonical_title = ? WHERE show_id = ?", (new_title, show_id))
                cur.execute(
                    "INSERT OR IGNORE INTO show_alias(alias_title, show_id) VALUES (?, ?)",
                    (rename_from, show_id),
                )
                con.commit()
            finally:
                con.close()

            sql_df.clear()
            load_lists.clear()
            st.success(f"Renamed '{rename_from}' to '{new_title}'.")

    st.markdown("### Merge two canonical shows (combine histories)")
    c1, c2 = st.columns(2)
    with c1:
        keep = st.selectbox("Keep (target canonical)", titles, key="merge_keep")
    with c2:
        merge = st.selectbox("Merge (source canonical)", titles, key="merge_src")

    if st.button("Merge these shows"):
        if keep == merge:
            st.error("Pick two different shows.")
        else:
            keep_id = int(shows.loc[shows["canonical_title"] == keep, "show_id"].iloc[0])
            src_id = int(shows.loc[shows["canonical_title"] == merge, "show_id"].iloc[0])

            con = get_con()
            try:
                cur = con.cursor()
                cur.execute("BEGIN;")

                cur.execute("UPDATE t10_entry SET show_id = ? WHERE show_id = ?", (keep_id, src_id))

                cur.execute("""
                    INSERT OR IGNORE INTO show_alias(alias_title, show_id)
                    SELECT alias_title, ? FROM show_alias WHERE show_id = ?
                """, (keep_id, src_id))

                cur.execute("INSERT OR IGNORE INTO show_alias(alias_title, show_id) VALUES (?, ?)", (merge, keep_id))
                cur.execute("DELETE FROM show WHERE show_id = ?", (src_id,))

                con.commit()
            finally:
                con.close()

            sql_df.clear()
            load_lists.clear()
            st.success(f"Merged '{merge}' into '{keep}'.")

    
    st.markdown("### Merge imprint labels (relabel imprint_1 / imprint_2)")
    with st.expander("Merge/rename an imprint", expanded=False):
        st.caption("Replaces one imprint label with another across all weeks (both imprint_1 and imprint_2).")

        imprints = fetch_distinct_imprints()
        if not imprints:
            st.info("No imprints found in the database.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                from_imp = st.selectbox("From (imprint to replace)", imprints, key="imp_merge_from")
            with c2:
                to_imp_pick = st.selectbox("To (existing imprint)", imprints, key="imp_merge_to_pick")

            to_imp_custom = st.text_input("Or type a new imprint label (optional)", value="", key="imp_merge_to_custom")
            to_imp = to_imp_custom.strip() if to_imp_custom.strip() else to_imp_pick

            if from_imp == to_imp:
                st.warning("Pick two different imprint labels.")
            else:
                preview = sql_df(
                    """
                    SELECT
                      SUM(CASE WHEN imprint_1 = :f THEN 1 ELSE 0 END) AS hits_imprint_1,
                      SUM(CASE WHEN imprint_2 = :f THEN 1 ELSE 0 END) AS hits_imprint_2
                    FROM t10_entry
                    """,
                    params={"f": from_imp},
                )
                h1 = int(preview.loc[0, "hits_imprint_1"]) if not preview.empty else 0
                h2 = int(preview.loc[0, "hits_imprint_2"]) if not preview.empty else 0
                st.write(f"Rows to change: imprint_1 = **{h1}**, imprint_2 = **{h2}**")

                if st.button("Merge imprint", type="primary", key="imp_merge_apply"):
                    con = get_con()
                    try:
                        cur = con.cursor()
                        cur.execute("BEGIN;")

                        cur.execute("UPDATE t10_entry SET imprint_1 = ? WHERE imprint_1 = ?", (to_imp, from_imp))
                        cur.execute("UPDATE t10_entry SET imprint_2 = ? WHERE imprint_2 = ?", (to_imp, from_imp))

                        # Normalize whitespace
                        cur.execute("UPDATE t10_entry SET imprint_1 = TRIM(imprint_1) WHERE imprint_1 IS NOT NULL;")
                        cur.execute("UPDATE t10_entry SET imprint_2 = TRIM(imprint_2) WHERE imprint_2 IS NOT NULL;")

                        # If imprint_1 empty but imprint_2 filled, shift up
                        cur.execute(
                            """
                            UPDATE t10_entry
                            SET imprint_1 = imprint_2, imprint_2 = NULL
                            WHERE (imprint_1 IS NULL OR TRIM(imprint_1) = '')
                              AND imprint_2 IS NOT NULL AND TRIM(imprint_2) <> ''
                            """
                        )

                        # If imprint_2 duplicates imprint_1, drop imprint_2
                        cur.execute(
                            """
                            UPDATE t10_entry
                            SET imprint_2 = NULL
                            WHERE imprint_1 IS NOT NULL AND TRIM(imprint_1) <> ''
                              AND imprint_2 IS NOT NULL AND TRIM(imprint_2) <> ''
                              AND imprint_1 = imprint_2
                            """
                        )

                        con.commit()
                        st.success(f"Merged imprint '{from_imp}' → '{to_imp}'.")
                    finally:
                        con.close()

                    # refresh cached lists/data
                    try:
                        sql_df.clear()
                    except Exception:
                        pass
                    try:
                        load_lists.clear()
                    except Exception:
                        pass
                    try:
                        fetch_distinct_imprints.clear()
                    except Exception:
                        pass
                    st.rerun()


    # Safety: refresh lists here so titles/shows are always defined (and up-to-date)
    try:
        shows, _ = load_lists()
    except Exception:
        shows = pd.DataFrame(columns=["show_id", "canonical_title"])
    titles = (
        shows["canonical_title"].astype(str).tolist()
        if isinstance(shows, pd.DataFrame) and "canonical_title" in shows.columns
        else []
    )

    st.markdown("### Set imprints for a show")
    with st.expander("Add / fix imprints for one show", expanded=False):
        st.caption(
            "Set imprint_1 and/or imprint_2 for a specific show across its weeks. "
            "This keeps both imprints when they differ (no forced consolidation)."
        )

        if not titles:
            st.info("No shows found.")
        else:
            sel_title = st.selectbox("Show", titles, key="imp_set_show")
            sel_id = int(shows.loc[shows["canonical_title"] == sel_title, "show_id"].iloc[0])

            stats = sql_df(
                """
                SELECT
                  COUNT(*) AS weeks,
                  SUM(CASE
                        WHEN (imprint_1 IS NULL OR TRIM(imprint_1) = '')
                         AND (imprint_2 IS NULL OR TRIM(imprint_2) = '' OR lower(TRIM(imprint_2)) IN ('(blank)','blank'))
                        THEN 1 ELSE 0 END) AS missing_both,
                  SUM(CASE
                        WHEN (imprint_1 IS NULL OR TRIM(imprint_1) = '')
                         AND (imprint_2 IS NOT NULL AND TRIM(imprint_2) <> '' AND lower(TRIM(imprint_2)) NOT IN ('(blank)','blank'))
                        THEN 1 ELSE 0 END) AS missing_imprint_1,
                  SUM(CASE
                        WHEN (imprint_2 IS NULL OR TRIM(imprint_2) = '' OR lower(TRIM(imprint_2)) IN ('(blank)','blank'))
                         AND (imprint_1 IS NOT NULL AND TRIM(imprint_1) <> '' AND lower(TRIM(imprint_1)) NOT IN ('(blank)','blank'))
                        THEN 1 ELSE 0 END) AS missing_imprint_2
                FROM t10_entry
                WHERE show_id = ?
                """,
                (sel_id,),
            )
            if not stats.empty:
                s_weeks = 0 if pd.isna(stats.loc[0,'weeks']) else int(stats.loc[0,'weeks'])
                s_missing_both = 0 if pd.isna(stats.loc[0,'missing_both']) else int(stats.loc[0,'missing_both'])
                s_missing_1 = 0 if pd.isna(stats.loc[0,'missing_imprint_1']) else int(stats.loc[0,'missing_imprint_1'])
                s_missing_2 = 0 if pd.isna(stats.loc[0,'missing_imprint_2']) else int(stats.loc[0,'missing_imprint_2'])
                st.write(
                    f"Weeks: **{s_weeks}** · "
                    f"Missing both: **{s_missing_both}** · "
                    f"Missing imprint_1: **{s_missing_1}** · "
                    f"Missing imprint_2: **{s_missing_2}**"
                )


            st.markdown("**Current imprint pairs in the data**")
            cur_pairs = sql_df(
                """
                SELECT DISTINCT
                  COALESCE(NULLIF(TRIM(imprint_1), ''), '(blank)') AS imprint_1,
                  COALESCE(NULLIF(TRIM(imprint_2), ''), '(blank)') AS imprint_2
                FROM t10_entry
                WHERE show_id = ?
                ORDER BY imprint_1, imprint_2
                """,
                (sel_id,),
            )
            st.dataframe(cur_pairs, width='stretch', hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                new_imp1 = st.text_input("Set imprint_1 to (optional)", value="", key="imp_set_1").strip()
            with c2:
                new_imp2 = st.text_input("Set imprint_2 to (optional)", value="", key="imp_set_2").strip()

            mode = st.radio(
                "Apply mode",
                ["Fill missing only", "Overwrite existing (dangerous)"],
                index=0,
                horizontal=True,
                key="imp_set_mode",
            )

            clear1 = False
            clear2 = False
            if mode == "Overwrite existing (dangerous)":
                c3, c4 = st.columns(2)
                with c3:
                    clear1 = st.checkbox("Clear imprint_1 (set blank)", value=False, key="imp_set_clear1")
                with c4:
                    clear2 = st.checkbox("Clear imprint_2 (set blank)", value=False, key="imp_set_clear2")


            if mode == "Fill missing only":
                st.caption("Tip: leave a field blank to avoid changing that imprint column.")
                fill_both = st.checkbox(
                    "Fill weeks where BOTH imprints are blank",
                    value=True,
                    key="imp_set_fill_both",
                )
                fill1 = st.checkbox(
                    "Also fill imprint_1 when blank (even if imprint_2 is present)",
                    value=False,
                    key="imp_set_fill1",
                )
                fill2 = st.checkbox(
                    "Also fill imprint_2 when blank (even if imprint_1 is present)",
                    value=False,
                    key="imp_set_fill2",
                )
                confirm_ok = True
            else:
                confirm_ok = st.checkbox(
                    "I understand this will overwrite imprints for ALL weeks of this show.",
                    value=False,
                    key="imp_set_confirm_overwrite",
                )
                fill_both = fill1 = fill2 = False  # not used

            if st.button("Apply imprint update", type="primary", key="imp_set_apply"):
                if not confirm_ok:
                    st.warning("Please confirm overwrite to proceed.")
                elif not (new_imp1 or new_imp2 or clear1 or clear2):
                    st.warning("Enter at least one imprint value to apply, or use the Clear checkbox(es).")
                else:
                    con = get_con()
                    try:
                        cur = con.cursor()
                        cur.execute("BEGIN;")

                        # Helper: conditions
                        cond_both_blank = (
                            "(imprint_1 IS NULL OR TRIM(imprint_1) = '' OR lower(TRIM(imprint_1)) IN ('(blank)','blank')) "
                            "AND (imprint_2 IS NULL OR TRIM(imprint_2) = '' OR lower(TRIM(imprint_2)) IN ('(blank)','blank'))"
                        )
                        cond_i1_blank_i2_present = (
                            "(imprint_1 IS NULL OR TRIM(imprint_1) = '' OR lower(TRIM(imprint_1)) IN ('(blank)','blank')) "
                            "AND (imprint_2 IS NOT NULL AND TRIM(imprint_2) <> '' AND lower(TRIM(imprint_2)) NOT IN ('(blank)','blank'))"
                        )
                        cond_i2_blank_i1_present = (
                            "(imprint_2 IS NULL OR TRIM(imprint_2) = '') "
                            "AND (imprint_1 IS NOT NULL AND TRIM(imprint_1) <> '' AND lower(TRIM(imprint_1)) NOT IN ('(blank)','blank'))"
                        )

                        if mode == "Fill missing only":
                            if fill_both:
                                if new_imp1:
                                    cur.execute(
                                        f"UPDATE t10_entry SET imprint_1 = ? WHERE show_id = ? AND {cond_both_blank}",
                                        (new_imp1, sel_id),
                                    )
                                if new_imp2:
                                    cur.execute(
                                        f"UPDATE t10_entry SET imprint_2 = ? WHERE show_id = ? AND {cond_both_blank}",
                                        (new_imp2, sel_id),
                                    )

                            if fill1 and new_imp1:
                                cur.execute(
                                    f"UPDATE t10_entry SET imprint_1 = ? WHERE show_id = ? AND {cond_i1_blank_i2_present}",
                                    (new_imp1, sel_id),
                                )

                            if fill2 and new_imp2:
                                cur.execute(
                                    f"UPDATE t10_entry SET imprint_2 = ? WHERE show_id = ? AND {cond_i2_blank_i1_present}",
                                    (new_imp2, sel_id),
                                )
                        else:
                            # Overwrite (allow explicit clearing to true blank/NULL)
                            if clear1:
                                cur.execute("UPDATE t10_entry SET imprint_1 = NULL WHERE show_id = ?", (sel_id,))
                            elif new_imp1:
                                cur.execute(
                                    "UPDATE t10_entry SET imprint_1 = ? WHERE show_id = ?",
                                    (new_imp1, sel_id),
                                )

                            if clear2:
                                cur.execute("UPDATE t10_entry SET imprint_2 = NULL WHERE show_id = ?", (sel_id,))
                            elif new_imp2:
                                cur.execute(
                                    "UPDATE t10_entry SET imprint_2 = ? WHERE show_id = ?",
                                    (new_imp2, sel_id),
                                )

                        # Normalize whitespace; remove exact duplicates only (keeps distinct pairs)
                        cur.execute("UPDATE t10_entry SET imprint_1 = TRIM(imprint_1) WHERE imprint_1 IS NOT NULL;")
                        cur.execute("UPDATE t10_entry SET imprint_2 = TRIM(imprint_2) WHERE imprint_2 IS NOT NULL;")
                        cur.execute(
                            """
                            UPDATE t10_entry
                            SET imprint_2 = NULL
                            WHERE show_id = ?
                              AND imprint_1 IS NOT NULL AND TRIM(imprint_1) <> ''
                              AND imprint_2 IS NOT NULL AND TRIM(imprint_2) <> ''
                              AND imprint_1 = imprint_2
                            """,
                            (sel_id,),
                        )

                        con.commit()
                    finally:
                        con.close()

                    try:
                        sql_df.clear()
                    except Exception:
                        pass
                    try:
                        fetch_distinct_imprints.clear()
                    except Exception:
                        pass
                    st.success("Imprints updated.")
                    st.rerun()

    st.markdown('---')
    
    st.markdown('---')
    st.markdown("### Export show list (show_id)")
    export_show_df = sql_df("SELECT show_id, canonical_title FROM show ORDER BY show_id")
    st.caption("Download a simple lookup of show_id ↔ canonical_title. (This is based on the `show` table.)")

    csv_bytes = export_show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download show_ids.csv",
        data=csv_bytes,
        file_name="show_ids.csv",
        mime="text/csv",
        key="dl_show_ids_csv",
    )

    # Excel (optional)
    try:
        import openpyxl  # type: ignore  # noqa: F401

        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            export_show_df.to_excel(writer, index=False, sheet_name="show_ids")
        st.download_button(
            "Download show_ids.xlsx",
            data=xlsx_buf.getvalue(),
            file_name="show_ids.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_show_ids_xlsx",
        )
    except Exception:
        st.info("Excel export requires `openpyxl`. Install it in your venv with: `pip install openpyxl`")
    st.markdown("### View aliases for a show")
    show_for_aliases = st.selectbox("Show", titles, key="alias_list_show")
    show_id = int(shows.loc[shows["canonical_title"] == show_for_aliases, "show_id"].iloc[0])
    alias_df = sql_df("SELECT alias_title FROM show_alias WHERE show_id = ? ORDER BY alias_title", (show_id,))
    st.dataframe(alias_df, width='stretch')


# ----------------------------
# Main
@st.cache_data(show_spinner=False)
def _load_records_base(db_path: str, db_mtime: float) -> pd.DataFrame:
    """Load chart rows for record calculations (weekly gross only; no gross bonuses). db_mtime busts cache."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
              date(e.week_ending) AS week_ending,
              e.week_number,
              e.rank,
              e.pos,
              e.show_id,
              s.canonical_title AS canonical_title,
              COALESCE(e.imprint_1,'(Unknown)') AS imprint_1,
              COALESCE(NULLIF(TRIM(e.imprint_2),''),'(None)') AS imprint_2,
              e.gross_millions AS gross_millions
            FROM t10_entry e
            JOIN show s ON s.show_id = e.show_id
            """,
            con,
        )
    finally:
        con.close()

    if df.empty:
        return df

    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["gross_millions"] = pd.to_numeric(df["gross_millions"], errors="coerce").fillna(0.0)

    # Collapse any accidental duplicates safely
    df = (
        df.groupby(
            ["show_id", "canonical_title", "week_ending", "week_ending_dt", "week_number", "rank", "pos"],
            as_index=False,
        )
        .agg(
            imprint_1=("imprint_1", "first"),
            imprint_2=("imprint_2", "first"),
            gross_millions=("gross_millions", "sum"),
        )
        .sort_values(["week_ending_dt", "rank", "pos"])
        .reset_index(drop=True)
    )

    # Flags for debut appearances (used for #1 debut records)
    # "Debut" means the show's first-ever appearance on the chart.
    first_dt = df.groupby("show_id")["week_ending_dt"].transform("min")
    df["is_debut"] = df["week_ending_dt"].eq(first_dt)
    df["is_1_debut"] = df["is_debut"] & df["rank"].eq(1)

    return df


def _fmt_date(x: Any) -> str:
    try:
        if pd.isna(x):
            return "—"
        return pd.to_datetime(x).date().isoformat()
    except Exception:
        return str(x)



def _record_progression(
    unique_week_winners: pd.DataFrame,
    latest_week_number: float | None,
    latest_dt: pd.Timestamp | None
) -> pd.DataFrame:
    """Given one winner per week (no ties), return a record progression table (strictly increasing gross).

    If available, carries through show_id / imprint_1 / imprint_2 from the winning rows.
    """
    out_cols = [
        "show_id",
        "canonical_title",
        "imprint_1",
        "imprint_2",
        "week_ending",
        "gross_millions",
        "length_weeks",
        "broken_week",
        "broken_by",
    ]

    if unique_week_winners.empty:
        return pd.DataFrame(columns=out_cols)

    df = unique_week_winners.copy()
    df = df.sort_values(["week_ending_dt"]).reset_index(drop=True)

    df["prev_record"] = df["gross_millions"].cummax().shift(1).fillna(-np.inf)
    events = df[df["gross_millions"] > df["prev_record"]].copy()
    if events.empty:
        return pd.DataFrame(columns=out_cols)

    # Next event (record breaker)
    events["next_week_ending_dt"] = events["week_ending_dt"].shift(-1)
    events["broken_week"] = events["next_week_ending_dt"].dt.strftime("%Y-%m-%d")
    events["broken_by"] = events["canonical_title"].shift(-1)

    def _len_weeks(row: pd.Series) -> int | None:
        cur_wn = row.get("week_number")
        cur_dt = row.get("week_ending_dt")
        nxt_dt = row.get("next_week_ending_dt")

        if pd.notna(cur_dt) and pd.notna(nxt_dt):
            return int(round((pd.to_datetime(nxt_dt) - pd.to_datetime(cur_dt)).days / 7.0))

        # Last record: use latest week info
        if pd.notna(cur_wn) and latest_week_number is not None and pd.notna(latest_week_number):
            return int(float(latest_week_number) - float(cur_wn) + 1)

        if pd.notna(cur_dt) and latest_dt is not None and pd.notna(latest_dt):
            return int(round((pd.to_datetime(latest_dt) - pd.to_datetime(cur_dt)).days / 7.0)) + 1

        return None

    events["length_weeks"] = events.apply(_len_weeks, axis=1)

    # Build output with optional columns
    base_cols = [c for c in ["show_id", "canonical_title", "imprint_1", "imprint_2", "week_ending", "gross_millions", "length_weeks", "broken_week", "broken_by"] if c in events.columns]
    out = events[base_cols].copy()
    for c in out_cols:
        if c not in out.columns:
            out[c] = None
    out = out[out_cols]
    return out



def _monthly_record_progression(unique_month_winners: pd.DataFrame, latest_month_ord: int | None) -> pd.DataFrame:
    """Record progression for monthly single-show gross totals.

    Input should contain one row per chart-month (ties removed), ordered by month.
    Columns expected: month, month_ord, show_id, canonical_title, imprint_1, imprint_2, gross_millions

    Output includes:
      - length_months: how many chart-months the record stood (inclusive of the record month)
      - broken_month: the next record-set month (or blank for the current record)
      - broken_by: show that broke it (or blank)
    """
    out_cols = [
        "show_id",
        "canonical_title",
        "imprint_1",
        "imprint_2",
        "month",
        "gross_millions",
        "length_months",
        "broken_month",
        "broken_by",
    ]

    if unique_month_winners is None or unique_month_winners.empty:
        return pd.DataFrame(columns=out_cols)

    df = unique_month_winners.copy()
    df = df.sort_values(["month_ord", "month", "canonical_title"], ascending=[True, True, True]).reset_index(drop=True)

    df["prev_record"] = pd.to_numeric(df["gross_millions"], errors="coerce").fillna(0.0).cummax().shift(1).fillna(-np.inf)
    events = df[pd.to_numeric(df["gross_millions"], errors="coerce").fillna(0.0) > df["prev_record"]].copy()
    if events.empty:
        return pd.DataFrame(columns=out_cols)

    events["next_month"] = events["month"].shift(-1)
    events["next_month_ord"] = events["month_ord"].shift(-1)
    events["broken_month"] = events["next_month"].fillna("")
    events["broken_by"] = events["canonical_title"].shift(-1).fillna("")

    def _len_months(row: pd.Series) -> int | None:
        cur_ord = row.get("month_ord")
        nxt_ord = row.get("next_month_ord")
        if pd.notna(cur_ord) and pd.notna(nxt_ord):
            return int(nxt_ord) - int(cur_ord)
        if pd.notna(cur_ord) and latest_month_ord is not None:
            return int(latest_month_ord) - int(cur_ord) + 1
        return None

    events["length_months"] = events.apply(_len_months, axis=1)

    base_cols = [c for c in ["show_id", "canonical_title", "imprint_1", "imprint_2", "month", "gross_millions", "length_months", "broken_month", "broken_by"] if c in events.columns]
    out = events[base_cols].copy()
    for c in out_cols:
        if c not in out.columns:
            out[c] = None
    out = out[out_cols]
    return out

def tab_records_achievements():
    st.subheader("Records and Achievements")
    st.caption("Grosses on this page use weekly gross only (no gross bonuses).")

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}.")
        return

    db_mtime = DB_PATH.stat().st_mtime
    base = _load_records_base(str(DB_PATH), db_mtime)
    if base.empty:
        st.info("No chart rows found.")
        return

    # Common helpers
    latest_week_number = None
    if base["week_number"].notna().any():
        latest_week_number = int(base["week_number"].max())

    latest_dt = None
    if base["week_ending_dt"].notna().any():
        latest_dt = pd.to_datetime(base["week_ending_dt"].max())

    def fmt_millions(x: Any) -> str:
        try:
            if pd.isna(x):
                return ""
            v = float(x)
            s = f"{v:,.1f}"
            if s.endswith(".0"):
                s = s[:-2]
            return s
        except Exception:
            return ""

    def fmt_month(p: Any) -> str:
        try:
            if pd.isna(p):
                return ""
            ts = pd.to_datetime(p)
            return ts.strftime("%Y-%m")
        except Exception:
            return str(p)

    # Base slices
    n1 = base[base["rank"].eq(1)].copy()
    debut_n1 = base[base["is_1_debut"]].copy()

    # Grossing-era filter for gross-based record tables (exclude pre-grossing era data)
    gross_start = pd.to_datetime(base.loc[base["gross_millions"].fillna(0).gt(0), "week_ending_dt"].min())
    if pd.isna(gross_start):
        gross_start = None

    if gross_start is None:
        base_gross = base[base["gross_millions"].fillna(0).gt(0)].copy()
    else:
        base_gross = base[(base["week_ending_dt"] >= gross_start) & (base["gross_millions"].fillna(0).gt(0))].copy()

    n1_gross = base_gross[base_gross["rank"].eq(1)].copy()
    debut_n1_gross = base_gross[base_gross["is_1_debut"]].copy()

    # Helper: unique top per week (exclude ties within a week)
    def unique_top_by_week(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        wk = df.groupby("week_ending", as_index=False)["gross_millions"].max().rename(columns={"gross_millions": "max_gross"})
        cand = df.merge(wk, on="week_ending", how="inner")
        cand = cand[cand["gross_millions"].eq(cand["max_gross"])].copy()
        counts = cand.groupby("week_ending")["show_id"].size()
        uniq_weeks = counts[counts.eq(1)].index.tolist()
        cand = cand[cand["week_ending"].isin(uniq_weeks)].copy()
        return cand.drop(columns=["max_gross"]).sort_values(["week_ending_dt"]).reset_index(drop=True)

    # -------------------------
    # 1) Most weeks at <position>
    # -------------------------
    st.markdown("### Most weeks at a position (rank)")
    c1, c2 = st.columns([1, 2])
    with c1:
        rank_pick = st.selectbox("Position (rank)", options=list(range(1, 11)), index=0, key="rec_rank_pick")
    with c2:
        top_n = st.slider("Top N", 5, 300, 25, key="rec_rank_topn")

    at_rank = base[base["rank"].eq(rank_pick)].copy()
    if at_rank.empty:
        st.info("No rows found for that rank.")
    else:
        agg = (
            at_rank.groupby(["show_id", "canonical_title"], as_index=False)
            .agg(
                imprint_1=("imprint_1", "first"),
                imprint_2=("imprint_2", "first"),
                weeks_at_rank=("week_ending", "nunique"),
                first_week=("week_ending_dt", "min"),
                last_week=("week_ending_dt", "max"),
            )
            .sort_values(["weeks_at_rank", "first_week", "canonical_title"], ascending=[False, True, True])
            .head(top_n)
            .reset_index(drop=True)
        )
        agg.insert(0, "Rank", np.arange(1, len(agg) + 1))
        disp = pd.DataFrame({
            "Rank": agg["Rank"],
            "Show": agg["canonical_title"],
            "Imprint 1": agg["imprint_1"],
            "Imprint 2": agg["imprint_2"],
            f"Total Career Weeks at #{rank_pick}": agg["weeks_at_rank"],
            f"First Career #{rank_pick}": agg["first_week"].apply(_fmt_date),
            f"Last #{rank_pick}": agg["last_week"].apply(_fmt_date),
        })
        st.dataframe(disp, width='stretch', hide_index=True)

    # -------------------------
    # 2) #1 week grossing record (progression)
    # -------------------------
    st.markdown("### List of shows holding #1 single-week grossing record")
    if n1_gross.empty:
        st.info("No #1 rows found.")
    else:
        w1 = unique_top_by_week(n1)
        prog = _record_progression(w1, latest_week_number, latest_dt)
        if prog.empty:
            st.info("No record progression found (possible ties or no gross data).")
        else:
            prog = prog.reset_index(drop=True)
            disp = pd.DataFrame({
                "#": np.arange(1, len(prog) + 1),
                "Show": prog["canonical_title"],
                "Imprint 1": prog["imprint_1"],
                "Imprint 2": prog["imprint_2"],
                "Total Length (in weeks)": prog["length_weeks"].astype("Int64"),
                "Week Record Set": prog["week_ending"].apply(_fmt_date),
                "Week Record Broken": prog["broken_week"].apply(_fmt_date),
                "Grosses (in millions)": prog["gross_millions"].apply(fmt_millions),
            })
            st.dataframe(disp, width='stretch', hide_index=True)

    

    # -------------------------
    # 2b) Single-show monthly grossing record (progression)
    # -------------------------
    st.markdown("### List of shows holding single-show monthly grossing record")

    # Compute chart-month totals using the shared monthly grossing logic.
    m_base = _apply_chart_month_logic(base_gross, week_dt_col="week_ending_dt", week_str_col="week_ending")

    if m_base.empty:
        st.info("No gross rows available for monthly aggregation.")
    else:
        month_totals = (
            m_base.groupby(["month", "month_ord", "show_id", "canonical_title", "imprint_1", "imprint_2"], as_index=False)
            .agg(gross_millions=("gross_millions", "sum"))
        )

        # Unique #1 per month (exclude ties)
        mx = month_totals.groupby("month", as_index=False)["gross_millions"].max().rename(columns={"gross_millions": "max_gross"})
        cand = month_totals.merge(mx, on="month", how="inner")
        cand = cand[cand["gross_millions"].eq(cand["max_gross"])].copy()
        counts = cand.groupby("month")["show_id"].size()
        uniq_months = counts[counts.eq(1)].index.tolist()
        uniq = cand[cand["month"].isin(uniq_months)].drop(columns=["max_gross"]).copy()

        latest_month_ord = int(month_totals["month_ord"].max()) if not month_totals.empty else None
        prog_m = _monthly_record_progression(uniq, latest_month_ord)

        if prog_m.empty:
            st.info("No monthly record progression found (possible ties or no gross data).")
        else:
            prog_m = prog_m.reset_index(drop=True)
            disp_m = pd.DataFrame({
                "#": np.arange(1, len(prog_m) + 1),
                "Show": prog_m["canonical_title"],
                "Imprint 1": prog_m["imprint_1"],
                "Imprint 2": prog_m["imprint_2"],
                "Total Length (in months)": prog_m["length_months"].astype("Int64"),
                "Month Record Set": prog_m["month"],
                "Month Record Broken": prog_m["broken_month"],
                "Grosses (in millions)": prog_m["gross_millions"].apply(fmt_millions),
            })
            st.dataframe(disp_m, width='stretch', hide_index=True)

    st.divider()

    # -------------------------
    # 3) #1 debut grossing record (progression)
    # -------------------------
    st.markdown("### List of shows holding #1 debut grossing record")
    if debut_n1_gross.empty:
        st.info("No #1 debuts found.")
    else:
        w1d = unique_top_by_week(debut_n1)
        prog = _record_progression(w1d, latest_week_number, latest_dt)
        if prog.empty:
            st.info("No record progression found (possible ties or no gross data).")
        else:
            prog = prog.reset_index(drop=True)
            disp = pd.DataFrame({
                "#": np.arange(1, len(prog) + 1),
                "Show": prog["canonical_title"],
                "Imprint 1": prog["imprint_1"],
                "Imprint 2": prog["imprint_2"],
                "Total Length (in weeks)": prog["length_weeks"].astype("Int64"),
                "Week Record Set": prog["week_ending"].apply(_fmt_date),
                "Week Record Broken": prog["broken_week"].apply(_fmt_date),
                "Grosses (in millions)": prog["gross_millions"].apply(fmt_millions),
            })
            st.dataframe(disp, width='stretch', hide_index=True)

    st.divider()

    # -------------------------
    # 4) #1 hat tricks (three consecutive #1 weeks)
    # -------------------------
    st.markdown("### List of T-10 Chart #1 hat tricks (three consecutive #1 weeks)")
    if n1_gross.empty:
        st.info("No #1 rows found.")
    else:
        # Precompute weekly lineups for "Perfect" checks (only when a full top-10 exists).
        week_lineups: dict[str, tuple] = {}
        try:
            top10 = base[base["rank"].between(1, 10)].copy()
            for wk, g in top10.groupby("week_ending"):
                g = g.sort_values("rank")
                lineup = tuple(g["show_id"].tolist())
                if len(lineup) == 10:
                    week_lineups[str(wk)] = lineup
        except Exception:
            week_lineups = {}

        # "Consecutive charts" means consecutive entries in the ordered list of chart weeks
        # present in the database (ignore date gaps / skipped weeks).
        n1_week = base[base["rank"].eq(1)].copy()
        n1_week = n1_week.sort_values(["week_ending_dt", "week_number"], na_position="last")

        week_order = (
            base[["week_ending", "week_ending_dt"]]
            .drop_duplicates()
            .sort_values("week_ending_dt")
            .reset_index(drop=True)
        )

        # week -> list of #1 show_ids (ties allowed)
        week_n1 = (
            n1_week.groupby("week_ending")["show_id"]
            .apply(lambda s: sorted({int(x) for x in s.dropna().tolist()}))
            .rename("show_ids")
            .reset_index()
        )

        week_meta = week_order.merge(week_n1, on="week_ending", how="left")
        week_meta["show_ids"] = week_meta["show_ids"].apply(lambda v: v if isinstance(v, list) else [])

        # show_id -> canonical_title lookup
        title_lookup = (
            base[["show_id", "canonical_title"]]
            .dropna()
            .drop_duplicates(subset=["show_id"])
            .set_index("show_id")["canonical_title"]
            .to_dict()
        )

        # Quick lookup of #1 gross per week per show (no bonuses) for summing 3-week totals
        n1_gross_lookup = n1_week.drop_duplicates(subset=["week_ending", "show_id"])[
            ["week_ending", "show_id", "gross_millions"]
        ].copy()

        hat_rows: list[dict[str, Any]] = []

        # Track streaks per show_id across consecutive chart weeks where that show appears at #1
        # (ties allowed). Record a hat trick only when a show FIRST reaches 3 consecutive charts at #1.
        streak_len_by_sid: dict[int, int] = {}
        streak_weeks_by_sid: dict[int, list[str]] = {}

        for _, wkrow in week_meta.iterrows():
            wk = str(wkrow["week_ending"])
            n1_sids = wkrow["show_ids"]

            # End streaks for shows that are no longer #1 this chart week
            active = set(streak_len_by_sid.keys())
            present = set(n1_sids)
            for sid in list(active - present):
                streak_len_by_sid.pop(sid, None)
                streak_weeks_by_sid.pop(sid, None)

            # Advance streaks for shows that are #1 this week (including ties)
            for sid in n1_sids:
                prev_len = streak_len_by_sid.get(sid, 0)
                prev_weeks = streak_weeks_by_sid.get(sid, [])
                new_len = prev_len + 1
                new_weeks = (prev_weeks + [wk])[-3:]

                streak_len_by_sid[sid] = new_len
                streak_weeks_by_sid[sid] = new_weeks

                # Only record once per streak, when it FIRST reaches 3.
                if new_len == 3:
                    weeks = new_weeks[:]  # 3 consecutive chart weeks (in chart order)
                    completed_week = weeks[-1]
                    title = title_lookup.get(sid, str(sid))

                    # Imprints for the primary show (use the completed week's #1 row when available)
                    prim = base[(base["week_ending"].eq(completed_week)) & (base["show_id"].eq(sid)) & (base["rank"].eq(1))]
                    if prim.empty:
                        prim = base[base["show_id"].eq(sid)].head(1)
                    imp1 = str(prim.iloc[0]["imprint_1"]) if not prim.empty else ""
                    imp2 = str(prim.iloc[0]["imprint_2"]) if (not prim.empty) and ("imprint_2" in prim.columns) else ""

                    # #2 show on completed week (may not exist if ranks skip due to tie-at-#1)
                    sec = base[(base["week_ending"].eq(completed_week)) & (base["rank"].eq(2))]
                    sec_title = str(sec.iloc[0]["canonical_title"]) if not sec.empty else ""

                    status = ""
                    if all(w in week_lineups for w in weeks):
                        if week_lineups[weeks[0]] == week_lineups[weeks[1]] == week_lineups[weeks[2]]:
                            status = "Perfect"

                    if status == "":
                        # "Shutout" means the #2 show is the same for all three weeks (when rank==2 exists).
                        sec_ids: list[int] = []
                        ok = True
                        for w in weeks:
                            s2 = base[(base["week_ending"].eq(w)) & (base["rank"].eq(2))]
                            if s2.empty:
                                ok = False
                                break
                            sec_ids.append(int(s2.iloc[0]["show_id"]))
                        if ok and len(set(sec_ids)) == 1:
                            status = "Shutout"

                    gross3 = None
                    try:
                        g3 = n1_gross_lookup[
                            (n1_gross_lookup["show_id"].eq(sid))
                            & (n1_gross_lookup["week_ending"].isin(weeks))
                        ]["gross_millions"]
                        if (len(g3) == 3) and (not g3.isna().any()):
                            gross3 = float(g3.sum())
                    except Exception:
                        gross3 = None

                    pair = str(title) if not sec_title else f"{title}/{sec_title}"
                    hat_rows.append(
                        {
                            "#1 Show/#2 Show": pair,
                            "Imprint 1": imp1,
                            "Imprint 2": imp2,
                            "Hat Trick Status": status,
                            "Hat Trick Week": _fmt_date(completed_week),
                            "Total Grosses (in millions)": fmt_millions(gross3) if gross3 is not None else "",
                            "_completed_dt": pd.to_datetime(completed_week),
                        }
                    )

        hat = pd.DataFrame(hat_rows)
        if hat.empty:
            st.info("No hat tricks found.")
        else:
            hat = hat.sort_values(["_completed_dt", "#1 Show/#2 Show"]).reset_index(drop=True)
            hat.insert(0, "#", np.arange(1, len(hat) + 1))
            disp = hat[["#", "#1 Show/#2 Show", "Imprint 1", "Imprint 2", "Hat Trick Status", "Hat Trick Week", "Total Grosses (in millions)"]].copy()
            st.dataframe(disp, width='stretch', hide_index=True)

    st.divider()

    st.markdown("### Record grosses for positions (no ties)")
    pos_rows: list[dict[str, Any]] = []
    for r in range(1, 11):
        sub = base_gross[base_gross["rank"].eq(r)].copy()
        if sub.empty:
            continue
        mx = float(sub["gross_millions"].max())
        winners = sub[sub["gross_millions"].eq(mx)]
        if len(winners) != 1:
            continue
        w = winners.iloc[0]
        pos_rows.append({
            "Rank": int(r),
            "Show": str(w["canonical_title"]),
            "Imprint 1": str(w.get("imprint_1", "")),
            "Imprint 2": str(w.get("imprint_2", "")),
            "Week": _fmt_date(w["week_ending"]),
            "Grosses (in millions)": fmt_millions(w["gross_millions"]),
        })
    pos_df = pd.DataFrame(pos_rows).sort_values(["Rank"]).reset_index(drop=True) if pos_rows else pd.DataFrame()
    if pos_df.empty:
        st.info("No unique max-by-rank rows found (ties may be blocking one or more ranks).")
    else:
        st.dataframe(pos_df, width='stretch', hide_index=True)

    st.divider()

    # -------------------------
    # 6) Earliest occurrence of the 1st, 2nd, 3rd... #1 show (by calendar-year order)
    # -------------------------
    st.markdown("### Earliest occurrence of different #1's")
    if n1.empty:
        st.info("No #1 rows found.")
    else:
        tmp = n1.copy()
        tmp["year"] = tmp["week_ending_dt"].dt.year

        # For each year, find the first week each show hit #1 in that year
        first_in_year = (
            tmp.groupby(["year", "show_id", "canonical_title"], as_index=False)
            .agg(
                first_n1=("week_ending_dt", "min"),
                imprint_1=("imprint_1", "first"),
                imprint_2=("imprint_2", "first"),
            )
            .sort_values(["year", "first_n1", "canonical_title"])
            .reset_index(drop=True)
        )

        # Order of distinct #1 shows within each calendar year
        first_in_year["Order"] = first_in_year.groupby("year").cumcount() + 1
        first_in_year["doy"] = first_in_year["first_n1"].dt.dayofyear

        # For each order (1st #1 show of a year, 2nd, 3rd, ...), pick the earliest day-of-year across all years.
        min_doy_by_order = first_in_year.groupby("Order")["doy"].min()
        winners = first_in_year[first_in_year["doy"].eq(first_in_year["Order"].map(min_doy_by_order))].copy()

        # Note "Earliest possible" when the date aligns with the theoretical minimum for weekly charts:
        # Jan 1 for Order 1, Jan 8 for Order 2, Jan 15 for Order 3, etc.
        winners["Note"] = np.where(winners["doy"].eq(1 + 7 * (winners["Order"] - 1)), "Earliest possible", "")
        winners = winners.sort_values(["Order", "first_n1", "canonical_title"]).reset_index(drop=True)

        rows = []
        for order, g in winners.groupby("Order", sort=True):
            for j, (_, r) in enumerate(g.iterrows()):
                rows.append({
                    "Note": r["Note"] if j == 0 else "",
                    "Order": str(int(order)) if j == 0 else "",
                    "Show": r["canonical_title"],
                    "Week Hit #1": _fmt_date(r["first_n1"]),
                })
        disp = pd.DataFrame(rows)
        if not disp.empty and "Order" in disp.columns:
            disp["Order"] = disp["Order"].astype(str)
        st.dataframe(disp, width='stretch', hide_index=True)

    st.divider()

    # -------------------------
    # 7) Most weeks at #1 by imprint (Imprint 1)
    # -------------------------

    # -------------------------
    # 7) Most weeks at #1 by imprint (Imprint 1 + Imprint 2 combined)
    # -------------------------
    st.markdown("### Most #1's by imprint")
    if n1.empty:
        st.info("No #1 rows found.")
    else:
        def _norm_imp(v: Any) -> str | None:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            low = s.lower()
            if low in {"(none)", "<none>", "none", "(unknown)", "unknown"}:
                return None
            return s

        imp_pairs = n1[["week_ending", "week_ending_dt", "imprint_1", "imprint_2"]].copy()
        imp_pairs["imp1"] = imp_pairs["imprint_1"].map(_norm_imp)
        imp_pairs["imp2"] = imp_pairs["imprint_2"].map(_norm_imp)

        # De-duplicate within a row (same imprint in both columns shouldn't double-count)
        imp_pairs["imprints"] = imp_pairs.apply(
            lambda r: sorted({x for x in [r["imp1"], r["imp2"]] if x is not None}),
            axis=1,
        )

        imp_long = (
            imp_pairs[["week_ending", "week_ending_dt", "imprints"]]
            .explode("imprints")
            .dropna(subset=["imprints"])
            .rename(columns={"imprints": "imprint"})
        )

        if imp_long.empty:
            st.info("No imprint data found for #1 rows.")
        else:
            imp = (
                imp_long.groupby("imprint", as_index=False)
                .agg(
                    total_weeks_at_1=("week_ending", "nunique"),
                    first_1=("week_ending_dt", "min"),
                    last_1=("week_ending_dt", "max"),
                )
                .sort_values(["total_weeks_at_1", "first_1", "imprint"], ascending=[False, True, True])
                .reset_index(drop=True)
            )
            imp.insert(0, "Rank", np.arange(1, len(imp) + 1))
            disp = pd.DataFrame({
                "Rank": imp["Rank"],
                "Imprint": imp["imprint"],
                "#1 Weeks": imp["total_weeks_at_1"],
                "Earliest #1": imp["first_1"].apply(_fmt_date),
                "Latest #1": imp["last_1"].apply(_fmt_date),
            })
            st.dataframe(disp, width='stretch', hide_index=True)

    st.divider()


    # -------------------------
    # 8) All #1 debuts
    # -------------------------
    st.markdown("### #1 debuts")
    if debut_n1.empty:
        st.info("No #1 debuts found.")
    else:
        deb = debut_n1.sort_values(["week_ending_dt"]).reset_index(drop=True).copy()
        deb.insert(0, "#", np.arange(1, len(deb) + 1))
        disp = pd.DataFrame({
            "#": deb["#"],
            "Show": deb["canonical_title"],
            "Imprint 1": deb["imprint_1"],
            "Imprint 2": deb["imprint_2"],
            "Week #": deb["week_number"].astype("Int64"),
            "Debut": deb["week_ending"].apply(_fmt_date),
            "Grosses (in millions)": deb["gross_millions"].apply(fmt_millions),
        })
        st.dataframe(disp, width='stretch', hide_index=True)

    st.divider()

    # -------------------------
    # 9) Biggest grossing #1 weeks by month (ties allowed)
    # -------------------------
    st.markdown("### Biggest grossing #1 weeks by month (ties allowed)")
    if n1_gross.empty:
        st.info("No grossing-era #1 rows found.")
    else:
        tmp = n1_gross.copy()
        tmp["month_num"] = tmp["week_ending_dt"].dt.month

        month_max = tmp.groupby("month_num")["gross_millions"].max()
        winners = tmp[tmp["gross_millions"].eq(tmp["month_num"].map(month_max))].copy()
        winners = winners.sort_values(["month_num", "gross_millions", "canonical_title"], ascending=[True, False, True]).reset_index(drop=True)

        rows = []
        for mnum, g in winners.groupby("month_num", sort=True):
            mlabel = pd.Timestamp(2016, int(mnum), 1).strftime("%B")
            for j, (_, r) in enumerate(g.iterrows()):
                rows.append({
                    "Month": mlabel if j == 0 else "",
                    "Show": r["canonical_title"],
                    "Imprint 1": r.get("imprint_1", ""),
                    "Imprint 2": r.get("imprint_2", ""),
                    "Week Record Set": _fmt_date(r["week_ending"]),
                    "Grosses (in millions)": fmt_millions(r["gross_millions"]),
                })
        disp = pd.DataFrame(rows)
        st.dataframe(disp, width='stretch', hide_index=True)

    st.divider()

    # -------------------------
    # 10) Biggest grossing #1 debuts by month (ties allowed)
    # -------------------------
    st.markdown("### Biggest grossing #1 debuts by month (ties allowed)")
    if debut_n1_gross.empty:
        st.info("No grossing-era #1 debuts found.")
    else:
        tmp = debut_n1_gross.copy()
        tmp["month_num"] = tmp["week_ending_dt"].dt.month

        month_max = tmp.groupby("month_num")["gross_millions"].max()
        winners = tmp[tmp["gross_millions"].eq(tmp["month_num"].map(month_max))].copy()
        winners = winners.sort_values(["month_num", "gross_millions", "canonical_title"], ascending=[True, False, True]).reset_index(drop=True)

        rows = []
        for mnum, g in winners.groupby("month_num", sort=True):
            mlabel = pd.Timestamp(2016, int(mnum), 1).strftime("%B")
            for j, (_, r) in enumerate(g.iterrows()):
                rows.append({
                    "Month": mlabel if j == 0 else "",
                    "Show": r["canonical_title"],
                    "Imprint 1": r.get("imprint_1", ""),
                    "Imprint 2": r.get("imprint_2", ""),
                    "Week Record Set": _fmt_date(r["week_ending"]),
                    "Grosses (in millions)": fmt_millions(r["gross_millions"]),
                })
        disp = pd.DataFrame(rows)
        st.dataframe(disp, width='stretch', hide_index=True)


# ----------------------------
def _t10_rank_years(rank: int) -> list[str]:
    dfy = sql_df(
        """
        SELECT DISTINCT strftime('%Y', week_ending) AS y
        FROM t10_entry
        WHERE rank = ?
        ORDER BY y DESC
        """,
        (int(rank),),
    )
    years = [str(y) for y in dfy["y"].dropna().tolist()] if not dfy.empty else []
    # Defensive: ensure unique + sorted desc
    years = sorted(list(dict.fromkeys(years)), reverse=True)
    return years



def _fetch_t10_rank_rows(rank: int, year: str | None = None) -> pd.DataFrame:
    params: list[Any] = [int(rank)]
    year_clause = ""
    if year and year != "All":
        year_clause = "AND strftime('%Y', e.week_ending) = ?"
        params.append(str(year))

    df = sql_df(
        f"""
        SELECT
          e.show_id,
          e.week_number,
          date(e.week_ending) AS week_ending,
          s.canonical_title,
          e.imprint_1,
          e.imprint_2,
          e.gross_millions AS base_gross_millions
        FROM t10_entry e
        JOIN show s ON s.show_id = e.show_id
        WHERE e.rank = ?
          {year_clause}
        ORDER BY date(e.week_ending) ASC, e.week_number ASC
        """,
        tuple(params),
    )

    if not df.empty:
        df["week_ending"] = _as_date_str(df["week_ending"])
    return df



def _streaks_for_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive-week streaks per show for a given rank table.

    Ties do NOT break streaks: if a show appears at the position in consecutive
    chart weeks (even as a co-#1 / co-#2), the streak continues.
    """
    if df.empty:
        return pd.DataFrame(columns=["canonical_title", "weeks", "start_week_ending", "end_week_ending"])

    d = df.copy()
    d["week_ending_dt"] = pd.to_datetime(d["week_ending"], errors="coerce")
    d = d.dropna(subset=["week_ending_dt"])

    # Prefer stable grouping by show_id when available (avoids title casing/dup issues).
    group_cols = ["show_id"] if "show_id" in d.columns else ["canonical_title"]

    # Remove any accidental duplicates (a show should only appear once per week at a given rank).
    d = d.drop_duplicates(subset=group_cols + ["week_ending_dt"])

    out_parts: list[pd.DataFrame] = []
    for _, g in d.groupby(group_cols, dropna=False):
        g = g.sort_values("week_ending_dt").copy()

        gap_days = g["week_ending_dt"].diff().dt.days
        gap_ok = gap_days.between(*CONSECUTIVE_DAY_TOLERANCE)
        new_streak = (~gap_ok).fillna(True)
        g["_streak_id"] = new_streak.cumsum()

        agg = (
            g.groupby("_streak_id", as_index=False)
            .agg(
                canonical_title=("canonical_title", "first"),
                weeks=("week_ending_dt", "size"),
                start_week_ending=("week_ending_dt", "min"),
                end_week_ending=("week_ending_dt", "max"),
            )
        )
        out_parts.append(agg)

    if not out_parts:
        return pd.DataFrame(columns=["canonical_title", "weeks", "start_week_ending", "end_week_ending"])

    streaks = pd.concat(out_parts, ignore_index=True)
    streaks = (
        streaks.sort_values(["weeks", "start_week_ending", "canonical_title"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    streaks["start_week_ending"] = _as_date_str(streaks["start_week_ending"])
    streaks["end_week_ending"] = _as_date_str(streaks["end_week_ending"])
    return streaks



def _totals_table(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[label, "weeks"])
    d = df.copy()
    if col in ("imprint_2",):
        d[col] = d[col].fillna("").astype(str).str.strip()
        d = d[d[col] != ""]
    out = (
        d.groupby(col, dropna=False)
        .size()
        .reset_index(name="weeks")
        .rename(columns={col: label})
        .sort_values(["weeks", label], ascending=[False, True])
        .reset_index(drop=True)
    )
    return out



def _render_t10_rank_view(rank: int, title: str) -> None:
    years = _t10_rank_years(rank)
    if not years:
        st.info("No data found.")
        return

    year = st.selectbox("Year", ["All"] + years, index=1 if years else 0, key=f"t10_rank_{rank}_year")

    df = _fetch_t10_rank_rows(rank, None if year == "All" else year)

    # Main listing
    if year == "All":
        st.caption(f"Listing all shows that reached #{rank}, grouped by year.")
        if df.empty:
            st.info("No results.")
        else:
            df["_year"] = df["week_ending"].astype(str).str.slice(0, 4)
            for y in sorted(df["_year"].unique().tolist(), reverse=True):
                st.markdown(f"#### {y}")
                dy = df[df["_year"] == y].copy()
                st.dataframe(
                    dy[["week_number", "week_ending", "canonical_title", "imprint_1", "imprint_2", "base_gross_millions"]],
                    width='stretch',
                    hide_index=True,
                )
    else:
        st.caption(f"Listing all shows that reached #{rank} in **{year}**.")
        if df.empty:
            st.info("No results for that year.")
        else:
            st.dataframe(
                df[["week_number", "week_ending", "canonical_title", "imprint_1", "imprint_2", "base_gross_millions"]],
                width='stretch',
                hide_index=True,
            )

    # Weeks-at sections
    st.markdown("---")
    st.markdown(f"### Weeks at #{rank} sections")

    streaks = _streaks_for_rank(df)
    st.markdown(f"#### Consecutive weeks at #{rank} (Show)")
    streaks_show = streaks[streaks["weeks"] >= 2].copy()
    if streaks_show.empty:
        st.write("No multi-week streaks found.")
    else:
        st.dataframe(streaks_show, width='stretch', hide_index=True)

    c1 = st.columns(3)
    with c1[0]:
        st.markdown(f"#### Total weeks at #{rank} (Imprint 1)")
        t1 = _totals_table(df, "imprint_1", "imprint_1")
        st.dataframe(t1, width='stretch', hide_index=True)
    with c1[1]:
        st.markdown(f"#### Total weeks at #{rank} (Imprint 2)")
        t2 = _totals_table(df, "imprint_2", "imprint_2")
        st.dataframe(t2, width='stretch', hide_index=True)
    
    with c1[2]:
        st.markdown(f"#### Total weeks at #{rank} (Show)")

        # For #1 Shows (year-specific), also show "career #1s through that year".
        if rank == 1 and year != "All":
            if df.empty:
                ts_disp = pd.DataFrame(
                    columns=["canonical_title", "weeks", f"career_weeks_at_#{rank} (through {year})"]
                )
            else:
                yd = df.copy()
                if "show_id" in yd.columns:
                    year_tot = (
                        yd.dropna(subset=["show_id"])
                        .groupby(["show_id", "canonical_title"], as_index=False)
                        .size()
                        .rename(columns={"size": "weeks"})
                    )
                else:
                    year_tot = _totals_table(yd, "canonical_title", "canonical_title").rename(columns={"weeks": "weeks"})

                # Career totals up through the end of the selected year.
                try:
                    y_int = int(str(year))
                except Exception:
                    y_int = None

                career_tot = pd.DataFrame(columns=["show_id", "career_weeks"])
                if y_int is not None and "show_id" in year_tot.columns:
                    cutoff = f"{y_int + 1:04d}-01-01"
                    career_tot = sql_df(
                        """
                        SELECT e.show_id, COUNT(*) AS career_weeks
                        FROM t10_entry e
                        WHERE e.rank = ?
                          AND date(e.week_ending) < date(?)
                        GROUP BY e.show_id
                        """,
                        (int(rank), cutoff),
                    )

                if "show_id" in year_tot.columns and not career_tot.empty:
                    out = year_tot.merge(career_tot, on="show_id", how="left")
                else:
                    out = year_tot.copy()
                    out["career_weeks"] = pd.NA

                # If career_tot missing (or show_id unavailable), fall back to that year's weeks.
                out["career_weeks"] = out["career_weeks"].fillna(out["weeks"]).astype("Int64")
                out = out.sort_values(["weeks", "canonical_title"], ascending=[False, True]).reset_index(drop=True)

                ts_disp = out[["canonical_title", "weeks", "career_weeks"]].rename(
                    columns={"career_weeks": f"career_weeks_at_#{rank} (through {year})"}
                )

            st.dataframe(ts_disp, width='stretch', hide_index=True)
        else:
            ts = _totals_table(df, "canonical_title", "canonical_title")
            st.dataframe(ts, width='stretch', hide_index=True)



def tab_t10_chart_number_shows() -> None:
    st.header("T-10 Chart #1 Shows")
    st.caption("All #1 and #2 shows, grouped by year, with streak and total summaries.")

    subtabs = st.tabs(["#1 Shows", "#2 Shows"])
    with subtabs[0]:
        _render_t10_rank_view(1, "#1 Shows")
    with subtabs[1]:
        _render_t10_rank_view(2, "#2 Shows")

# ----------------------------
# New tab: Hall of Fame
# ----------------------------

# Tunables (badges / inductees)
HOF_G1_SHOW_GROSS = 10000.0  # "millions" units
HOF_P1_SHOW_POINTS = 2000.0
HOF_CG1_COMPANY_GROSS = 50000.0
HOF_CP1_COMPANY_POINTS = 20000.0
HOF_DEEP_BENCH_YEAR_POINTS_X = 150.0
HOF_DEEP_BENCH_MIN_SHOWS = 8

HOF_INDUCT_MIN_WEEKS = 10  # minimum chart weeks to show up in inductees (to avoid noise)
HOF_TABLE_MAX_BADGES = 4

_HOF_BADGE_ORDER_SHOW = [
    "Crown Collector",
    "#1 Legend",
    "Chart Emperor",
    "Top 10 Ironman",
    "Points Machine",
    "Box Office Beast",
    "Peak Monster",
    "Consistency King",
    "Steady Climber",
    "Comeback Kid",
    "Gatekeeper",
    "One-Hit Titan",
]

_HOF_BADGE_ORDER_COMPANY = [
    "Points Empire",
    "Box Office Titan",
    "#1 Machine",
    "Dynasty Year",
    "Hit Factory",
    "Deep Bench",
    "Era Staple",
    "Seasonal Specialists",
]


def _hof_inverse_rank_points(rank: float | int | None) -> float:
    """Pre-grossing era (or missing gross): #1=100, #2=90, ..."""
    if rank is None or pd.isna(rank):
        return np.nan
    r = int(rank)
    return float(max(0, 100 - 10 * (r - 1)))


def _hof_compute_week_points(df: pd.DataFrame) -> pd.DataFrame:
    """Add ye_points column: gross-era uses % of #1 gross; pre-gross-era uses inverse rank points."""
    if df.empty:
        df = df.copy()
        df["ye_points"] = pd.Series(dtype=float)
        return df

    out = df.copy()
    out["week_ending_dt"] = pd.to_datetime(out["week_ending"], errors="coerce")
    out["gross_millions"] = pd.to_numeric(out.get("gross_millions"), errors="coerce")

    # top1 gross per week (only meaningful in gross-era)
    wk = out.groupby("week_ending_dt", dropna=False)
    top1 = (
        out[out["rank"] == 1]
        .groupby("week_ending_dt", dropna=False)["gross_millions"]
        .max()
        .rename("top1_gross")
    )
    out = out.join(top1, on="week_ending_dt")

    gross_era = out["week_ending_dt"].dt.date >= GROSS_TRACKING_START
    has_gross = out["gross_millions"].notna() & out["top1_gross"].notna() & (out["top1_gross"] > 0)

    out["ye_points"] = np.nan

    # Gross era points
    mask = gross_era & has_gross
    out.loc[mask, "ye_points"] = (out.loc[mask, "gross_millions"] / out.loc[mask, "top1_gross"]) * 100.0

    # Pre-gross era OR missing gross -> inverse rank points
    mask2 = ~mask
    out.loc[mask2, "ye_points"] = out.loc[mask2, "rank"].apply(_hof_inverse_rank_points)

    out = out.drop(columns=["top1_gross"])
    return out


@st.cache_data(show_spinner=False)
def _load_hof_weekly_base(db_path: str, db_mtime: float) -> pd.DataFrame:
    """All weekly rows needed for Hall of Fame (includes bonus gross + ye_points)."""
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
              e.week_ending,
              e.week_number,
              e.rank,
              e.pos,
              e.show_id,
              s.canonical_title,
              COALESCE(NULLIF(TRIM(e.imprint_1), ''), '(Unknown)') AS imprint_1,
              COALESCE(NULLIF(TRIM(e.imprint_2), ''), '(Unknown)') AS imprint_2,
              e.gross_millions AS base_gross_millions,
              (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions
            FROM t10_entry e
            JOIN show s ON s.show_id = e.show_id
            LEFT JOIN (
              SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
              FROM gross_bonus
              GROUP BY show_id, week_ending
            ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
            WHERE e.week_ending IS NOT NULL
            """,
            con,
        )
    finally:
        con.close()

    if df.empty:
        return df

    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["base_gross_millions"] = pd.to_numeric(df.get("base_gross_millions"), errors="coerce")

    df["year"] = df["week_ending_dt"].dt.year.astype("Int64")
    df["month"] = df["week_ending_dt"].dt.month.astype("Int64")
    df["quarter"] = df["week_ending_dt"].dt.quarter.astype("Int64")

    # Points
    df = _hof_compute_week_points(df)

    return df


def _hof_apply_filters(df: pd.DataFrame, date_min: str | None, date_max: str | None, rank_min: int, rank_max: int) -> pd.DataFrame:
    out = df
    if date_min:
        dmin = pd.to_datetime(date_min, errors="coerce")
        if pd.notna(dmin):
            out = out[out["week_ending_dt"] >= dmin]
    if date_max:
        dmax = pd.to_datetime(date_max, errors="coerce")
        if pd.notna(dmax):
            out = out[out["week_ending_dt"] <= dmax]
    out = out[(out["rank"] >= int(rank_min)) & (out["rank"] <= int(rank_max))]
    return out.copy()


def _hof_company_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Explode imprint_1 + imprint_2 into a single company universe.

    - Counts imprint_1 rows
    - Counts imprint_2 rows
    - Avoids double-counting when imprint_2 == imprint_1 for the same chart row
    - Hall of Fame: suppress '(Unknown)' / blank companies so they don't dominate leaderboards
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    base = df.copy()

    # Defensive: upstream merges can create duplicate column names; keep first occurrence
    # so column selection yields a Series (not a DataFrame).
    if base.columns.duplicated().any():
        base = base.loc[:, ~base.columns.duplicated()].copy()

    # If it's already a company-universe frame (has 'company' and no imprint columns),
    # just normalize + filter.
    if ("company" in base.columns) and ("imprint_1" not in base.columns) and ("imprint_2" not in base.columns):
        out = base.copy()
        out["company"] = out["company"].fillna("(Unknown)").astype("string")
        _c = out["company"].astype("string").str.strip()
        out = out[(_c != "(Unknown)") & (_c != "")].copy()
        return out

    # Ensure expected imprint columns exist
    if "imprint_1" not in base.columns:
        base["imprint_1"] = "(Unknown)"
    if "imprint_2" not in base.columns:
        base["imprint_2"] = "(Unknown)"

    base["imprint_1"] = base["imprint_1"].fillna("(Unknown)").astype("string")
    base["imprint_2"] = base["imprint_2"].fillna("(Unknown)").astype("string")

    # Drop any pre-existing 'company' column to avoid duplicates after rename
    if "company" in base.columns:
        base = base.drop(columns=["company"])

    a = base.rename(columns={"imprint_1": "company"}).copy()
    b = base.rename(columns={"imprint_2": "company"}).copy()

    # Remove rows where imprint_2 equals imprint_1 (row-wise) to avoid double counting.
    b = b[b["company"].astype("string").str.strip() != b["imprint_1"].astype("string").str.strip()]

    out = pd.concat([a, b], ignore_index=True)
    out["company"] = out["company"].fillna("(Unknown)").astype("string")

    # Hall of Fame: suppress unknown/blank companies so they don't dominate leaderboards.
    _c = out["company"].astype("string").str.strip()
    out = out[(_c != "(Unknown)") & (_c != "")].copy()
    return out



def _hof_agg_shows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(["show_id", "canonical_title"], dropna=False)

    out = g.agg(
        weeks_charting=("week_ending", "nunique"),
        top10_weeks=("rank", lambda s: int((pd.to_numeric(s, errors="coerce") <= 10).sum())),
        top5_weeks=("rank", lambda s: int((pd.to_numeric(s, errors="coerce") <= 5).sum())),
        weeks_at_1=("rank", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
        peak_rank=("rank", lambda s: int(pd.to_numeric(s, errors="coerce").min())),
        avg_rank=("rank", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        median_rank=("rank", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
        total_gross_millions=("gross_millions", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        total_ye_points=("ye_points", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        first_week=("week_ending", "min"),
        last_week=("week_ending", "max"),
    ).reset_index()

    return out


def _hof_agg_companies(df_company: pd.DataFrame) -> pd.DataFrame:
    if df_company.empty:
        return pd.DataFrame()

    g = df_company.groupby(["company"], dropna=False)

    out = g.agg(
        entries=("week_ending", "count"),
        unique_shows_charted=("show_id", "nunique"),
        weeks_charting=("week_ending", "nunique"),
        weeks_at_1_sum=("rank", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
        total_gross_millions=("gross_millions", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        total_ye_points=("ye_points", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        first_week=("week_ending", "min"),
        last_week=("week_ending", "max"),
    ).reset_index()

    return out


def _hof_badge_pack_show(row: dict[str, Any], flags: dict[str, bool]) -> list[tuple[str, str, str]]:
    """Return list of (emoji, label, earned_how)."""
    badges: list[tuple[str, str, str]] = []

    weeks_at_1 = float(row.get("weeks_at_1", 0) or 0)
    weeks_charting = float(row.get("weeks_charting", 0) or 0)
    top10_weeks = float(row.get("top10_weeks", 0) or 0)
    top5_weeks = float(row.get("top5_weeks", 0) or 0)
    peak_rank = int(row.get("peak_rank", 999) or 999)
    med_rank = float(row.get("median_rank", 999) or 999)
    total_gross = float(row.get("total_gross_millions", 0) or 0)
    total_points = float(row.get("total_ye_points", 0) or 0)

    # Dominance
    if weeks_at_1 >= 10:
        badges.append(("🏆", "#1 Legend", f"Weeks at #1: {int(weeks_at_1)}"))
    if weeks_at_1 >= 25:
        badges.append(("🥇", "Crown Collector", f"Weeks at #1: {int(weeks_at_1)}"))
    if peak_rank == 1 and weeks_charting >= 20:
        badges.append(("⚡", "Peak Monster", f"Peak #1, weeks charting: {int(weeks_charting)}"))

    # Longevity
    if weeks_charting >= 52:
        badges.append(("👑", "Chart Emperor", f"Weeks charting: {int(weeks_charting)}"))
    if top10_weeks >= 104:
        badges.append(("🧱", "Top 10 Ironman", f"Top 10 weeks: {int(top10_weeks)}"))

    # Consistency
    if weeks_charting >= 20 and (top5_weeks / max(1.0, weeks_charting)) >= 0.70:
        badges.append(("🎯", "Consistency King", f"Top 5 rate: {top5_weeks/max(1.0,weeks_charting):.0%} over {int(weeks_charting)} weeks"))
    if weeks_charting >= 20 and med_rank <= 5:
        badges.append(("🧊", "Steady Climber", f"Median rank: {med_rank:.1f} over {int(weeks_charting)} weeks"))

    # Money / points
    if total_gross >= HOF_G1_SHOW_GROSS:
        badges.append(("💰", "Box Office Beast", f"Total gross: {total_gross:,.0f}M"))
    if total_points >= HOF_P1_SHOW_POINTS:
        badges.append(("📈", "Points Machine", f"Total points: {total_points:,.0f}"))

    # Narrative / wings
    if flags.get("comeback", False):
        badges.append(("🪃", "Comeback Kid", "Returned after a long absence and charted in multiple distinct runs."))
    if flags.get("gatekeeper", False):
        badges.append(("🚪", "Gatekeeper", "Huge Top 10 presence without ever hitting #1."))
    if flags.get("onehit", False):
        badges.append(("🧨", "One-Hit Titan", "Exactly one week at #1 with a short overall run."))

    return badges


def _hof_badge_pack_company(row: dict[str, Any], flags: dict[str, bool]) -> list[tuple[str, str, str]]:
    badges: list[tuple[str, str, str]] = []

    uniq = float(row.get("unique_shows_charted", 0) or 0)
    weeks1 = float(row.get("weeks_at_1_sum", 0) or 0)
    gross = float(row.get("total_gross_millions", 0) or 0)
    points = float(row.get("total_ye_points", 0) or 0)
    years_distinct = float(row.get("years_distinct", 0) or 0)

    if uniq >= 25:
        badges.append(("🏢", "Hit Factory", f"Unique shows charted: {int(uniq)}"))
    if flags.get("deep_bench", False):
        badges.append(("🧠", "Deep Bench", "Had at least one year with a big cluster of strong point-getters."))
    if weeks1 >= 52:
        badges.append(("🥇", "#1 Machine", f"Weeks at #1 (sum): {int(weeks1)}"))
    if gross >= HOF_CG1_COMPANY_GROSS:
        badges.append(("💼", "Box Office Titan", f"Total gross: {gross:,.0f}M"))
    if points >= HOF_CP1_COMPANY_POINTS:
        badges.append(("📊", "Points Empire", f"Total points: {points:,.0f}"))
    if flags.get("dynasty_year", False):
        badges.append(("🏰", "Dynasty Year", "Won at least one year-long championship."))
    if years_distinct >= 10:
        badges.append(("🗓️", "Era Staple", f"Distinct years with meaningful presence: {int(years_distinct)}"))
    if flags.get("seasonal", False):
        badges.append(("🌡️", "Seasonal Specialists", "Won at least one seasonal bucket title."))

    return badges


def _hof_pick_badges_for_table(badges: list[tuple[str, str, str]], order: list[str]) -> tuple[str, int]:
    """Return (rendered_badge_str, extra_count)."""
    if not badges:
        return ("", 0)

    by_label = {lbl: (emo, lbl, how) for (emo, lbl, how) in badges}
    picked: list[str] = []
    for lbl in order:
        if lbl in by_label:
            emo = by_label[lbl][0]
            picked.append(f"{emo} {lbl}")
        if len(picked) >= HOF_TABLE_MAX_BADGES:
            break

    extra = max(0, len(badges) - len(picked))
    return (" · ".join(picked) + (f"  +{extra}" if extra else ""), extra)


def _hof_wing_gatekeepers(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    out = agg[
        (agg["top10_weeks"] >= 40) &
        (agg["weeks_at_1"] == 0) &
        (agg["peak_rank"].isin([2, 3]))
    ].copy()
    out["gatekeeper"] = True
    return out.sort_values(["top10_weeks", "weeks_charting"], ascending=False)


def _hof_wing_one_hit_titans(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg
    out = agg[(agg["weeks_at_1"] == 1) & (agg["weeks_charting"] <= 6)].copy()
    out["onehit"] = True
    return out.sort_values(["weeks_charting", "total_ye_points"], ascending=False)


def _hof_wing_comeback_kids(df: pd.DataFrame, gap_weeks: int = 13, min_runs: int = 2) -> pd.DataFrame:
    """Return per-show comeback metadata: max_gap_weeks, runs, total_weeks."""
    if df.empty:
        return pd.DataFrame()

    use_weeknum = df["week_number"].notna().sum() > 0
    out_rows = []

    for (sid, title), g in df.groupby(["show_id", "canonical_title"], dropna=False):
        gg = g.sort_values(["week_number" if use_weeknum else "week_ending_dt"]).copy()

        if use_weeknum:
            idx = pd.to_numeric(gg["week_number"], errors="coerce").astype("Int64")
            diffs = idx.diff()
            gaps = diffs.fillna(1).astype("float")
        else:
            dt = pd.to_datetime(gg["week_ending_dt"], errors="coerce")
            gaps = dt.diff().dt.days.div(7.0)

        gaps = pd.to_numeric(gaps, errors="coerce").fillna(1.0)

        # New run whenever gap >= gap_weeks
        new_run = (gaps >= float(gap_weeks)).astype(int)
        run_id = new_run.cumsum()

        runs = int(run_id.nunique())
        if runs < min_runs:
            continue

        max_gap = float(gaps.max())
        if max_gap < float(gap_weeks):
            continue

        out_rows.append({
            "show_id": sid,
            "canonical_title": title,
            "runs": runs,
            "max_gap_weeks": max_gap,
            "weeks_charting": int(gg["week_ending"].nunique()),
        })

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out["comeback"] = True
    return out.sort_values(["max_gap_weeks", "runs", "weeks_charting"], ascending=False)


def _hof_wing_dynasty_champions(df: pd.DataFrame, entity: str = "show") -> pd.DataFrame:
    """Return multi-year champion streaks for show/company by year using ye_points (fallback to gross)."""
    if df.empty:
        return pd.DataFrame()

    if entity == "company":
        base = _hof_company_universe(df)
        grp_cols = ["year", "company"]
        label_col = "company"
    else:
        base = df
        grp_cols = ["year", "canonical_title"]
        label_col = "canonical_title"

    base = base.dropna(subset=["year"])
    if base.empty:
        return pd.DataFrame()

    yearly = (
        base.groupby(grp_cols, dropna=False)
        .agg(points=("ye_points", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
             gross=("gross_millions", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())))
        .reset_index()
    )
    # champion per year by points, tie-breaker gross
    yearly = yearly.sort_values(["year", "points", "gross"], ascending=[True, False, False])
    champs = yearly.groupby("year", as_index=False).first()
    champs = champs.sort_values("year")

    # compute streaks of consecutive years
    champs["prev_year"] = champs["year"].shift(1)
    champs["prev_label"] = champs[label_col].shift(1)
    champs["new_streak"] = (champs["year"] != (champs["prev_year"] + 1)) | (champs[label_col] != champs["prev_label"])
    champs["streak_id"] = champs["new_streak"].cumsum()

    streaks = (
        champs.groupby(["streak_id", label_col], dropna=False)
        .agg(start_year=("year", "min"), end_year=("year", "max"), years=("year", "count"))
        .reset_index()
    )
    streaks = streaks[streaks["years"] >= 2].copy()
    if streaks.empty:
        return pd.DataFrame(columns=["label","start_year","end_year","years"])

    streaks["label"] = streaks[label_col].astype("string")
    return streaks.sort_values(["years", "end_year"], ascending=False)[["label", "start_year", "end_year", "years"]]


def _hof_company_deep_bench_flags(df_company: pd.DataFrame) -> dict[str, bool]:
    """Return {company: deep_bench_bool} based on year-level depth."""
    if df_company.empty:
        return {}

    # points per (company, show, year)
    base = df_company.dropna(subset=["year"]).copy()
    base["ye_points"] = pd.to_numeric(base["ye_points"], errors="coerce").fillna(0.0)

    show_year = (
        base.groupby(["company", "show_id", "year"], dropna=False)["ye_points"]
        .sum()
        .reset_index(name="year_points")
    )
    show_year["is_strong"] = show_year["year_points"] >= float(HOF_DEEP_BENCH_YEAR_POINTS_X)

    depth = (
        show_year.groupby(["company", "year"], dropna=False)["is_strong"]
        .sum()
        .reset_index(name="strong_shows")
    )

    best = depth.groupby("company", dropna=False)["strong_shows"].max()
    return {c: bool(v >= HOF_DEEP_BENCH_MIN_SHOWS) for c, v in best.items()}


def _hof_company_dynasty_year_flags(df_company: pd.DataFrame) -> dict[str, bool]:
    """Return {company: dynasty_year_bool} if the company is a year champion at least once.

    Accepts company-universe weekly rows (must include a 'company' column). If a show-week
    frame is passed accidentally, we will explode it into the company universe.
    """
    if df_company is None or df_company.empty:
        return {}

    base = df_company
    if "company" not in base.columns:
        base = _hof_company_universe(base)

    base = base.dropna(subset=["year"]).copy()
    if base.empty:
        return {}

    # Per (year, company) totals
    base["ye_points"] = pd.to_numeric(base.get("ye_points"), errors="coerce").fillna(0.0)
    base["gross_millions"] = pd.to_numeric(base.get("gross_millions"), errors="coerce").fillna(0.0)

    yearly = (
        base.groupby(["year", "company"], dropna=False)
        .agg(points=("ye_points", "sum"), gross=("gross_millions", "sum"))
        .reset_index()
    )
    yearly = yearly.sort_values(["year", "points", "gross"], ascending=[True, False, False])

    # Champion per year (top points; tiebreak gross)
    champs = yearly.groupby("year", as_index=False).first()
    return {str(c): True for c in champs["company"].dropna().tolist()}



def _hof_holiday_winners_by_year(
    df_rows: pd.DataFrame,
    all_week_endings: list[date],
    holidays: Optional[dict[str, Callable[[int], date]]] = None,
    gross_col: str = "gross_millions",
) -> pd.DataFrame:
    """Return per-year holiday winners within the provided rows.

    Uses ye_points as primary sort key, then total_gross_millions, then best rank.
    Only includes holidays whose computed week_ending exists in all_week_endings AND appears in df_rows.
    """
    if holidays is None:
        holidays = HOLIDAYS

    if df_rows is None or df_rows.empty:
        return pd.DataFrame(
            columns=[
                "holiday",
                "year",
                "week_ending",
                "show_id",
                "canonical_title",
                "rank",
                "ye_points",
                "total_gross_millions",
                "imprint_1",
                "imprint_2",
            ]
        )

    df = df_rows.copy()
    if "week_ending" not in df.columns:
        return pd.DataFrame()

    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["year"] = pd.to_numeric(df.get("year", df["week_ending_dt"].dt.year), errors="coerce")
    if "ye_points" in df.columns:
        df["ye_points"] = pd.to_numeric(df["ye_points"], errors="coerce").fillna(0.0)
    else:
        df["ye_points"] = 0.0


    # Gross used for holiday totals / tiebreak
    _src = gross_col if (gross_col and (gross_col in df.columns)) else ("gross_millions" if "gross_millions" in df.columns else None)
    if _src is None:
        df["total_gross_millions"] = 0.0
    else:
        df["total_gross_millions"] = pd.to_numeric(df[_src], errors="coerce").fillna(0.0)
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    years = sorted([int(y) for y in df["year"].dropna().unique().tolist()])
    if not years:
        return pd.DataFrame()

    all_we = [d for d in (all_week_endings or []) if isinstance(d, date)]
    if not all_we:
        all_we = sorted(df["week_ending_dt"].dt.date.dropna().unique().tolist())

    out_rows = []
    for hol_name, hol_fn in (holidays or {}).items():
        for y in years:
            try:
                hol_dt = hol_fn(int(y))
            except Exception:
                continue

            we = holiday_week_ending_for_date(all_we, hol_dt, hol_name)
            if we is None:
                continue

            sel = df[df["week_ending_dt"].dt.date == we]
            if sel.empty:
                continue

            sort_cols = ["ye_points", "total_gross_millions"]
            asc = [False, False]
            if "rank" in sel.columns:
                sort_cols.append("rank")
                asc.append(True)

            best = sel.sort_values(sort_cols, ascending=asc, kind="mergesort").iloc[0]

            out_rows.append(
                {
                    "holiday": str(hol_name),
                    "year": int(y),
                    "week_ending": we.isoformat(),
                    "show_id": int(best["show_id"]) if pd.notna(best.get("show_id", None)) else None,
                    "canonical_title": str(best.get("canonical_title", "")),
                    "rank": int(best["rank"]) if pd.notna(best.get("rank", None)) else None,
                    "ye_points": float(best.get("ye_points", 0.0)),
                    "total_gross_millions": float(best.get("total_gross_millions", 0.0)),
                    "imprint_1": best.get("imprint_1", None),
                    "imprint_2": best.get("imprint_2", None),
                }
            )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    out = out.sort_values(["holiday", "year"], ascending=[True, False]).reset_index(drop=True)
    return out


def _hof_holiday_champions_from_winners(winners: pd.DataFrame) -> pd.DataFrame:
    """From per-year winners, compute one 'champion' show per holiday (by total ye_points)."""
    if winners is None or winners.empty:
        return pd.DataFrame(columns=["holiday", "champ_show", "years_won", "total_points", "avg_rank", "total_gross_millions"])

    w = winners.copy()
    w["ye_points"] = pd.to_numeric(w.get("ye_points", 0.0), errors="coerce").fillna(0.0)
    w["total_gross_millions"] = pd.to_numeric(w.get("total_gross_millions", 0.0), errors="coerce").fillna(0.0)
    if "rank" in w.columns:
        w["rank"] = pd.to_numeric(w["rank"], errors="coerce")

    g = (
        w.groupby(["holiday", "canonical_title"], dropna=False)
        .agg(
            years_won=("year", "nunique"),
            total_points=("ye_points", "sum"),
            avg_rank=("rank", "mean"),
            total_gross_millions=("total_gross_millions", "sum"),
        )
        .reset_index()
        .sort_values(["holiday", "total_points", "years_won"], ascending=[True, False, False])
    )

    champs = g.groupby("holiday", as_index=False).first().rename(columns={"canonical_title": "champ_show"})
    champs["avg_rank"] = champs["avg_rank"].round(2)
    champs["total_points"] = champs["total_points"].round(2)
    champs["total_gross_millions"] = champs["total_gross_millions"].round(2)
    return champs


def tab_hall_of_fame():
    st.subheader("Hall of Fame")
    st.caption("A narrative-first museum of dominance, longevity, and weird, fun chart lore — powered by your ye_points system.")

    db_path = str(DB_PATH)
    db_mtime = DB_PATH.stat().st_mtime
    base = _load_hof_weekly_base(db_path, db_mtime)

    if base is None or base.empty:
        st.info("No chart rows found yet.")
        return

    # All known week_endings (used to map holidays -> chart weeks reliably)
    try:
        _we = pd.to_datetime(base["week_ending"], errors="coerce").dt.date
        all_week_endings = sorted([d for d in _we.dropna().unique().tolist() if isinstance(d, date)])
    except Exception:
        all_week_endings = []

    with st.sidebar:
        st.header("Hall of Fame filters")
        date_min = st.text_input("Start date (YYYY-MM-DD)", value="", key="hof_date_min")
        date_max = st.text_input("End date (YYYY-MM-DD)", value="", key="hof_date_max")
        rank_min, rank_max = st.slider("Rank range (HOF)", 1, 17, (1, 10), key="hof_rank_range")
        top_n = st.slider("Top N (tables)", 10, 300, 50, step=10, key="hof_top_n")
        comeback_gap = st.selectbox("Comeback gap (weeks)", [13, 26], index=0, key="hof_comeback_gap")
        badge_scope = st.radio("Badges scope", ["Lifetime", "Current filters"], index=0, key="hof_badge_scope")

    hof_load = st.checkbox(
        "Load Hall of Fame results",
        value=False,
        key="hof_load_results",
        help="Hall of Fame is one of the heaviest sections in the app. Leave this off on mobile until you want to use it.",
    )
    if not hof_load:
        st.info("Turn on 'Load Hall of Fame results' when you want to run Hall of Fame calculations.")
        return

    df_filt = _hof_apply_filters(
        base,
        date_min.strip() or None,
        date_max.strip() or None,
        int(rank_min),
        int(rank_max),
    )

    # Lifetime aggregates for badges / profile toggles
    agg_life = _hof_agg_shows(base)
    agg_cur = _hof_agg_shows(df_filt)

    # Wings (computed on current-filter rows for coherence)
    wing_gate = _hof_wing_gatekeepers(agg_cur)
    wing_onehit = _hof_wing_one_hit_titans(agg_cur)
    wing_comeback = _hof_wing_comeback_kids(df_filt, gap_weeks=int(comeback_gap), min_runs=2)

    gate_ids = set(wing_gate["show_id"].tolist()) if not wing_gate.empty else set()
    onehit_ids = set(wing_onehit["show_id"].tolist()) if not wing_onehit.empty else set()
    comeback_ids = set(wing_comeback["show_id"].tolist()) if not wing_comeback.empty else set()

    # Company universe + flags
    comp_life_rows = _hof_company_universe(base)
    comp_cur_rows = _hof_company_universe(df_filt)
    comp_agg_life = _hof_agg_companies(comp_life_rows)
    comp_agg_cur = _hof_agg_companies(comp_cur_rows)

    deep_bench_flags = _hof_company_deep_bench_flags(comp_cur_rows if badge_scope == "Current filters" else comp_life_rows)
    dynasty_year_flags = _hof_company_dynasty_year_flags(comp_cur_rows if badge_scope == "Current filters" else comp_life_rows)

    # Distinct years for "Era Staple" (based on year_points>=200)
    def _years_distinct(company_rows: pd.DataFrame) -> dict[str, int]:
        if company_rows.empty:
            return {}
        tmp = company_rows.dropna(subset=["year"]).copy()
        tmp["ye_points"] = pd.to_numeric(tmp["ye_points"], errors="coerce").fillna(0.0)
        yr = tmp.groupby(["company", "year"])["ye_points"].sum().reset_index(name="year_points")
        yr = yr[yr["year_points"] >= 200.0]
        return yr.groupby("company")["year"].nunique().to_dict()

    years_distinct = _years_distinct(comp_cur_rows if badge_scope == "Current filters" else comp_life_rows)

    # Which aggregate to use for badges / views
    agg_for_badges = agg_life if badge_scope == "Lifetime" else agg_cur
    comp_for_badges = comp_agg_life if badge_scope == "Lifetime" else comp_agg_cur

    hof_section = st.selectbox(
        "Hall of Fame section",
        ["Inductees", "Leaderboards", "Wings", "Profiles", "Years & Seasons"],
        index=0,
        key="hof_section",
    )

    # -------------------
    # Inductees
    # -------------------
    if hof_section == "Inductees":
        st.markdown("### Inductees")
        mode = st.radio("Inductees type", ["Shows", "Companies"], horizontal=True, index=0, key="hof_ind_type")

        if mode == "Shows":
            a = agg_cur.copy()
            if a.empty:
                st.info("No shows match your current filters.")
            else:
                a = a[a["weeks_charting"] >= int(HOF_INDUCT_MIN_WEEKS)].copy()

                # simple deterministic score
                a["hof_score"] = (
                    a["total_ye_points"]
                    + a["weeks_at_1"] * 50.0
                    + a["top10_weeks"] * 5.0
                    + a["total_gross_millions"] * 0.10
                )

                # badges per row
                flags_map = {}
                for sid in a["show_id"].tolist():
                    flags_map[int(sid)] = {
                        "comeback": int(sid) in comeback_ids,
                        "gatekeeper": int(sid) in gate_ids,
                        "onehit": int(sid) in onehit_ids,
                    }

                # attach badge summary
                badge_strs = []
                for _, r in a.iterrows():
                    sid = int(r["show_id"])
                    ref_row = agg_for_badges[agg_for_badges["show_id"] == sid]
                    rr = (ref_row.iloc[0].to_dict() if not ref_row.empty else r.to_dict())
                    badges = _hof_badge_pack_show(rr, flags_map.get(sid, {}))
                    btxt, _ = _hof_pick_badges_for_table(badges, _HOF_BADGE_ORDER_SHOW)
                    badge_strs.append(btxt)
                a["badges"] = badge_strs

                out = a.sort_values(["hof_score", "total_ye_points"], ascending=False).head(int(top_n))
                show_cols = [
                    "hof_score",
                    "canonical_title",
                    "badges",
                    "total_ye_points",
                    "weeks_at_1",
                    "weeks_charting",
                    "total_gross_millions",
                    "peak_rank",
                    "first_week",
                    "last_week",
                ]
                st.dataframe(out[show_cols], width='stretch', hide_index=True)

        else:
            c = comp_agg_cur.copy()
            if c.empty:
                st.info("No companies match your current filters.")
            else:
                c["years_distinct"] = c["company"].map(years_distinct).fillna(0).astype(int)

                # score
                c["hof_score"] = (
                    c["total_ye_points"]
                    + c["weeks_at_1_sum"] * 25.0
                    + c["unique_shows_charted"] * 30.0
                    + c["total_gross_millions"] * 0.05
                )

                badge_strs = []
                for _, r in c.iterrows():
                    company = str(r["company"])
                    ref = comp_for_badges[comp_for_badges["company"] == company]
                    rr = (ref.iloc[0].to_dict() if not ref.empty else r.to_dict())
                    rr["years_distinct"] = years_distinct.get(company, 0)
                    flags = {
                        "deep_bench": deep_bench_flags.get(company, False),
                        "dynasty_year": dynasty_year_flags.get(company, False),
                        "seasonal": False,
                    }
                    badges = _hof_badge_pack_company(rr, flags)
                    btxt, _ = _hof_pick_badges_for_table(badges, _HOF_BADGE_ORDER_COMPANY)
                    badge_strs.append(btxt)

                c["badges"] = badge_strs
                out = c.sort_values(["hof_score", "total_ye_points"], ascending=False).head(int(top_n))
                cols = [
                    "hof_score",
                    "company",
                    "badges",
                    "total_ye_points",
                    "weeks_at_1_sum",
                    "unique_shows_charted",
                    "total_gross_millions",
                    "years_distinct",
                    "first_week",
                    "last_week",
                ]
                st.dataframe(out[cols], width='stretch', hide_index=True)

    # -------------------
    # Leaderboards
    # -------------------
    if hof_section == "Leaderboards":
        st.markdown("### Leaderboards")
        st.caption("Fast leaderboards — your usual ‘who’s the best at *this* one thing’ view.")

        a = agg_cur.copy()
        if a.empty:
            st.info("No shows match your current filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Shows — most weeks at #1**")
                st.dataframe(
                    a.sort_values(["weeks_at_1", "total_ye_points"], ascending=False)
                    .head(int(top_n))[["canonical_title", "weeks_at_1", "weeks_charting", "total_ye_points"]],
                    width='stretch',
                    hide_index=True,
                )
                st.markdown("**Shows — most total points**")
                st.dataframe(
                    a.sort_values(["total_ye_points", "weeks_charting"], ascending=False)
                    .head(int(top_n))[["canonical_title", "total_ye_points", "weeks_at_1", "weeks_charting"]],
                    width='stretch',
                    hide_index=True,
                )
            with col2:
                st.markdown("**Shows — most weeks charting**")
                st.dataframe(
                    a.sort_values(["weeks_charting", "total_ye_points"], ascending=False)
                    .head(int(top_n))[["canonical_title", "weeks_charting", "peak_rank", "total_ye_points"]],
                    width='stretch',
                    hide_index=True,
                )
                st.markdown("**Shows — most total gross**")
                st.dataframe(
                    a.sort_values(["total_gross_millions", "total_ye_points"], ascending=False)
                    .head(int(top_n))[["canonical_title", "total_gross_millions", "weeks_charting", "total_ye_points"]],
                    width='stretch',
                    hide_index=True,
                )

        st.divider()

        c = comp_agg_cur.copy()
        if c.empty:
            st.info("No companies match your current filters.")
        else:
            c["years_distinct"] = c["company"].map(years_distinct).fillna(0).astype(int)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Companies — most total points**")
                st.dataframe(
                    c.sort_values(["total_ye_points"], ascending=False)
                    .head(int(top_n))[["company", "total_ye_points", "unique_shows_charted", "weeks_at_1_sum"]],
                    width='stretch',
                    hide_index=True,
                )
                st.markdown("**Companies — most weeks at #1 (sum)**")
                st.dataframe(
                    c.sort_values(["weeks_at_1_sum", "total_ye_points"], ascending=False)
                    .head(int(top_n))[["company", "weeks_at_1_sum", "total_ye_points", "unique_shows_charted"]],
                    width='stretch',
                    hide_index=True,
                )
            with col2:
                st.markdown("**Companies — most total gross**")
                st.dataframe(
                    c.sort_values(["total_gross_millions", "total_ye_points"], ascending=False)
                    .head(int(top_n))[["company", "total_gross_millions", "total_ye_points", "unique_shows_charted"]],
                    width='stretch',
                    hide_index=True,
                )
                st.markdown("**Companies — deepest catalog**")
                st.dataframe(
                    c.sort_values(["unique_shows_charted", "total_ye_points"], ascending=False)
                    .head(int(top_n))[["company", "unique_shows_charted", "total_ye_points", "years_distinct"]],
                    width='stretch',
                    hide_index=True,
                )

    # -------------------
    # Wings
    # -------------------
    if hof_section == "Wings":
        st.markdown("### Wings")
        st.caption("The museum exhibits. Each one is a different flavor of greatness (or chaos).")

        # Expanders default behavior
        with st.expander("Dynasties", expanded=True):
            with st.expander("Company Dynasties (Year)", expanded=True):
                dco = _hof_wing_dynasty_champions(comp_cur_rows, entity="company")
                if dco.empty:
                    st.info("No multi-year company champion streaks in this filter range.")
                else:
                    st.dataframe(dco, width='stretch', hide_index=True)

            with st.expander("Show Dynasties (Multi-year)", expanded=False):
                dsh = _hof_wing_dynasty_champions(df_filt, entity="show")
                if dsh.empty:
                    st.info("No multi-year show champion streaks in this filter range.")
                else:
                    st.dataframe(dsh, width='stretch', hide_index=True)

        with st.expander("Prime Runs", expanded=False):
            st.caption("Longest streaks of being elite (rank ≤ 3) inside the current filters.")
            if df_filt.empty:
                st.info("No rows match your filters.")
            else:
                use_weeknum = df_filt["week_number"].notna().sum() > 0
                rows = []
                for (sid, title), g in df_filt.groupby(["show_id", "canonical_title"], dropna=False):
                    gg = g.sort_values(["week_number" if use_weeknum else "week_ending_dt"]).copy()
                    elite = (pd.to_numeric(gg["rank"], errors="coerce") <= 3).astype(int)
                    # compute longest consecutive 1s
                    longest = 0
                    cur = 0
                    for v in elite.tolist():
                        if v == 1:
                            cur += 1
                            longest = max(longest, cur)
                        else:
                            cur = 0
                    if longest >= 6:
                        rows.append({"show_id": sid, "canonical_title": title, "prime_run_weeks(rank<=3)": longest})
                d = pd.DataFrame(rows).sort_values("prime_run_weeks(rank<=3)", ascending=False).head(int(top_n))
                if d.empty:
                    st.info("No prime runs found (try widening date/rank filters).")
                else:
                    st.dataframe(d, width='stretch', hide_index=True)

        with st.expander("Gatekeepers", expanded=False):
            st.caption("Massive Top 10 presence, peak rank #2–#3, *never* hit #1.")
            if wing_gate.empty:
                st.info("No gatekeepers found in this filter range.")
            else:
                st.dataframe(
                    wing_gate.head(int(top_n))[["canonical_title", "top10_weeks", "weeks_charting", "peak_rank", "total_ye_points"]],
                    width='stretch',
                    hide_index=True,
                )

        with st.expander("Comeback Kids", expanded=False):
            st.caption(f"Shows with distinct runs separated by a **{int(comeback_gap)}+ week** gap.")
            if wing_comeback.empty:
                st.info("No comeback kids found in this filter range.")
            else:
                st.dataframe(
                    wing_comeback.head(int(top_n))[["canonical_title", "runs", "max_gap_weeks", "weeks_charting"]],
                    width='stretch',
                    hide_index=True,
                )

        with st.expander("Seasonal Bosses", expanded=False):
            st.caption("Month & quarter champs across all years in the current filter range (by total ye_points).")
            tmp = df_filt.dropna(subset=["month", "quarter", "year"]).copy()
            if tmp.empty:
                st.info("Not enough dated rows for seasonal titles.")
            else:
                m = (
                    tmp.groupby(["month", "canonical_title"], dropna=False)["ye_points"]
                    .sum()
                    .reset_index(name="points")
                    .sort_values(["month", "points"], ascending=[True, False])
                    .groupby("month", as_index=False)
                    .first()
                )
                q = (
                    tmp.groupby(["quarter", "canonical_title"], dropna=False)["ye_points"]
                    .sum()
                    .reset_index(name="points")
                    .sort_values(["quarter", "points"], ascending=[True, False])
                    .groupby("quarter", as_index=False)
                    .first()
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Month champs**")
                    st.dataframe(m, width='stretch', hide_index=True)
                with col2:
                    st.markdown("**Quarter champs**")
                    st.dataframe(q, width='stretch', hide_index=True)

        with st.expander("One-Hit Titans", expanded=False):
            st.caption("Exactly 1 week at #1, and ≤ 6 total weeks charting.")
            if wing_onehit.empty:
                st.info("No one-hit titans found in this filter range.")
            else:
                st.dataframe(
                    wing_onehit.head(int(top_n))[["canonical_title", "weeks_at_1", "weeks_charting", "total_ye_points", "total_gross_millions"]],
                    width='stretch',
                    hide_index=True,
                )

    # -------------------
    # Profiles
    # -------------------
    if hof_section == "Profiles":
        st.markdown("### Profiles")
        st.caption("Pick a show or company and see the full badge wall + core stats.")

        mode = st.radio("Profile type", ["Show", "Company"], horizontal=True, index=0, key="hof_profile_type")
        if mode == "Show":
            titles = agg_cur.sort_values("canonical_title")["canonical_title"].tolist()
            if not titles:
                st.info("No shows match your current filters.")
            else:
                pick = st.selectbox("Show", titles, key="hof_profile_show")
                row = agg_cur[agg_cur["canonical_title"] == pick].iloc[0].to_dict()

                sid = int(row["show_id"])
                flags = {
                    "comeback": sid in comeback_ids,
                    "gatekeeper": sid in gate_ids,
                    "onehit": sid in onehit_ids,
                }
                badge_row = (agg_for_badges[agg_for_badges["show_id"] == sid].iloc[0].to_dict()
                             if not agg_for_badges[agg_for_badges["show_id"] == sid].empty else row)
                badges = _hof_badge_pack_show(badge_row, flags)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Weeks charting", int(row.get("weeks_charting", 0)))
                c2.metric("Weeks at #1", int(row.get("weeks_at_1", 0)))
                c3.metric("Peak rank", int(row.get("peak_rank", 999)))
                c4.metric("Total points", f"{float(row.get('total_ye_points', 0)):,.0f}")

                st.write({
                    "Total gross (M)": float(row.get("total_gross_millions", 0)),
                    "Top 10 weeks": int(row.get("top10_weeks", 0)),
                    "Top 5 weeks": int(row.get("top5_weeks", 0)),
                    "Median rank": float(row.get("median_rank", np.nan)),
                    "First week": row.get("first_week"),
                    "Last week": row.get("last_week"),
                })

                st.markdown("#### Badges")
                if not badges:
                    st.caption("No badges earned under this scope.")
                else:
                    # Full badge wall
                    for emo, lbl, how in sorted(badges, key=lambda t: _HOF_BADGE_ORDER_SHOW.index(t[1]) if t[1] in _HOF_BADGE_ORDER_SHOW else 999):
                        st.write(f"{emo} **{lbl}** — {how}")

                with st.expander("Weekly rows (current filters)", expanded=False):
                    led = df_filt[df_filt["show_id"] == sid].sort_values("week_ending_dt", ascending=False)
                    if led.empty:
                        st.caption("No rows for this show inside your current filters.")
                    else:
                        st.dataframe(
                            led[["week_ending", "rank", "pos", "gross_millions", "ye_points", "imprint_1", "imprint_2"]],
                            width='stretch',
                            hide_index=True,
                        )

        else:
            comps = comp_agg_cur.sort_values("company")["company"].tolist()
            if not comps:
                st.info("No companies match your current filters.")
            else:
                pick = st.selectbox("Company", comps, key="hof_profile_company")
                row = comp_agg_cur[comp_agg_cur["company"] == pick].iloc[0].to_dict()
                row["years_distinct"] = years_distinct.get(pick, 0)

                flags = {
                    "deep_bench": deep_bench_flags.get(pick, False),
                    "dynasty_year": dynasty_year_flags.get(pick, False),
                    "seasonal": False,
                }

                ref = comp_for_badges[comp_for_badges["company"] == pick]
                badge_row = (ref.iloc[0].to_dict() if not ref.empty else row)
                badge_row["years_distinct"] = years_distinct.get(pick, 0)
                badges = _hof_badge_pack_company(badge_row, flags)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Unique shows", int(row.get("unique_shows_charted", 0)))
                c2.metric("Weeks at #1 (sum)", int(row.get("weeks_at_1_sum", 0)))
                c3.metric("Total points", f"{float(row.get('total_ye_points', 0)):,.0f}")
                c4.metric("Total gross (M)", f"{float(row.get('total_gross_millions', 0)):,.0f}")

                st.write({
                    "Entries": int(row.get("entries", 0)),
                    "Distinct years (>=200 pts)": int(row.get("years_distinct", 0)),
                    "First week": row.get("first_week"),
                    "Last week": row.get("last_week"),
                })

                st.markdown("#### Badges")
                if not badges:
                    st.caption("No badges earned under this scope.")
                else:
                    for emo, lbl, how in sorted(badges, key=lambda t: _HOF_BADGE_ORDER_COMPANY.index(t[1]) if t[1] in _HOF_BADGE_ORDER_COMPANY else 999):
                        st.write(f"{emo} **{lbl}** — {how}")

                with st.expander("Company rows (current filters)", expanded=False):
                    led = comp_cur_rows[comp_cur_rows["company"] == pick].sort_values("week_ending_dt", ascending=False)
                    if led.empty:
                        st.caption("No rows for this company inside your current filters.")
                    else:
                        st.dataframe(
                            led[["week_ending", "canonical_title", "rank", "gross_millions", "ye_points"]],
                            width='stretch',
                            hide_index=True,
                        )

    # -------------------
    # Years & Seasons
    # -------------------
    if hof_section == "Years & Seasons":
        st.markdown("### Years & Seasons")
        st.caption("Who owned which year? And what does each month/quarter ‘belong’ to?")

        tmp = df_filt.dropna(subset=["year"]).copy()
        if tmp.empty:
            st.info("Not enough dated rows in this filter range.")
        else:
            y = (
                tmp.groupby(["year", "canonical_title"], dropna=False)["ye_points"]
                .sum()
                .reset_index(name="points")
                .sort_values(["year", "points"], ascending=[True, False])
            )
            champs = y.groupby("year", as_index=False).first()
            champs = champs.rename(columns={"canonical_title": "year_champ_show", "points": "champ_points"})
            st.markdown("**Year champs (shows)**")
            st.dataframe(champs.sort_values("year", ascending=False), width='stretch', hide_index=True)

            st.divider()

            yc = _hof_company_universe(tmp)
            y2 = (
                yc.groupby(["year", "company"], dropna=False)["ye_points"]
                .sum()
                .reset_index(name="points")
                .sort_values(["year", "points"], ascending=[True, False])
            )
            champsc = y2.groupby("year", as_index=False).first().rename(columns={"company": "year_champ_company", "points": "champ_points"})
            st.markdown("**Year champs (companies)**")
            st.dataframe(champsc.sort_values("year", ascending=False), width='stretch', hide_index=True)

            st.divider()
            with st.expander("Holiday Bosses", expanded=False):
                st.caption("Best-performing shows on each holiday’s chart week (based on ye_points within your current Hall of Fame filters).")

                gross_mode = st.radio(
                    "Gross used for holiday totals / tiebreak",
                    ["Base gross (no bonuses)", "Total gross (base + bonuses)"],
                    index=0,
                    horizontal=True,
                )
                gross_col = "base_gross_millions" if gross_mode.startswith("Base") else "gross_millions"
                gross_label = "base_gross_millions" if gross_col == "base_gross_millions" else "gross_millions"

                winners = _hof_holiday_winners_by_year(df_filt, all_week_endings, holidays=HOLIDAYS, gross_col=gross_col)
                if winners.empty:
                    st.info("No holiday chart-weeks found in this filter range.")
                else:
                    champs_h = _hof_holiday_champions_from_winners(winners)

                    champs_disp = champs_h.rename(columns={"total_gross_millions": gross_label})
                    winners_disp = winners.rename(columns={"total_gross_millions": gross_label})

                    st.markdown("**Holiday champions (shows)**")
                    st.dataframe(
                        champs_disp.sort_values(["holiday"], ascending=True),
                        width="stretch",
                        hide_index=True,
                    )

                    st.markdown("**Holiday winners by year**")
                    st.dataframe(
                        winners_disp.sort_values(["holiday", "year"], ascending=[True, False]),
                        width="stretch",
                        hide_index=True,
                    )


# ----------------------------
# New tab: Streak Analytics
# ----------------------------
def tab_streak_analytics():
    st.subheader("Streak Analytics")
    st.caption("Longest consecutive-week streaks for a show at a given rank. (Uses week_number when available.)")

    shows, _ = load_lists()

    with st.sidebar:
        st.header("Streak filters")
        date_min = st.text_input("Start date (YYYY-MM-DD)      ", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)        ", value="")
        rank_min, rank_max = st.slider("Rank range (streaks)", 1, 17, (1, 10))
        top_n = st.slider("Top N (streaks)", 5, 200, 25)

    filters = FilterSpec(date_min.strip() or None, date_max.strip() or None, int(rank_min), int(rank_max))
    where, params = build_where(filters, "e")

    rows = sql_df(f"""
        SELECT
          e.week_ending,
          e.week_number,
          e.rank,
          e.pos,
          e.show_id,
          s.canonical_title
        FROM t10_entry e
        JOIN show s ON s.show_id = e.show_id
        WHERE {where}
        ORDER BY e.week_number ASC, e.rank ASC, e.pos ASC
    """, tuple(params))

    if rows.empty:
        st.info("No rows match your filters.")
        return

    streaks = compute_longest_streaks(rows)
    if streaks.empty:
        st.info("Not enough data to compute streaks.")
        return

    rank_tab, charted_tab = st.tabs(["By Rank", "Longest Charted Runs"])

    with rank_tab:
        st.markdown("### Longest streaks by rank")
        ranks = sorted(streaks["rank"].dropna().unique().tolist())
        rank_pick = st.selectbox("Rank", ranks, index=0)

        block = streaks[streaks["rank"] == rank_pick].head(int(top_n)).copy()
        st.dataframe(block, width='stretch')

        st.divider()
        st.markdown("### Per-show streak breakdown")
        title_pick = st.selectbox("Show (canonical)", shows["canonical_title"].tolist(), key="streak_show_pick")
        show_id = int(shows.loc[shows["canonical_title"] == title_pick, "show_id"].iloc[0])

        show_block = streaks[streaks["show_id"] == show_id].sort_values(["rank"]).copy()
        if show_block.empty:
            st.info("No streak data for this show in the selected filters.")
        else:
            st.dataframe(show_block, width='stretch')

            st.markdown("### Quick peek: raw weeks for this show (filtered)")
            # Useful for validating consecutive week_number behavior
            show_rows = rows[rows["show_id"] == show_id].copy()
            show_rows["week_ending"] = _as_date_str(show_rows["week_ending"])
            show_rows_disp = show_rows.sort_values(["week_number", "rank", "pos"]).drop(columns=["pos"], errors="ignore")
            st.dataframe(show_rows_disp, width='stretch')

    with charted_tab:
        st.markdown("### Longest consecutive charted runs")
        run_df = compute_longest_charted_runs(rows)
        if run_df.empty:
            st.info("No charted-run data available in the selected filters.")
        else:
            run_show = run_df.head(int(top_n)).copy().rename(columns={
                "canonical_title": "Show",
                "run_len": "Weeks",
                "start_week_ending": "Start",
                "end_week_ending": "End",
            })
            run_show.insert(0, "Rank", range(1, len(run_show) + 1))
            st.dataframe(run_show[["Rank", "Show", "Weeks", "Start", "End"]], width='stretch', hide_index=True)

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("SQLite + FTS search, per-show analytics, company analytics, and movement/grossing charts. (Ties supported.)")

    # Make the top tab bar horizontally scrollable (so you can reach "Admin" on smaller screens)
    st.markdown(
        """
        <style>
        /* Streamlit tabs: allow horizontal scrolling instead of wrapping/cropping */
        .stTabs [data-baseweb="tab-list"],
        div[data-baseweb="tab-list"] {
            overflow-x: auto !important;
            overflow-y: hidden !important;
            flex-wrap: nowrap !important;
            white-space: nowrap !important;
            scrollbar-width: thin; /* Firefox */
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar,
        div[data-baseweb="tab-list"]::-webkit-scrollbar {
            height: 8px;
        }
        .stTabs [data-baseweb="tab-list"] button,
        div[data-baseweb="tab-list"] button {
            white-space: nowrap !important;
            flex: 0 0 auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # Keep Admin last.
    tabs = st.tabs([
        "Search",
        "Show Detail",
        "Compare Two Shows",
        "Companies",
        "Analytics",
        "Gross Races",
        "Monthly T-25 (SMPS)",
        "Year-End (SMPS)",
        "Grossing Milestones",
        "Grossing Trends",
        "Show Trends",
        "T-10 Chart #1 Shows",
        "Streak Analytics",
        "Holidays",
        "Records and Achievements",
        "Hall of Fame",
        "Admin",
    ])

    with tabs[0]:
        tab_search()
    with tabs[1]:
        tab_show_detail()
    with tabs[2]:
        tab_compare_two_shows()
    with tabs[3]:
        tab_companies()
    with tabs[4]:
        tab_analytics()
    with tabs[5]:
        tab_gross_races()
    with tabs[6]:
        tab_monthly_smps_t25()
    with tabs[7]:
        tab_year_end_smps_t35()
    with tabs[8]:
        tab_grossing_milestones()
    with tabs[9]:
        tab_grossing_trends()
    with tabs[10]:
        tab_show_trends()
    with tabs[11]:
        tab_t10_chart_number_shows()
    with tabs[12]:
        tab_streak_analytics()
    with tabs[13]:
        tab_holidays()
    with tabs[14]:
        tab_records_achievements()
    with tabs[15]:
        tab_hall_of_fame()
    with tabs[16]:
        tab_admin()


if __name__ == "__main__":
    main()
