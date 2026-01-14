#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Callable

from datetime import date, timedelta

import numpy as np
import pandas as pd
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

def fetch_entries(filters: FilterSpec, fts_query: str | None = None, limit: int = 1000) -> pd.DataFrame:
    where, params = build_where(filters, "e")
    params2 = list(params)

    if fts_query and fts_query.strip():
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
        params2 = [fts_query.strip()] + params2 + [int(limit)]
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



def fetch_company_entries(company: str, filters: FilterSpec, limit: int = 2000) -> pd.DataFrame:
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
    WHERE COALESCE(e.imprint_1,'(Unknown)') = ?
      AND {where}
    ORDER BY e.week_ending DESC, e.rank ASC, e.pos ASC
    LIMIT ?
    """
    df = sql_df(sql, tuple([company] + params + [int(limit)]))
    if not df.empty:
        df["week_ending"] = _as_date_str(df["week_ending"])
    return df


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
def _consecutive_by_week_number(wn: pd.Series) -> pd.Series:
    """Boolean series: True where current row continues a streak from prior row."""
    d = wn.diff()
    return d.eq(1)

def _consecutive_by_date(week_ending_str: pd.Series) -> pd.Series:
    """Boolean series: True where current row continues a streak from prior row (by ~7 day spacing)."""
    dt = pd.to_datetime(week_ending_str, errors="coerce")
    dd = dt.diff().dt.days
    lo, hi = CONSECUTIVE_DAY_TOLERANCE
    return dd.between(lo, hi)

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
        * If the holiday is Sun/Mon/Tue/Wed -> use the previous weekend (Saturday before)
        * If the holiday is Thu/Fri/Sat     -> use the following weekend (Saturday on/after)
      Example: Independence Day (07-04) on Thursday -> use week_ending 07-06.
    - Thanksgiving: use the following week_ending (Saturday after Thanksgiving).
      Example: Thanksgiving 11-23 -> use 11-25.
    - Weekend/Monday holidays (Easter, Memorial Day, Labor Day, MLK Day, Presidents Day):
      use the weekend the holiday is part of (Saturday before).
      Example: Easter 04-17 -> use 04-16.
    - Fallback: if no prior/next exists (edge years), use the closest available week_ending.
    """
    if not all_week_endings:
        return None

    weeks = sorted(all_week_endings)

    def prev_week_ending(d: date) -> Optional[date]:
        prev = None
        for we in weeks:
            if we < d:
                prev = we
            else:
                break
        return prev

    def next_week_ending(d: date) -> Optional[date]:
        for we in weeks:
            if we >= d:
                return we
        return None

    def closest_week_ending(d: date) -> date:
        return min(weeks, key=lambda we: abs((we - d).days))

    name = (holiday_name or "").strip()

    # Thanksgiving: always the following week ending
    if name.startswith("Thanksgiving"):
        we = next_week_ending(holiday_dt)
        return we if we is not None else closest_week_ending(holiday_dt)

    # Weekend/Monday-style holidays: Saturday before
    if (
        name.startswith("Easter")
        or name.startswith("Memorial Day")
        or name.startswith("Labor Day")
        or name.startswith("Martin Luther King")
        or name.startswith("Presidents Day")
    ):
        we = prev_week_ending(holiday_dt)
        return we if we is not None else closest_week_ending(holiday_dt)

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
        if wd in (6, 0, 1, 2):  # Sun/Mon/Tue/Wed
            we = prev_week_ending(holiday_dt)
            return we if we is not None else closest_week_ending(holiday_dt)
        else:  # Thu/Fri/Sat
            we = next_week_ending(holiday_dt)
            return we if we is not None else closest_week_ending(holiday_dt)

    # Default: previous chart as-of holiday date
    we = prev_week_ending(holiday_dt)
    return we if we is not None else closest_week_ending(holiday_dt)





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
        use_container_width=True,
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
        use_container_width=True,
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

    subtabs = st.tabs(["Overview", "Momentum", "Longevity", "Market Structure", "Imprints/Companies", "Anomalies & Eras"])

    # ----------------------------
    # Overview
    # ----------------------------
    with subtabs[0]:
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
                st.dataframe(show_tbl, use_container_width=True)

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
                st.dataframe(pos_stats, use_container_width=True)

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
        # Use Top 2 regardless of chosen rank_scope
        df_top2 = df[df["rank"].between(1, 2)].copy()
        if df_top2.empty:
            st.info("Not enough data for #1 vs #2.")
        else:
            pivot = (
                df_top2.groupby(["week_ending_dt", "rank"])["gross_use"]
                .mean()
                .reset_index()
                .pivot(index="week_ending_dt", columns="rank", values="gross_use")
                .sort_index()
            )
            if 1 not in pivot.columns or 2 not in pivot.columns:
                st.info("Not enough data for both #1 and #2 weeks.")
            else:
                ratio = (pivot[1] / pivot[2].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                diff = pivot[1] - pivot[2]
                fig = plt.figure()
                plt.plot(diff.index, diff.values)
                plt.title("#1 premium (difference: #1 − #2)")
                st.pyplot(fig, clear_figure=True)

                fig = plt.figure()
                plt.plot(ratio.index, ratio.values)
                plt.title("#1 premium (ratio: #1 ÷ #2)")
                st.pyplot(fig, clear_figure=True)

    # ----------------------------
    # Momentum
    # ----------------------------
    with subtabs[1]:
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
                st.dataframe(tbl, use_container_width=True)

                st.markdown("#### Biggest week-over-week $ gains")
                tbl = show_week.sort_values("abs_change", ascending=False).head(topn)[
                    ["canonical_title", "week_ending_dt", "prev_gross", "gross_use", "abs_change"]
                ].copy()
                tbl["week_ending_dt"] = tbl["week_ending_dt"].dt.date
                st.dataframe(tbl, use_container_width=True)

            with c2:
                st.markdown("#### Biggest week-over-week drops")
                tbl = show_week.sort_values("abs_change", ascending=True).head(topn)[
                    ["canonical_title", "week_ending_dt", "prev_gross", "gross_use", "abs_change"]
                ].copy()
                tbl["week_ending_dt"] = tbl["week_ending_dt"].dt.date
                st.dataframe(tbl, use_container_width=True)

                st.markdown("#### Biggest 4-appearance moves (net change)")
                show_week["gross_4_ago"] = show_week.groupby("show_id")["gross_use"].shift(4)
                show_week["net_4"] = show_week["gross_use"] - show_week["gross_4_ago"]
                tbl = show_week.dropna(subset=["gross_4_ago"]).sort_values("net_4", ascending=False).head(topn)[
                    ["canonical_title", "week_ending_dt", "gross_4_ago", "gross_use", "net_4"]
                ].copy()
                tbl["week_ending_dt"] = tbl["week_ending_dt"].dt.date
                st.dataframe(tbl, use_container_width=True)

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
            st.dataframe(slope_df, use_container_width=True)

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
            st.dataframe(reb_df, use_container_width=True)

    # ----------------------------
    # Longevity
    # ----------------------------
    with subtabs[2]:
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
                    st.dataframe(peaks.sort_values("peak_gross", ascending=False).head(200), use_container_width=True)

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
                        st.dataframe(half_df.sort_values("weeks_to_half", ascending=False).head(200), use_container_width=True)

    # ----------------------------
    # Market Structure
    # ----------------------------
    with subtabs[3]:
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
            st.dataframe(newest[["canonical_title", "first_week"]], use_container_width=True)

    # ----------------------------
    # Imprints / Companies
    # ----------------------------
    with subtabs[4]:
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
            st.dataframe(share_tbl.head(50), use_container_width=True)

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
                st.dataframe(mom.head(50), use_container_width=True)
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
            st.dataframe(hits.head(50), use_container_width=True)

    # ----------------------------
    # Anomalies & Eras
    # ----------------------------
    with subtabs[5]:
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
                st.dataframe(out_disp, use_container_width=True)

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
                st.dataframe(eras_df, use_container_width=True)

def tab_search():
    st.subheader("Search")
    with st.sidebar:
        st.header("Search filters")
        fts = st.text_input("Full-text search (FTS)", placeholder="e.g. Nickelodeon AND (school OR kids)")
        date_min = st.text_input("Start date (YYYY-MM-DD)", value="", key="gt_date_min")
        date_max = st.text_input("End date (YYYY-MM-DD)", value="", key="gt_date_max")
        rank_min, rank_max = st.slider("Rank range", 1, 50, (1, 10))
        limit = st.slider("Max results", 50, 10000, 1000, step=50)

    filters = FilterSpec(
        date_min=date_min.strip() or None,
        date_max=date_max.strip() or None,
        rank_min=int(rank_min),
        rank_max=int(rank_max),
    )
    df = fetch_entries(filters, fts_query=fts, limit=int(limit))

    st.write(f"Results: **{len(df)}**")
    st.dataframe(df, use_container_width=True)

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
        rank_min, rank_max = st.slider("Rank range (show)", 1, 50, (1, 10))

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
            st.dataframe(led, use_container_width=True, hide_index=True)

    df = fetch_show_entries(show_id, filters)
    st.dataframe(df, use_container_width=True)

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
        rank_min, rank_max = st.slider("Rank range (compare)", 1, 50, (1, 10))
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
        st.dataframe(overlap, use_container_width=True)

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
    st.subheader("Company view (Imprint 1)")
    _, companies = load_lists()
    company = st.selectbox("Company (Imprint 1)", companies["company"].tolist())

    with st.sidebar:
        st.header("Company filters")
        date_min = st.text_input("Start date (YYYY-MM-DD)    ", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)     ", value="")
        rank_min, rank_max = st.slider("Rank range (company)", 1, 50, (1, 10))

    filters = FilterSpec(date_min.strip() or None, date_max.strip() or None, int(rank_min), int(rank_max))

    stat = sql_df("SELECT * FROM v_company_stats WHERE company = ?", (company,))
    if not stat.empty:
        s = stat.iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entries", int(s["entries"]))
        c2.metric("Unique shows", int(s["unique_shows"]))
        c3.metric("Total gross (M)", float(s["total_gross_millions"]))
        c4.metric("Avg gross (M)", None if pd.isna(s["avg_gross_millions"]) else float(s["avg_gross_millions"]))

    df = fetch_company_entries(company, filters)
    st.dataframe(df, use_container_width=True)


def tab_analytics():
    st.subheader("Analytics (grossing + movement)")
    st.caption("All metrics are computed from stored rows.")

    with st.sidebar:
        st.header("Analytics filters")
        date_min = st.text_input("Start date (YYYY-MM-DD)     ", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)      ", value="")
        rank_min, rank_max = st.slider("Rank range (analytics)", 1, 50, (1, 10))
        top_n = st.slider("Top N", 5, 50, 15)

    filters = FilterSpec(date_min.strip() or None, date_max.strip() or None, int(rank_min), int(rank_max))

    where, params = build_where(filters, "e")
    df = sql_df(f"""
        SELECT
          e.week_ending,
          e.week_number,
          e.rank,
          e.pos,
          e.last_week,
          e.gross_millions AS base_gross_millions,
          COALESCE(gb.bonus_millions, 0) AS bonus_millions,
          (e.gross_millions + COALESCE(gb.bonus_millions, 0)) AS gross_millions,
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
    """, tuple(params))

    if df.empty:
        st.info("No rows match your filters.")
        return

    df["week_ending"] = _as_date_str(df["week_ending"])
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["year"] = df["week_ending_dt"].dt.year

    dg = df.dropna(subset=["gross_millions"]).copy()
    for col in ["gross_millions", "base_gross_millions", "bonus_millions"]:
        if col in dg.columns:
            dg[col] = pd.to_numeric(dg[col], errors="coerce")

    ignore_bonus = st.checkbox(
        "Ignore gross bonuses in most analytics (Top shows/companies + yearly totals still include bonuses)",
        value=False,
    )
    gross_col = "base_gross_millions" if ignore_bonus else "gross_millions"
    gross_label = "Base gross (millions)" if gross_col == "base_gross_millions" else "Gross + bonuses (millions)"
    if ignore_bonus:
        st.caption(
            "Bonuses are excluded from most charts/totals below, except Top shows, Top companies, and Yearly gross totals (those always include bonuses)."
        )
    st.markdown("### Total gross over time (weekly sum)")
    if dg.empty:
        st.warning("No gross values in the selected range.")
        return

    weekly = (

        dg.groupby("week_ending", as_index=False)[gross_col]

        .sum()

        .rename(columns={gross_col: "gross_millions"})

        .sort_values("week_ending")

    )
    plot_line_dates(weekly["week_ending"], weekly["gross_millions"], "Week Ending", "Total Gross (Millions)")

    # Keep this INSIDE the tab so 'weekly' is defined
    if st.checkbox("Show Top Gross Weeks"):
        chart_top_gross_weeks(weekly, n=20)

    st.markdown("### Rolling average total gross")
    win = st.slider("Rolling window (weeks)", 2, 52, 13)
    w2 = weekly.copy()
    w2["roll"] = w2["gross_millions"].rolling(win, min_periods=max(1, win // 3)).mean()
    plot_line_dates(w2["week_ending"], w2["roll"], "Week Ending", f"{win}-week avg gross (Millions)")

    st.markdown("### Rank vs Gross (scatter)")
    plot_scatter(dg["rank"].astype(float), dg[gross_col].astype(float), "Rank", gross_label)

    st.markdown("### Top companies by total gross")
    top_comp = dg.groupby("company", as_index=False)["gross_millions"].sum()
    top_comp = top_comp.sort_values("gross_millions", ascending=False).head(int(top_n))
    st.dataframe(top_comp, use_container_width=True)
    plot_barh(top_comp["company"][::-1], top_comp["gross_millions"][::-1], "Total Gross (Millions)", "Company")

    st.markdown("### Gross distribution")
    plot_hist(dg[gross_col].astype(float), bins=30, xlabel=gross_label, ylabel="Count")

    st.markdown("### Yearly gross totals")
    yearly = dg.groupby("year", as_index=False)["gross_millions"].sum().sort_values("year")
    st.dataframe(yearly, use_container_width=True)
    fig = plt.figure()
    plt.plot(yearly["year"], yearly["gross_millions"])
    plt.xlabel("Year")
    plt.ylabel("Total Gross (Millions)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



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
              SUM(c.base_gross_millions) AS base_gross_millions,
              SUM(c.bonus_millions) AS bonus_millions,
              SUM(c.base_gross_millions + c.bonus_millions) AS gross_millions
            FROM combined c
            JOIN show s ON s.show_id = c.show_id
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


def tab_gross_races():
    st.subheader("Gross Races")
    st.caption("All charts on this page include weekly gross **plus all gross bonuses** (annual + quarter bonuses included via gross_bonus).")

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}.")
        return

    db_mtime = DB_PATH.stat().st_mtime
    base = _load_gross_races_base(str(DB_PATH), db_mtime)
    if base.empty:
        st.info("No gross data found.")
        return

    base["week_ending_dt"] = pd.to_datetime(base["week_ending"], errors="coerce")
    # Exclude pre-gross-tracking years (you started tracking grosses the week ending 2001-03-17)
    base = base[base["week_ending_dt"] >= pd.Timestamp(GROSS_TRACKING_START)].copy()
    if base.empty:
        st.info("No gross rows found on/after the gross-tracking start date (2001-03-17).")
        return

    latest_dt = pd.to_datetime(base["week_ending_dt"].max())
    latest_date = latest_dt.date()

    meta = _load_show_meta_for_gross_races(str(DB_PATH), db_mtime)

    # -------------------------
    # 1) All-Time Gross Races Chart (unlimited rank)
    # -------------------------
    st.markdown("### All-Time Gross Races Chart")
    all_time = base.groupby(["show_id", "canonical_title"], as_index=False)["gross_millions"].sum()
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

    st.caption("Unlimited rank: every show with any gross is included.")
    all_time_disp = all_time[["rank", "canonical_title", "imprint_1", "imprint_2", "debut_date", "gross_millions"]].copy()
    st.dataframe(all_time_disp, use_container_width=True, hide_index=True)

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

        leaders = leaders[["rank", "canonical_title", "imprint_1", "imprint_2", "cum_gross_millions"]].copy()

        st.caption(f"Leaderboard for **{pick_dt.year}** (through **{pick_dt.isoformat()}**)")
        st.dataframe(leaders, use_container_width=True, hide_index=True)

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

    leaders_q = leaders_q[["rank", "canonical_title", "imprint_1", "imprint_2", "cum_gross_millions"]].copy()

    st.caption(f"Leaderboard for **Q{int(quarter)} {int(year_pick)}** (through **{wk_dt.date().isoformat()}**)")
    st.dataframe(leaders_q, use_container_width=True, hide_index=True)

    top_kq = st.slider("Shows to plot (quarter)", 2, min(50, int(len(leaders_q))), min(10, int(len(leaders_q))), key="q_race_topk")
    top_titles_q = leaders_q.head(int(top_kq))["canonical_title"].tolist()

    pivq = qdf[qdf["canonical_title"].isin(top_titles_q)].copy()
    pivq = pivq.pivot_table(index="week_ending_dt", columns="canonical_title", values="cum_gross_millions", aggfunc="max").sort_index()
    pivq = pivq.reindex(pd.to_datetime(qweeks)).ffill()

    series_by_label_q = {c: pivq[c] for c in pivq.columns}
    _plot_multi_line(list(pivq.index), series_by_label_q, "Week Ending", "Cumulative Gross (Millions)")

    st.divider()

    # -------------------------
    # 4) Monthly Gross Races (28th → 27th)
    # -------------------------
    st.markdown("### Monthly Gross Races")
    st.caption("Cumulative grosses reset on the 28th of each month (e.g., Dec 28 → Jan 27).")

    pick_m_dt = st.date_input(
        "As-of date (pick any date to view that cycle's race)",
        value=latest_date,
        min_value=GROSS_TRACKING_START,
        max_value=latest_date,
        key="gross_races_month_pick",
    )
    through_m_dt = pd.to_datetime(pick_m_dt)

    # Cycle start (28th)
    if through_m_dt.day >= 28:
        cycle_start = pd.Timestamp(year=int(through_m_dt.year), month=int(through_m_dt.month), day=28)
    else:
        prev = through_m_dt - pd.DateOffset(months=1)
        cycle_start = pd.Timestamp(year=int(prev.year), month=int(prev.month), day=28)
    cycle_end_excl = cycle_start + pd.DateOffset(months=1)
    cycle_end_incl = (cycle_end_excl - pd.Timedelta(days=1)).date()

    mbase = base[(base["week_ending_dt"] >= cycle_start) & (base["week_ending_dt"] < cycle_end_excl)].copy()
    mbase = mbase[mbase["week_ending_dt"] <= through_m_dt].copy()

    if mbase.empty:
        st.info("No monthly data available for that date range (gross-tracking era filter applied).")
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

        st.caption(
            f"Leaderboard for cycle **{cycle_start.date().isoformat()} → {cycle_end_incl.isoformat()}** "
            f"(through **{through_m_dt.date().isoformat()}**)"
        )
        st.dataframe(leaders_m, use_container_width=True, hide_index=True)

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
        rank_min, rank_max = st.slider("Rank range (streaks)", 1, 50, (1, 10))
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

    st.markdown("### Longest streaks by rank")
    ranks = sorted(streaks["rank"].dropna().unique().tolist())
    rank_pick = st.selectbox("Rank", ranks, index=0)

    block = streaks[streaks["rank"] == rank_pick].head(int(top_n)).copy()
    st.dataframe(block, use_container_width=True)

    st.divider()
    st.markdown("### Per-show streak breakdown")
    title_pick = st.selectbox("Show (canonical)", shows["canonical_title"].tolist(), key="streak_show_pick")
    show_id = int(shows.loc[shows["canonical_title"] == title_pick, "show_id"].iloc[0])

    show_block = streaks[streaks["show_id"] == show_id].sort_values(["rank"]).copy()
    if show_block.empty:
        st.info("No streak data for this show in the selected filters.")
        return

    st.dataframe(show_block, use_container_width=True)

    st.markdown("### Quick peek: raw weeks for this show (filtered)")
    # Useful for validating consecutive week_number behavior
    show_rows = rows[rows["show_id"] == show_id].copy()
    show_rows["week_ending"] = _as_date_str(show_rows["week_ending"])
    st.dataframe(show_rows.sort_values(["week_number", "rank", "pos"]), use_container_width=True)


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
    st.dataframe(out, use_container_width=True)

    miss = out["#1_show(s)"].isna().sum() if not out.empty else 0
    if miss:
        st.warning(
            f"{miss} year(s) had no #1 record for the computed holiday-week. "
            "This usually means your database doesn’t have that week, or rank=1 is missing for that week."
        )

    with st.expander("How the holiday week is chosen"):
        st.write(
            "- Fixed-date holidays (New Year's, Valentine's, Independence Day, Halloween, Christmas):\n"
            "  - Sun/Mon/Tue/Wed → previous weekend (Saturday before)\n"
            "  - Thu/Fri/Sat → following weekend (Saturday on/after)\n"
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

    st.markdown("### View aliases for a show")
    show_for_aliases = st.selectbox("Show", titles, key="alias_list_show")
    show_id = int(shows.loc[shows["canonical_title"] == show_for_aliases, "show_id"].iloc[0])
    alias_df = sql_df("SELECT alias_title FROM show_alias WHERE show_id = ? ORDER BY alias_title", (show_id,))
    st.dataframe(alias_df, use_container_width=True)


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
        st.dataframe(disp, use_container_width=True, hide_index=True)

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
            st.dataframe(disp, use_container_width=True, hide_index=True)

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
            st.dataframe(disp, use_container_width=True, hide_index=True)

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
            st.dataframe(disp, use_container_width=True, hide_index=True)

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
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

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
                    "Order": int(order) if j == 0 else "",
                    "Show": r["canonical_title"],
                    "Week Hit #1": _fmt_date(r["first_n1"]),
                })
        disp = pd.DataFrame(rows)
        st.dataframe(disp, use_container_width=True, hide_index=True)

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
            st.dataframe(disp, use_container_width=True, hide_index=True)

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
        st.dataframe(disp, use_container_width=True, hide_index=True)

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
        st.dataframe(disp, use_container_width=True, hide_index=True)

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
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ----------------------------
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
        "Grossing Milestones",
        "Grossing Trends",
        "Streak Analytics",
        "Holidays",
        "Records and Achievements",
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
        tab_grossing_milestones()
    with tabs[7]:
        tab_grossing_trends()
    with tabs[8]:
        tab_streak_analytics()
    with tabs[9]:
        tab_holidays()
    with tabs[10]:
        tab_records_achievements()
    with tabs[11]:
        tab_admin()


if __name__ == "__main__":
    main()
