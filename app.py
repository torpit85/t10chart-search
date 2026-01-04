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
    # Stats computed from rows + gross_bonus, so they match gross races / analytics totals.
    return sql_df(
        """
        WITH gb AS (
          SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
          FROM gross_bonus
          GROUP BY show_id, week_ending
        ),
        rows AS (
          SELECT
            e.week_ending,
            e.rank,
            (COALESCE(e.gross_millions, 0) + COALESCE(gb.bonus_millions, 0)) AS gross_millions
          FROM t10_entry e
          LEFT JOIN gb
            ON gb.show_id = e.show_id
           AND gb.week_ending = e.week_ending
          WHERE e.show_id = ?
        )
        SELECT
          COUNT(DISTINCT date(week_ending)) AS weeks_on_chart,
          MIN(rank) AS peak_rank,
          MIN(date(week_ending)) AS first_appearance,
          MAX(date(week_ending)) AS last_appearance,
          SUM(gross_millions) AS total_gross_millions,
          AVG(gross_millions) AS avg_gross_millions,
          AVG(rank) AS avg_rank
        FROM rows
        """,
        (show_id,),
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
    "Valentine's Day (Feb 14)": lambda y: date(y, 2, 14),
    "Easter (variable)": easter_date,
    "Memorial Day (last Mon in May)": lambda y: last_weekday_of_month(y, 5, 0),
    "Independence Day (Jul 4)": lambda y: date(y, 7, 4),
    "Labor Day (1st Mon in Sep)": lambda y: nth_weekday_of_month(y, 9, 0, 1),
    "Halloween (Oct 31)": lambda y: date(y, 10, 31),
    "Thanksgiving (4th Thu in Nov)": lambda y: nth_weekday_of_month(y, 11, 3, 4),
    "Christmas Day (Dec 25)": lambda y: date(y, 12, 25),
}

def holiday_week_ending_for_date(all_week_endings: list[date], holiday_dt: date) -> Optional[date]:
    """
    Choose the week_ending such that holiday_dt is within the 7-day window ending on week_ending:
      week_start = week_ending - 6 days
    """
    if not all_week_endings:
        return None
    weeks = sorted(all_week_endings)
    for we in weeks:
        if we >= holiday_dt and (we - holiday_dt).days <= 6:
            return we
    return min(weeks, key=lambda we: abs((we - holiday_dt).days))


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
            SELECT
              e.week_ending,
              e.show_id,
              s.canonical_title AS canonical_title,
              COALESCE(e.gross_millions, 0) AS base_gross_millions,
              COALESCE(gb.bonus_millions, 0) AS bonus_millions,
              (COALESCE(e.gross_millions, 0) + COALESCE(gb.bonus_millions, 0)) AS gross_millions
            FROM t10_entry e
            LEFT JOIN (
              SELECT show_id, week_ending, SUM(bonus_millions) AS bonus_millions
              FROM gross_bonus
              GROUP BY show_id, week_ending
            ) gb ON gb.show_id = e.show_id AND gb.week_ending = e.week_ending
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

    st.markdown("#### Top 20 earliest to reach this milestone")
    top20 = hit.head(20).copy()
    st.dataframe(
        top20[["canonical_title", "week_ending", "cumulative_gross_millions", "week_gross_millions"]],
        use_container_width=True,
        hide_index=True,
    )
    if not top20.empty:
        w = top20.iloc[0]
        st.success(
            f"Earliest: **{w['canonical_title']}** reached **{fmt_int(pick_big)}** on **{w['week_ending']}** "
            f"(cumulative: **{w['cumulative_gross_millions']:.1f}**)."
        )

# ----------------------------
# UI Tabs
# ----------------------------
def tab_search():
    st.subheader("Search")
    with st.sidebar:
        st.header("Search filters")
        fts = st.text_input("Full-text search (FTS)", placeholder="e.g. Nickelodeon AND (school OR kids)")
        date_min = st.text_input("Start date (YYYY-MM-DD)", value="")
        date_max = st.text_input("End date (YYYY-MM-DD)", value="")
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
        c1.metric("Weeks on chart", int(s["weeks_on_chart"]))
        c2.metric("Peak rank", int(s["peak_rank"]))
        c3.metric("First appearance", str(s["first_appearance"]))
        c4.metric("Last appearance", str(s["last_appearance"]))
        st.write({
            "Total gross (M)": float(s["total_gross_millions"]),
            "Avg gross (M)": None if pd.isna(s["avg_gross_millions"]) else float(s["avg_gross_millions"]),
            "Avg rank": float(s["avg_rank"]),
        })

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

    # -------------------------
    # 1) All-Time Gross Races Chart (unlimited rank)
    # -------------------------
    st.markdown("### All-Time Gross Races Chart")
    all_time = base.groupby("canonical_title", as_index=False)["gross_millions"].sum()
    all_time = all_time[all_time["gross_millions"] > 0].copy()
    all_time = all_time.sort_values("gross_millions", ascending=False).reset_index(drop=True)
    all_time.insert(0, "rank", np.arange(1, len(all_time) + 1))

    st.caption("Unlimited rank: every show with any gross is included.")
    st.dataframe(all_time, use_container_width=True, hide_index=True)

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
        leaders = last[["canonical_title", "cum_gross_millions"]].copy()
        leaders = leaders.sort_values("cum_gross_millions", ascending=False).reset_index(drop=True)
        leaders.insert(0, "rank", np.arange(1, len(leaders) + 1))

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
    leaders_q = lastq[["canonical_title", "cum_gross_millions"]].copy()
    leaders_q = leaders_q.sort_values("cum_gross_millions", ascending=False).reset_index(drop=True)
    leaders_q.insert(0, "rank", np.arange(1, len(leaders_q) + 1))

    st.caption(f"Leaderboard for **Q{int(quarter)} {int(year_pick)}** (through **{wk_dt.date().isoformat()}**)")
    st.dataframe(leaders_q, use_container_width=True, hide_index=True)

    top_kq = st.slider("Shows to plot (quarter)", 2, min(50, int(len(leaders_q))), min(10, int(len(leaders_q))), key="q_race_topk")
    top_titles_q = leaders_q.head(int(top_kq))["canonical_title"].tolist()

    pivq = qdf[qdf["canonical_title"].isin(top_titles_q)].copy()
    pivq = pivq.pivot_table(index="week_ending_dt", columns="canonical_title", values="cum_gross_millions", aggfunc="max").sort_index()
    pivq = pivq.reindex(pd.to_datetime(qweeks)).ffill()

    series_by_label_q = {c: pivq[c] for c in pivq.columns}
    _plot_multi_line(list(pivq.index), series_by_label_q, "Week Ending", "Cumulative Gross (Millions)")


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
        we = holiday_week_ending_for_date(week_endings, hdt)
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
            "- The holiday is assigned to the chart week whose Week Ending is the first date on/after the holiday, within 6 days.\n"
            "- If no such week exists, the closest Week Ending date is used."
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

    st.markdown("### View aliases for a show")
    show_for_aliases = st.selectbox("Show", titles, key="alias_list_show")
    show_id = int(shows.loc[shows["canonical_title"] == show_for_aliases, "show_id"].iloc[0])
    alias_df = sql_df("SELECT alias_title FROM show_alias WHERE show_id = ? ORDER BY alias_title", (show_id,))
    st.dataframe(alias_df, use_container_width=True)


# ----------------------------
# Main
# ----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("SQLite + FTS search, per-show analytics, company analytics, and movement/grossing charts. (Ties supported.)")

    # Original tabs + 2 new tabs inserted before Admin (so Admin stays last)
    tabs = st.tabs(["Search", "Show Detail", "Compare Two Shows", "Companies", "Analytics", "Gross Races", "Grossing Milestones", "Streak Analytics", "Holidays", "Admin"])
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
        tab_streak_analytics()
    with tabs[8]:
        tab_holidays()
    with tabs[9]:
        tab_admin()



if __name__ == "__main__":
    main()
