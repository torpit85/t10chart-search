#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
          e.gross_millions
        FROM t10_fts f
        JOIN t10_entry e ON e.id = f.rowid
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
          e.gross_millions
        FROM t10_entry e
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
      e.gross_millions
    FROM t10_entry e
    WHERE e.show_id = ?
      AND {where}
    ORDER BY e.week_number ASC, e.rank ASC, e.pos ASC
    """
    df = sql_df(sql, tuple([show_id] + params))
    if not df.empty:
        df["week_ending"] = _as_date_str(df["week_ending"])
    return df

def fetch_show_stats(show_id: int) -> pd.DataFrame:
    return sql_df("SELECT * FROM v_show_stats WHERE show_id = ?", (show_id,))

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
      e.gross_millions
    FROM t10_entry e
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
          e.gross_millions,
          COALESCE(e.imprint_1,'(Unknown)') AS company,
          s.canonical_title
        FROM t10_entry e
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
    dg["gross_millions"] = pd.to_numeric(dg["gross_millions"], errors="coerce")

    st.markdown("### Total gross over time (weekly sum)")
    if dg.empty:
        st.warning("No gross values in the selected range.")
        return

    weekly = dg.groupby("week_ending", as_index=False)["gross_millions"].sum().sort_values("week_ending")
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
    plot_scatter(dg["rank"].astype(float), dg["gross_millions"].astype(float), "Rank", "Gross (Millions)")

    st.markdown("### Top shows by total gross")
    top_shows = dg.groupby("canonical_title", as_index=False)["gross_millions"].sum()
    top_shows = top_shows.sort_values("gross_millions", ascending=False).head(int(top_n))
    st.dataframe(top_shows, use_container_width=True)
    plot_barh(top_shows["canonical_title"][::-1], top_shows["gross_millions"][::-1], "Total Gross (Millions)", "Show")

    st.markdown("### Top companies by total gross")
    top_comp = dg.groupby("company", as_index=False)["gross_millions"].sum()
    top_comp = top_comp.sort_values("gross_millions", ascending=False).head(int(top_n))
    st.dataframe(top_comp, use_container_width=True)
    plot_barh(top_comp["company"][::-1], top_comp["gross_millions"][::-1], "Total Gross (Millions)", "Company")

    st.markdown("### Gross distribution")
    plot_hist(dg["gross_millions"].astype(float), bins=30, xlabel="Gross (Millions)", ylabel="Count")

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

    tabs = st.tabs(["Search", "Show Detail", "Compare Two Shows", "Companies", "Analytics", "Admin"])
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
        tab_admin()


if __name__ == "__main__":
    main()

