# charts.py
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def chart_top_gross_weeks(weekly: pd.DataFrame, n: int = 20):
    top = weekly.sort_values("gross_millions", ascending=False).head(n)
    st.dataframe(top, use_container_width=True)
    fig = plt.figure()
    plt.barh(top["week_ending"][::-1], top["gross_millions"][::-1])
    plt.xlabel("Total Gross (Millions)")
    plt.ylabel("Week Ending")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

