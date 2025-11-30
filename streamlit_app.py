import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
import altair as alt

st.set_page_config(page_title="Ozone Forecast Dashboard", layout="wide")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

@st.cache_data(ttl=300)
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=20)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_predictions(limit=200, start=None, end=None):
    params = {"limit": limit}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    r = requests.get(f"{API_URL}/predict", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]) 
    df = df.sort_values("date")
    return df

def metric_cards(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", f"{len(df):,}")
    with col2:
        st.metric("Actual avg", f"{df['actualValue'].mean():.1f}")
    with col3:
        st.metric("Predicted avg", f"{df['predictedValue'].mean():.1f}")
    with col4:
        st.metric("MAE", f"{np.abs(df['actualValue']-df['predictedValue']).mean():.1f}")

def line_chart(df):
    mdf = df.melt(id_vars="date", value_vars=["actualValue", "predictedValue"], var_name="series", value_name="value")
    chart = alt.Chart(mdf).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Ozone μg/m³"),
        color=alt.Color("series:N", title="Series")
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

def heatmap(df):
    hdf = df.copy()
    hdf["dow"] = hdf["date"].dt.dayofweek
    hdf["week"] = hdf["date"].dt.isocalendar().week.astype(int)
    chart = alt.Chart(hdf).mark_rect().encode(
        x=alt.X("week:O", title="Week"),
        y=alt.Y("dow:O", title="Day", sort=[0,1,2,3,4,5,6]),
        color=alt.Color("predictedValue:Q", title="μg/m³", scale=alt.Scale(scheme="redyellowgreen")),
        tooltip=["date:T", alt.Tooltip("predictedValue:Q", format=".1f")]
    ).properties(height=180)
    st.altair_chart(chart, use_container_width=True)

def daily_table(df):
    st.dataframe(df.tail(30)[["date", "actualValue", "predictedValue"]].rename(columns={"date": "Date", "actualValue": "Actual", "predictedValue": "Predicted"}), use_container_width=True)

def layout():
    st.title("Ozone Forecast Dashboard")
    health = get_health()
    status_col, limit_col, range_col = st.columns([1,1,3])
    with status_col:
        if health and health.get("status") == "ok":
            st.success(f"Backend OK • modelLoaded={health.get('modelLoaded')} • count={health.get('count')}")
        else:
            st.error("Backend unavailable")
    with limit_col:
        limit = st.slider("Limit", 30, 500, 200)
    with range_col:
        df_all = get_predictions(limit=500)
        min_d = df_all["date"].min().date()
        max_d = df_all["date"].max().date()
        start_d, end_d = st.slider("Date range", min_value=min_d, max_value=max_d, value=(max_d, max_d))
    df = get_predictions(limit=limit, start=str(start_d), end=str(end_d))

    metric_cards(df)

    lc1, lc2 = st.columns([3,2])
    with lc1:
        st.subheader("Trend")
        line_chart(df)
    with lc2:
        st.subheader("Heatmap")
        heatmap(df)

    st.subheader("Daily values")
    daily_table(df)

if __name__ == "__main__":
    layout()

