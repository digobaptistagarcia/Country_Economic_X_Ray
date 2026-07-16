"""
Country Economic X-Ray
A consulting-grade Streamlit page for exploring long-run living standards
and productivity dynamics using Penn World Table (PWT) data.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Country Economic X-Ray",
    layout="wide",
)

DATA_PATH = "pwt_clean.parquet"

# Fixed decade breakpoints used across the page for era segmentation
ERA_BOUNDS = [
    (None, 1989, "Pre-1990"),
    (1990, 1999, "1990s"),
    (2000, 2009, "2000s"),
    (2010, 2019, "2010s"),
    (2020, None, "2020s"),
]

ERA_COLORS = [
    "rgba(99, 110, 250, 0.10)",
    "rgba(239, 85, 59, 0.10)",
    "rgba(0, 204, 150, 0.10)",
    "rgba(171, 99, 250, 0.10)",
    "rgba(255, 161, 90, 0.10)",
]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values(["countrycode", "year"]).reset_index(drop=True)
    return df


def cagr(first_value: float, last_value: float, n_years: int) -> float | None:
    """Compound annual growth rate between two points, n_years apart."""
    if n_years <= 0 or first_value is None or last_value is None:
        return None
    if pd.isna(first_value) or pd.isna(last_value) or first_value <= 0:
        return None
    return (last_value / first_value) ** (1 / n_years) - 1


def resolve_era_bounds(min_year: int, max_year: int):
    """Clip the fixed era breakpoints to the years actually available."""
    resolved = []
    for start, end, label in ERA_BOUNDS:
        s = min_year if start is None else max(start, min_year)
        e = max_year if end is None else min(end, max_year)
        if s <= e:
            resolved.append((s, e, label))
    return resolved


def era_cagr_series(df: pd.DataFrame, value_col: str, bounds):
    """Compute CAGR per era band for a given value column (drops NaNs)."""
    results = []
    for s, e, label in bounds:
        window = df[(df["year"] >= s) & (df["year"] <= e)].dropna(subset=[value_col])
        if len(window) < 2:
            results.append((s, e, label, None))
            continue
        first_row = window.iloc[0]
        last_row = window.iloc[-1]
        n_years = last_row["year"] - first_row["year"]
        g = cagr(first_row[value_col], last_row[value_col], n_years)
        results.append((int(first_row["year"]), int(last_row["year"]), label, g))
    return results


# --------------------------------------------------------------------------
# Chart builders — charts 4-9 (each isolated so it can be reused/tested
# independently, matching v1's inline-figure pattern otherwise)
# --------------------------------------------------------------------------
def build_government_size_chart(cdf: pd.DataFrame) -> go.Figure:
    """Chart 4 — Government Size: csh_g (share of government spending in GDP)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf["year"],
            y=cdf["csh_g"] * 100,
            mode="lines",
            name="Government share of GDP",
            line=dict(color="#AB63FA", width=2),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="Government spending (% of GDP)",
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        height=420,
    )
    return fig


def build_work_effort_chart(cdf: pd.DataFrame) -> go.Figure:
    """Chart 5 — Work Effort: avh (average annual hours worked per worker)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf["year"],
            y=cdf["avh"],
            mode="lines",
            name="Average annual hours worked",
            line=dict(color="#FFA15A", width=2),
            hovertemplate="%{x}: %{y:,.0f} hrs<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="Average annual hours worked per worker",
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        height=420,
    )
    return fig


def build_human_capital_chart(cdf: pd.DataFrame) -> go.Figure:
    """Chart 6 — Human Capital: hc (human capital index)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf["year"],
            y=cdf["hc"],
            mode="lines",
            name="Human capital index",
            line=dict(color="#19D3F3", width=2),
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="Human capital index (education & skills, 1=baseline)",
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        height=420,
    )
    return fig


def build_investment_rate_chart(cdf: pd.DataFrame) -> go.Figure:
    """Chart 7 — Investment Rate: csh_i (share of capital investment in GDP)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf["year"],
            y=cdf["csh_i"] * 100,
            mode="lines",
            name="Investment share of GDP",
            line=dict(color="#00CC96", width=2),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="Capital investment (% of GDP)",
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        height=420,
    )
    return fig


def build_income_split_chart(cdf: pd.DataFrame) -> go.Figure:
    """Chart 8 — Income Split: labsh (labor share of income)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf["year"],
            y=cdf["labsh"] * 100,
            mode="lines",
            name="Labor share of income",
            line=dict(color="#EF553B", width=2),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="Labor share of income (%)",
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        height=420,
        yaxis_range=[0, 100],
    )
    return fig


def build_trade_openness_chart(cdf: pd.DataFrame) -> go.Figure:
    """Chart 9 — Trade Openness: csh_x (exports) vs. |csh_m| (imports), stacked."""
    exports = cdf["csh_x"] * 100
    imports = cdf["csh_m"].abs() * 100  # csh_m is stored as a negative share in PWT

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cdf["year"],
            y=exports,
            mode="lines",
            name="Exports (% of GDP)",
            stackgroup="one",
            line=dict(color="#636EFA", width=0.5),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cdf["year"],
            y=imports,
            mode="lines",
            name="Imports (% of GDP)",
            stackgroup="one",
            line=dict(color="#EF553B", width=0.5),
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title="Trade flows (% of GDP, stacked)",
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# --------------------------------------------------------------------------
# Load data + sidebar/country selector
# --------------------------------------------------------------------------
df = load_data(DATA_PATH)

country_options = (
    df[["country", "countrycode"]]
    .drop_duplicates()
    .sort_values("country")
    .reset_index(drop=True)
)
labels = country_options["country"].tolist()
default_idx = int(country_options.index[country_options["countrycode"] == "AUS"][0]) if (
    "AUS" in country_options["countrycode"].values
) else 0

st.title("Country Economic X-Ray")
st.caption(
    "Long-run living standards and productivity diagnostics, built on the Penn World Table (PWT 10.01)."
)

selected_country = st.selectbox("Country", options=labels, index=default_idx)
code = country_options.loc[country_options["country"] == selected_country, "countrycode"].iloc[0]

cdf = df[df["countrycode"] == code].sort_values("year").reset_index(drop=True)

if cdf.empty:
    st.warning("No data available for this country.")
    st.stop()

min_year, max_year = int(cdf["year"].min()), int(cdf["year"].max())
bounds = resolve_era_bounds(min_year, max_year)

st.divider()

# --------------------------------------------------------------------------
# Chart 1 — Living Standards Over Time
# --------------------------------------------------------------------------
st.subheader("1. Living Standards Over Time")
st.markdown(
    "Real GDP per capita is the standard proxy for a population's average material "
    "living standard. Tracking it over the full available history shows the "
    "long-run trajectory of a country's development — and how far current levels "
    "sit from historical trend."
)

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=cdf["year"],
        y=cdf["gdp_pc"],
        mode="lines",
        name="GDP per capita",
        line=dict(color="#636EFA", width=2),
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
    )
)
fig1.update_layout(
    yaxis_title="Real GDP per capita (constant 2017 intl-$)",
    xaxis_title="Year",
    hovermode="x unified",
    margin=dict(t=20, b=20),
    height=420,
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 2 — Golden Era vs Now
# --------------------------------------------------------------------------
st.subheader("2. Golden Era vs Now")
st.markdown(
    "Splitting the same GDP-per-capita series into decade-length eras and "
    "computing the compound annual growth rate (CAGR) within each band isolates "
    "which periods actually drove today's living standards — and whether recent "
    "growth is keeping pace with a country's own history."
)

era_gdp = era_cagr_series(cdf, "gdp_pc", bounds)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=cdf["year"],
        y=cdf["gdp_pc"],
        mode="lines",
        name="GDP per capita",
        line=dict(color="#636EFA", width=2),
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
    )
)

y_max = cdf["gdp_pc"].max()
for i, (s, e, label, g) in enumerate(era_gdp):
    fig2.add_vrect(
        x0=s, x1=e,
        fillcolor=ERA_COLORS[i % len(ERA_COLORS)],
        line_width=0,
    )
    annotation = f"{label}<br>{g*100:+.1f}%pa" if g is not None else f"{label}<br>n/a"
    fig2.add_annotation(
        x=(s + e) / 2,
        y=y_max * 1.04,
        text=annotation,
        showarrow=False,
        font=dict(size=11),
        align="center",
    )

fig2.update_layout(
    yaxis_title="Real GDP per capita (constant 2017 intl-$)",
    xaxis_title="Year",
    hovermode="x unified",
    margin=dict(t=60, b=20),
    height=460,
    yaxis_range=[0, y_max * 1.15],
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 3 — Productivity Growth
# --------------------------------------------------------------------------
st.subheader("3. Productivity Growth")

tfp_col = "rtfpna"
tfp_label = "Real TFP at constant national prices (2017=1)"
if cdf["rtfpna"].isna().all() and not cdf["ctfp"].isna().all():
    tfp_col = "ctfp"
    tfp_label = "TFP at current PPPs (2017=1) — fallback series"

st.markdown(
    "Total factor productivity (TFP) captures how efficiently a country converts "
    "capital and labor into output — the key long-run driver of GDP-per-capita "
    "growth once population and investment effects are stripped out. The rolling "
    "average smooths year-to-year noise to reveal the underlying trend."
)

tfp_df = cdf.dropna(subset=[tfp_col]).copy()

if tfp_df.empty:
    st.info("No TFP data (rtfpna or ctfp) available for this country.")
else:
    tfp_df["rolling_10y"] = tfp_df[tfp_col].rolling(window=10, min_periods=3).mean()

    tfp_min_year, tfp_max_year = int(tfp_df["year"].min()), int(tfp_df["year"].max())
    tfp_bounds = resolve_era_bounds(tfp_min_year, tfp_max_year)
    era_tfp = era_cagr_series(tfp_df, tfp_col, tfp_bounds)

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=tfp_df["year"],
            y=tfp_df[tfp_col],
            mode="lines",
            name=tfp_label,
            line=dict(color="#00CC96", width=1.5),
            opacity=0.55,
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=tfp_df["year"],
            y=tfp_df["rolling_10y"],
            mode="lines",
            name="10-year rolling average",
            line=dict(color="#EF553B", width=2.5),
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )

    tfp_y_max = tfp_df[tfp_col].max()
    for s, e, label, g in era_tfp:
        if g is None:
            continue
        annotation = f"{s}-{e} {g*100:+.1f}%pa"
        fig3.add_annotation(
            x=(s + e) / 2,
            y=tfp_y_max * 1.06,
            text=annotation,
            showarrow=False,
            font=dict(size=10.5),
            align="center",
        )

    fig3.update_layout(
        yaxis_title=tfp_label,
        xaxis_title="Year",
        hovermode="x unified",
        margin=dict(t=60, b=20),
        height=460,
        yaxis_range=[0, tfp_y_max * 1.18],
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 4 — Government Size
# --------------------------------------------------------------------------
st.subheader("4. Government Size")
st.markdown(
    "The share of government spending in GDP indicates the size of the public "
    "sector relative to the overall economy. A rising share can reflect expanding "
    "public services, but persistently high levels also raise crowding-out risk — "
    "less room for private investment and slower long-run growth."
)
st.plotly_chart(build_government_size_chart(cdf), use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 5 — Work Effort
# --------------------------------------------------------------------------
st.subheader("5. Work Effort")
st.markdown(
    "Average annual hours worked shows whether a country's living standards are "
    "built on working more, rather than on higher output per hour worked. A "
    "country that relies on long hours to sustain GDP per capita has less "
    "productivity cushion than one where the same living standard comes from "
    "fewer, more productive hours."
)
if cdf["avh"].isna().all():
    st.info("Data not available for this country.")
else:
    st.plotly_chart(build_work_effort_chart(cdf), use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 6 — Human Capital
# --------------------------------------------------------------------------
st.subheader("6. Human Capital")
st.markdown(
    "The human capital index captures the skills and education embedded in the "
    "workforce, a key structural driver of productivity growth. Unlike "
    "hours-based gains, improvements in human capital tend to be durable and "
    "compound over time."
)
if cdf["hc"].isna().all():
    st.info("Data not available for this country.")
else:
    st.plotly_chart(build_human_capital_chart(cdf), use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 7 — Investment Rate
# --------------------------------------------------------------------------
st.subheader("7. Investment Rate")
st.markdown(
    "Capital investment as a share of GDP reflects how much of current output is "
    "being redirected into future productive capacity — plant, equipment, and "
    "infrastructure. Sustained low investment rates constrain how fast an economy "
    "can grow its capital stock going forward."
)
st.plotly_chart(build_investment_rate_chart(cdf), use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 8 — Income Split
# --------------------------------------------------------------------------
st.subheader("8. Income Split")
st.markdown(
    "The labor share of income shows how national output is divided between "
    "workers and capital owners. A declining labor share means growth is "
    "increasingly accruing to capital — a structural shift with direct "
    "implications for inequality and consumption-driven demand."
)
st.plotly_chart(build_income_split_chart(cdf), use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Chart 9 — Trade Openness
# --------------------------------------------------------------------------
st.subheader("9. Trade Openness")
st.markdown(
    "Exports and imports as shares of GDP, stacked together, show how exposed "
    "the economy is to external demand and supply shocks. A high combined share "
    "means growth and stability are more tightly linked to global conditions "
    "outside the country's direct control."
)
st.plotly_chart(build_trade_openness_chart(cdf), use_container_width=True)

st.divider()
st.caption("Source: Penn World Table 10.01 (Feenstra, Inklaar & Timmer). Data as loaded from pwt_clean.parquet.")
