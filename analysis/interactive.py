"""Interactive dashboard for optimizer-agent experiment summaries.

Reads ``experiment_results_<version>.csv`` and
``experiment_results_summary_<version>.csv`` from ``--data-dir``. Build those
files from optimizer-agent runs with ``collect_results.py``.

Run:
    python analysis/collect_results.py --experiments-dir experiments --version adv8
    python analysis/interactive.py --data-dir experiments --version adv8
"""

import argparse
import math
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "optimizer-agent · Experiment Explorer"

# Set from CLI in __main__.
DATA_DIR = Path(".")
VERSION = "optagent"

PERFORMANCE_METRICS = [
    "combined_score",
    "fast_1",
    "fast_2",
    "correctness",
    "compiled",
    "loading_success",
    "runtime_custom_ms",
    "runtime_original_ms",
    "new_combined_score",
]
TOKEN_METRICS = ["prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"]
METRICS = PERFORMANCE_METRICS + TOKEN_METRICS
METRIC_LABELS = {
    "combined_score": "Score / speedup",
    "fast_1": "Fast @ 1x",
    "fast_2": "Fast @ 2x",
    "correctness": "Correctness",
    "compiled": "Compiled",
    "loading_success": "Loading success",
    "runtime_custom_ms": "Candidate runtime (ms)",
    "runtime_original_ms": "Reference runtime (ms)",
    "new_combined_score": "Runtime improvement (ms)",
    "prompt_tokens": "Prompt tokens",
    "completion_tokens": "Completion tokens",
    "total_tokens": "Total tokens",
    "llm_calls": "LLM calls",
}
SCORE_COLUMNS = [
    "compiled",
    "loading_success",
    "correctness",
    "runtime_custom_ms",
    "runtime_original_ms",
    "combined_score",
    "fast_1",
    "fast_2",
    "new_combined_score",
]
TOKEN_COLUMNS = ["prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"]
PALETTE = [
    "#58a6ff",
    "#f78166",
    "#3fb950",
    "#d2a8ff",
    "#ffa657",
    "#79c0ff",
    "#7ee787",
    "#ff7b72",
]
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

APP_CSS = """
body { background: #0d1117; color: #c9d1d9; }
.app-title { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; margin: 0; }
.app-subtitle { color: #8b949e; margin: 4px 0 0; }
.kpi-label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.kpi-value { font-size: 26px; font-weight: 700; color: #f0f6fc; margin-top: 4px; }
.kpi-sublabel { color: #6e7681; font-size: 12px; min-height: 16px; }
.control-label { color: #8b949e; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.Select-control, .Select-menu-outer { background-color: #0d1117 !important; color: #c9d1d9 !important; }
.Select-value-label, .Select-placeholder { color: #c9d1d9 !important; }
.tab--selected { border-top: 2px solid #58a6ff !important; color: #f0f6fc !important; }
.js-plotly-plot .plotly .modebar { display: none !important; }
"""

app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>""" + APP_CSS + """</style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""
CARD_STYLE = {
    "backgroundColor": "#161b22",
    "border": "1px solid #30363d",
    "borderRadius": "14px",
    "boxShadow": "0 8px 24px rgba(0,0,0,0.22)",
}
CONTROL_STYLE = {
    "backgroundColor": "#161b22",
    "border": "1px solid #30363d",
    "borderRadius": "14px",
    "padding": "18px 20px",
}
CHART_STYLE = {"height": "520px"}
SMALL_CHART_STYLE = {"height": "390px"}


def parse_method(method_name):
    params = {}
    for part in str(method_name).split("__"):
        if "=" in part:
            key, value = part.split("=", 1)
            params[key] = value
    return params


def method_label(method_name):
    params = parse_method(method_name)
    bits = [params.get("met", str(method_name))]
    if params.get("samp") and params["samp"] != "1":
        bits.append(f"{params['samp']} samples")
    if params.get("el") and params["el"] != "1":
        bits.append(f"{params['el']} cells")
    if params.get("strat") == "True":
        bits.append("strategy")
    if params.get("feed") == "True":
        bits.append("feedback")
    return " · ".join(bits)


def enrich(df):
    if df.empty:
        return df
    df = df.copy()
    df["params"] = df["method_name"].apply(parse_method)
    df["met"] = df["params"].apply(lambda item: item.get("met", ""))
    df["samp"] = df["params"].apply(lambda item: item.get("samp", ""))
    df["el"] = df["params"].apply(lambda item: item.get("el", ""))
    df["strat"] = df["params"].apply(lambda item: item.get("strat", ""))
    df["feed"] = df["params"].apply(lambda item: item.get("feed", ""))
    df["method_label"] = df["method_name"].apply(method_label)
    for column in set(SCORE_COLUMNS + TOKEN_COLUMNS):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "iteration" in df.columns:
        df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    score_cols = [c for c in SCORE_COLUMNS if c in df.columns]
    if score_cols:
        df[score_cols] = df[score_cols].replace([np.inf, -np.inf], np.nan)
    token_cols = [c for c in TOKEN_COLUMNS if c in df.columns]
    if token_cols:
        df[token_cols] = df[token_cols].fillna(0)
    return df


def load_summary():
    path = DATA_DIR / f"experiment_results_summary_{VERSION}.csv"
    return enrich(pd.read_csv(path))


def load_raw():
    path = DATA_DIR / f"experiment_results_{VERSION}.csv"
    return enrich(pd.read_csv(path))


def prepare_summary(cumulative=False):
    df = load_summary()
    if cumulative and not df.empty:
        score_cols = [c for c in SCORE_COLUMNS if c in df.columns]
        token_cols = [c for c in TOKEN_COLUMNS if c in df.columns]
        if score_cols:
            df[score_cols] = df.groupby("method_name")[score_cols].cummax()
        if token_cols:
            df[token_cols] = df.groupby("method_name")[token_cols].cumsum()
    return df


def filter_frame(df, methods, strategies, feedbacks):
    if df.empty:
        return df
    if methods:
        df = df[df["method_name"].isin(methods)]
    if strategies:
        df = df[df["strat"].isin(strategies)]
    if feedbacks:
        df = df[df["feed"].isin(feedbacks)]
    return df


def method_options():
    df = load_summary()
    if df.empty:
        return []
    options = (
        df[["method_name", "method_label"]]
        .drop_duplicates()
        .sort_values("method_label")
    )
    return [
        {"label": row.method_label, "value": row.method_name}
        for row in options.itertuples()
    ]


def finite_values(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values


def kpi_values():
    summary = load_summary()
    raw = load_raw()
    tasks = raw["problem_id"].nunique() if "problem_id" in raw.columns else 0
    iterations = (
        int(summary["iteration"].max()) + 1
        if "iteration" in summary.columns and not summary["iteration"].dropna().empty
        else 0
    )
    scores = finite_values(raw["combined_score"]) if "combined_score" in raw.columns else pd.Series(dtype=float)
    best = float(scores.max()) if not scores.empty else None
    tokens = float(summary["total_tokens"].sum()) if "total_tokens" in summary.columns else 0.0
    calls = float(summary["llm_calls"].sum()) if "llm_calls" in summary.columns else 0.0
    return tasks, iterations, best, tokens, calls


def kpi_card(label, value, sublabel=""):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="kpi-label"),
                html.Div(value, className="kpi-value"),
                html.Div(sublabel, className="kpi-sublabel"),
            ],
            className="p-3",
        ),
        style=CARD_STYLE,
        className="h-100",
    )


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=15, color="#8b949e"),
    )
    return style_figure(fig)


def style_figure(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", family="Inter, system-ui, sans-serif"),
        margin=dict(l=45, r=24, t=48, b=38),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        height=height,
    )
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#30363d")
    return fig


def build_layout():
    tasks, iterations, best, tokens, calls = kpi_values()
    method_values = [option["value"] for option in method_options()]
    best_text = f"{best:,.3f}" if best is not None else "—"
    token_text = f"{tokens:,.0f}" if tokens > 0 else "—"
    calls_text = f"{calls:,.0f}" if calls > 0 else "—"

    return dbc.Container(
        fluid=True,
        style={"maxWidth": "1600px", "padding": "24px 28px 40px"},
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("optimizer-agent", className="app-title"),
                            html.P(
                                "Experiment explorer · scores, correctness and token cost",
                                className="app-subtitle",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Badge(f"run {VERSION}", color="primary", className="me-2"),
                            dbc.Badge(f"{tasks} tasks", color="secondary", className="me-2"),
                            dbc.Badge(f"{iterations} iterations", color="secondary"),
                        ],
                        md=4,
                        className="d-flex align-items-center justify-content-end gap-2",
                    ),
                ],
                className="mb-4 align-items-center",
            ),
            dbc.Row(
                [
                    dbc.Col(kpi_card("Tasks", f"{tasks:,}", "evaluated"), md=2),
                    dbc.Col(kpi_card("Iterations", f"{iterations:,}", "search rounds"), md=2),
                    dbc.Col(kpi_card("Best score", best_text, "best candidate"), md=3),
                    dbc.Col(kpi_card("Total tokens", token_text, "prompt + completion"), md=3),
                    dbc.Col(kpi_card("LLM calls", calls_text, "successful requests"), md=2),
                ],
                className="g-3 mb-4",
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Metric", className="control-label"),
                                    dcc.Dropdown(
                                        id="metric-dropdown",
                                        options=[
                                            {"label": METRIC_LABELS[m], "value": m}
                                            for m in METRICS
                                        ],
                                        value="combined_score",
                                        clearable=False,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div("View", className="control-label"),
                                    dcc.RadioItems(
                                        id="view-filter",
                                        options=[
                                            {"label": "Per-iteration", "value": "iteration"},
                                            {"label": "Cumulative", "value": "cumulative"},
                                        ],
                                        value="iteration",
                                        inline=True,
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Methods", className="control-label"),
                                    dcc.Dropdown(
                                        id="method-filter",
                                        options=method_options(),
                                        value=method_values,
                                        multi=True,
                                        placeholder="All methods",
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Strategy", className="control-label"),
                                    dcc.Dropdown(
                                        id="strat-filter",
                                        options=[
                                            {"label": "with strategy", "value": "True"},
                                            {"label": "without strategy", "value": "False"},
                                        ],
                                        multi=True,
                                        placeholder="All",
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Feedback", className="control-label"),
                                    dcc.Dropdown(
                                        id="feed-filter",
                                        options=[
                                            {"label": "with feedback", "value": "True"},
                                            {"label": "without feedback", "value": "False"},
                                        ],
                                        multi=True,
                                        placeholder="All",
                                    ),
                                ],
                                md=2,
                            ),
                        ],
                        className="g-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Color by", className="control-label"),
                                    dcc.Dropdown(
                                        id="color-grouping",
                                        options=[
                                            {"label": "Method", "value": "method"},
                                            {"label": "Strategy", "value": "strategy"},
                                            {"label": "Feedback", "value": "feedback"},
                                        ],
                                        value="method",
                                        clearable=False,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Scale", className="control-label"),
                                    dcc.Checklist(
                                        id="log-scale",
                                        options=[{"label": "Log Y axis", "value": "log"}],
                                        value=[],
                                        inline=True,
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                html.Div(
                                    "Tip: use the Tasks and Tokens tabs for per-task breakdowns.",
                                    className="app-subtitle mt-2",
                                ),
                                md=7,
                            ),
                        ],
                        className="g-3 mt-1",
                    ),
                ],
                style=CONTROL_STYLE,
                className="mb-4",
            ),
            dcc.Tabs(
                id="tabs",
                value="trends",
                children=[
                    dcc.Tab(
                        label="Trends",
                        value="trends",
                        children=[
                            dcc.Graph(
                                id="trend-chart",
                                config=GRAPH_CONFIG,
                                style=CHART_STYLE,
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Tasks",
                        value="tasks",
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="task-chart",
                                            config=GRAPH_CONFIG,
                                            style=CHART_STYLE,
                                        ),
                                        lg=7,
                                    ),
                                    dbc.Col(
                                        html.Div(id="task-table"),
                                        lg=5,
                                    ),
                                ],
                                className="g-3 mt-1",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Tokens",
                        value="tokens",
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="token-scatter",
                                            config=GRAPH_CONFIG,
                                            style=SMALL_CHART_STYLE,
                                        ),
                                        lg=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="token-trend",
                                            config=GRAPH_CONFIG,
                                            style=SMALL_CHART_STYLE,
                                        ),
                                        lg=6,
                                    ),
                                ],
                                className="g-3 mt-1",
                            )
                        ],
                    ),
                ],
            ),
            html.Div(
                f"Source: {DATA_DIR} · version {VERSION}",
                className="app-subtitle mt-4",
            ),
        ],
    )


@app.callback(
    Output("trend-chart", "figure"),
    [
        Input("metric-dropdown", "value"),
        Input("view-filter", "value"),
        Input("method-filter", "value"),
        Input("strat-filter", "value"),
        Input("feed-filter", "value"),
        Input("color-grouping", "value"),
        Input("log-scale", "value"),
    ],
)
def update_trend(metric, view, methods, strategies, feedbacks, color_by, log_scale):
    df = filter_frame(
        prepare_summary(cumulative=(view == "cumulative")),
        methods,
        strategies,
        feedbacks,
    )
    if df.empty or metric not in df.columns:
        return empty_figure("No data for this selection")

    df = df.dropna(subset=[metric])
    if df.empty:
        return empty_figure("No finite values for this metric")

    group_key = {
        "method": lambda row: row["method_label"],
        "strategy": lambda row: "strategy" if row["strat"] == "True" else "no strategy",
        "feedback": lambda row: "feedback" if row["feed"] == "True" else "no feedback",
    }.get(color_by, lambda row: row["method_label"])

    groups = []
    for _, row in df.iterrows():
        key = group_key(row)
        if key not in groups:
            groups.append(key)

    fig = go.Figure()
    for group_index, group in enumerate(groups):
        subset = df[df.apply(lambda row: group_key(row) == group, axis=1)]
        for method_index, (method, method_df) in enumerate(
            subset.groupby("method_name", sort=False)
        ):
            method_df = method_df.sort_values("iteration")
            fig.add_trace(
                go.Scatter(
                    x=method_df["iteration"],
                    y=method_df[metric],
                    mode="lines+markers",
                    name=method_label(method),
                    legendgroup=group,
                    showlegend=method_index == 0,
                    line=dict(
                        color=PALETTE[group_index % len(PALETTE)],
                        width=2.5,
                    ),
                    marker=dict(size=7, line=dict(width=1, color="#0d1117")),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "iteration %{x}<br>"
                        f"{METRIC_LABELS.get(metric, metric)}: %{{y:,.3f}}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(
            text=f"{METRIC_LABELS.get(metric, metric)} vs iteration",
            font=dict(size=18),
        ),
        xaxis_title="Iteration",
        yaxis_title=METRIC_LABELS.get(metric, metric),
    )
    fig.update_xaxes(dtick=1)
    if log_scale and "log" in log_scale and (df[metric] > 0).all():
        fig.update_yaxes(type="log")
    elif metric in {"fast_1", "fast_2"}:
        fig.update_yaxes(range=[0, 1.05])
    if metric in {"combined_score", "fast_1", "fast_2"}:
        values = finite_values(df[metric])
        if not values.empty and values.min() >= 0 and values.max() > 1.0:
            fig.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="#8b949e",
                annotation_text="baseline parity",
                annotation_position="top left",
            )
    return style_figure(fig, height=560)


def task_data(methods, strategies, feedbacks):
    raw = filter_frame(load_raw(), methods, strategies, feedbacks)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    if "combined_score" not in raw.columns:
        return pd.DataFrame()
    valid = raw.copy()
    valid["combined_score"] = pd.to_numeric(valid["combined_score"], errors="coerce")
    valid = valid.replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna(subset=["combined_score"])
    grouped = valid.groupby(["problem_id", "method_name"], as_index=False).agg(
        best_speedup=("combined_score", "max"),
        tokens=("total_tokens", "sum"),
        calls=("llm_calls", "sum"),
        correctness=("correctness", "mean"),
    )
    grouped["method_label"] = grouped["method_name"].apply(method_label)
    grouped["valid"] = True
    all_pairs = raw[["problem_id", "method_name"]].drop_duplicates()
    all_pairs["method_label"] = all_pairs["method_name"].apply(method_label)
    result = all_pairs.merge(
        grouped[
            ["problem_id", "method_name", "best_speedup", "tokens", "calls", "correctness"]
        ],
        on=["problem_id", "method_name"],
        how="left",
    )
    result["valid"] = result["best_speedup"].notna()
    result["best_speedup"] = result["best_speedup"].fillna(0.0)
    result["tokens"] = result["tokens"].fillna(0.0)
    result["calls"] = result["calls"].fillna(0.0)
    result["correctness"] = result["correctness"].fillna(0.0)
    return result


@app.callback(
    Output("task-chart", "figure"),
    [
        Input("method-filter", "value"),
        Input("strat-filter", "value"),
        Input("feed-filter", "value"),
    ],
)
def update_task_chart(methods, strategies, feedbacks):
    df = task_data(methods, strategies, feedbacks)
    if df.empty:
        return empty_figure("No per-task data")
    valid = df[df["valid"]].sort_values("best_speedup", ascending=True)
    if valid.empty:
        return empty_figure("No valid candidate for the selected tasks")
    fig = go.Figure(
        go.Bar(
            x=valid["best_speedup"],
            y=valid["problem_id"],
            orientation="h",
            marker_color=[
                PALETTE[i % len(PALETTE)] for i in range(len(valid))
            ],
            hovertemplate="<b>%{y}</b><br>best score: %{x:,.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Best score per task", font=dict(size=18)),
        xaxis_title="Best score / speedup",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_xaxes(rangemode="tozero")
    return style_figure(fig, height=560)


@app.callback(
    Output("task-table", "children"),
    [
        Input("method-filter", "value"),
        Input("strat-filter", "value"),
        Input("feed-filter", "value"),
    ],
)
def update_task_table(methods, strategies, feedbacks):
    df = task_data(methods, strategies, feedbacks)
    if df.empty:
        return html.Div("No per-task data.", className="app-subtitle")
    table = df.copy()
    table["Task"] = table["problem_id"]
    table["Method"] = table["method_label"]
    table["Best score"] = table.apply(
        lambda row: f"{row['best_speedup']:,.3f}" if row["valid"] else "invalid",
        axis=1,
    )
    table["Tokens"] = table["tokens"].apply(lambda value: f"{value:,.0f}")
    table["Calls"] = table["calls"].apply(lambda value: f"{value:,.0f}")
    table["Correct"] = table["correctness"].apply(lambda value: f"{value:.0%}")
    display = table[
        ["Task", "Method", "Best score", "Tokens", "Calls", "Correct"]
    ].sort_values(["Method", "Best score"], ascending=[True, False])
    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": column, "id": column} for column in display.columns],
        style_as_list_view=True,
        style_header={
            "backgroundColor": "#161b22",
            "color": "#8b949e",
            "fontWeight": "600",
            "border": "1px solid #30363d",
        },
        style_cell={
            "backgroundColor": "#0d1117",
            "color": "#c9d1d9",
            "border": "1px solid #21262d",
            "padding": "8px 10px",
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": "12px",
            "textAlign": "left",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": "{Best score} = invalid"},
                "color": "#f78166",
            }
        ],
        page_size=12,
        sort_action="native",
        style_table={"overflowX": "auto"},
    )


@app.callback(
    Output("token-scatter", "figure"),
    [
        Input("method-filter", "value"),
        Input("strat-filter", "value"),
        Input("feed-filter", "value"),
    ],
)
def update_token_scatter(methods, strategies, feedbacks):
    df = task_data(methods, strategies, feedbacks)
    if df.empty:
        return empty_figure("No token data")
    df = df[df["tokens"] > 0]
    if df.empty:
        return empty_figure("Token tracking was not available for this run")
    fig = go.Figure()
    for index, (method, subset) in enumerate(df.groupby("method_label")):
        fig.add_trace(
            go.Scatter(
                x=subset["tokens"],
                y=subset["best_speedup"],
                mode="markers",
                name=method,
                marker=dict(
                    size=np.clip(np.sqrt(subset["calls"].to_numpy() + 1) * 7, 8, 30),
                    color=PALETTE[index % len(PALETTE)],
                    line=dict(width=1, color="#0d1117"),
                    opacity=0.9,
                ),
                text=subset["problem_id"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "tokens: %{x:,.0f}<br>"
                    "best score: %{y:,.3f}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(text="Token cost vs performance", font=dict(size=18)),
        xaxis_title="Total tokens",
        yaxis_title="Best score / speedup",
        showlegend=True,
    )
    fig.update_xaxes(tickformat=",.0f")
    return style_figure(fig, height=430)


@app.callback(
    Output("token-trend", "figure"),
    [
        Input("method-filter", "value"),
        Input("strat-filter", "value"),
        Input("feed-filter", "value"),
    ],
)
def update_token_trend(methods, strategies, feedbacks):
    df = filter_frame(load_summary(), methods, strategies, feedbacks)
    if df.empty or "total_tokens" not in df.columns:
        return empty_figure("No token data")
    df = df.groupby(["method_name", "method_label", "iteration"], as_index=False).agg(
        prompt=("prompt_tokens", "sum"),
        completion=("completion_tokens", "sum"),
        total=("total_tokens", "sum"),
        calls=("llm_calls", "sum"),
    )
    if df["total"].sum() <= 0:
        return empty_figure("Token tracking was not available for this run")
    fig = go.Figure()
    for index, (label, subset) in enumerate(df.groupby("method_label")):
        subset = subset.sort_values("iteration")
        fig.add_trace(
            go.Bar(
                x=subset["iteration"],
                y=subset["total"],
                name=label,
                marker_color=PALETTE[index % len(PALETTE)],
                customdata=np.stack(
                    [subset["prompt"], subset["completion"], subset["calls"]], axis=-1
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "iteration %{x}<br>"
                    "total: %{y:,.0f}<br>"
                    "prompt: %{customdata[0]:,.0f}<br>"
                    "completion: %{customdata[1]:,.0f}<br>"
                    "calls: %{customdata[2]:,.0f}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(text="Token usage per iteration", font=dict(size=18)),
        xaxis_title="Iteration",
        yaxis_title="Tokens",
        barmode="group",
        showlegend=True,
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=",.0f")
    return style_figure(fig, height=430)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing experiment_results_*.csv files.",
    )
    parser.add_argument(
        "--version",
        default="optagent",
        help="Suffix used by collect_results.py.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir).resolve()
    VERSION = args.version
    app.layout = build_layout()
    app.run(host=args.host, port=args.port, debug=args.debug)
