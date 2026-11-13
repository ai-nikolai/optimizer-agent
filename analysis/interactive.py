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
RUN_LABEL = "optagent"

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
    "combined_score": "Combined score",
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
    "#7aa2f7",
    "#bb9af7",
    "#9ece6a",
    "#e0af68",
    "#7dcfff",
    "#f7768e",
    "#ff9e64",
    "#73daca",
]
GRAPH_CONFIG = {"displayModeBar": False, "responsive": True}

APP_CSS = """
body { background: #0b0f14; color: #e6edf3; font-family: Inter, system-ui, sans-serif; line-height: 1.5; }
.app-title { font-size: 32px; font-weight: 750; letter-spacing: -0.6px; margin: 0; color: #f0f6fc; }
.app-subtitle { color: #9aa4b2; margin: 6px 0 0; font-size: 14px; }
.section-title { color: #e6edf3; font-size: 14px; font-weight: 700; letter-spacing: .02em; }
.kpi-label { color: #9aa4b2; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }
.kpi-value { font-size: 30px; font-weight: 750; color: #f0f6fc; margin-top: 6px; line-height: 1.2; }
.kpi-sublabel { color: #6b7280; font-size: 12px; margin-top: 2px; min-height: 16px; }
.control-label { color: #9aa4b2; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px; }
.dash-dropdown-wrapper { background-color: #121821 !important; color: #e6edf3 !important; }
.dash-dropdown-trigger { background-color: #121821 !important; border: 1px solid #2a3a4f !important; border-radius: 10px !important; min-height: 42px !important; color: #e6edf3 !important; box-shadow: none !important; }
.dash-dropdown-trigger:hover, .dash-dropdown-trigger:focus-within { border-color: #7aa2f7 !important; box-shadow: 0 0 0 1px rgba(122,162,247,0.35) !important; }
.dash-dropdown-value { background-color: transparent !important; color: #e6edf3 !important; }
.dash-dropdown-placeholder { color: #6b7280 !important; }
.dash-dropdown-value-item { background-color: #1f2937 !important; border: 1px solid #2a3a4f !important; color: #e6edf3 !important; border-radius: 6px !important; }
.dash-dropdown-content { background-color: #121821 !important; border: 1px solid #2a3a4f !important; border-radius: 10px !important; box-shadow: 0 18px 40px rgba(0,0,0,0.45) !important; overflow: hidden !important; }
.dash-dropdown-options { background-color: #121821 !important; }
.dash-dropdown-option { background-color: #121821 !important; color: #e6edf3 !important; padding: 9px 12px !important; cursor: pointer !important; }
.dash-dropdown-option:hover, .dash-dropdown-option[aria-selected="true"], .dash-dropdown-option[data-selected="true"] { background-color: #1f2937 !important; color: #f0f6fc !important; }
.dash-dropdown-search-container, .dash-dropdown-search { background-color: #0b0f14 !important; color: #e6edf3 !important; }
.dash-dropdown-search input { background-color: #0b0f14 !important; color: #e6edf3 !important; }
.dash-dropdown-clear, .dash-dropdown-trigger-icon, .dash-dropdown-action-button { color: #9aa4b2 !important; }
.dash-checklist label, .dash-radio-items label { color: #d6dee9 !important; }
.Select-control { background-color: #121821 !important; border-color: #2a3a4f !important; color: #e6edf3 !important; }
.Select-menu-outer { background-color: #121821 !important; border: 1px solid #2a3a4f !important; color: #e6edf3 !important; }
.Select-option { background-color: #121821 !important; color: #e6edf3 !important; }
.Select-option:hover, .Select-option.is-focused { background-color: #1f2937 !important; }
.Select-value-label, .Select-placeholder { color: #e6edf3 !important; }
.tab-container { border-bottom: 1px solid #243044 !important; }
.tab { background: transparent !important; border: none !important; color: #9aa4b2 !important; }
.tab--selected { border-top: 2px solid #7aa2f7 !important; color: #f0f6fc !important; }
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
    "backgroundColor": "#121821",
    "border": "1px solid #243044",
    "borderRadius": "16px",
    "boxShadow": "0 14px 34px rgba(0,0,0,0.22)",
}
CONTROL_STYLE = {
    "backgroundColor": "#121821",
    "border": "1px solid #243044",
    "borderRadius": "16px",
    "padding": "22px 24px",
}
CHART_CARD_STYLE = {
    **CARD_STYLE,
    "padding": "8px",
}
HEADER_STYLE = {
    "background": "linear-gradient(135deg, rgba(122,162,247,0.16), rgba(187,154,247,0.05))",
    "border": "1px solid #243044",
    "borderRadius": "18px",
    "padding": "24px 28px",
}
TAB_STYLE = {
    "padding": "12px 20px",
    "backgroundColor": "transparent",
    "border": "none",
    "color": "#9aa4b2",
    "fontWeight": "600",
    "fontSize": "14px",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": "#f0f6fc",
    "borderTop": "2px solid #7aa2f7",
    "backgroundColor": "rgba(122,162,247,0.06)",
    "borderRadius": "8px 8px 0 0",
}
BADGE_STYLE = {
    "backgroundColor": "#1f2937",
    "color": "#e6edf3",
    "border": "1px solid #243044",
    "padding": "6px 10px",
    "fontWeight": "600",
    "fontSize": "12px",
}
CHART_STYLE = {"height": "600px"}
SMALL_CHART_STYLE = {"height": "440px"}


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


def load_lineage():
    path = DATA_DIR / f"lineage_{VERSION}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for column in [
        "child_iteration",
        "child_sample",
        "child_score",
        "child_tokens",
        "parent_iteration",
        "parent_sample",
        "parent_score",
        "delta",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def lineage_task_options():
    df = load_lineage()
    if df is None or df.empty:
        return []
    return [
        {"label": task, "value": task}
        for task in sorted(df["task"].dropna().unique())
    ]


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


def kpi_card(label, value, sublabel="", accent="#7aa2f7"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="kpi-label"),
                html.Div(value, className="kpi-value"),
                html.Div(sublabel, className="kpi-sublabel"),
            ],
            className="p-0",
        ),
        style={
            **CARD_STYLE,
            "borderTop": f"3px solid {accent}",
            "padding": "18px 20px",
        },
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
        font=dict(
            color="#d6dee9",
            family="Inter, system-ui, sans-serif",
            size=12,
        ),
        margin=dict(l=64, r=32, t=68, b=96),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#121821",
            bordercolor="#243044",
            font=dict(color="#e6edf3", size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(18,24,33,0.65)",
            bordercolor="#243044",
            borderwidth=1,
            font=dict(size=12),
        ),
        height=height,
    )
    fig.update_layout(title=dict(x=0.01, xanchor="left"))
    fig.update_xaxes(
        gridcolor="#1b2532",
        zerolinecolor="#243044",
        automargin=True,
        title_font=dict(size=13),
        tickfont=dict(size=12),
    )
    fig.update_yaxes(
        gridcolor="#1b2532",
        zerolinecolor="#243044",
        automargin=True,
        title_font=dict(size=13),
        tickfont=dict(size=12),
    )
    return fig


def build_layout():
    tasks, iterations, best, tokens, calls = kpi_values()
    method_values = [option["value"] for option in method_options()]
    lineage_options = lineage_task_options()
    lineage_default = lineage_options[0]["value"] if lineage_options else None
    best_text = f"{best:,.3f}" if best is not None else "—"
    token_text = f"{tokens:,.0f}" if tokens > 0 else "—"
    calls_text = f"{calls:,.0f}" if calls > 0 else "—"

    return dbc.Container(
        fluid=True,
        style={"maxWidth": "1560px", "padding": "32px 36px 48px"},
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
                            dbc.Badge(f"run {RUN_LABEL}", style=BADGE_STYLE, className="me-2"),
                            dbc.Badge(f"{tasks} tasks", style=BADGE_STYLE, className="me-2"),
                            dbc.Badge(f"{iterations} iterations", style=BADGE_STYLE),
                        ],
                        md=4,
                        className="d-flex align-items-center justify-content-end gap-2",
                    ),
                ],
                style=HEADER_STYLE,
                className="mb-4 align-items-center g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        kpi_card("Tasks", f"{tasks:,}", "evaluated", "#7aa2f7"),
                        md=2,
                    ),
                    dbc.Col(
                        kpi_card("Iterations", f"{iterations:,}", "search rounds", "#bb9af7"),
                        md=2,
                    ),
                    dbc.Col(
                        kpi_card("Best score", best_text, "combined score", "#9ece6a"),
                        md=3,
                    ),
                    dbc.Col(
                        kpi_card(
                            "Total tokens",
                            token_text,
                            "prompt + completion",
                            "#e0af68",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        kpi_card(
                            "LLM calls",
                            calls_text,
                            "successful requests",
                            "#7dcfff",
                        ),
                        md=2,
                    ),
                ],
                className="g-4 mb-4",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Controls", className="section-title"),
                            html.Div(
                                "Filter the run and choose how to read the results.",
                                className="app-subtitle",
                            ),
                        ],
                        className="mb-3",
                    ),
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
                        className="g-4",
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
                                    "Tip: Tasks, Tokens and Lineage provide per-task and parent→child breakdowns.",
                                    className="app-subtitle mt-2",
                                ),
                                md=7,
                            ),
                        ],
                        className="g-4 mt-1",
                    ),
                    html.Div(
                        "Combined score is evaluator-defined and higher-is-better: "
                        "AlgoTune uses speedup; KernelBench legacy uses negative runtime; "
                        "hardened KernelBench uses speedup.",
                        className="app-subtitle mt-3",
                    ),
                ],
                style=CONTROL_STYLE,
                className="mb-4",
            ),
            dcc.Tabs(
                id="tabs",
                value="trends",
                parent_style={
                    "borderBottom": "1px solid #243044",
                    "marginBottom": "20px",
                },
                children=[
                    dcc.Tab(
                        label="Trends",
                        value="trends",
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                        children=[
                            dbc.Card(
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="trend-chart",
                                        config=GRAPH_CONFIG,
                                        style=CHART_STYLE,
                                    ),
                                    className="p-2",
                                ),
                                style=CHART_CARD_STYLE,
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Tasks",
                        value="tasks",
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    id="task-chart",
                                                    config=GRAPH_CONFIG,
                                                    style=CHART_STYLE,
                                                ),
                                                className="p-2",
                                            ),
                                            style=CHART_CARD_STYLE,
                                        ),
                                        lg=7,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                html.Div(id="task-table"),
                                                className="p-3",
                                            ),
                                            style=CHART_CARD_STYLE,
                                        ),
                                        lg=5,
                                    ),
                                ],
                                className="g-4 mt-1",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Tokens",
                        value="tokens",
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    id="token-scatter",
                                                    config=GRAPH_CONFIG,
                                                    style=SMALL_CHART_STYLE,
                                                ),
                                                className="p-2",
                                            ),
                                            style=CHART_CARD_STYLE,
                                        ),
                                        lg=6,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    id="token-trend",
                                                    config=GRAPH_CONFIG,
                                                    style=SMALL_CHART_STYLE,
                                                ),
                                                className="p-2",
                                            ),
                                            style=CHART_CARD_STYLE,
                                        ),
                                        lg=6,
                                    ),
                                ],
                                className="g-4 mt-1",
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Lineage",
                        value="lineage",
                        style=TAB_STYLE,
                        selected_style=TAB_SELECTED_STYLE,
                        children=[
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div("Task", className="control-label"),
                                            dcc.Dropdown(
                                                id="lineage-task",
                                                options=lineage_options,
                                                value=lineage_default,
                                                clearable=False,
                                                placeholder="Select a task",
                                            ),
                                        ],
                                        md=4,
                                        className="mb-3",
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            "Parent → child candidate links, with score deltas.",
                                            className="app-subtitle mt-4",
                                        ),
                                        md=8,
                                    ),
                                ],
                                className="g-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    id="lineage-graph",
                                                    config=GRAPH_CONFIG,
                                                    style=CHART_STYLE,
                                                ),
                                                className="p-2",
                                            ),
                                            style=CHART_CARD_STYLE,
                                        ),
                                        lg=7,
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                html.Div(id="lineage-table"),
                                                className="p-3",
                                            ),
                                            style=CHART_CARD_STYLE,
                                        ),
                                        lg=5,
                                    ),
                                ],
                                className="g-4 mt-1",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                f"Source: {DATA_DIR} · files {VERSION} · {RUN_LABEL}",
                className="app-subtitle mt-5",
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
                    marker=dict(size=7, line=dict(width=1, color="#0b0f14")),
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
    method_labels = list(valid["method_label"].unique())
    method_colors = {
        label: PALETTE[index % len(PALETTE)]
        for index, label in enumerate(method_labels)
    }
    fig = go.Figure(
        go.Bar(
            x=valid["best_speedup"],
            y=valid["problem_id"],
            orientation="h",
            marker=dict(
                color=[method_colors[label] for label in valid["method_label"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>combined score: %{x:,.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Best combined score per task", font=dict(size=18)),
        xaxis_title="Best combined score",
        yaxis_title="",
        showlegend=False,
        bargap=0.38,
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
    table["Combined score"] = table.apply(
        lambda row: f"{row['best_speedup']:,.3f}" if row["valid"] else "invalid",
        axis=1,
    )
    table["Tokens"] = table["tokens"].apply(lambda value: f"{value:,.0f}")
    table["Calls"] = table["calls"].apply(lambda value: f"{value:,.0f}")
    table["Correct"] = table["correctness"].apply(lambda value: f"{value:.0%}")
    display = table[
        ["Task", "Method", "Combined score", "Tokens", "Calls", "Correct"]
    ].sort_values(["Method", "Combined score"], ascending=[True, False])
    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": column, "id": column} for column in display.columns],
        style_as_list_view=True,
        style_header={
            "backgroundColor": "#121821",
            "color": "#9aa4b2",
            "fontWeight": "700",
            "fontSize": "12px",
            "border": "1px solid #243044",
        },
        style_cell={
            "backgroundColor": "#0b0f14",
            "color": "#d6dee9",
            "border": "1px solid #1b2532",
            "padding": "9px 11px",
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": "12.5px",
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
                    line=dict(width=1, color="#0b0f14"),
                    opacity=0.9,
                ),
                text=subset["problem_id"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "tokens: %{x:,.0f}<br>"
                    "combined score: %{y:,.3f}"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(text="Token cost vs performance", font=dict(size=18)),
        xaxis_title="Total tokens",
        yaxis_title="Best combined score",
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
        bargap=0.28,
        showlegend=True,
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=",.0f")
    return style_figure(fig, height=430)


@app.callback(
    Output("lineage-graph", "figure"),
    [Input("lineage-task", "value")],
)
def update_lineage_graph(task):
    df = load_lineage()
    if df is None or df.empty:
        return empty_figure(
            "No lineage data. Generate it with: python analysis/collect_lineage.py "
            "--experiments-dir <run-dir> --version <version>"
        )
    if task:
        df = df[df["task"] == task]
    if df.empty:
        return empty_figure("No lineage rows for this task")

    df = df.sort_values(["child_iteration", "child_sample"]).copy()
    edge_x, edge_y = [], []
    for row in df.itertuples():
        if (
            pd.notna(row.parent_iteration)
            and pd.notna(row.parent_score)
            and pd.notna(row.child_score)
        ):
            edge_x.extend([row.parent_iteration, row.child_iteration, None])
            edge_y.extend([row.parent_score, row.child_score, None])

    fig = go.Figure()
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="#30363d", width=1.6),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    valid = df["child_valid"].astype(str).str.lower().isin(["true", "1"])
    colors = ["#9ece6a" if value else "#f7768e" for value in valid]
    custom = list(
        zip(
            df["task"],
            df["child_iteration"],
            df["child_sample"],
            df["parent_iteration"].fillna(-1),
            df["parent_sample"].fillna(-1),
            df["delta"].fillna(0.0),
            df["match"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["child_iteration"],
            y=df["child_score"],
            mode="markers",
            marker=dict(
                size=11,
                color=colors,
                line=dict(width=1, color="#0b0f14"),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "child: iteration %{customdata[1]} · sample %{customdata[2]}<br>"
                "score: %{y:.3f}<br>"
                "parent: iteration %{customdata[3]} · sample %{customdata[4]}<br>"
                "delta: %{customdata[5]:+.3f}<br>"
                "match: %{customdata[6]}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(text=f"Candidate lineage · {task}", font=dict(size=18)),
        xaxis_title="Iteration",
        yaxis_title="Combined score",
        showlegend=False,
    )
    fig.update_xaxes(dtick=1)
    scores = finite_values(df["child_score"])
    if not scores.empty and scores.min() >= 0 and scores.max() > 1.0:
        fig.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="#8b949e",
            annotation_text="baseline parity",
            annotation_position="top left",
        )
    fig = style_figure(fig, height=560)
    fig.update_layout(hovermode="closest")
    return fig


@app.callback(
    Output("lineage-table", "children"),
    [Input("lineage-task", "value")],
)
def update_lineage_table(task):
    df = load_lineage()
    if df is None or df.empty:
        return html.Div(
            "No lineage data. Run `analysis/collect_lineage.py` for this run.",
            className="app-subtitle",
        )
    if task:
        df = df[df["task"] == task]
    if df.empty:
        return html.Div("No lineage rows for this task.", className="app-subtitle")

    display = df.sort_values(["child_iteration", "child_sample"]).copy()
    display["Child"] = display.apply(
        lambda row: f"it {int(row['child_iteration'])} · s {int(row['child_sample'])}",
        axis=1,
    )
    display["Child score"] = display["child_score"].apply(
        lambda value: f"{value:,.3f}" if pd.notna(value) else "—"
    )
    display["Parent"] = display.apply(
        lambda row: (
            "baseline"
            if row["match"] == "baseline"
            else f"it {int(row['parent_iteration'])} · s {int(row['parent_sample'])}"
            if pd.notna(row["parent_iteration"]) and pd.notna(row["parent_sample"])
            else "—"
        ),
        axis=1,
    )
    display["Parent score"] = display["parent_score"].apply(
        lambda value: f"{value:,.3f}" if pd.notna(value) else "—"
    )
    display["Δ score"] = display["delta"].apply(
        lambda value: f"{value:+.3f}" if pd.notna(value) else "—"
    )
    display["Match"] = display["match"]
    display["Tokens"] = display["child_tokens"].apply(
        lambda value: f"{value:,.0f}" if pd.notna(value) else "—"
    )
    columns = [
        "Child",
        "Child score",
        "Parent",
        "Parent score",
        "Δ score",
        "Match",
        "Tokens",
    ]
    return dash_table.DataTable(
        data=display[columns].to_dict("records"),
        columns=[{"name": column, "id": column} for column in columns],
        style_as_list_view=True,
        style_header={
            "backgroundColor": "#121821",
            "color": "#9aa4b2",
            "fontWeight": "700",
            "fontSize": "12px",
            "border": "1px solid #243044",
        },
        style_cell={
            "backgroundColor": "#0b0f14",
            "color": "#d6dee9",
            "border": "1px solid #1b2532",
            "padding": "9px 11px",
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": "12.5px",
            "textAlign": "left",
        },
        page_size=12,
        sort_action="native",
        style_table={"overflowX": "auto"},
    )


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
    parser.add_argument(
        "--label",
        default=None,
        help="Display label for the run (defaults to --version).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir).resolve()
    VERSION = args.version
    RUN_LABEL = args.label or args.version
    app.title = f"optimizer-agent · {RUN_LABEL}"
    app.layout = build_layout()
    app.run(host=args.host, port=args.port, debug=args.debug)
