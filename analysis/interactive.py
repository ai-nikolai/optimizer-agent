
"""Interactive viewer for optimizer-agent experiment summaries.

Reads ``experiment_results_summary_<version>.csv`` and
``experiment_results_summary_agg_<version>.csv`` from ``--data-dir``.
Build those files from optimizer-agent runs with ``collect_results.py``.

Run:
    python analysis/collect_results.py --experiments-dir experiments
    python analysis/interactive.py --data-dir experiments --version optagent
"""

import argparse
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "optimizer-agent Interactive Metric Viewer"

# Loaded from CLI arguments.
DATA_DIR = Path(".")
VERSION = "v4_2_0_0"
metrics = ['compiled', 'loading_success', 'correctness', 'runtime_custom_ms', 'runtime_original_ms', 'combined_score', 'fast_1', 'fast_2', 'new_combined_score', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'llm_calls']

# Parse method components
def parse_method(method_name):
    # Expects format like "met=bestN_last__is=1__samp=3__el=3__strat=True__feed=False"
    parts = method_name.split('__')
    params = {}
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            params[key] = value
    return params

def load_data(aggregated=False):
    if aggregated:
        name = f"experiment_results_summary_agg_{VERSION}.csv"
    else:
        name = f"experiment_results_summary_{VERSION}.csv"

    df = pd.read_csv(DATA_DIR / name)

    # Extract filterable attributes from method name
    df['params'] = df['method_name'].apply(parse_method)
    df['met'] = df['params'].apply(lambda x: x.get('met', ''))
    df['is'] = df['params'].apply(lambda x: x.get('is', ''))
    df['samp'] = df['params'].apply(lambda x: x.get('samp', ''))
    df['el'] = df['params'].apply(lambda x: x.get('el', ''))
    df['strat'] = df['params'].apply(lambda x: x.get('strat', ''))
    df['feed'] = df['params'].apply(lambda x: x.get('feed', ''))

    # Define metrics to display
    return df

app.layout = html.Div([
    html.H1("optimizer-agent Interactive Metric Viewer"),
    # Create a modern, card-based layout with boxes around inputs
    html.Div([
        html.H4("Filters", style={'color': '#ffffff', 'margin-bottom': '10px'}),
        html.Div([
            html.Div([
            html.Label("Metric", style={'font-weight': 'bold', 'color': '#ecf0f1'}),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[{'label': m, 'value': m} for m in metrics],
        value=metrics[0],
                clearable=False,
                style={'background-color': '#34495e', 'color': '#ecf0f1'}
                )
            ], style={'flex': '1', 'margin-right': '10px'}),
            html.Div([
            html.Label("Strategy Filter", style={'font-weight': 'bold', 'color': '#ecf0f1'}),
    dcc.RadioItems(
        id='strat-filter',
        options=[
            {'label': 'strat=True', 'value': 'True'},
            {'label': 'strat=False', 'value': 'False'},
            {'label': 'both', 'value': 'both'}
        ],
                value='both',
                labelStyle={'display': 'block', 'color': '#ecf0f1'}
                )
            ], style={'flex': '1', 'margin-right': '10px'}),
            html.Div([
            html.Label("Feedback Filter", style={'font-weight': 'bold', 'color': '#ecf0f1'}),
    dcc.RadioItems(
        id='feed-filter',
        options=[
            {'label': 'feed=True', 'value': 'True'},
            {'label': 'feed=False', 'value': 'False'},
            {'label': 'both', 'value': 'both'}
        ],
                value='both',
                labelStyle={'display': 'block', 'color': '#ecf0f1'}
                )
            ], style={'flex': '1', 'margin-right': '10px'}),
            html.Div([
            html.Label("Aggregation Filter", style={'font-weight': 'bold', 'color': '#ecf0f1'}),
    dcc.RadioItems(
        id='agg-filter',
        options=[
            {'label': 'Aggregated', 'value': 'agg'},
            {'label': 'Individual', 'value': 'ind'},
        ],
                value='agg',
                labelStyle={'display': 'block', 'color': '#ecf0f1'}
                )
            ], style={'flex': '1', 'margin-right': '10px'}),
            html.Div([
            html.Label("Color Grouping", style={'font-weight': 'bold', 'color': '#ecf0f1'}),
    dcc.RadioItems(
        id='color-grouping',
        options=[
            {'label': 'Methods', 'value': 'methods'},
            {'label': 'Strategy', 'value': 'strategy'},
            {'label': 'Feedback', 'value': 'feedback'},
            {'label': 'All', 'value': 'all'}
        ],
                value='all',
                labelStyle={'display': 'block', 'color': '#ecf0f1'}
                )
            ], style={'flex': '1', 'margin-right': '10px'}),
            html.Div([
            html.Label("Method Selection", style={'font-weight': 'bold', 'color': '#ecf0f1'}),
    dcc.Checklist(
        id='method-filter',
        options=[
            {'label': 'bestN_last', 'value': 'bestN_last'},
            {'label': 'bestN_all', 'value': 'bestN_all'},
            {'label': 'map_elites', 'value': 'map_elites'},
            {'label': 'all', 'value': 'all'}
        ],
                value=['all'],
                labelStyle={'display': 'block', 'color': '#ecf0f1'}
                )
            ], style={'flex': '1', 'margin-right': '10px'}),    
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'padding': '15px', 'border': '1px solid #3498db', 'border-radius': '5px', 'margin-bottom': '20px'})
    ]),
    dcc.Graph(id='line-chart')
])

@app.callback(
    Output('line-chart', 'figure'),
        [
            Input('metric-dropdown', 'value'),
            Input('strat-filter', 'value'),
            Input('feed-filter', 'value'),
            Input('method-filter', 'value'),
            Input('agg-filter', 'value'),
            Input('color-grouping', 'value')
        ]
    )
def update_graph(selected_metric, strat_val, feed_val, met_val, agg_val, color_group):
    if agg_val=="agg":
        filtered_df = load_data(True)
    else:
        filtered_df = load_data(False)

    if 'all' not in (met_val or []):
        filtered_df = filtered_df[filtered_df['met'].isin(met_val)]

    if strat_val != 'both':
        filtered_df = filtered_df[filtered_df['strat'] == strat_val]

    if feed_val != 'both':
        filtered_df = filtered_df[filtered_df['feed'] == feed_val]
    
    # Determine x and y axis max values based on the selected metric
    if 'iteration' in filtered_df.columns and not filtered_df.empty:
        x_max = float(filtered_df['iteration'].max()) + 1
    else:
        x_max = 6
    y_max = filtered_df[selected_metric].max()*1.1 if selected_metric in filtered_df.columns else 1

    # Create figure
    fig = go.Figure()
    color_map = {
        "False" : "#d62728",  # red
        "True" : "#1f77b4",   # blue
        "bestN_last" : "#ff7f0e",  # orange
        "bestN_all" : "#2ca02c",   # green
        "map_elites" : "#9467bd",  # purple
    }
    for method in filtered_df['method_name'].unique():
        method_data = filtered_df[filtered_df['method_name'] == method]
        method_params = parse_method(method)

        # Determine color based on selected grouping
        if color_group == 'methods':
            group_key = method_params['met']
        elif color_group == 'strategy':
            group_key = method_params['strat']
        elif color_group == 'feedback':
            group_key = method_params['feed']
        else:  # all
            group_key = method

        # Assign color per group
        if group_key not in color_map:
            color_map[group_key] = None  # will be assigned by plotly
        color = color_map[group_key]

        fig.add_trace(go.Scatter(
            x=method_data['iteration'],
            y=method_data[selected_metric],
            mode='lines+markers',
            name=method,
            line=dict(color=color)
        ))

    fig.update_layout(
        title=f"{selected_metric} vs. Iteration",
        xaxis_title="Iteration",
        yaxis_title=selected_metric,
        hovermode='closest',
    )

    # Update layout to set the x and y axis ranges
    fig.update_layout(
        xaxis=dict(range=[0, x_max]),
        yaxis=dict(range=[0, y_max])
    )

    # Display the color_map when color_group is not 'all'
    # Create an annotation box to display the color mapping
    if color_group != "all":
        color_boxes = []
        for group, color in color_map.items():
            if color is not None:
                color_boxes.append(f"<span style='color:{color}'>&#9632;</span> {group}")

        legend_text = "<br>".join(color_boxes)

        fig.add_annotation(
            text=legend_text,
            xref="paper", yref="paper",
            x=1.0, y=1.0,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            xanchor="right", yanchor="top",
            borderpad=5
        )
        # # Optionally, add an annotation with the color map
        # fig.add_annotation(
        #     text="Color Mapping:<br>" + "<br>".join([f"{k}: {v}" for k, v in color_map.items() if v is not None]),
        #     xref="paper", yref="paper",
        #     x=1.02, y=1.0,
        #     showarrow=False,
        #     align="left",
        #     bgcolor="rgba(255,255,255,0.8)",
        #     bordercolor="black",
        #     borderpad=4
        # )

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing experiment_results_summary_*.csv files.",
    )
    parser.add_argument(
        "--version",
        default="v4_2_0_0",
        help="Suffix used by collect_results.py (default: v4_2_0_0).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir).resolve()
    VERSION = args.version
    app.run(host=args.host, port=args.port, debug=args.debug)