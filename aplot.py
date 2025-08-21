import plotly.graph_objects as go
import numpy as np


def aplot_scatter(data, mode="markers", size=2):
    fig = go.Figure()
    for d in data:
        x = d[:, 0]
        y = d[:, 1]
        plt = go.Scatter(x=x, y=y, mode=mode, marker=dict(size=size))

    fig.add_traces(plt)
    fig.update_yaxes(scaleanchor="x1")
    return fig
