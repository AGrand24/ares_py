import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from ares_py.plot.plt import plt_meas, plt_electrodes, plt_dtm, plt_z_check
from plotly.subplots import make_subplots


def fig_meas_data(ert):
    dtick = get_dtick(ert)
    plt = [
        plt_meas(ert),
        plt_meas(ert, clr="res_log"),
        plt_meas(ert, clr="ep"),
        plt_meas(ert, clr="std"),
    ]
    fig = go.Figure()

    # fig.add_trace(plt_electrodes(ert))
    fig.add_traces(plt)

    plt = plt_z_check(ert, visible="legendonly")[:2]
    fig.add_traces(plt)

    fig.update_yaxes(scaleanchor="x1", scaleratio=1)
    fig.update_layout(
        xaxis=dict(showgrid=True, dtick=dtick, side="top", showticklabels=True),
        yaxis=dict(showgrid=True, dtick=dtick),
    )

    fig.update_layout(width=1600, height=900)
    fp_out = Path(ert.fp_load).name.replace(".2dm", ".html")
    fp_out = "output/" + fp_out
    fig.write_html(fp_out)
    return fig


def get_dtick(ert):
    dtick = ert.el_space
    k = 1 + (np.nanmax(ert.data.iloc[:, :4]) // 100)
    dtick *= k
    return dtick


def fig_dtm(dtm):
    plt = plt_dtm(dtm)
    fig = go.Figure()

    fig = fig.add_trace(plt)
    fig.update_yaxes(
        scaleanchor="x1",
        scaleratio=1,
    )
    return fig


def fig_z_check(ert):
    plt = plt_z_check(ert)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = fig.add_traces(plt[:2])
    fig = fig.add_trace(plt[2], secondary_y=True)
    fig = fig.add_trace(plt[3], secondary_y=True)

    fig = fig.update_yaxes(secondary_y=False, scaleanchor="x1", scaleratio=1)
    fig = fig.update_yaxes(secondary_y=True, scaleanchor="x1", scaleratio=5)
    return fig
