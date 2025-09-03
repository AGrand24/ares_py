import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from ares_py.plot.plt import (
    plt_meas,
    plt_electrodes,
    plt_dtm,
    plt_topo,
    plt_meas_3d,
    plt_dtm_3d,
    plt_sec_3d,
)
from plotly.subplots import make_subplots
from ares_py.geometry.dtm import dtm_clip_3d


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

    plt = plt_topo(ert, visible="legendonly")[:2]
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


def fig_meas_data_3d(data, sec, cs_res, dtm, fp_out):
    plt = [
        plt_meas_3d(data, cs_res),
        plt_meas_3d(data, cs_res, clr="res_log"),
        plt_meas_3d(data, cs_res, clr="ep"),
        plt_meas_3d(data, cs_res, clr="std"),
    ]
    fig = go.Figure()

    # fig.add_trace(plt_electrodes(ert))
    fig.add_traces(plt)

    df_dtm = dtm_clip_3d(dtm, data)

    plt = plt_dtm_3d(df_dtm)
    fig = fig.add_trace(plt)

    plt = plt_sec_3d(sec)
    fig = fig.add_trace(plt)

    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))

    camera = dict(eye=dict(x=1, y=1, z=1))

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.update_layout(width=1600, height=900)
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
    plt = plt_topo(ert)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = fig.add_traces(plt[:2])
    fig = fig.add_trace(plt[2], secondary_y=True)
    fig = fig.add_trace(plt[3], secondary_y=True)

    fig = fig.update_yaxes(secondary_y=False, scaleanchor="x1", scaleratio=20)
    fig = fig.update_yaxes(secondary_y=True, scaleanchor="x1", scaleratio=5)
    return fig
