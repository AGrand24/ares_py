from ares_py.plot.plt import plt_meas, plt_electrodes
import plotly.graph_objects as go


def fig_meas_data(ert):
    dtick = get_dtick(ert)
    plt = [
        plt_meas(ert),
        plt_meas(ert, clr="res_log"),
        plt_meas(ert, clr="ep"),
        plt_meas(ert, clr="std"),
    ]
    fig = go.Figure()

    fig.add_trace(plt_electrodes(ert))
    fig.add_traces(plt)

    fig.update_yaxes(scaleanchor="x1", scaleratio=1)
    fig.update_layout(
        xaxis=dict(showgrid=True, dtick=dtick, side="top", showticklabels=True),
        yaxis1=dict(showgrid=True, dtick=dtick),
    )
    fig.show()

    fig.update_layout(width=1600, height=900)
    fig.write_html("tmp/test.html")


def get_dtick(ert):
    dtick = ert.el_space

    k = 1 + (ert.el.max() // 100)
    dtick *= k
    return dtick
