from ares_py.plot.plt import plt_meas
import plotly.graph_objects as go


def fig_meas_data(ert):
    plt = [plt_meas(ert), plt_meas(ert, clr="res_log"), plt_meas(ert, clr="sp")]
    fig = go.Figure()
    fig.add_traces(plt)
    fig.update_yaxes(scaleanchor="x1", scaleratio=1)
    fig.update_layout(
        xaxis=dict(
            showgrid=True, dtick=ert.el_space * 2, side="top", showticklabels=True
        ),
        yaxis1=dict(showgrid=True, dtick=ert.el_space * 2),
    )
    fig.show()
    fig.update_layout(width=1600, height=900)
    fig.write_html("tmp/test.html")
