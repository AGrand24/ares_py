import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"


def plt_meas(ert, topo=False):
    if topo == False:
        x = ert.x_meas
        y = ert.doi

    hovertxt = ert.meas[:, 0]
    colorbar = dict(tickvals=ert.cs.tickvals, ticktext=ert.cs.ticktext)

    marker = dict(
        color=ert.meas[:, 0],
        colorscale=ert.cs_res["color_scale"],
        size=10,
        symbol="square",
        cmin=ert.cs_res["crange"][0],
        cmax=ert.cs_res["crange"][0],
        showscale=True,
        colorbar=colorbar,
    )

    plt = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        hovertext=hovertxt,
        name=ert.fp_load,
        marker=marker,
    )
    return plt
