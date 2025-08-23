import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

pio.templates.default = "plotly_dark"


def plt_meas(ert, topo=False, clr="res"):

    clr_def = {
        "res": {
            "data": ert.data["res"],
            "cd": "%{cd[5]}",
            "cs": ert.cs_res["color_scale"],
            "cmin": ert.cs_res["crange"][0],
            "cmax": ert.cs_res["crange"][1],
            "showscale": False,
            "colorbar": dict(
                tickvals=ert.cs_res["tickvals"],
                ticktext=ert.cs_res["ticktxt"],
            ),
            "visible": True,
        },
        "res_log": {
            "data": np.log10(ert.data["res"].clip(lower=1)),
            "cd": "%{cd[5]}",
            "cs": "Turbo",
            "cmin": None,
            "cmax": None,
            "showscale": False,
            "colorbar": dict(
                tickvals=ert.cs_res["tickvals"],
                ticktext=ert.cs_res["ticktxt"],
            ),
            "visible": "legendonly",
        },
        "sp": {
            "data": ert.data["ep"],
            "cd": "%{cd[8]}",
            "cs": "RdBu_r",
            "cmin": -500,
            "cmax": 500,
            "showscale": False,
            "colorbar": dict(
                tickvals=ert.cs_res["tickvals"],
                ticktext=ert.cs_res["ticktxt"],
            ),
            "visible": "legendonly",
        },
    }
    clr_def = clr_def[clr]

    if topo == False:
        x = ert.xm
        y = ert.doi

    cd = ert.data.copy().fillna(-1)
    cd["i"] = (1000 / cd["i"]).round()
    cd[["v", "ep"]] = cd[["v", "ep"]].round()
    cd = cd.values

    ht = "%{x}, %{y}, "
    ht += clr_def["cd"]
    ht = [ht]
    ht.append("R: %{cd[5]}, V: %{cd[6]}, I: %{cd[7]}")
    ht.append("SP: %{cd[8]}, Vout: %{cd[10]}")
    ht.append("a: %{cd[13]}, n: %{cd[14]}")
    ht = ("<br>").join(ht)
    ht = ht.replace("cd[", "customdata[")

    marker = dict(
        color=clr_def["data"],
        colorscale=clr_def["cs"],
        size=10,
        symbol="square",
        cmin=clr_def["cmin"],
        cmax=clr_def["cmax"],
        showscale=clr_def["showscale"],
        colorbar=clr_def["colorbar"],
    )
    plt = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        customdata=cd,
        hovertemplate=ht,
        name=clr,
        marker=marker,
        visible=clr_def["visible"],
        showlegend=True,
    )
    return plt
