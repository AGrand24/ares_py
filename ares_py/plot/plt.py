import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd

pio.templates.default = "plotly_dark"


def plt_meas(ert, clr="res"):
    df = ert.data.copy()

    if not clr in ["res"]:
        df = df.loc[df["res"] > 0]
    x = df["ld_hor"]
    y = df["z"]

    clr_def = {
        "res": {
            "data": df["res"],
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
            "data": np.log10(df["res"].clip(lower=1)),
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
        "ep": {
            "data": df["ep"],
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
        "std": {
            "data": df["std"],
            "cd": "%{cd[9]}",
            "cs": "Temps",
            "cmin": 0,
            "cmax": 1,
            "showscale": False,
            "colorbar": dict(
                tickvals=ert.cs_res["tickvals"],
                ticktext=ert.cs_res["ticktxt"],
            ),
            "visible": "legendonly",
        },
    }
    clr_def = clr_def[clr]

    cd = df.copy().fillna(-1)
    cd["i"] = (1000 / cd["i"]).round()

    round = [("res", 0), ("v", 0), ("ep", 0), ("i", 0), ("a", 0), ("n", 2), ("z", 0)]
    for col, d in round:
        cd[col] = np.round(cd[col], d)
        cd[col] = cd[col].astype(str).str.rstrip(".0")

    cd["std"] = np.round(cd["std"], 1)

    cd = cd.astype(str)
    cd = cd.values

    ht = [clr_def["cd"]]
    ht.append("%{x}, %{y}, %{cd[16]} ")
    ht.append("%{cd[0]} %{cd[1]} %{cd[2]} %{cd[3]}")
    ht.append("R: %{cd[5]}, V: %{cd[6]}, I: %{cd[7]}")
    ht.append("SP: %{cd[8]}, Vout: %{cd[10]}")
    ht.append("a: %{cd[13]}, n: %{cd[14]}")
    ht.append("std: %{cd[9]}, ch: %{cd[11]} %{cd[12]}")
    ht.append("ID: %{cd[17]}")
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


def plt_electrodes(ert, topo=False):

    xn = ert.sec[:, 0] + 1
    sec = ert.sec[:, 1] + 1
    n_sec = ert.sec[:, 2] + 1

    x = ert.sec[:, 0] * ert.el_space
    y = np.tile(4, xn.shape[0])

    cd = np.column_stack([x, y - 4, xn, sec, n_sec])
    ht = "x: %{cd[0]}, z: %{cd[1]}<br>n: %{cd[2]}, sec: %{cd[3]}<br>n sec: %{cd[4]}"
    ht = ht.replace("cd", "customdata")

    plt = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=7,
            color=sec,
            colorscale="Rainbow",
            line=dict(width=1, color="#6B6B6B"),
        ),
        name="electrodes",
        customdata=cd,
        hovertemplate=ht,
    )

    return plt
